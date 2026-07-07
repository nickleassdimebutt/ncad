#!/usr/bin/env python3
"""topo.py — 3D SIMP topology optimization for genloop's structural engine.

Deterministic, dependency-light (numpy/scipy/scikit-image/matplotlib).
Classic compliance minimization: H8 voxel elements, SIMP penalization,
density filtering, optimality-criteria updates.

Usage:  python topo.py <config.json> <outdir>

Config:
{
  "domain_mm":  [120, 40, 60],
  "resolution": [60, 20, 30],          # elements per axis (voxels ~cubic)
  "volfrac": 0.3, "penal": 3.0, "rmin": 1.5, "max_iters": 50,
  "supports": [ {"box": [0, 0, 0, 0.02, 1, 1]} ],              # normalized
  "loads":    [ {"box": [0.95, 0, 0, 1, 1, 0.08], "dir": [0, 0, -1]} ]
}

Per iteration it prints one JSON line {"iter", "compliance", "vol", "change",
"preview"} to stdout (line-buffered — the server streams these live) and
writes <outdir>/iterNNN.png. Afterward: result.stl, result.json, final.png.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
from skimage import measure

# CHOLMOD (supernodal sparse Cholesky, SuiteSparse via scikit-sparse) solves the
# SPD stiffness system ~14x faster than scipy's single-threaded spsolve and uses
# the BLAS threads — the FEA solve is ~96% of each iteration. Optional: falls back
# to spsolve/CG cleanly if the wheel isn't present.
try:
    from sksparse.cholmod import cho_factor, CholmodNotPositiveDefiniteError
    HAVE_CHOLMOD = True
except Exception:
    HAVE_CHOLMOD = False

E0, EMIN, NU = 1.0, 1e-9, 0.3


# ------------------------------------------------------------ element matrix

def hexa_ke(a, b, c):
    """24x24 stiffness matrix of an a x b x c H8 element (E=1, nu=NU),
    2x2x2 Gauss integration. Node order: (0,0,0)(1,0,0)(1,1,0)(0,1,0)
    (0,0,1)(1,0,1)(1,1,1)(0,1,1) in local (x,y,z)."""
    C = np.zeros((6, 6))
    lam = NU / ((1 + NU) * (1 - 2 * NU))
    mu = 1 / (2 * (1 + NU))
    C[:3, :3] = lam
    C[np.diag_indices(3)] = lam + 2 * mu
    C[3:, 3:] = np.diag([mu, mu, mu])
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    gp = [(-1 / np.sqrt(3), 1.0), (1 / np.sqrt(3), 1.0)]
    KE = np.zeros((24, 24))
    detJ = (a / 2) * (b / 2) * (c / 2)
    for xi, wx in gp:
        for eta, wy in gp:
            for zeta, wz in gp:
                B = np.zeros((6, 24))
                for i, (cx, cy, cz) in enumerate(corners):
                    sx, sy, sz = 2 * cx - 1, 2 * cy - 1, 2 * cz - 1
                    # dN/dlocal for N = (1+sx*xi)(1+sy*eta)(1+sz*zeta)/8
                    dNdx = sx * (1 + sy * eta) * (1 + sz * zeta) / 8 * (2 / a)
                    dNdy = (1 + sx * xi) * sy * (1 + sz * zeta) / 8 * (2 / b)
                    dNdz = (1 + sx * xi) * (1 + sy * eta) * sz / 8 * (2 / c)
                    B[0, 3 * i] = dNdx
                    B[1, 3 * i + 1] = dNdy
                    B[2, 3 * i + 2] = dNdz
                    B[3, 3 * i] = dNdy
                    B[3, 3 * i + 1] = dNdx
                    B[4, 3 * i + 1] = dNdz
                    B[4, 3 * i + 2] = dNdy
                    B[5, 3 * i] = dNdz
                    B[5, 3 * i + 2] = dNdx
                KE += wx * wy * wz * detJ * (B.T @ C @ B)
    return KE


# ------------------------------------------------------------------ helpers

def node_id(ix, iy, iz, ny, nz):
    return (ix * (ny + 1) + iy) * (nz + 1) + iz


def nodes_in_box(box, nx, ny, nz):
    """Normalized [x0,y0,z0,x1,y1,z1] box → node ids (grid coords /n)."""
    x0, y0, z0, x1, y1, z1 = box
    eps = 1e-9
    ids = []
    for ix in range(nx + 1):
        fx = ix / nx
        if not (x0 - eps <= fx <= x1 + eps):
            continue
        for iy in range(ny + 1):
            fy = iy / ny
            if not (y0 - eps <= fy <= y1 + eps):
                continue
            for iz in range(nz + 1):
                fz = iz / nz
                if z0 - eps <= fz <= z1 + eps:
                    ids.append(node_id(ix, iy, iz, ny, nz))
    return np.array(ids, dtype=np.int64)


def elements_in_box(box, nx, ny, nz):
    """Normalized [x0,y0,z0,x1,y1,z1] box → boolean (nx,ny,nz) ELEMENT mask, by
    element centre. Used for Phase-2 keep regions (solid/void/preserve)."""
    x0, y0, z0, x1, y1, z1 = box
    cx = (np.arange(nx) + 0.5) / nx
    cy = (np.arange(ny) + 0.5) / ny
    cz = (np.arange(nz) + 0.5) / nz
    mx = (cx >= x0) & (cx <= x1)
    my = (cy >= y0) & (cy <= y1)
    mz = (cz >= z0) & (cz <= z1)
    return mx[:, None, None] & my[None, :, None] & mz[None, None, :]


def write_stl(path, verts, faces):
    """Binary STL from marching-cubes output."""
    tri = verts[faces]                                     # (n, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    with open(path, "wb") as f:
        f.write(b"genloop topo".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            f.write(struct.pack("<3f", *n[i]))
            for v in tri[i]:
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")


def mesh_volume(verts, faces):
    tri = verts[faces]
    return float(abs(np.einsum("ij,ij->i", tri[:, 0],
                               np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


MAX_STL_TRIS = 2_000_000          # voxelization is pure-python per-triangle → bound the work


def read_stl(path):
    """Read a binary OR ASCII STL → (verts (3F,3), faces (F,3)). Phase 3: turns an
    uploaded part into geometry we can voxelize. No dedup — per-triangle vertices."""
    data = Path(path).read_bytes()
    head = data[:5].lower()
    if head == b"solid" and b"facet" in data[:2048].lower():          # ASCII
        vs = [[float(t[1]), float(t[2]), float(t[3])]
              for ln in data.decode("ascii", "ignore").splitlines()
              for t in [ln.split()] if len(t) >= 4 and t[0] == "vertex"]
        v = np.asarray(vs, float).reshape(-1, 3)
        if len(v) // 3 > MAX_STL_TRIS:
            raise ValueError(f"mesh too complex ({len(v) // 3} triangles, cap {MAX_STL_TRIS}) — decimate it")
    else:                                                             # binary
        n = struct.unpack("<I", data[80:84])[0]
        if n > MAX_STL_TRIS:
            raise ValueError(f"mesh too complex ({n} triangles, cap {MAX_STL_TRIS}) — decimate it")
        rec = np.frombuffer(data[84:84 + n * 50], dtype=np.uint8)
        if rec.size < n * 50:
            raise ValueError("truncated STL")
        floats = rec.reshape(n, 50)[:, :48].copy().view("<f4").reshape(n, 12)
        v = floats[:, 3:12].reshape(n * 3, 3).astype(float)
    if len(v) < 3:
        raise ValueError("no triangles in STL")
    return v, np.arange(len(v)).reshape(-1, 3)


def voxelize_mesh(verts, faces, dims, res):
    """Ray-parity voxelization of a watertight mesh → bool occupancy (nx,ny,nz).
    An element centre is inside if a +z ray from it crosses an odd number of triangles
    above it. `verts` are assumed shifted into the [0,dims] frame. Pure numpy — no deps."""
    Lx, Ly, Lz = dims
    nx, ny, nz = res
    # sample slightly off cell centre so a +z ray never grazes an axis-aligned face/edge
    # exactly (that double-counts shared edges → parity artifacts on CAD meshes).
    xs = (np.arange(nx) + 0.5007) * Lx / nx
    ys = (np.arange(ny) + 0.4993) * Ly / ny
    zs = (np.arange(nz) + 0.5) * Lz / nz
    tri = verts[faces]
    cols = [[[] for _ in range(ny)] for _ in range(nx)]               # z-crossings per column
    for t in tri:
        ax, ay, az = t[0]; bx, by, bz = t[1]; cx, cy, cz = t[2]
        i0 = max(0, int(np.floor(min(ax, bx, cx) / Lx * nx - 0.5)))
        i1 = min(nx - 1, int(np.ceil(max(ax, bx, cx) / Lx * nx + 0.5)))
        j0 = max(0, int(np.floor(min(ay, by, cy) / Ly * ny - 0.5)))
        j1 = min(ny - 1, int(np.ceil(max(ay, by, cy) / Ly * ny + 0.5)))
        if i1 < i0 or j1 < j0:
            continue
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        for i in range(i0, i1 + 1):
            px = xs[i]
            for j in range(j0, j1 + 1):
                py = ys[j]
                w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
                w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
                if w0 >= 0 and w1 >= 0 and (w0 + w1) <= 1:
                    cols[i][j].append(w0 * az + w1 * bz + (1 - w0 - w1) * cz)
    occ = np.zeros((nx, ny, nz), bool)
    for i in range(nx):
        for j in range(ny):
            zl = cols[i][j]
            if zl:
                zl = np.sort(np.asarray(zl))
                occ[i, j] = ((len(zl) - np.searchsorted(zl, zs)) % 2) == 1   # odd = inside
    return occ


def is_watertight(faces):
    """Every undirected edge must appear exactly twice."""
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _pixel_map(ax, fig, dims):
    """Slice 14 pixel-map contract. Projects world (mm) points to NORMALIZED image
    coordinates (u,v) ∈ [0,1] with origin at the TOP-LEFT of the PNG — resolution- and
    CSS-scale-independent, so the client scales by the displayed <img> size. The camera
    is fixed (elev/azim), so the projection is deterministic. Returns a dict the client
    overlays as SVG leader lines + labels; see index.html drawCallouts()."""
    Lx, Ly, Lz = dims
    M = ax.get_proj()                                    # valid after fig.canvas.draw()
    inv = fig.transFigure.inverted()

    def uv(x, y, z):
        xp, yp, _ = proj3d.proj_transform(x, y, z, M)
        dx, dy = ax.transData.transform((xp, yp))        # display px (bottom-left origin)
        u, v = inv.transform((dx, dy))                   # figure fraction (bottom-left)
        return [round(float(u), 4), round(float(1.0 - v), 4)]   # → top-left origin

    m = 0.10 * max(Lx, Ly, Lz)                           # float dim lines just off the box
    spans = [("x", "L", Lx, (0, -m, -m), (Lx, -m, -m)),
             ("y", "W", Ly, (Lx + m, 0, -m), (Lx + m, Ly, -m)),
             ("z", "H", Lz, (Lx + m, -m, 0), (Lx + m, -m, Lz))]
    corners = [(x, y, z) for x in (0, Lx) for y in (0, Ly) for z in (0, Lz)]
    return {
        "space": "normalized", "origin": "top-left", "units": "mm",
        "camera": {"elev": 24, "azim": -58}, "domain_mm": [Lx, Ly, Lz],
        "bbox": [{"world": [x, y, z], "uv": uv(x, y, z)} for (x, y, z) in corners],
        "dims": [{"axis": a, "label": lb, "mm": round(float(mm), 2),
                  "world": [list(p0), list(p1)], "uv": [uv(*p0), uv(*p1)],
                  "mid_uv": uv((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2,
                              (p0[2] + p1[2]) / 2)}
                 for (a, lb, mm, p0, p1) in spans],
    }


def render_iso(verts, faces, dims, path, title, silhouette=None,
               color="#7ba7c9", edge="none", edge_lw=0.0, map_path=None):
    """One 3D frame from a FIXED camera with FIXED axis limits, so the
    structure visibly evolves in place across iteration frames. `color`/`edge`
    come from the active style's palette (edge shows only when edge_lw>0 — used
    by the faceted mesh finish). If `map_path` is set, also writes the Slice 14
    pixel-map sidecar (world→image projection) beside the PNG."""
    Lx, Ly, Lz = dims
    fig = plt.figure(figsize=(8, 6), facecolor="#111")
    ax = fig.add_subplot(projection="3d", facecolor="#111")
    if len(faces):
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        color=color, edgecolor=(edge if edge_lw > 0 else "none"),
                        linewidth=edge_lw, shade=True)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect((Lx, Ly, Lz))
    ax.view_init(elev=24, azim=-58)
    ax.axis("off")
    if silhouette is not None:                       # small side-view inset
        ins = fig.add_axes([0.74, 0.72, 0.24, 0.24])
        ins.imshow(silhouette.T, cmap="inferno", origin="lower", vmin=0, vmax=1)
        ins.set_title("side x-ray", color="#888", fontsize=7)
        ins.axis("off")
    fig.suptitle(title, color="#ddd", fontsize=10)
    fig.savefig(path, dpi=95)                             # also forces a draw
    if map_path is not None:                              # Slice 14 pixel-map sidecar
        Path(map_path).write_text(json.dumps(_pixel_map(ax, dims=dims, fig=fig), indent=1))
    plt.close(fig)


def density_mesh(x, dims):
    """Marching cubes on the current density field. Early fields are nearly
    uniform (no voxel reaches 0.5 yet), so the iso level adapts: start at the
    field's own upper range and converge to the final 0.5 threshold."""
    nx, ny, nz = x.shape
    Lx, Ly, Lz = dims
    iso = min(0.5, float(np.quantile(x, 0.85)))
    iso = min(iso, float(x.max()) - 1e-4)
    if iso <= 1e-4:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    pad = np.pad(x, 1)
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
    verts, faces, _, _ = measure.marching_cubes(pad, iso, spacing=(hx, hy, hz))
    verts -= [hx * 0.5, hy * 0.5, hz * 0.5]
    return verts, faces


def symmetrize(x, mode):
    """Enforce a spatial symmetry on the density field (S3 style lever). Averaging
    a field with its reflections preserves the mean, so the volfrac constraint the
    OC step just imposed stays satisfied. nx = width (x), ny = depth (y)."""
    if mode == "mirror":
        return 0.5 * (x + x[::-1])                                   # left-right (x)
    if mode in ("axial", "mirror-axial"):
        return 0.25 * (x + x[::-1] + x[:, ::-1] + x[::-1, ::-1])     # 4-fold (x + y)
    return x


def _tpms_field(kind, kx1, ky1, kz1):
    """Implicit field g(x,y,z) for a triply-periodic minimal surface, evaluated on
    broadcast 1-D wave arrays kx1=(k*x)[nx,1,1], ky1=(k*y)[1,ny,1], kz1=(k*z)[1,1,nz]
    (numpy broadcasts the products up to the full grid — no meshgrid materialized).
    The zero level set {g=0} is the minimal surface; g>0 is one labyrinth, |g|<iso a
    thin double-wall sheet. Period along each axis = 2π/k = the cell size."""
    sx, cx = np.sin(kx1), np.cos(kx1)
    sy, cy = np.sin(ky1), np.cos(ky1)
    sz, cz = np.sin(kz1), np.cos(kz1)
    if kind == "schwarz_p":
        return cx + cy + cz
    if kind == "diamond":                                # Schwarz D
        return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
    return sx * cy + sy * cz + sz * cx                    # gyroid (default)


def _fine_grid(dims, cell_mm, vpc, max_vox):
    """Choose an independent fine voxel grid for lattice evaluation: `vpc` voxels
    per cell, uniformly coarsened only if it would exceed the memory budget. The
    lattice MUST be resolved on its own grid (not the coarse SIMP grid) — that is the
    whole reason the old coarse `g>0` hack pinched thin walls to single voxels and went
    non-manifold. Returns (fnx,fny,fnz, h, eff_vpc) or None if the cell is too fine to
    keep a manifold wall inside the budget."""
    Lx, Ly, Lz = dims
    cell = max(float(cell_mm), 1e-3)
    h = cell / max(vpc, 6)
    res = lambda hh: tuple(max(1, int(round(L / hh))) for L in (Lx, Ly, Lz))
    fnx, fny, fnz = res(h)
    while fnx * fny * fnz > max_vox:
        h *= 1.26                                        # ~2x volume budget per step
        fnx, fny, fnz = res(h)
    eff_vpc = cell / h
    if eff_vpc < 5.0:                                    # walls would be < ~2 voxels → bail
        return None
    return fnx, fny, fnz, h, eff_vpc


def tpms_lattice(solid, dims, cell_mm, kind="gyroid", mode="sheet",
                 wall_mm=None, skin_mm=None, vpc=10, max_vox=10_000_000, protect=None):
    """Real TPMS / periodic lattice primitive. Infills the optimized `solid` (coarse
    bool, nx,ny,nz) with a lattice evaluated on its OWN fine grid, then keeps a solid
    outer skin so the struts anchor into a shell and the body stays one connected,
    watertight solid. Watertight BY CONSTRUCTION: the fine grid guarantees every wall
    is ≥ ~3 voxels, so marching-cubes never sees the single-voxel pinch that made the
    coarse hack non-manifold.

    kind  : 'gyroid' | 'schwarz_p' | 'diamond'  (smooth TPMS)
            'ribs'    → cross-hatched internal stiffening walls (2-axis)
            'tracery' → cubic strut skeleton along cell edges (gothic openwork)
    mode  : 'sheet'   → |g|<iso  thin double-wall (stiff, light — the printed default)
            'network' → g>0      solid ligament labyrinth (chunkier)
    Returns (body_bool_fine, fine_h_xyz, info) or (None, None, info) if it can't be
    made printable at the memory budget (caller then ships the plain solid)."""
    Lx, Ly, Lz = dims
    grid = _fine_grid(dims, cell_mm, vpc, max_vox)
    if grid is None:
        return None, None, {"reason": "cell too fine for domain at memory budget"}
    fnx, fny, fnz, h, eff_vpc = grid
    cell = max(float(cell_mm), 1e-3)

    # occupancy on the fine grid: trilinear upsample of the blocky coarse solid
    nx, ny, nz = solid.shape
    occ = ndimage.zoom(solid.astype(np.float32), (fnx / nx, fny / ny, fnz / nz),
                       order=1) >= 0.5
    if not occ.any():
        return None, None, {"reason": "empty after resample"}

    k = 2 * np.pi / cell
    kx1 = (k * (np.arange(fnx) + 0.5) * Lx / fnx).reshape(fnx, 1, 1)
    ky1 = (k * (np.arange(fny) + 0.5) * Ly / fny).reshape(1, fny, 1)
    kz1 = (k * (np.arange(fnz) + 0.5) * Lz / fnz).reshape(1, 1, fnz)

    if kind in ("ribs", "tracery"):
        # thin periodic walls: |sin(k·axis)| < s  ⇒  planes every half-cell.
        s = np.sin(np.pi * min(max((wall_mm or 3.5 * h), 3.0 * h), 0.45 * cell) / cell)
        wx = np.abs(np.sin(kx1)) < s
        wy = np.abs(np.sin(ky1)) < s
        if kind == "ribs":
            lat = wx | wy                                # 2-axis stiffening grid
        else:
            wz = np.abs(np.sin(kz1)) < s
            lat = (wx & wy) | (wy & wz) | (wx & wz)      # struts along cell edges
    else:
        g = _tpms_field(kind, kx1, ky1, kz1)
        if mode == "network":
            lat = g > 0.0
        else:                                            # sheet: |g|<iso, wall ≈ 2·iso/k
            w = max((wall_mm or 0.16 * cell), 3.0 * h)   # ≥3 fine voxels (printable floor)
            lat = np.abs(g) < 0.5 * k * w

    # solid skin so the lattice is enclosed and connected to the boundary
    sk = max((skin_mm or 0.06 * min(Lx, Ly, Lz)), 2.0 * h)
    skin_vox = max(1, int(round(sk / h)))
    interior = ndimage.binary_erosion(occ, iterations=skin_vox)
    body = occ & ((occ & ~interior) | lat)

    pf = None
    if protect is not None and protect.any():             # Phase 2: keep-solid regions
        pf = ndimage.zoom(protect.astype(np.float32),     # stay fully solid (un-latticed)
                          (fnx / solid.shape[0], fny / solid.shape[1], fnz / solid.shape[2]),
                          order=1) >= 0.5
        body = body | (occ & pf)

    lb, nn = ndimage.label(body)                          # keep largest + any protected island
    if nn > 1:                                            # (never drop a forced keep-region)
        counts = np.bincount(lb.ravel())
        keep = {int(np.argmax(counts[1:]) + 1)}
        if pf is not None:
            keep |= {int(l) for l in np.unique(lb[pf]) if l > 0}
        body = np.isin(lb, list(keep))
    return body, (Lx / fnx, Ly / fny, Lz / fnz), {
        "kind": kind, "mode": mode, "cell_mm": round(cell, 3),
        "eff_vpc": round(eff_vpc, 1), "fine_res": [fnx, fny, fnz],
        "skin_vox": int(skin_vox), "solid_fraction": round(float(body.mean()), 4)}


def taubin_smooth(verts, faces, iters=12, lam=0.5, mu=-0.53, frozen=None):
    """S5: shrink-free Laplacian (Taubin λ|μ) smoothing of the marching-cubes mesh
    — turns the voxel staircase into continuous organic surfaces (Bones/Evolved).
    Connectivity (faces) is untouched, so watertightness is preserved. The λ/μ
    pair inflates back what a plain Laplacian would shrink. `frozen` (bool per vertex,
    Phase 2 preserve regions) pins those vertices to their original position so a
    preserved flat face stays flat."""
    if not len(faces) or not len(verts):
        return verts
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])                       # undirected adjacency
    W = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    W.data[:] = 1.0
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    v = verts.astype(float).copy()
    orig = v.copy()
    for _ in range(iters):
        for step in (lam, mu):
            v = v + step * (W.dot(v) / deg[:, None] - v)
            if frozen is not None:
                v[frozen] = orig[frozen]            # pin preserved vertices
    return v


# ---------------------------------------------------------------- main solve

def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    Lx, Ly, Lz = cfg["domain_mm"]
    nx, ny, nz = cfg["resolution"]
    volfrac = float(cfg.get("volfrac", 0.3))
    penal = float(cfg.get("penal", 3.0))
    rmin = float(cfg.get("rmin", 1.5))
    max_iters = int(cfg.get("max_iters", 50))
    pal = cfg.get("palette") or {}
    rcol = pal.get("render_hex", "#7ba7c9")     # style surface color (S1)
    ecol = pal.get("edge_hex", "#7ba7c9")       # style edge color (used by faceted finish, S5)
    symmetry = cfg.get("symmetry", "none")      # S3: enforced each iteration
    z_bias = float(cfg.get("z_bias", 1.0))      # S3: >1 biases toward vertical members
    mesh_finish = cfg.get("mesh_finish", "as-is")  # S5: final-mesh smooth vs faceted
    lattice = bool(cfg.get("lattice"))          # S6: TPMS/periodic infill (Lattice/Gothic)
    lattice_cell = cfg.get("lattice_cell")      # mm; None → auto
    lattice_kind = cfg.get("lattice_kind", "gyroid")   # gyroid|schwarz_p|diamond|ribs|tracery
    lattice_mode = cfg.get("lattice_mode", "sheet")    # sheet (thin double-wall) | network (ligaments)

    nele = nx * ny * nz
    ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    KE = hexa_ke(Lx / nx, Ly / ny, Lz / nz)

    # element -> 24 dofs
    elx, ely, elz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                indexing="ij")
    elx, ely, elz = elx.ravel(), ely.ravel(), elz.ravel()
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    edof = np.zeros((nele, 24), dtype=np.int64)
    for i, (dx, dy, dz) in enumerate(corners):
        nid = ((elx + dx) * (ny + 1) + (ely + dy)) * (nz + 1) + (elz + dz)
        edof[:, 3 * i] = 3 * nid
        edof[:, 3 * i + 1] = 3 * nid + 1
        edof[:, 3 * i + 2] = 3 * nid + 2
    iK = np.kron(edof, np.ones((1, 24), dtype=np.int64)).ravel()
    jK = np.kron(edof, np.ones((24, 1), dtype=np.int64)).T.reshape(24, -1).T.ravel()
    # (iK, jK) pair every dof with every dof per element, row-major

    fixed = np.unique(np.concatenate([
        np.concatenate([3 * nodes_in_box(s["box"], nx, ny, nz) + d for d in (0, 1, 2)])
        for s in cfg["supports"]]))
    free = np.setdiff1d(np.arange(ndof), fixed)
    if not len(fixed):
        raise SystemExit("no support nodes matched — check supports[].box")

    fixed_nodes = set((fixed // 3).tolist())
    F = np.zeros(ndof)
    for ld in cfg["loads"]:
        ids = nodes_in_box(ld["box"], nx, ny, nz)
        # loads landing on supported nodes would be silently absorbed by the
        # supports — normalize over FREE loaded nodes only
        ids = np.array([i for i in ids if i not in fixed_nodes], dtype=np.int64)
        if not len(ids):
            raise SystemExit(f"no free load nodes matched box {ld['box']} "
                             "(load region overlaps the supports entirely)")
        d = np.array(ld["dir"], dtype=float)
        d /= np.linalg.norm(d) or 1.0
        for axis in range(3):
            F[3 * ids + axis] += d[axis] / len(ids)

    # Phase 2 keep-regions: the user pins parts of the domain while the rest
    # optimizes freely. solid/preserve → forced material (density 1) each iteration;
    # void → forced empty (density 0); preserve additionally stays flat (no mesh
    # smoothing) and un-latticed. void wins any overlap.
    solid_mask = np.zeros((nx, ny, nz), bool)
    void_mask = np.zeros((nx, ny, nz), bool)
    preserve_mask = np.zeros((nx, ny, nz), bool)
    preserve_boxes = []
    for r in (cfg.get("regions") or []):
        b = r.get("box")
        if not (isinstance(b, (list, tuple)) and len(b) == 6):
            continue
        m = elements_in_box([float(v) for v in b], nx, ny, nz)
        rule = r.get("rule", "solid")
        if rule == "void":
            void_mask |= m
        elif rule == "preserve":
            solid_mask |= m
            preserve_mask |= m
            preserve_boxes.append([float(v) for v in b])
        else:                                    # solid / keep-filled
            solid_mask |= m
    solid_mask &= ~void_mask                      # void takes precedence
    preserve_mask &= ~void_mask
    has_passive = bool(solid_mask.any() or void_mask.any())

    # Phase 3: an uploaded part becomes the DESIGN DOMAIN — material may only exist
    # where the mesh is. Voxelize it and force everything outside to void, so the
    # solver redesigns *inside the user's envelope* (lightweight it, keep the shell).
    domain_mask = None
    domain_stl = cfg.get("domain_stl")
    if domain_stl:
        try:
            dv, df = read_stl(domain_stl)
            dv = dv - dv.min(axis=0)                   # bbox min → origin (aligns with domain_mm)
            # honest: ray-parity voxelization assumes a watertight mesh. Weld coincident
            # verts and check — an open/non-manifold upload gives an imprecise domain.
            _, inv = np.unique(np.round(dv, 4), axis=0, return_inverse=True)
            if not is_watertight(inv[df]):
                print(json.dumps({"warn": "uploaded mesh is not watertight — the design domain "
                      "is a best-effort voxelization and may be imprecise"}), flush=True)
            domain_mask = voxelize_mesh(dv, df, (Lx, Ly, Lz), (nx, ny, nz))
            if float(domain_mask.mean()) < 0.005:      # nothing usable → don't pretend it worked
                print(json.dumps({"warn": "uploaded mesh didn't voxelize into a usable domain "
                      "(too thin or open) — redesigned the bounding box instead"}), flush=True)
                domain_mask = None
        except Exception as exc:
            print(json.dumps({"warn": f"couldn't use the uploaded mesh as the domain ({exc}) "
                  "— used the bounding box"}), flush=True)
            domain_mask = None
    if domain_mask is not None:
        void_mask |= ~domain_mask                      # no material outside the part
        solid_mask &= domain_mask                      # keep-solid/preserve only inside it
        preserve_mask &= domain_mask
        has_passive = True
        if symmetry != "none":                         # mirroring an arbitrary uploaded shape is
            symmetry = "none"                          # nonsensical AND drifts the domain-relative
        # honest: a default/edited BC on a bbox face the part doesn't actually reach (a sphere,
        # a rounded or recessed part) would fix/load pure void → a physically meaningless floating
        # result that otherwise reports as a clean pass. Warn so it's visible + fixable in the recipe.
        for s in cfg["supports"]:
            if not (elements_in_box(s["box"], nx, ny, nz) & domain_mask).any():
                print(json.dumps({"warn": "a fixed/support region doesn't touch the uploaded part "
                      "— the result may be a floating structure; move the fixed face in the recipe"}), flush=True)
        for ld in cfg["loads"]:
            if not (elements_in_box(ld["box"], nx, ny, nz) & domain_mask).any():
                print(json.dumps({"warn": "a load region doesn't touch the uploaded part — the load "
                      "won't transfer into it; move it in the recipe"}), flush=True)
    active = domain_mask                               # designable region (None ⇒ the whole box);
    # volfrac + reporting are RELATIVE to this so an uploaded part with low occupancy isn't an
    # impossible target measured over the full bounding box.

    seed = cfg.get("seed")
    asym = float(cfg.get("asymmetry", 0.0))
    if seed is not None and asym > 0:
        # CREATIVITY Axis 2: perturb the SIMP starting field so the optimizer
        # converges to a DIFFERENT valid local optimum (topology optimization is
        # non-convex). The OC update re-imposes the volfrac mean every iteration,
        # so this only steers WHICH optimum, not how much material ends up used.
        rng = np.random.default_rng(int(seed))
        x = np.clip(volfrac * (1.0 + asym * (rng.random((nx, ny, nz)) - 0.5) * 2.0),
                    0.01, 1.0)
    else:
        x = np.full((nx, ny, nz), volfrac)
    if domain_mask is not None:
        x = np.where(domain_mask, x, 0.0)             # start empty outside the uploaded part
    change, comp = 1.0, 0.0
    for it in range(1, max_iters + 1):
        xp = EMIN + x.ravel() ** penal * (E0 - EMIN)
        sK = (KE.ravel()[None, :] * xp[:, None]).ravel()
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        Kf = (K[free][:, free]).tocsc()
        U = np.zeros(ndof)
        # The FEA solve is ~96% of each iteration. CHOLMOD (supernodal Cholesky,
        # multithreaded) is the fast path; spsolve with the symmetric-friendly
        # MMD_AT_PLUS_A ordering (~1.5x over COLAMD) / preconditioned CG are the
        # dependency-free fallbacks.
        if HAVE_CHOLMOD:
            try:
                U[free] = cho_factor(Kf).solve(F[free]).ravel()
            except CholmodNotPositiveDefiniteError:      # ill-conditioned → exact fallback
                U[free] = spsolve(Kf, F[free], permc_spec="MMD_AT_PLUS_A")
        elif ndof <= 60_000:
            U[free] = spsolve(Kf, F[free], permc_spec="MMD_AT_PLUS_A")
        else:
            from scipy.sparse import diags
            Minv = diags(1.0 / Kf.diagonal())
            sol, info = cg(Kf, F[free], rtol=1e-6, maxiter=8000, M=Minv)
            if info != 0:                    # non-converged CG would silently
                sol = spsolve(Kf, F[free], permc_spec="MMD_AT_PLUS_A")  # fall back
            U[free] = sol
        Ue = U[edof]
        ce = np.einsum("ij,jk,ik->i", Ue, KE, Ue).reshape(nx, ny, nz)
        comp = float((xp.reshape(nx, ny, nz) * ce).sum())
        dc = -penal * x ** (penal - 1) * (E0 - EMIN) * ce
        # density-weighted sensitivity smoothing — uniform-box approximation
        # of the classic cone filter (adequate here; upgrade to a true cone
        # kernel if mesh-dependency shows). Window must be ODD and symmetric.
        size = max(1, 2 * int(np.ceil(rmin)) - 1)
        # S3: z_bias>1 widens the vertical filter window → members merge upward,
        # favouring tall/vertical load paths (Gothic verticality).
        size_z = max(1, 2 * int(np.ceil(rmin * z_bias)) - 1)
        win = (size, size, size_z)
        dc = ndimage.uniform_filter(dc * x, size=win) / \
            np.maximum(1e-6, ndimage.uniform_filter(x, size=win))
        if active is not None:              # Phase 3: no incentive to place material outside the part
            dc = np.where(active, dc, 0.0)
        # optimality criteria update with bisection — volume measured over the ACTIVE region
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(x * np.sqrt(np.maximum(1e-12, -dc) / lmid),
                           np.maximum(0.0, x - move), np.minimum(1.0, x + move))
            frac = float(xnew[active].mean()) if active is not None else float(xnew.mean())
            if frac > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = symmetrize(xnew, symmetry)      # S3: enforce style symmetry (mean-preserving)
        if has_passive:                     # Phase 2: pin keep-regions after the update
            x[solid_mask] = 1.0
            x[void_mask] = 0.0
        vfrac = float(x[active].mean()) if active is not None else float(x.mean())   # domain-relative
        png = out / f"iter{it:03d}.png"
        pv, pf = density_mesh(x, (Lx, Ly, Lz))
        render_iso(pv, pf, (Lx, Ly, Lz), png,
                   f"iter {it}  compliance {comp:.3f}  vol {vfrac:.3f}",
                   silhouette=x.max(axis=1), color=rcol, edge=ecol)
        print(json.dumps({"iter": it, "compliance": round(comp, 4),
                          "vol": round(vfrac, 4),
                          "change": round(change, 4), "preview": png.name}), flush=True)
        if it >= 10 and change < 0.02:
            break

    # ---- extract final geometry: threshold, keep the largest component — PLUS any
    # component holding a forced keep-region (a keep-solid/preserve boss must never be
    # silently dropped just because it optimized into its own island).
    thr = 0.5
    solid = x >= thr
    lab, n = ndimage.label(solid)
    if n > 1:
        counts = np.bincount(lab.ravel())
        keep = {int(np.argmax(counts[1:]) + 1)}
        if has_passive:
            keep |= {int(l) for l in np.unique(lab[solid_mask | preserve_mask]) if l > 0}
        solid = np.isin(lab, list(keep))
    final_pieces = int(ndimage.label(solid)[1])       # honest one-piece-or-not count
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz

    # ---- real TPMS / periodic lattice, evaluated on its own FINE grid so it's
    # watertight by construction. Still BEST-EFFORT: if it somehow can't be made a
    # closed manifold, ship the plain solid — a styled part is never a broken STL.
    lattice_applied = False
    lattice_info = None
    fine_mesh = None
    if lattice and solid.any():
        cell = (float(lattice_cell) if lattice_cell
                else max(min(Lx, Ly, Lz) / 6.0, 8 * max(hx, hy, hz)))
        body, fh, lattice_info = tpms_lattice(solid, (Lx, Ly, Lz), cell,
                                              kind=lattice_kind, mode=lattice_mode,
                                              protect=solid_mask if has_passive else None)
        if body is not None and body.any():
            fhx, fhy, fhz = fh
            tv, tf, _, _ = measure.marching_cubes(np.pad(body.astype(np.float32), 1),
                                                  0.5, spacing=(fhx, fhy, fhz))
            if len(tf) and is_watertight(tf):
                tv -= [fhx * 0.5, fhy * 0.5, fhz * 0.5]
                fine_mesh, lattice_applied = (tv, tf), True

    if fine_mesh is not None:                         # lattice mesh: do NOT smooth — it
        verts, faces = fine_mesh                      # would erase the thin walls; the
    elif solid.any():                                 # fine grid already reads continuous
        pad = np.pad(solid.astype(float), 1)          # plain-solid path (coarse grid)
        verts, faces, _, _ = measure.marching_cubes(pad, 0.5, spacing=(hx, hy, hz))
        verts -= [hx * 0.5, hy * 0.5, hz * 0.5]       # unpad + half-voxel (cell-centered)
        if mesh_finish in ("smooth", "g2-polish") and len(faces):   # S5: polish staircase
            frozen = None
            if preserve_boxes:                            # Phase 2: keep preserved faces flat
                frozen = np.zeros(len(verts), bool)
                for bx in preserve_boxes:
                    frozen |= ((verts[:, 0] >= bx[0] * Lx) & (verts[:, 0] <= bx[3] * Lx) &
                               (verts[:, 1] >= bx[1] * Ly) & (verts[:, 1] <= bx[4] * Ly) &
                               (verts[:, 2] >= bx[2] * Lz) & (verts[:, 2] <= bx[5] * Lz))
            verts = taubin_smooth(verts, faces, iters=16 if mesh_finish == "g2-polish" else 8,
                                  frozen=frozen)
    else:                                             # degenerate (no material) → empty mesh
        verts, faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    write_stl(out / "result.stl", verts, faces)
    vfinal = float(x[active].mean()) if active is not None else float(x.mean())   # domain-relative
    render_iso(verts, faces, (Lx, Ly, Lz), out / "final.png",
               f"final  compliance {comp:.3f}  vol {vfinal:.3f}", color=rcol, edge=ecol,
               edge_lw=0.25 if mesh_finish == "faceted" else 0.0,
               map_path=out / "final.map.json")           # Slice 14 pixel-map contract
    watertight = bool(len(faces)) and is_watertight(faces)
    result = {
        "iterations": it, "compliance": comp,
        "vol_fraction": float(x[active].mean()) if active is not None else float(x.mean()),
        "solid_fraction_after_threshold": float(solid.mean()),
        "components": final_pieces, "raw_components": int(n),
        "kept_largest": bool(n > final_pieces),
        "watertight": watertight, "lattice_applied": bool(lattice_applied),
        "lattice": lattice_info,
        "regions": ({"solid_vox": int(solid_mask.sum()), "void_vox": int(void_mask.sum()),
                     "preserve_vox": int(preserve_mask.sum())}
                    if (has_passive or preserve_mask.any()) else None),
        "domain_upload": ({"occupancy_frac": round(float(domain_mask.mean()), 4)}
                          if domain_mask is not None else None),
        "mesh_volume_mm3": mesh_volume(verts, faces) if watertight else None,
        "domain_mm": [Lx, Ly, Lz], "resolution": [nx, ny, nz],
        "stl": "result.stl", "final_png": "final.png", "final_map": "final.map.json",
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"done": True, **result}), flush=True)


if __name__ == "__main__":
    main()
