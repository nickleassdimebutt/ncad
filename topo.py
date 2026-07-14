#!/usr/bin/env python3
"""topo.py — 3D SIMP topology optimization for nCAD's structural engine.

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
import heapq
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import proj3d
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial import cKDTree
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


def voxels_from_tris(tris, nx, ny, nz):
    """Per-face keep region: rasterize a triangle patch (normalized [0,1] coords,
    shape (T,3,3)) into a thin ELEMENT surface-shell mask by dense barycentric
    sampling. Unlike a box, this captures the *exact* footprint of a clicked face
    (an L-shaped pad, a partial side) rather than its whole bounding slab."""
    mask = np.zeros((nx, ny, nz), bool)
    tris = np.asarray(tris, float)
    if tris.ndim == 2 and tris.shape[1] == 9:              # wire format: rows of 9 → (T,3,3)
        tris = tris.reshape(-1, 3, 3)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
        return mask
    dims = np.array([nx, ny, nz], float)
    step = 0.4 * min(1.0 / nx, 1.0 / ny, 1.0 / nz)         # ~0.4 cell → no gaps
    for a, b, c in tris:
        e1, e2 = b - a, c - a
        span = max(np.linalg.norm(e1), np.linalg.norm(e2), 1e-9)
        n = int(min(64, max(2, np.ceil(span / step))))     # samples per edge
        u = np.linspace(0.0, 1.0, n)
        uu, vv = np.meshgrid(u, u)
        keep = (uu + vv) <= 1.0                             # barycentric triangle
        pts = a + uu[keep][:, None] * e1 + vv[keep][:, None] * e2
        idx = np.clip(np.floor(pts * dims).astype(int), 0, dims.astype(int) - 1)
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def write_stl(path, verts, faces):
    """Binary STL from marching-cubes output."""
    tri = verts[faces]                                     # (n, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    with open(path, "wb") as f:
        f.write(b"nCAD topo".ljust(80, b"\0"))
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
    fig = plt.figure(figsize=(8, 6), facecolor="#080b0f")
    ax = fig.add_subplot(projection="3d", facecolor="#080b0f")
    if len(faces):
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        color=color, edgecolor=(edge if edge_lw > 0 else "none"),
                        linewidth=edge_lw, shade=True, antialiased=False,
                        lightsource=LightSource(azdeg=315, altdeg=45))
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


def density_iso_level(x, active=None):
    """The exact isovalue used by both live previews and eligible final surfaces."""
    values = x[active] if active is not None else x
    return min(0.5, float(np.quantile(values, 0.85)), float(x.max()) - 1e-4)


def density_mesh(x, dims, detail=1, active=None):
    """Marching cubes on the current density field. `detail`>1 upsamples the CONTINUOUS
    field (trilinear) before marching → a smooth, sub-voxel surface from the coarse solve
    instead of the voxel staircase. Early fields are nearly uniform (no voxel reaches 0.5
    yet), so the iso adapts: start at the active field's own upper range, converge to 0.5.
    For an uploaded design domain, excluding the forced-zero exterior is essential: once
    the solve becomes sparse, a whole-grid quantile collapses toward zero and the preview
    misleadingly renders nearly the entire original envelope."""
    Lx, Ly, Lz = dims
    iso = density_iso_level(x, active)
    if iso <= 1e-4:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    field = x
    if detail > 1:                                    # upsample + de-rib (see final-mesh note)
        field = ndimage.gaussian_filter(ndimage.zoom(x, detail, order=1), sigma=0.7 * detail)
    iso = min(iso, float(field.max()) - 1e-4)          # smoothing can lower thin peaks below 0.5
    if iso <= 1e-4:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    fnx, fny, fnz = field.shape
    hx, hy, hz = Lx / fnx, Ly / fny, Lz / fnz
    pad = np.pad(field, 1)
    verts, faces, _, _ = measure.marching_cubes(pad, iso, spacing=(hx, hy, hz))
    verts -= [hx * 0.5, hy * 0.5, hz * 0.5]
    return verts, faces


_FACE_NEIGHBORS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                   (0, 1, 0), (0, 0, -1), (0, 0, 1))


def connected_to(mask, seeds):
    """Return the face-connected part of `mask` that touches any seed voxel."""
    labels, _ = ndimage.label(mask)
    seed_labels = np.unique(labels[seeds & mask])
    seed_labels = seed_labels[seed_labels > 0]
    return np.isin(labels, seed_labels) if len(seed_labels) else np.zeros_like(mask)


def cheapest_density_path(active, density, sources, targets, existing):
    """Find a six-connected path through the domain, preferring optimized high density.

    SIMP uses a tiny nonzero stiffness in void, so a valid continuous density solution can
    still become disconnected after projection to a printable solid. This path is not a new
    arbitrary straight bridge: it follows the density field the FEA/OC solve already ranked.
    """
    shape = active.shape
    n = active.size
    active_f = active.ravel()
    density_f = density.ravel()
    existing_f = existing.ravel()
    target_f = targets.ravel()
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=np.int64)
    heap = []
    for idx in np.flatnonzero(sources & active):
        idx = int(idx)
        dist[idx] = 0.0
        heapq.heappush(heap, (0.0, idx))
    end = -1
    while heap:
        cost, idx = heapq.heappop(heap)
        if cost != dist[idx]:
            continue
        if target_f[idx]:
            end = idx
            break
        i, j, k = np.unravel_index(idx, shape)
        for di, dj, dk in _FACE_NEIGHBORS:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                continue
            nxt = int(np.ravel_multi_index((ni, nj, nk), shape))
            if not active_f[nxt]:
                continue
            step = 0.001 if existing_f[nxt] else 0.05 + (1.0 - density_f[nxt]) ** 4
            alt = cost + step
            if alt < dist[nxt]:
                dist[nxt], prev[nxt] = alt, idx
                heapq.heappush(heap, (alt, nxt))
    path = np.zeros(n, bool)
    while end >= 0:
        path[end] = True
        end = int(prev[end])
    return path.reshape(shape)


def grow_connected(mask, eligible, density, desired_count):
    """Grow a solid through its highest-density face neighbors to restore a volume target."""
    current = int((mask & eligible).sum())
    if current >= desired_count:
        return mask
    result = mask.copy()
    queued = np.zeros(mask.shape, bool)
    heap = []

    def queue_neighbors(idx):
        i, j, k = idx
        for di, dj, dk in _FACE_NEIGHBORS:
            ni, nj, nk = i + di, j + dj, k + dk
            if (0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and
                    0 <= nk < mask.shape[2] and eligible[ni, nj, nk] and
                    not result[ni, nj, nk] and not queued[ni, nj, nk]):
                queued[ni, nj, nk] = True
                heapq.heappush(heap, (-float(density[ni, nj, nk]), ni, nj, nk))

    for idx in np.argwhere(result):
        queue_neighbors(tuple(idx))
    while heap and current < desired_count:
        _, i, j, k = heapq.heappop(heap)
        if result[i, j, k]:
            continue
        result[i, j, k] = True
        current += 1
        queue_neighbors((i, j, k))
    return result


def continuous_density_solid(density, active, support, load_masks):
    """Return the preview's thresholded body and whether it is already production-safe."""
    iso = density_iso_level(density, active)
    solid = (density >= iso) & active if iso > 1e-4 else np.zeros_like(active)
    _, pieces = ndimage.label(solid)
    support_body = connected_to(solid, support)
    loads_connected = bool(support_body.any()) and all(
        bool((support_body & load_pad).any()) for load_pad in load_masks)
    return solid, bool(pieces == 1 and loads_connected), iso


def printable_projection(density, active, design, passive, void, support, load_masks,
                         design_fraction):
    """Project a density field into a connected, volume-aware printable voxel body."""
    desired = int(round(design_fraction * int(design.sum())))
    solid = passive.copy()
    if desired > 0:
        design_ids = np.flatnonzero(design)
        take = min(desired, len(design_ids))
        ranked = np.argpartition(density.ravel()[design_ids], -take)[-take:]
        solid.ravel()[design_ids[ranked]] = True
    solid &= active
    selected_before_paths = int(solid.sum())
    for load_pad in load_masks:
        source_body = connected_to(solid, support)
        if (source_body & load_pad).any():
            continue
        path = cheapest_density_path(active, density, source_body, load_pad, solid)
        if path.any():
            # The one-voxel centerline is structural bookkeeping; thicken it once so surface
            # smoothing cannot make a load path disappear from the exported mesh.
            solid |= ndimage.binary_dilation(path) & active & ~void
    connectivity_added = max(0, int(solid.sum()) - selected_before_paths)

    labels, raw_pieces = ndimage.label(solid)
    raw_voxels = int(solid.sum())
    if raw_pieces > 1:
        keep = {int(v) for v in np.unique(labels[support]) if v > 0}
        keep |= {int(v) for v in np.unique(labels[passive]) if v > 0}
        if not keep:
            counts = np.bincount(labels.ravel())
            keep = {int(np.argmax(counts[1:]) + 1)}
        solid = np.isin(labels, list(keep))
    solid = grow_connected(solid, design, density, desired)
    return solid, {
        "raw_components": int(raw_pieces),
        "raw_solid_voxels": raw_voxels,
        "connectivity_added_voxels": connectivity_added,
    }


def binary_surface_mesh(solid, dims, detail=1):
    """Smooth a printable voxel projection into the same watertight mesh used in previews."""
    Lx, Ly, Lz = dims
    field = solid.astype(float)
    d = detail
    while d > 1 and field.size * d ** 3 > 15_000_000:
        d -= 1
    if d > 1:
        field = ndimage.zoom(field, d, order=1)
        field = ndimage.gaussian_filter(field, sigma=0.45 * d)
    if not field.any() or float(field.max()) <= 0.5:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    sx, sy, sz = Lx / field.shape[0], Ly / field.shape[1], Lz / field.shape[2]
    verts, faces, _, _ = measure.marching_cubes(
        np.pad(field.astype(np.float32), 1), 0.5, spacing=(sx, sy, sz))
    verts -= [sx * 0.5, sy * 0.5, sz * 0.5]
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


def surface_region_vertices(verts, faces, regions, rules=("preserve",)):
    """Map normalized 3D retain selections back to welded source-mesh vertices.

    Triangle selections are matched by surface-centroid proximity so selections made on a
    lightly smoothed revision still find the original source. Box-only legacy regions remain
    supported, but exact drawn selections do not leak through to the opposite side.
    """
    selected = np.zeros(len(verts), bool)
    vmin = verts.min(axis=0)
    span = np.maximum(verts.max(axis=0) - vmin, 1e-9)
    norm_verts = (verts - vmin) / span
    targets = []
    for region in regions or []:
        if region.get("rule") not in rules:
            continue
        tris = region.get("tris")
        if isinstance(tris, list) and tris:
            try:
                a = np.asarray(tris, float).reshape(-1, 3, 3)
                targets.append(a.mean(axis=1))
            except (TypeError, ValueError):
                pass
            continue
        box = region.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 6:
            lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
            selected |= np.all((norm_verts >= lo) & (norm_verts <= hi), axis=1)
    if targets:
        target = np.vstack(targets)
        centres = norm_verts[faces].mean(axis=1)
        distance, _ = cKDTree(target).query(centres, k=1, distance_upper_bound=.006)
        hit = np.isfinite(distance)
        if hit.any():
            selected[np.unique(faces[hit])] = True
    return selected


def outward_vertex_normals(verts, faces):
    tri = verts[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(verts, dtype=float)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, length[:, None], out=np.zeros_like(normals),
                        where=length[:, None] > 1e-12)
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
    if signed < 0:
        normals *= -1.0
    return normals


HIGH_DETAIL_TEXTURES = {
    "diamond", "crosshatch", "starburst", "grid", "triangles", "topo",
    "herringbone", "carbon", "leather", "bark", "sandstone", "guilloche", "circuit",
    "leopard",
}


def texture_sampling_edge(settings):
    """Largest mesh edge that can still represent the selected relief field.

    Texture functions are sampled at mesh vertices; leaving an imported 24 mm triangle under
    an 8 mm repeat turns every motif into the same three-corner ramp. The pattern supplies a
    safe base rate, then the user-facing surface-detail setting increases it for close inspection.
    The face cap prevents an unexpectedly tiny repeat from producing an unbounded mesh.
    """
    repeat = max(float(settings.get("scale_mm", 8.0)), 1e-6)
    pattern = settings.get("pattern")
    # High-detail fields use twelve samples per repeat at Balanced. Leopard's internal
    # spot spacing is already denser than its user-facing feature size, so giving it a
    # second 1.5x multiplier made Fine request an impractical 27 samples per feature and
    # hit the face cap before it could finish even the coarser passes.
    base = 12.0 if pattern in HIGH_DETAIL_TEXTURES else 8.0
    detail = settings.get("detail", "fine")
    samples = base * {"balanced": 1.0, "fine": 1.5, "ultra": 2.0}.get(detail, 1.5)
    return max(0.2, repeat / samples)


def texture_face_cap(settings):
    """Bound high-detail exports while allowing fine vector-like relief to be represented."""
    return {"balanced": 600_000, "fine": 1_400_000, "ultra": 2_400_000}.get(
        settings.get("detail", "fine"), 1_400_000)


def subdivide_texture_mesh(verts, faces, selected, max_edge, max_faces=600_000):
    """Conforming midpoint subdivision for relief sampling.

    Every marked undirected edge is split in both incident faces, so the result stays watertight
    (no T-junctions). Selection weights are interpolated onto midpoint vertices, preserving a
    drawn texture mask rather than expanding it to the whole refined mesh.
    """
    vertices = np.asarray(verts, float).tolist()
    weights = np.asarray(selected, float).tolist()
    refined = np.asarray(faces, int)
    requested_edge = float(max_edge)
    # An imported surface needs substantially more triangles than the ideal equilateral
    # area estimate because its original edges, creases, and adaptive split transitions
    # remain in the mesh. Clamp impossible requests *before* choosing split edges. This
    # keeps detail monotonic: asking for Ultra can resolve to the same cap-limited edge as
    # Fine, but can never take a different early split path and finish visibly coarser.
    source = np.asarray(verts, float)[refined]
    surface_area = .5 * np.linalg.norm(
        np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]), axis=1).sum()
    capacity_edge = np.sqrt(max(surface_area, 0.0) * 8.0 / max(int(max_faces), 1))
    max_edge = max(requested_edge, float(capacity_edge))
    limited = max_edge > requested_edge * (1.0 + 1e-9)
    rounds = 0
    edge_slots = ((0, 1), (1, 2), (2, 0))

    for _ in range(12):
        array = np.asarray(vertices, float)
        lengths = np.stack([
            np.linalg.norm(array[refined[:, i]] - array[refined[:, j]], axis=1)
            for i, j in edge_slots
        ], axis=1)
        marked = lengths > max_edge
        if not marked.any():
            break
        projected_faces = int(len(refined) + marked.sum())
        if projected_faces > max_faces:
            limited = True
            break

        marked_edges = []
        for edge_index, (i, j) in enumerate(edge_slots):
            edge_faces = refined[marked[:, edge_index]][:, [i, j]]
            if len(edge_faces):
                marked_edges.append(np.sort(edge_faces, axis=1))
        unique_edges = np.unique(np.vstack(marked_edges), axis=0)
        midpoint = {}
        for a, b in unique_edges:
            a, b = int(a), int(b)
            midpoint[(a, b)] = len(vertices)
            vertices.append(((array[a] + array[b]) * .5).tolist())
            weights.append((weights[a] + weights[b]) * .5)

        next_faces = []
        for triangle, split in zip(refined, marked):
            a, b, c = (int(x) for x in triangle)
            m0 = midpoint.get((min(a, b), max(a, b)))
            m1 = midpoint.get((min(b, c), max(b, c)))
            m2 = midpoint.get((min(c, a), max(c, a)))
            case = int(split[0]) + 2 * int(split[1]) + 4 * int(split[2])
            if case == 0:
                next_faces.append((a, b, c))
            elif case == 1:
                next_faces.extend(((a, m0, c), (m0, b, c)))
            elif case == 2:
                next_faces.extend(((b, m1, a), (m1, c, a)))
            elif case == 4:
                next_faces.extend(((c, m2, b), (m2, a, b)))
            elif case == 3:
                next_faces.extend(((m0, b, m1), (a, m0, m1), (a, m1, c)))
            elif case == 6:
                next_faces.extend(((m1, c, m2), (b, m1, m2), (b, m2, a)))
            elif case == 5:
                next_faces.extend(((m2, a, m0), (c, m2, m0), (c, m0, b)))
            else:
                next_faces.extend(((a, m0, m2), (m0, b, m1),
                                   (m2, m1, c), (m0, m1, m2)))
        refined = np.asarray(next_faces, int)
        rounds += 1

    array = np.asarray(vertices, float)
    if len(refined):
        final_edges = np.vstack([refined[:, [0, 1]], refined[:, [1, 2]], refined[:, [2, 0]]])
        achieved = float(np.linalg.norm(array[final_edges[:, 0]] -
                                        array[final_edges[:, 1]], axis=1).max())
    else:
        achieved = 0.0
    return array, refined, np.asarray(weights) >= .5, {
        "sampling_edge_mm": requested_edge,
        "effective_sampling_edge_mm": float(max_edge), "max_edge_mm": achieved,
        "subdivision_rounds": rounds, "mesh_vertices": len(array),
        "mesh_faces": len(refined), "sampling_limited": limited,
    }


def texture_hash(ix, iy, salt=0.0):
    """Deterministic 2D hash used by aperiodic procedural surface patterns."""
    return np.mod(np.sin(ix * 127.1 + iy * 311.7 + salt) * 43758.5453, 1.0)


def leopard_pattern(u, v, feature_size, period_u=None, orientation=0.0):
    """Dense aperiodic leopard spots, optionally closed across one large U period.

    A wrapped body needs one join, but repeating a small texture tile would defeat the
    aesthetic. When ``period_u`` is supplied we hash one circumference-wide strip and
    repeat only its boundary cells, so the seam closes without a short visible repeat.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    # Natural leopard markings are a crowded mixture of small solid islands and larger
    # broken rosettes. Use `feature_size` as the perceived mark size rather than as a
    # sparse one-mark tile, with the centres packed more tightly than that distance.
    target_spacing = feature_size * .48
    if period_u is not None and period_u > 1e-9:
        columns = max(3, int(round(period_u / target_spacing)))
        spacing_u = period_u / columns
        su = np.mod(u, period_u) / spacing_u
    else:
        columns = None
        su = u / target_spacing
    sv = v / target_spacing
    cell_u, cell_v = np.floor(su), np.floor(sv)
    relief = np.zeros_like(su, dtype=float)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            gx, gy = cell_u + ox, cell_v + oy
            # Hash the wrapped column but place it in the neighboring unwrapped copy. Points
            # on either side of U=0 therefore see the same jittered spots and displacement.
            hash_x = np.mod(gx, columns) if columns is not None else gx
            present = texture_hash(hash_x, gy, 1.7) > .02
            hx = texture_hash(hash_x, gy, 7.1)
            hy = texture_hash(hash_x, gy, 13.9)
            size = texture_hash(hash_x, gy, 23.3)
            aspect = texture_hash(hash_x, gy, 31.7)
            phase = texture_hash(hash_x, gy, 47.9) * 2.0 * np.pi
            solid = texture_hash(hash_x, gy, 59.3) < .78

            dx = su - (gx + .18 + .64 * hx)
            dy = sv - (gy + .18 + .64 * hy)
            angle = phase + orientation + .35 * np.sin(hash_x * .73 + gy * 1.17)
            ca, sa = np.cos(angle), np.sin(angle)
            rx = ca * dx + sa * dy
            ry = -sa * dx + ca * dy
            major = .45 + .11 * size
            minor = major * (.42 + .52 * aspect)
            radius = np.hypot(rx / major, ry / minor)
            theta = np.arctan2(ry / minor, rx / major)
            irregular = radius * (1.0 + .10 * np.sin(3.0 * theta + phase)
                                  + .06 * np.sin(5.0 * theta - 1.7 * phase)
                                  + .03 * np.sin(7.0 * theta + 2.3 * phase))

            outer_t = np.clip((1.14 - irregular) / .22, 0.0, 1.0)
            outer = outer_t * outer_t * (3.0 - 2.0 * outer_t)
            inner_t = np.clip((.40 - irregular) / .14, 0.0, 1.0)
            inner = inner_t * inner_t * (3.0 - 2.0 * inner_t)
            ring = outer * (1.0 - inner)
            gap_angle = (texture_hash(hash_x, gy, 71.1) - .5) * 2.0 * np.pi
            gap_delta = np.abs(np.arctan2(np.sin(theta - gap_angle),
                                          np.cos(theta - gap_angle)))
            gap_width = .10 + .22 * texture_hash(hash_x, gy, 79.7)
            gap_t = np.clip((gap_delta - gap_width) / .30, 0.0, 1.0)
            gap_gate = gap_t * gap_t * (3.0 - 2.0 * gap_t)
            ring *= .08 + .92 * gap_gate

            # A small detached island beside some rosettes makes their broken outline read
            # like the irregular C and U marks in real leopard coats instead of tidy ovals.
            satellite = np.exp(-((rx / (major * .28) - 1.23) ** 2
                                 + (ry / (minor * .42) + .12) ** 2) * 2.0)
            has_satellite = texture_hash(hash_x, gy, 91.3) > .42
            centre_dot = np.exp(-(irregular / .27) ** 4)
            has_centre = texture_hash(hash_x, gy, 101.9) > .86
            rosette = np.maximum(ring, np.where(has_satellite, .86 * satellite, 0.0))
            rosette = np.maximum(rosette, np.where(has_centre, .76 * centre_dot, 0.0))
            mark = np.where(solid, outer, rosette)
            relief = np.maximum(relief, np.where(present, mark, 0.0))
    return np.clip(relief, 0.0, 1.0)


def texture_pattern(pattern, u, v, repeat):
    """Deterministic 0..1 relief fields. Coordinates/feature size are millimetres."""
    x, y = 2.0 * np.pi * u / repeat, 2.0 * np.pi * v / repeat
    if pattern == "ribs":
        return .5 + .5 * np.cos(x)
    if pattern == "waves":
        return .5 + .5 * np.sin(x + .65 * np.sin(y))
    if pattern == "diamond":
        fu = np.mod(2.0 * u / repeat + .5, 1.0) - .5
        fv = np.mod(2.0 * v / repeat + .5, 1.0) - .5
        return np.clip(1.0 - 2.0 * (np.abs(fu) + np.abs(fv)), 0.0, 1.0)
    if pattern == "dimples":
        fu = np.mod(u / repeat + .5, 1.0) - .5
        fv = np.mod(v / repeat + .5, 1.0) - .5
        return np.clip(1.0 - np.hypot(fu, fv) / .46, 0.0, 1.0) ** 2
    if pattern == "honeycomb":
        # True flat-top hex-cell walls. `repeat` is the visible flat-to-flat cell spacing,
        # not an internal two-column tile width; the old repeat/3 radius made a requested
        # 4 mm honeycomb only about 2.3 mm wide and read as dense surface noise.
        radius = repeat / 1.7320508
        step_x, step_y = 1.5 * radius, 1.7320508 * radius
        column0 = np.rint(u / step_x)
        wall_distance = np.full_like(np.asarray(u, float), np.inf, dtype=float)
        for dc in (-1, 0, 1):
            column = column0 + dc
            stagger = .5 * np.mod(column, 2.0)
            row0 = np.rint(v / step_y - stagger)
            for dr in (-1, 0, 1):
                row = row0 + dr
                dx = np.abs(u - column * step_x)
                dy = np.abs(v - (row + stagger) * step_y)
                hex_radius = np.maximum(.8660254 * dx + .5 * dy, dy)
                wall_distance = np.minimum(wall_distance,
                                           np.abs(hex_radius - .8660254 * radius))
        return np.exp(-(wall_distance / max(.10 * repeat, 1e-6)) ** 2)
    if pattern == "leopard":
        return leopard_pattern(u, v, repeat)
    if pattern == "checker":
        return np.mod(np.floor(2.0 * u / repeat) + np.floor(2.0 * v / repeat), 2.0)
    if pattern == "chevron":
        phase = x + 1.35 * np.abs(np.mod(y / (2.0 * np.pi), 1.0) - .5) * 2.0 * np.pi
        return .5 + .5 * np.cos(phase)
    if pattern == "crosshatch":
        return np.clip(np.abs(np.sin(x + y)) ** 9 + np.abs(np.sin(x - y)) ** 9, 0.0, 1.0)
    if pattern == "basket":
        cell_u = np.floor(2.0 * u / repeat)
        cell_v = np.floor(2.0 * v / repeat)
        horizontal = np.mod(cell_u + cell_v, 2.0) == 0
        return np.where(horizontal, .25 + .75 * np.abs(np.cos(y)),
                        .25 + .75 * np.abs(np.cos(x)))
    if pattern == "brick":
        row = np.floor(2.0 * v / repeat)
        fu = np.abs(np.mod(2.0 * u / repeat + .5 * np.mod(row, 2.0), 1.0) - .5)
        fv = np.abs(np.mod(2.0 * v / repeat, 1.0) - .5)
        return np.clip(np.minimum((.5 - fu) / .1, (.5 - fv) / .14), 0.0, 1.0)
    if pattern == "scales":
        row = np.floor(2.0 * v / repeat)
        fu = np.mod(2.0 * u / repeat + .5 * np.mod(row, 2.0), 1.0) - .5
        fv = np.mod(2.0 * v / repeat, 1.0)
        radius = np.hypot(fu, fv - .03)
        return np.clip(1.0 - np.abs(radius - .44) / .1, 0.0, 1.0)
    if pattern == "rings":
        return .5 + .5 * np.cos(2.0 * np.pi * np.hypot(u, v) / repeat)
    if pattern == "starburst":
        # A starburst is a repeated rosette, not one global polar singularity.  Keeping
        # the polar coordinates local to each repeat cell makes repeat size meaningful
        # and prevents giant converging bands when the field wraps around a body.
        fu = np.mod(u / repeat + .5, 1.0) - .5
        fv = np.mod(v / repeat + .5, 1.0) - .5
        radius = np.hypot(fu, fv)
        spokes = np.abs(np.cos(8.0 * np.arctan2(fv, fu))) ** 4
        rosette = .35 + .65 * (.5 + .5 * np.cos(2.0 * np.pi * radius))
        # atan2 has no unique value at the polar centre. Fade its angular component to
        # a constant there, and again at cell borders, so adjacent tiles meet with no
        # pinwheel spike or triangle-to-triangle jump.
        core_t = np.clip(radius / .14, 0.0, 1.0)
        core_fade = core_t * core_t * (3.0 - 2.0 * core_t)
        edge_distance = .5 - np.maximum(np.abs(fu), np.abs(fv))
        edge_t = np.clip(edge_distance / .1, 0.0, 1.0)
        edge_fade = edge_t * edge_t * (3.0 - 2.0 * edge_t)
        return np.clip(.5 + core_fade * edge_fade * (spokes * rosette - .5), 0.0, 1.0)
    if pattern == "grid":
        return np.maximum(np.abs(np.cos(x)) ** 10, np.abs(np.cos(y)) ** 10)
    if pattern == "triangles":
        a = np.abs(np.cos(x)) ** 10
        b = np.abs(np.cos(.5 * x + .8660254 * y)) ** 10
        c = np.abs(np.cos(.5 * x - .8660254 * y)) ** 10
        return np.maximum.reduce([a, b, c])
    if pattern == "woodgrain":
        fu = np.mod(u / (2.0 * repeat) + .5, 1.0) - .5
        fv = np.mod(v / repeat + .5, 1.0) - .5
        rings = 12.0 * np.pi * np.hypot(fu, 1.8 * fv)
        return .5 + .5 * np.cos(rings + .8 * np.sin(2.0 * np.pi * fu)
                               + .3 * np.sin(4.0 * np.pi * fv))
    if pattern == "stipple":
        row = np.floor(2.0 * v / repeat)
        fu = np.mod(2.0 * u / repeat + .5 * np.mod(row, 2.0), 1.0) - .5
        fv = np.mod(2.0 * v / repeat, 1.0) - .5
        return np.exp(-18.0 * (fu * fu + fv * fv))
    if pattern == "voronoi":
        su, sv = u / repeat, v / repeat
        iu, iv = np.floor(su), np.floor(sv)
        d1 = np.full_like(su, np.inf, dtype=float)
        d2 = np.full_like(su, np.inf, dtype=float)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                gx, gy = iu + ox, iv + oy
                hx = np.mod(np.sin(gx * 127.1 + gy * 311.7) * 43758.5453, 1.0)
                hy = np.mod(np.sin(gx * 269.5 + gy * 183.3) * 24634.6345, 1.0)
                distance = np.hypot(gx + hx - su, gy + hy - sv)
                nearer = distance < d1
                d2 = np.where(nearer, d1, np.minimum(d2, distance))
                d1 = np.minimum(d1, distance)
        return np.clip(1.0 - (d2 - d1) * 6.0, 0.0, 1.0)
    if pattern == "topo":
        terrain = np.sin(.42 * x) + .72 * np.sin(.36 * y) + .38 * np.sin(.23 * x + .31 * y)
        return np.abs(np.sin(4.5 * terrain)) ** 10
    if pattern == "herringbone":
        row = np.floor(2.0 * v / repeat)
        phase = np.where(np.mod(row, 2.0) == 0, x + y, x - y)
        return np.abs(np.cos(phase)) ** 8
    if pattern == "tread":
        row = np.floor(2.0 * v / repeat)
        fu = np.mod(2.0 * u / repeat + .5 * np.mod(row, 2.0) + .5, 1.0) - .5
        fv = np.mod(2.0 * v / repeat + .5, 1.0) - .5
        centreline = .52 * np.abs(fu) - .13
        lug = np.exp(-105.0 * (fv - centreline) ** 2)
        end_fade = np.clip((.48 - np.abs(fu)) / .08, 0.0, 1.0)
        return lug * end_fade
    if pattern == "carbon":
        # 2×2 twill: one diagonal tow remains visually dominant while the crossing tow
        # passes beneath it. The old implementation switched between two cosine fields in
        # square cells; their contour lines produced the concentric diamonds seen on curved
        # parts instead of anything resembling the diagonal preview.
        pitch = repeat * .5
        dominant = u - v
        crossing = u + v
        stripe = (.5 + .5 * np.cos(2.0 * np.pi * dominant / pitch)) ** 2
        under = (.5 + .5 * np.cos(2.0 * np.pi * crossing / pitch)) ** 5
        stripe_index = np.floor(dominant / pitch + .5)
        along = np.mod(crossing / pitch + .5 * stripe_index, 1.0)
        crown = .72 + .28 * (.5 + .5 * np.cos(2.0 * np.pi * along))
        return .08 + .92 * np.maximum(stripe * crown, .24 * under)
    if pattern == "hammered":
        value = (.5 + .22 * np.sin(1.3 * x + .7 * np.sin(y))
                 + .18 * np.cos(1.7 * y - .4 * np.sin(1.2 * x))
                 + .1 * np.sin(2.8 * x - 2.1 * y))
        return np.clip(value, 0.0, 1.0)
    if pattern == "leather":
        grain = (.48 + .18 * np.sin(1.7 * x + .6 * np.sin(.8 * y))
                 + .14 * np.cos(2.3 * y + .35 * np.sin(x))
                 + .12 * np.abs(np.sin(3.4 * x - 2.6 * y)))
        return np.clip(grain, 0.0, 1.0)
    if pattern == "bark":
        phase = x + 1.05 * np.sin(.31 * y) + .38 * np.sin(1.73 * y)
        return np.abs(np.sin(phase)) ** 2.4
    if pattern == "sandstone":
        value = (.5 + .16 * np.sin(3.1 * x) * np.sin(2.7 * y)
                 + .13 * np.cos(4.3 * x + 3.7 * y)
                 + .11 * np.sin(6.1 * x - 5.3 * y))
        return np.clip(value, 0.0, 1.0)
    if pattern == "droplets":
        su, sv = u / repeat, v / repeat
        row = np.floor(sv)
        fu = np.mod(su + .5 * np.mod(row, 2.0), 1.0) - .5
        fv = np.mod(sv, 1.0) - .5
        size = .18 + .1 * np.mod(np.sin(np.floor(su) * 91.7 + row * 57.3) * 7131.3, 1.0)
        return np.exp(-(fu * fu + fv * fv) / np.maximum(size * size, 1e-6))
    if pattern == "guilloche":
        wave_a = np.sin(2.0 * y + .9 * np.sin(x))
        wave_b = np.sin(2.0 * y - .9 * np.sin(x))
        return np.clip(np.exp(-18.0 * wave_a * wave_a) +
                       np.exp(-18.0 * wave_b * wave_b), 0.0, 1.0)
    if pattern == "circuit":
        horizontal = np.exp(-55.0 * np.sin(y) ** 2) * (.35 + .65 * (np.sin(.5 * x) > -.2))
        vertical = np.exp(-55.0 * np.sin(x + .5 * np.floor(y / np.pi)) ** 2) * (.35 + .65 * (np.cos(.5 * y) > -.1))
        fu, fv = np.mod(u / repeat + .5, 1.0) - .5, np.mod(v / repeat + .5, 1.0) - .5
        pads = np.exp(-24.0 * (fu * fu + fv * fv))
        return np.clip(np.maximum.reduce([horizontal, vertical, pads]), 0.0, 1.0)
    # pebbled: a deterministic multi-frequency field, not random per revision.
    value = (.48 + .22 * np.sin(x) * np.sin(1.17 * y)
             + .18 * np.sin(1.91 * x + .37) * np.cos(1.53 * y)
             + .12 * np.cos(2.73 * x - 1.31 * y))
    return np.clip(value, 0.0, 1.0)


def texture_falloff(mask, faces, rings=3):
    """Feather the relief over a few triangle rings so region edges stay watertight."""
    weight = mask.astype(float)
    if not mask.any() or mask.all():
        return weight
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    crossing = mask[edges[:, 0]] != mask[edges[:, 1]]
    boundary = np.unique(edges[crossing][mask[edges[crossing, 0]], 0]) if crossing.any() else np.array([], int)
    if crossing.any():
        other = edges[crossing][mask[edges[crossing, 1]], 1]
        boundary = np.unique(np.concatenate([boundary, other]))
    frontier = np.zeros(len(mask), bool); frontier[boundary] = True
    weight[frontier] = 1.0 / (rings + 1)
    remaining = mask & ~frontier
    for ring in range(2, rings + 1):
        touching = np.zeros(len(mask), bool)
        ef = frontier[edges[:, 0]] & remaining[edges[:, 1]]
        touching[edges[ef, 1]] = True
        ef = frontier[edges[:, 1]] & remaining[edges[:, 0]]
        touching[edges[ef, 0]] = True
        if not touching.any():
            break
        weight[touching] = ring / (rings + 1)
        remaining &= ~touching
        frontier = touching
    return weight


def texture_crease_falloff(verts, faces, rings=2, crease_degrees=50.0):
    """Fade relief near hard edges where one shared displacement direction is undefined."""
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, length[:, None], out=np.zeros_like(normals),
                        where=length[:, None] > 1e-12)
    first_face = {}
    crease = np.zeros(len(verts), bool)
    for face_index, triangle in enumerate(faces):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            other = first_face.get(edge)
            if other is None:
                first_face[edge] = face_index
            elif np.dot(normals[face_index], normals[other]) < np.cos(np.deg2rad(crease_degrees)):
                crease[list(edge)] = True
    if not crease.any():
        return np.ones(len(verts), float)

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(verts), len(verts))).tocsr()
    adjacency.data[:] = 1.0
    weight = np.ones(len(verts), float)
    weight[crease] = 0.0
    visited = crease.copy()
    frontier = crease.copy()
    for ring in range(1, rings + 1):
        touching = np.asarray(adjacency.dot(frontier.astype(float))).ravel() > 0
        touching &= ~visited
        weight[touching] = ring / (rings + 1)
        visited |= touching
        frontier = touching
    return weight


def surface_texture_field(verts, normals, settings):
    """Map one coherent 2D height field onto the surface in the requested frame."""
    repeat = float(settings["scale_mm"])
    angle = np.deg2rad(float(settings.get("angle_deg", 0.0)))
    ca, sa = np.cos(angle), np.sin(angle)
    def oriented(u, v):
        return texture_pattern(settings["pattern"], ca * u - sa * v,
                               sa * u + ca * v, repeat)
    mapping = settings.get("mapping", "wrap")
    if mapping == "front":
        return oriented(verts[:, 0], verts[:, 2])
    if mapping == "side":
        return oriented(verts[:, 1], verts[:, 2])
    if mapping == "top":
        return oriented(verts[:, 0], verts[:, 1])
    if mapping == "wrap":
        center = (verts.min(axis=0) + verts.max(axis=0)) * .5
        radial = np.hypot(verts[:, 0] - center[0], verts[:, 1] - center[1])
        useful = radial[radial > max(float(radial.max()) * .08, 1e-9)]
        mean_radius = float(np.median(useful)) if len(useful) else max(float(radial.max()), repeat)
        circumference = 2.0 * np.pi * mean_radius
        # Honeycomb repeats horizontally after two staggered columns (sqrt(3) cell spacings).
        # Other motifs include one- and two-repeat tiles, so an even repeat count closes both.
        tile_width = repeat * 1.7320508 if settings.get("pattern") == "honeycomb" else repeat
        if settings.get("pattern") == "honeycomb":
            repeats = max(1, int(round(circumference / tile_width)))
        else:
            repeats = 2 * max(1, int(round(circumference / (2.0 * tile_width))))
        # Front is −Y in nCAD. Put the sole cylindrical seam at the back (+Y), not a side.
        theta = np.arctan2(verts[:, 0] - center[0], -(verts[:, 1] - center[1]))
        wrap_width = repeats * tile_width
        u = (theta + np.pi) / (2.0 * np.pi) * wrap_width
        if settings.get("pattern") == "leopard":
            wrapped = leopard_pattern(u, verts[:, 2] - center[2], repeat,
                                      period_u=wrap_width, orientation=angle)
        else:
            wrapped = oriented(u, verts[:, 2] - center[2])
        # A cylindrical map has zero 2D area on horizontal faces (z is constant), reducing
        # honeycomb, guilloché, and similar motifs to one-dimensional stripes. Project caps
        # in X/Y and blend by surface orientation. This also removes pole convergence on heads.
        cap = oriented(verts[:, 0] - center[0], verts[:, 1] - center[1])
        cap_t = np.clip((np.abs(normals[:, 2]) - .55) / .30, 0.0, 1.0)
        cap_weight = cap_t * cap_t * (3.0 - 2.0 * cap_t)
        return wrapped * (1.0 - cap_weight) + cap * cap_weight
    # Legacy triplanar mapping: useful for irregular solids, but structured patterns can
    # visibly change orientation where the dominant projection plane changes.
    sharp = np.abs(normals) ** 4
    sharp /= np.maximum(sharp.sum(axis=1, keepdims=True), 1e-12)
    xy = oriented(verts[:, 0], verts[:, 1])
    yz = oriented(verts[:, 1], verts[:, 2])
    zx = oriented(verts[:, 2], verts[:, 0])
    return sharp[:, 2] * xy + sharp[:, 0] * yz + sharp[:, 1] * zx


def texture_projection_falloff(normals, mapping):
    """Keep planar projections on the face pair they actually describe.

    A planar X-Z field is constant through Y. Applying it indiscriminately to a closed
    solid therefore turns spots into long stripes around its sides. Front/Back,
    Left/Right, and Top/Bottom are paired planar mappings: retain the field on surfaces
    aligned with that face pair, then feather it out before the surface turns the corner.
    Seamless wrap and triplanar mappings already cover the whole exterior coherently.
    """
    axis = {"front": 1, "side": 0, "top": 2}.get(mapping)
    if axis is None:
        return np.ones(len(normals), float)
    alignment = np.abs(normals[:, axis])
    weight = np.clip((alignment - .15) / .50, 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)


def apply_surface_texture(verts, faces, selected, settings):
    normals = outward_vertex_normals(verts, faces)
    relief = surface_texture_field(verts, normals, settings)
    if settings.get("pattern") == "starburst":
        # Polar spokes can be finer than an imported mesh's local triangle spacing.
        # One adjacency-weighted sample is a geometric antialias pass: it removes
        # alternating pinwheel vertices while retaining the full requested peak height.
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.vstack([edges, edges[:, ::-1]])
        adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                               shape=(len(verts), len(verts))).tocsr()
        adjacency.data[:] = 1.0
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        degree[degree == 0] = 1.0
        relief = .35 * relief + .65 * (adjacency.dot(relief) / degree)
        peak = float(relief.max())
        if peak > 1e-12:
            relief /= peak
    projection = texture_projection_falloff(normals, settings.get("mapping", "wrap"))
    fade = (texture_falloff(selected, faces) * texture_crease_falloff(verts, faces)
            * projection)
    direction = 1.0 if settings.get("mode") == "emboss" else -1.0
    displacement = direction * float(settings["height_mm"]) * relief * fade
    base_tri = verts[faces]
    base_face_normals = np.cross(base_tri[:, 1] - base_tri[:, 0],
                                 base_tri[:, 2] - base_tri[:, 0])
    base_area = np.linalg.norm(base_face_normals, axis=1)
    safety_scale = np.ones(len(verts), float)
    initial_bad = 0
    remaining_bad = 0
    candidate = verts + normals * displacement[:, None]
    for attempt in range(16):
        tri = candidate[faces]
        face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(face_normals, axis=1)
        alignment = np.einsum("ij,ij->i", base_face_normals, face_normals) / np.maximum(
            base_area * area, 1e-20)
        bad = (alignment <= 0.0) | (area < base_area * .02)
        remaining_bad = int(bad.sum())
        if attempt == 0:
            initial_bad = remaining_bad
        if not remaining_bad:
            break
        safety_scale[np.unique(faces[bad])] *= .5
        candidate = verts + normals * (displacement * safety_scale)[:, None]
    return candidate, normals, fade * safety_scale, {
        "face_flips_prevented": initial_bad,
        "relief_limited_vertices": int((safety_scale < .999).sum()),
        "minimum_relief_scale": float(safety_scale.min()),
        "unsafe_faces_remaining": remaining_bad,
    }


def emit_original_baseline(cfg, out, dims, color, edge, mesh_finish):
    """Emit the uploaded mesh unchanged as the 100%-material starting point.

    The first upload is an inspection/editing baseline, not an optimization result. Running
    it through voxelization, SIMP, thresholding, marching cubes, or smoothing would alter the
    source before the user has identified its functional faces. Translating its bounding-box
    minimum to the renderer's origin and welding duplicate STL vertices do not change its surface.
    """
    src = cfg.get("domain_stl")
    if not src:
        raise SystemExit("an original upload baseline needs domain_stl")
    verts, faces = read_stl(src)
    verts = verts - verts.min(axis=0)

    # STL stores triangles independently. Weld coincident copies for an honest watertightness
    # check and a smaller render mesh. Rounded coordinates are keys only; retained coordinates
    # come directly from the source mesh.
    _, first, inverse = np.unique(np.round(verts, 6), axis=0,
                                  return_index=True, return_inverse=True)
    verts = verts[first]
    faces = inverse[faces]
    nondegenerate = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) &
                     (faces[:, 2] != faces[:, 0]))
    faces = faces[nondegenerate]
    if not len(faces):
        raise SystemExit("uploaded mesh has no usable triangles")

    watertight = is_watertight(faces)
    write_stl(out / "result.stl", verts, faces)
    title = "uploaded original  material 100%"
    render_iso(verts, faces, dims, out / "iter001.png", title,
               color=color, edge=edge,
               edge_lw=0.25 if mesh_finish == "faceted" else 0.0)
    render_iso(verts, faces, dims, out / "final.png", title,
               color=color, edge=edge,
               edge_lw=0.25 if mesh_finish == "faceted" else 0.0,
               map_path=out / "final.map.json")
    result = {
        "iterations": 1, "compliance": None, "vol_fraction": 1.0,
        "solid_fraction_after_threshold": 1.0,
        "components": 1, "raw_components": 1, "kept_largest": False,
        "raw_solid_voxels": None, "kept_solid_voxels": None,
        "retained_solid_fraction": 1.0,
        "watertight": watertight, "lattice_applied": False, "lattice": None,
        "regions": None, "domain_upload": {"source_preserved": True},
        "mesh_volume_mm3": mesh_volume(verts, faces) if watertight else None,
        "domain_mm": list(dims), "resolution": list(cfg["resolution"]),
        "stl": "result.stl", "final_png": "final.png", "final_map": "final.map.json",
        "baseline_original": True,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"iter": 1, "compliance": None, "vol": 1.0,
                      "change": 0.0, "preview": "iter001.png"}), flush=True)
    print(json.dumps({"done": True, **result}), flush=True)


def emit_mass_constrained_form(cfg, out, dims, color, edge):
    """Refine an uploaded surface without inventing a structural load case.

    This is deliberately a surface/form operation, not fake topology optimization.  The
    source mesh supplies both the geometry and the reference mass.  Taubin relaxation is
    applied as far as the user's allowed mass deviation permits; protected regions remain
    fixed.  Because mass is volume×density, an unverifiable/open STL is rejected instead of
    pretending that its tolerance was honored.
    """
    src = cfg.get("domain_stl")
    if not src:
        raise SystemExit("a form revision needs an uploaded STL/STEP source")
    raw_verts, raw_faces = read_stl(src)
    raw_verts = raw_verts - raw_verts.min(axis=0)
    _, first, inverse = np.unique(np.round(raw_verts, 6), axis=0,
                                  return_index=True, return_inverse=True)
    original = raw_verts[first]
    faces = inverse[raw_faces]
    good = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) &
            (faces[:, 2] != faces[:, 0]))
    faces = faces[good]
    if not len(faces) or not is_watertight(faces):
        raise SystemExit("mass-constrained form revision needs a watertight source mesh so "
                         "its original mass can be measured")

    source_volume = mesh_volume(original, faces)
    if source_volume <= 1e-9:
        raise SystemExit("source mesh has no measurable enclosed volume")
    tolerance = max(0.0, min(25.0, float(cfg.get("mass_tolerance_pct", 2.0))))
    operation = cfg.get("form_operation", "facet")
    operation = operation if operation in ("relax", "smooth", "polish", "facet") else "facet"

    regions = cfg.get("regions") or []
    frozen = surface_region_vertices(original, faces, regions, rules=("solid", "preserve"))
    texture_selected = surface_region_vertices(original, faces, regions, rules=("preserve",))

    if operation == "facet":
        candidate = original.copy()
    else:
        iterations = {"relax": 4, "smooth": 10, "polish": 20}[operation]
        candidate = taubin_smooth(original, faces, iters=iterations,
                                  frozen=frozen if frozen.any() else None)

    def mass_change(verts):
        return (mesh_volume(verts, faces) / source_volume - 1.0) * 100.0

    strength = 1.0
    if abs(mass_change(candidate)) > tolerance + 1e-9:
        lo, hi = 0.0, 1.0
        for _ in range(28):
            mid = (lo + hi) * 0.5
            trial = original + mid * (candidate - original)
            if abs(mass_change(trial)) <= tolerance:
                lo = mid
            else:
                hi = mid
        strength = lo
        candidate = original + strength * (candidate - original)

    texture = cfg.get("surface_texture")
    compensation_mm = 0.0
    compensation_scale = 1.0
    texture_sampling = {}
    texture_safety = {}
    if texture:
        if not texture_selected.any():
            texture_selected[:] = True                    # no mask means the entire exterior
        candidate, faces, texture_selected, texture_sampling = subdivide_texture_mesh(
            candidate, faces, texture_selected, texture_sampling_edge(texture),
            max_faces=texture_face_cap(texture))
        texture_base = candidate.copy()
        candidate, texture_normals, texture_fade, texture_safety = apply_surface_texture(
            candidate, faces, texture_selected, texture)
        textured_volume = mesh_volume(candidate, faces)
        textured_change = (textured_volume / source_volume - 1.0) * 100.0
        if abs(textured_change) > tolerance + 1e-9:
            pre_compensation = candidate.copy()
            target_change = np.sign(textured_change) * tolerance
            target_volume = source_volume * (1.0 + target_change / 100.0)
            if texture_selected.all():
                # Whole-body normal offsets fold sharp corners back through themselves. Scale
                # only the underlying body, then add the exact millimetre relief back unchanged.
                # Bisection reaches the requested volume without shrinking the configured height.
                center = (texture_base.min(axis=0) + texture_base.max(axis=0)) * .5
                relief_delta = pre_compensation - texture_base

                def scaled_body(scale):
                    return center + (texture_base - center) * scale + relief_delta

                lo, hi = .5, 1.5
                for _ in range(40):
                    mid = (lo + hi) * .5
                    if mesh_volume(scaled_body(mid), faces) < target_volume:
                        lo = mid
                    else:
                        hi = mid
                compensation_scale = (lo + hi) * .5
                candidate = scaled_body(compensation_scale)
                scaled_base = center + (texture_base - center) * compensation_scale
                compensation_mm = float(np.linalg.norm(scaled_base - texture_base, axis=1).max())
            else:
                # A masked texture compensates only outside its retained region, preserving
                # both the protected surface and the requested relief height.
                comp_dir = -1.0 if textured_volume > target_volume else 1.0
                comp_weight = (~texture_selected).astype(float)
                base = candidate.copy()

                def compensated(offset):
                    return base + texture_normals * (comp_dir * offset * comp_weight)[:, None]

                def reached(volume):
                    return volume <= target_volume if comp_dir < 0 else volume >= target_volume

                hi = .02
                max_offset = max(float(texture["height_mm"]) * 4.0,
                                 float(np.max(np.ptp(original, axis=0))) * .08, .5)
                while hi < max_offset and not reached(mesh_volume(compensated(hi), faces)):
                    hi *= 2.0
                hi = min(hi, max_offset)
                if not reached(mesh_volume(compensated(hi), faces)):
                    raise SystemExit("requested texture height cannot satisfy the selected mass tolerance; "
                                     "reduce height, enlarge the retained region, or allow more mass change")
                lo = 0.0
                for _ in range(32):
                    mid = (lo + hi) * .5
                    if reached(mesh_volume(compensated(mid), faces)):
                        hi = mid
                    else:
                        lo = mid
                compensation_mm = hi
                candidate = compensated(hi)

            before = pre_compensation[faces]
            after = candidate[faces]
            before_normal = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
            after_normal = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
            alignment = np.einsum("ij,ij->i", before_normal, after_normal)
            if np.any(alignment <= 0.0):
                raise SystemExit("mass compensation would fold texture triangles at the mask boundary; "
                                 "increase mass tolerance, reduce texture height, or enlarge the mask")

    output_volume = mesh_volume(candidate, faces)
    change_pct = (output_volume / source_volume - 1.0) * 100.0
    ratio = output_volume / source_volume
    write_stl(out / "result.stl", candidate, faces)
    mode = f"mass-constrained-{operation}"
    relief_title = (f"  {texture['mode']} {texture['pattern']} {texture['height_mm']:g}mm"
                    f" {texture.get('angle_deg', 0):g}deg {texture.get('mapping', 'wrap')}"
                    if texture else "")
    title = f"{operation}{relief_title}  mass {change_pct:+.2f}%  allowed ±{tolerance:g}%"
    # Texture tessellation adds sampling triangles, not intentional visible facets.
    edge_lw = 0.35 if operation == "facet" and not texture else 0.0
    render_iso(candidate, faces, dims, out / "iter001.png", title,
               color=color, edge=edge, edge_lw=edge_lw)
    render_iso(candidate, faces, dims, out / "final.png", title,
               color=color, edge=edge, edge_lw=edge_lw,
               map_path=out / "final.map.json")
    texture_scope = None
    if texture:
        if texture_selected.all():
            texture_scope = {"front": "front/back-facing surfaces",
                             "side": "left/right-facing surfaces",
                             "top": "top/bottom-facing surfaces"}.get(
                                 texture.get("mapping"), "whole exterior")
        else:
            texture_scope = "retained regions"
    result = {
        "iterations": 1, "compliance": None, "vol_fraction": ratio,
        "design_vol_fraction": ratio, "requested_design_vol_fraction": 1.0,
        "passive_fraction": 0.0, "load_cases": 0,
        "best_iteration": None, "selected_iteration": None,
        "surface_mode": mode, "surface_iso": None,
        "surface_voxel_fraction": None, "connectivity_added_voxels": 0,
        "solid_fraction_after_threshold": ratio,
        "components": 1, "raw_components": 1, "kept_largest": False,
        "raw_solid_voxels": None, "kept_solid_voxels": None,
        "retained_solid_fraction": 1.0, "watertight": True,
        "lattice_applied": False, "lattice": None,
        "regions": {"protected_surfaces": int(frozen.sum())} if frozen.any() else None,
        "domain_upload": {"source_preserved": False},
        "source_volume_mm3": source_volume, "mesh_volume_mm3": output_volume,
        "mass_change_pct": change_pct, "mass_tolerance_pct": tolerance,
        "form_strength": strength, "revision_mode": "form",
        "surface_texture": ({**texture,
                             "scope": texture_scope,
                             "selected_vertices": int(texture_selected.sum()),
                             "mass_compensation_mm": compensation_mm,
                             "mass_compensation_scale": compensation_scale,
                             **texture_sampling, **texture_safety} if texture else None),
        "domain_mm": list(dims), "resolution": list(cfg["resolution"]),
        "stl": "result.stl", "final_png": "final.png", "final_map": "final.map.json",
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"iter": 1, "compliance": None, "vol": ratio,
                      "change": 0.0, "preview": "iter001.png"}), flush=True)
    print(json.dumps({"done": True, **result}), flush=True)


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
    # mesh detail: upsample the CONTINUOUS density before marching so the surface is sub-voxel
    # smooth (decoupled from the coarse solve grid), NOT the binary staircase. Faceted styles
    # want the low-poly crystalline look, so they stay at 1×.
    mesh_detail = 1 if mesh_finish == "faceted" else max(1, min(3, int(cfg.get("mesh_detail", 2))))
    lattice = bool(cfg.get("lattice"))          # S6: TPMS/periodic infill (Lattice/Gothic)
    lattice_cell = cfg.get("lattice_cell")      # mm; None → auto
    lattice_kind = cfg.get("lattice_kind", "gyroid")   # gyroid|schwarz_p|diamond|ribs|tracery
    lattice_mode = cfg.get("lattice_mode", "sheet")    # sheet (thin double-wall) | network (ligaments)

    if cfg.get("preserve_original") and volfrac >= 0.999:
        emit_original_baseline(cfg, out, (Lx, Ly, Lz), rcol, ecol, mesh_finish)
        return
    if cfg.get("revision_mode") == "form":
        emit_mass_constrained_form(cfg, out, (Lx, Ly, Lz), rcol, ecol)
        return

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
        has_box = isinstance(b, (list, tuple)) and len(b) == 6
        tris = r.get("tris")
        rule = r.get("rule", "solid")
        if tris:                                 # per-face patch → exact surface shell
            m = voxels_from_tris(tris, nx, ny, nz)
            if not m.any():
                continue
            if rule == "solid":                 # structural pads need a load-transfer band;
                m = ndimage.binary_dilation(m)   # preserve keeps only the sampled surface layer
        elif has_box:                            # legacy axis-aligned box
            m = elements_in_box([float(v) for v in b], nx, ny, nz)
        else:
            continue
        if rule == "void":
            void_mask |= m
        elif rule == "preserve":
            solid_mask |= m
            preserve_mask |= m
            if has_box:                          # pin these verts flat during smoothing
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
    domain_surface = None
    domain_stl = cfg.get("domain_stl")
    if domain_stl:
        try:
            dv, df = read_stl(domain_stl)
            dv = dv - dv.min(axis=0)                   # bbox min → origin (aligns with domain_mm)
            # honest: ray-parity voxelization assumes a watertight mesh. Weld coincident
            # verts and check — an open/non-manifold upload gives an imprecise domain.
            _, inv = np.unique(np.round(dv, 4), axis=0, return_inverse=True)
            _, first6, render_inv = np.unique(np.round(dv, 6), axis=0,
                                               return_index=True, return_inverse=True)
            domain_surface = (dv[first6], render_inv[df])
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
            lm = (ndimage.binary_dilation(voxels_from_tris(ld["tris"], nx, ny, nz))
                  if ld.get("tris") else elements_in_box(ld["box"], nx, ny, nz))
            if not (lm & domain_mask).any():
                print(json.dumps({"warn": "a load region doesn't touch the uploaded part — the load "
                      "won't transfer into it; move it in the recipe"}), flush=True)

    # Loads and supports are non-design interface pads in a meaningful topology solve: material
    # may evolve between them, but the optimizer must not erode the places where force enters or
    # leaves the part. A one-cell dilation gives the surface nodes a robust transfer band.
    support_interface_mask = np.zeros((nx, ny, nz), bool)
    for s in cfg["supports"]:
        support_interface_mask |= elements_in_box(s["box"], nx, ny, nz)
    raw_load_masks = []
    for ld in cfg["loads"]:
        lm = (voxels_from_tris(ld["tris"], nx, ny, nz)
              if ld.get("tris") else elements_in_box(ld["box"], nx, ny, nz))
        raw_load_masks.append(lm)
    support_interface_mask = ndimage.binary_dilation(support_interface_mask)
    load_interface_masks = [ndimage.binary_dilation(m) for m in raw_load_masks]
    interface_mask = support_interface_mask.copy()
    for m in load_interface_masks:
        interface_mask |= m
    if domain_mask is not None:
        support_interface_mask &= domain_mask
        load_interface_masks = [m & domain_mask for m in load_interface_masks]
        interface_mask &= domain_mask
    support_interface_mask &= ~void_mask
    load_interface_masks = [m & ~void_mask for m in load_interface_masks]
    interface_mask &= ~void_mask                       # an explicit keep-void still wins (and warns)
    solid_mask |= interface_mask
    solid_mask &= ~void_mask
    has_passive = bool(solid_mask.any() or void_mask.any())

    active = domain_mask                               # designable region (None ⇒ the whole box);
    # volfrac + reporting are RELATIVE to this so an uploaded part with low occupancy isn't an
    # impossible target measured over the full bounding box.

    # Boundary conditions must live on the design domain too. Previously an uploaded part was
    # optimized inside `domain_mask`, but loads/supports were still selected from every grid node
    # in their bounding boxes. A thin/curved part therefore put most of the force on empty-space
    # nodes connected only by EMIN stiffness. Compliance was dominated by that artificial void,
    # the density field fragmented, and final component cleanup appeared to destroy a good part.
    domain_nodes = None
    if domain_mask is not None:
        domain_nodes = np.unique((edof[domain_mask.ravel(), 0::3] // 3).ravel())

    def boundary_nodes(entry):
        tris = entry.get("tris")
        if tris:
            mask = ndimage.binary_dilation(voxels_from_tris(tris, nx, ny, nz))
            if domain_mask is not None:
                mask &= domain_mask
            if mask.any():
                return np.unique((edof[mask.ravel(), 0::3] // 3).ravel())
        ids = nodes_in_box(entry["box"], nx, ny, nz)
        if domain_nodes is not None:
            ids = np.intersect1d(ids, domain_nodes, assume_unique=False)
        return ids

    support_nodes = [boundary_nodes(s) for s in cfg["supports"]]
    support_nodes = [ids for ids in support_nodes if len(ids)]
    if not support_nodes:
        raise SystemExit("no support nodes touch the design domain — check supports[].box")
    fixed_node_ids = np.unique(np.concatenate(support_nodes))
    fixed = np.concatenate([3 * fixed_node_ids + d for d in (0, 1, 2)])
    fixed = np.unique(fixed)
    free = np.setdiff1d(np.arange(ndof), fixed)

    fixed_nodes = set(fixed_node_ids.tolist())
    load_cases = []
    for ld in cfg["loads"]:
        ids = boundary_nodes(ld)
        # Loads landing on supported nodes would be silently absorbed by the supports;
        # normalize over free loaded nodes on the actual design domain only.
        ids = np.array([i for i in ids if i not in fixed_nodes], dtype=np.int64)
        if not len(ids):
            raise SystemExit(f"no free load nodes touch the design domain in box {ld['box']} "
                             "(move the load region or separate it from the supports)")
        d = np.array(ld["dir"], dtype=float)
        d /= np.linalg.norm(d) or 1.0
        case = np.zeros(ndof)
        for axis in range(3):
            case[3 * ids + axis] = d[axis] / len(ids)
        load_cases.append(case)
    F = np.column_stack(load_cases)                    # independent cases: no cancellation

    active_grid = domain_mask if domain_mask is not None else np.ones((nx, ny, nz), bool)
    design_mask = active_grid & ~solid_mask & ~void_mask
    active_count = int(active_grid.sum())
    design_count = int(design_mask.sum())
    passive_fraction = float(solid_mask[active_grid].mean()) if active_count else 0.0
    if not design_count:
        print(json.dumps({"warn": "solid/preserve/interface zones consume the entire design domain; "
                          "there is no free region left for generative topology"}), flush=True)

    seed = cfg.get("seed")
    asym = float(cfg.get("asymmetry", 0.0))
    seed_bias = None
    if seed is not None and asym > 0:
        # CREATIVITY Axis 2: bias sensitivities (not the initial density) so every solve still
        # begins from the intact domain but can settle into a different non-convex optimum.
        rng = np.random.default_rng(int(seed))
        seed_bias = 1.0 + 0.15 * asym * (rng.random((nx, ny, nz)) - 0.5) * 2.0

    # Continuation is essential for a generative solve. Starting directly at 10% density
    # destroys the domain before compliance has identified a load path. Start intact and
    # progressively remove only the FREE design material while SIMP penalization ramps up.
    x = np.where(active_grid, 1.0, 0.0)
    x[void_mask] = 0.0
    x[solid_mask] = 1.0

    def finish_surface(verts, faces):
        """Apply the configured finish identically to printable previews and final output."""
        if mesh_finish not in ("smooth", "g2-polish") or not len(faces):
            return verts
        frozen = None
        if preserve_boxes:
            frozen = np.zeros(len(verts), bool)
            for bx in preserve_boxes:
                frozen |= ((verts[:, 0] >= bx[0] * Lx) & (verts[:, 0] <= bx[3] * Lx) &
                           (verts[:, 1] >= bx[1] * Ly) & (verts[:, 1] <= bx[4] * Ly) &
                           (verts[:, 2] >= bx[2] * Lz) & (verts[:, 2] <= bx[5] * Lz))
        return taubin_smooth(verts, faces,
                             iters=16 if mesh_finish == "g2-polish" else 8,
                             frozen=frozen)

    ramp_iters = max(2, min(max_iters, int(round(max_iters * 0.45))))
    change, comp = 1.0, 0.0
    best_x, best_comp, best_iteration = None, np.inf, None
    for it in range(1, max_iters + 1):
        progress = min(1.0, (it - 1) / max(1, ramp_iters - 1))
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        design_target = 1.0 - (1.0 - volfrac) * smooth_progress
        penal_now = 1.0 + (penal - 1.0) * smooth_progress

        xp = EMIN + x.ravel() ** penal_now * (E0 - EMIN)
        sK = (KE.ravel()[None, :] * xp[:, None]).ravel()
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        Kf = (K[free][:, free]).tocsc()
        rhs = F[free, :]
        U = np.zeros((ndof, F.shape[1]))
        # The FEA solve is ~96% of each iteration. CHOLMOD (supernodal Cholesky,
        # multithreaded) is the fast path; spsolve with the symmetric-friendly
        # MMD_AT_PLUS_A ordering (~1.5x over COLAMD) / preconditioned CG are the
        # dependency-free fallbacks.
        if HAVE_CHOLMOD:
            try:
                sol = cho_factor(Kf).solve(rhs)
            except CholmodNotPositiveDefiniteError:      # ill-conditioned → exact fallback
                sol = spsolve(Kf, rhs, permc_spec="MMD_AT_PLUS_A")
        elif ndof <= 60_000:
            sol = spsolve(Kf, rhs, permc_spec="MMD_AT_PLUS_A")
        else:
            from scipy.sparse import diags
            Minv = diags(1.0 / Kf.diagonal())
            cols = []
            for case_i in range(rhs.shape[1]):
                case_sol, info = cg(Kf, rhs[:, case_i], rtol=1e-6, maxiter=8000, M=Minv)
                if info != 0:                # non-converged CG would silently
                    case_sol = spsolve(Kf, rhs[:, case_i], permc_spec="MMD_AT_PLUS_A")
                cols.append(case_sol)
            sol = np.column_stack(cols)
        sol = np.asarray(sol)
        if sol.ndim == 1:
            sol = sol[:, None]
        U[free, :] = sol
        Ue = U[edof]
        ce_cases = np.einsum("eic,ij,ejc->ec", Ue, KE, Ue)
        ce = ce_cases.sum(axis=1).reshape(nx, ny, nz)
        comp = float((xp.reshape(nx, ny, nz) * ce).sum())
        current_design_v = float(x[design_mask].mean()) if design_count else 0.0
        # Sensitivity filters and aggressive low-volume targets are not guaranteed to improve
        # monotonically. Retain the stiffest target-feasible state instead of blindly shipping
        # the last oscillation—a major source of the old final-frame geometry jump.
        target_tolerance = max(0.005, 0.02 * volfrac)
        if (progress >= 1.0 and current_design_v <= volfrac + target_tolerance and
                comp < best_comp):
            best_x, best_comp, best_iteration = x.copy(), comp, it
        dc = -penal_now * x ** (penal_now - 1) * (E0 - EMIN) * ce
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
        if seed_bias is not None:
            dc *= seed_bias
        dc = np.where(design_mask, dc, 0.0)

        # Optimality-criteria bisection applies the requested material fraction only to the
        # designable region. Solid, preserve, load, and support pads are non-design material;
        # counting them against the target previously starved every connecting load path.
        xnew = x.copy()
        if design_count:
            l1, l2, move = 0.0, 1e9, 0.12
            while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
                lmid = 0.5 * (l1 + l2)
                candidate = np.clip(x * np.sqrt(np.maximum(1e-12, -dc) / lmid),
                                    np.maximum(0.0, x - move), np.minimum(1.0, x + move))
                frac = float(candidate[design_mask].mean())
                if frac > design_target:
                    l1 = lmid
                else:
                    l2 = lmid
            xnew[design_mask] = candidate[design_mask]
        change = float(np.abs(xnew - x).max())
        x = symmetrize(xnew, symmetry)      # S3: enforce style symmetry (mean-preserving)
        x[~active_grid] = 0.0
        x[solid_mask] = 1.0
        x[void_mask] = 0.0
        design_vfrac = float(x[design_mask].mean()) if design_count else 0.0
        vfrac = float(x[active_grid].mean())                         # total domain-relative material
        png = out / f"iter{it:03d}.png"
        preview_x, preview_comp, preview_iteration = x, comp, it
        if (progress >= 1.0 and best_x is not None and
                comp > best_comp * 1.02):
            preview_x, preview_comp, preview_iteration = best_x, best_comp, best_iteration
        if it == 1 and domain_surface is not None:           # first frame is the intact source
            pv, pf = domain_surface
        elif progress >= 1.0 and not lattice:
            preview_solid, preview_continuous, _ = continuous_density_solid(
                preview_x, active_grid, support_interface_mask, load_interface_masks)
            if preview_continuous:
                pv, pf = density_mesh(preview_x, (Lx, Ly, Lz), detail=mesh_detail,
                                      active=active)
            else:
                preview_solid, _ = printable_projection(
                    preview_x, active_grid, design_mask, solid_mask, void_mask,
                    support_interface_mask, load_interface_masks, volfrac)
                pv, pf = binary_surface_mesh(preview_solid, (Lx, Ly, Lz), mesh_detail)
            pv = finish_surface(pv, pf)
        else:
            pv, pf = density_mesh(preview_x, (Lx, Ly, Lz), detail=mesh_detail,
                                  active=active)
        source_note = (f"  best {preview_iteration}" if preview_iteration != it else "")
        render_iso(pv, pf, (Lx, Ly, Lz), png,
                   f"iter {it}{source_note}  compliance {preview_comp:.3f}  vol {vfrac:.3f}",
                   silhouette=preview_x.max(axis=1), color=rcol, edge=ecol)
        print(json.dumps({"iter": it, "compliance": round(comp, 4),
                          "vol": round(vfrac, 4), "design_vol": round(design_vfrac, 4),
                          "target_design_vol": round(design_target, 4),
                          "penal": round(penal_now, 3), "load_cases": F.shape[1],
                          "preview_iteration": preview_iteration,
                          "change": round(change, 4), "preview": png.name}), flush=True)
        if progress >= 1.0 and it >= 10 and change < 0.02:
            break

    # Prefer the converged last state when it is within 2% of the historical best. Selecting a
    # visually different old iterate for a sub-percent compliance gain made "final" jump backward
    # even when the visible solve had stabilized. A materially degraded tail still falls back to
    # the best target-feasible state, which late previews already show as best-so-far.
    selected_iteration = it
    if best_x is not None and comp > best_comp * 1.02:
        x, comp, selected_iteration = best_x, best_comp, best_iteration

    # Use the exact continuous preview surface when it is already one load-bearing component.
    # Only invoke volume projection/connectivity repair for a genuinely unsafe density surface.
    solid, continuous_surface, surface_iso = continuous_density_solid(
        x, active_grid, support_interface_mask, load_interface_masks)
    surface_mode = "continuous-density" if continuous_surface and not lattice else "projected"
    if surface_mode == "continuous-density":
        n = 1
        raw_solid_voxels = int(solid.sum())
        projection_info = {"connectivity_added_voxels": 0}
    else:
        solid, projection_info = printable_projection(
            x, active_grid, design_mask, solid_mask, void_mask,
            support_interface_mask, load_interface_masks, volfrac)
        n = int(projection_info["raw_components"])
        raw_solid_voxels = int(projection_info["raw_solid_voxels"])
    connectivity_added_voxels = int(projection_info["connectivity_added_voxels"])
    kept_solid_voxels = int(solid.sum())
    retained_solid_fraction = (kept_solid_voxels / raw_solid_voxels
                               if raw_solid_voxels else 1.0)
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
    elif surface_mode == "continuous-density":       # exact same field/iso/detail as late preview
        verts, faces = density_mesh(x, (Lx, Ly, Lz), detail=mesh_detail, active=active)
        verts = finish_surface(verts, faces)
    elif solid.any():                                 # repaired low-density fallback
        verts, faces = binary_surface_mesh(solid, (Lx, Ly, Lz), mesh_detail)
        verts = finish_surface(verts, faces)
    else:                                             # degenerate (no material) → empty mesh
        verts, faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    write_stl(out / "result.stl", verts, faces)
    density_vfinal = float(x[active_grid].mean())
    density_design_final = float(x[design_mask].mean()) if design_count else 0.0
    surface_vfinal = float(solid[active_grid].mean())
    surface_design_final = float(solid[design_mask].mean()) if design_count else 0.0
    # The optimization constraint and live charts describe integrated density, not a particular
    # display isovalue. Keep that metric continuous across the last iteration and final frame;
    # report surface occupancy separately for diagnostics.
    vfinal = density_vfinal
    design_final = density_design_final
    render_iso(verts, faces, (Lx, Ly, Lz), out / "final.png",
               f"final  compliance {comp:.3f}  vol {vfinal:.3f}", color=rcol, edge=ecol,
               edge_lw=0.25 if mesh_finish == "faceted" else 0.0,
               map_path=out / "final.map.json")           # Slice 14 pixel-map contract
    watertight = bool(len(faces)) and is_watertight(faces)
    result = {
        "iterations": it, "compliance": comp,
        "vol_fraction": vfinal,
        "design_vol_fraction": design_final,
        "requested_design_vol_fraction": volfrac,
        "optimized_density_vol_fraction": density_vfinal,
        "optimized_density_design_vol_fraction": density_design_final,
        "passive_fraction": passive_fraction,
        "load_cases": int(F.shape[1]),
        "best_iteration": best_iteration,
        "selected_iteration": selected_iteration,
        "surface_mode": surface_mode,
        "surface_iso": surface_iso,
        "surface_voxel_fraction": surface_vfinal,
        "surface_design_voxel_fraction": surface_design_final,
        "connectivity_added_voxels": connectivity_added_voxels,
        "solid_fraction_after_threshold": float(solid.mean()),
        "components": final_pieces, "raw_components": int(n),
        "kept_largest": bool(n > final_pieces),
        "raw_solid_voxels": raw_solid_voxels,
        "kept_solid_voxels": kept_solid_voxels,
        "retained_solid_fraction": retained_solid_fraction,
        "watertight": watertight, "lattice_applied": bool(lattice_applied),
        "lattice": lattice_info,
        "regions": ({"solid_vox": int(solid_mask.sum()), "void_vox": int(void_mask.sum()),
                     "preserve_vox": int(preserve_mask.sum()),
                     "interface_vox": int(interface_mask.sum())}
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
