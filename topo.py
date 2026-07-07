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


def is_watertight(faces):
    """Every undirected edge must appear exactly twice."""
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def render_iso(verts, faces, dims, path, title, silhouette=None,
               color="#7ba7c9", edge="none", edge_lw=0.0):
    """One 3D frame from a FIXED camera with FIXED axis limits, so the
    structure visibly evolves in place across iteration frames. `color`/`edge`
    come from the active style's palette (edge shows only when edge_lw>0 — used
    by the faceted mesh finish)."""
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
    fig.savefig(path, dpi=95)
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


def gyroid_infill(solid, dims, cell_mm, skin_vox=2):
    """S6 feature language 'lattice': hollow the SOLID interior with a gyroid TPMS
    lattice while keeping a solid outer skin — the printed-light 'solid skin +
    exposed lattice windows' look. Returns the perforated bool mask (still one
    connected body via the skin). solid: bool (nx,ny,nz)."""
    nx, ny, nz = solid.shape
    Lx, Ly, Lz = dims
    xs = (np.arange(nx) + 0.5) * Lx / nx
    ys = (np.arange(ny) + 0.5) * Ly / ny
    zs = (np.arange(nz) + 0.5) * Lz / nz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    k = 2 * np.pi / max(cell_mm, 1e-6)
    g = (np.sin(k * X) * np.cos(k * Y) + np.sin(k * Y) * np.cos(k * Z)
         + np.sin(k * Z) * np.cos(k * X))
    # ONE gyroid labyrinth (g>0) — a thick interconnected SOLID network, not a thin
    # |g|<t shell. A solid region stays manifold through marching-cubes (watertight),
    # where thin walls pinch to single voxels and go non-manifold.
    net = g > 0.0
    interior = ndimage.binary_erosion(solid, iterations=max(1, skin_vox))
    skin = solid & ~interior                       # keep the boundary shell solid
    return solid & (skin | net)


def taubin_smooth(verts, faces, iters=12, lam=0.5, mu=-0.53):
    """S5: shrink-free Laplacian (Taubin λ|μ) smoothing of the marching-cubes mesh
    — turns the voxel staircase into continuous organic surfaces (Bones/Evolved).
    Connectivity (faces) is untouched, so watertightness is preserved. The λ/μ
    pair inflates back what a plain Laplacian would shrink."""
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
    for _ in range(iters):
        for step in (lam, mu):
            v = v + step * (W.dot(v) / deg[:, None] - v)
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
    lattice = bool(cfg.get("lattice"))          # S6: gyroid infill (Lattice/Gyroid style)
    lattice_cell = cfg.get("lattice_cell")      # mm; None → auto

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
        # optimality criteria update with bisection
        l1, l2, move = 0.0, 1e9, 0.2
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(x * np.sqrt(np.maximum(1e-12, -dc) / lmid),
                           np.maximum(0.0, x - move), np.minimum(1.0, x + move))
            if xnew.mean() > volfrac:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x = symmetrize(xnew, symmetry)      # S3: enforce style symmetry (mean-preserving)
        png = out / f"iter{it:03d}.png"
        pv, pf = density_mesh(x, (Lx, Ly, Lz))
        render_iso(pv, pf, (Lx, Ly, Lz), png,
                   f"iter {it}  compliance {comp:.3f}  vol {x.mean():.3f}",
                   silhouette=x.max(axis=1), color=rcol, edge=ecol)
        print(json.dumps({"iter": it, "compliance": round(comp, 4),
                          "vol": round(float(x.mean()), 4),
                          "change": round(change, 4), "preview": png.name}), flush=True)
        if it >= 10 and change < 0.02:
            break

    # ---- extract final geometry: threshold, keep largest component
    thr = 0.5
    solid = x >= thr
    lab, n = ndimage.label(solid)
    if n > 1:
        keep = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        solid = lab == keep
    hx, hy, hz = Lx / nx, Ly / ny, Lz / nz

    def largest(mask):
        lb, nn = ndimage.label(mask)
        return (lb == (np.argmax(np.bincount(lb.ravel())[1:]) + 1)) if nn > 1 else mask, nn

    # S6 lattice is BEST-EFFORT: a gyroid carved into a thin optimized structure can
    # go non-manifold (unprintable). Try it; keep it only if the mesh stays watertight,
    # otherwise ship the solid — a styled part is never a broken STL.
    lattice_applied = False
    body = solid
    if lattice and solid.any():
        cell = float(lattice_cell) if lattice_cell else max(min(Lx, Ly, Lz) / 5.0,
                                                            4 * max(hx, hy, hz))
        latt, _ = largest(gyroid_infill(solid, (Lx, Ly, Lz), cell, skin_vox=1))
        tv, tf, _, _ = measure.marching_cubes(np.pad(latt.astype(float), 1), 0.5, spacing=(hx, hy, hz))
        if is_watertight(tf):
            body, lattice_applied = latt, True
    solid = body
    if solid.any():                                   # marching_cubes needs the level
        pad = np.pad(solid.astype(float), 1)          # inside the data range — an empty
        verts, faces, _, _ = measure.marching_cubes(pad, 0.5, spacing=(hx, hy, hz))
        verts -= [hx * 0.5, hy * 0.5, hz * 0.5]       # unpad + half-voxel (cell-centered)
    else:                                             # degenerate (no material) → empty mesh
        verts, faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    if mesh_finish in ("smooth", "g2-polish") and len(faces):    # S5: polish the staircase
        verts = taubin_smooth(verts, faces, iters=16 if mesh_finish == "g2-polish" else 8)
    write_stl(out / "result.stl", verts, faces)
    render_iso(verts, faces, (Lx, Ly, Lz), out / "final.png",
               f"final  compliance {comp:.3f}  vol {x.mean():.3f}", color=rcol, edge=ecol,
               edge_lw=0.25 if mesh_finish == "faceted" else 0.0)
    watertight = bool(len(faces)) and is_watertight(faces)
    result = {
        "iterations": it, "compliance": comp, "vol_fraction": float(x.mean()),
        "solid_fraction_after_threshold": float(solid.mean()),
        "components": int(n), "kept_largest": bool(n > 1),
        "watertight": watertight, "lattice_applied": bool(lattice_applied),
        "mesh_volume_mm3": mesh_volume(verts, faces) if watertight else None,
        "domain_mm": [Lx, Ly, Lz], "resolution": [nx, ny, nz],
        "stl": "result.stl", "final_png": "final.png",
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"done": True, **result}), flush=True)


if __name__ == "__main__":
    main()
