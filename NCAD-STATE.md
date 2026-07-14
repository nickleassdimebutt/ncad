# nCAD — implementation state & continuation handoff

Working reference for continuing the nCAD build across a context compaction.
nCAD = a **local generative-CAD web app** for Nick (solo hardware maker). All AI
is the LOCAL model (qwen3.6-35b via LM Studio) used ONLY to configure/interpret;
ALL geometry is deterministic. One GPU → jobs run strictly serial.

## Locations & commands
- GIT: `~/Documents/GitHub/nCAD/` is the local repo on branch `main`, tracking
  `git@github.com:nickleassdimebutt/ncad.git`. Source only (.gitignore excludes
  .venv/__pycache__/.bak/.DS_Store/server.log).
- App dir: `~/Documents/GitHub/nCAD/` — `server.py` (Python stdlib HTTP, port 8765),
  `index.html` (single-file vanilla JS/CSS UI), `topo.py` (SIMP solver),
  `templates/` (parametric templates + manifest.json).
- Generated projects/workspace: `~/Documents/CAD/` by default, configurable with
  `NCAD_CAD_ROOT`.
- Engine venv: `~/Documents/CAD/.venv` by default (agentcad 0.3.1 + build123d 0.10).
  Solver venv: `~/Documents/GitHub/nCAD/.venv` by default
  (numpy/scipy/scikit-image/matplotlib; scikit-sparse optional).
- Bootstrap: `bin/bootstrap` (macOS/Linux) or `bin/bootstrap.ps1` (Windows) recreates
  disposable venvs from requirements manifests.
- Launch: `ncad` (opens UI, starts server if down) — `~/.local/bin/ncad` should point
  to `~/Documents/GitHub/nCAD/bin/ncad`. Desktop shortcut: `~/Desktop/nCAD.command`.
  Old UI backup: `index.html.bak-*`.
- Preview during dev: start from `~/Documents/GitHub/nCAD`; the server reads index.html
  fresh per request (HTML edits need only a reload; server.py edits need a restart).
- Design proposal (the spec we're building to): Artifact
  https://claude.ai/code/artifact/9423c285-c1d3-4d33-9354-495a4af110ba ; full text at
  `/private/tmp/.../scratchpad/ncad-spec.md` (may be gone post-session) — key points below.

## Architecture
**Two engines:**
- STACK (parametric): LLM picks a template (plate/flange/enclosure) + params ONCE →
  deterministic loop `agentcad run --params` → versioned STEP + 4-view PNG + metrics,
  verifies vs spec (bbox + cyl features), auto-corrects. ~12s. Output: STEP.
- STRUCTURAL (topology opt / SIMP): LLM translates brief → supports+loads+volfrac ONCE
  → `topo.py` evolves a voxel density field ~50 iters, each rendered as a fixed-camera
  3D isosurface PNG. ~2min. Output: watertight STL.

**Revisions are LLM-FREE**: edit the recipe (job.config) → POST /revise → deterministic
re-run → new job `<base>-rN`, nested under parent by explicit `root` id.

## Backend API (server.py)
- `GET /api/jobs` → [public_job]. public_job fields: id,name,state,iterations,brief,bbox,
  bbox_tol,features,objective,max_iters,model,generator,created,updated,started,root,config,log(last 80).
- `GET /api/workspace` → external agentcad output (read-only tray).
- `GET /api/status` → {now, active:{id,name,state,generator,started}|null, queue:[ids]}.
  active comes from `ACTIVE_JOB["id"]` (set by worker on dequeue — honest during init).
- `POST /api/jobs` → create+queue (expert path). `POST /api/jobs/{id}/revise` {config} → new -rN job.
  `POST /api/jobs/{id}/cancel`.
- Guided sessions: `POST /api/sessions` {name,request} → {id,questions}. `POST .../answers`
  {answers} → synthesize spec ONLY (stage 'drafted'), returns {spec,model}, does NOT run.
  `POST .../run` {engine?,bbox?,features?,objective?} → submits confirmed job (stage-gated:
  requires 'drafted', idempotent, atomic 'submitting' claim). GET `/api/sessions/{id}`.
- job.config shapes: structural `{engine:'structural',domain_mm:[x,y,z],volfrac,supports:[{box:[6]}],loads:[{box:[6],dir:[3]}]}`;
  stack `{engine:'stack',template,params:{...}}`. Boxes are NORMALIZED 0..1 [x0,y0,z0,x1,y1,z1].
  z is VERTICAL; gravity=[0,0,-1]; z-min=floor, z-max=top.
- Structural config: `llm_structural_config()` (one LLM call, retries) with
  `heuristic_structural()` fallback that is now ARCHETYPE-AWARE (table→floor+top load,
  bridge→both ends, column→floor+top, hook→top, else cantilever/shelf).

## Frontend (index.html) — architecture
Three-zone shell: rail (job tree, nested revisions by `root`) | morphing center stage
(new→drafting→draft(confirm gate)→job) | docked recipe panel + metrics + engineer log.
State object `S`: jobs,status,sel,selIter,session,mode,expanded,stageKind,stageJob,dockKey,
film,play,pendingJob. **Clobber-safe rendering discipline (critical):** poll() every 2-4s
calls renderChrome() (tree+gpu, full rebuild ok) + renderMain(); the STAGE only remounts on
kind/job change (never on poll while editing the gate); the DOCK recipe only rebuilds when
`dockKey` (id+state+config) changes; filmstrip DIFF-APPENDS (S.film[id] count). GPU strip
ticks 1/s. `escAttr()` for all attribute values (not `esc()`); file hrefs use encodeURI.
`mergeJobs()` upserts by id + preserves pendingJob → optimistic append can't dup/drop.

## DONE & verified (slices 1–6 of the proposal)
1. Three-zone shell + Recipe-as-spine (Guided/Expert toggle deleted). 2. Engine segmented
control at confirm gate (auto-choose + override). 3. Zero-blank state machine (last frame
under wash; failure keeps frame+editable recipe). 4. Honest optimistic queue + GPU strip
(live clock+ETA). 5. Nested -rN revision tree (by root id) + config-diff. 6. Filled
configuring wait (skeleton "Reading your brief…"). Guided interview (units convert, engine
auto-pick), recipe editor + LLM-free revision, workspace watcher — all working.
Codex-reviewed twice; 10 findings fixed (poll-clobber, escAttr XSS, /run stage-gate,
optimistic merge, ACTIVE_JOB, root nesting, failReason). Shape-aware structural fallback +
improved LLM prompt (fixed "table came out as upside-down shelf": LLM returned empty →
cantilever fallback; now archetype-aware + more tokens + "no <think> block").

## DONE 2026-07-06 (session 2): Quality + Creativity knobs + slices 7–13
All verified live (browser + real qwen fan-out). Backend: `QUALITY_PRESETS`
(draft 8k/50, balanced 16k/90, fine 32k/130) drives `topo_resolution(domain,target)`
+ max_iters; `CREATIVITY_PRESETS` (focused/explore/wild). `llm_structural_variants`
(one array call → N distinct concepts, robust fallback to single+seed diversity),
`variant_combos` (concepts×seeds, index 0 = pristine anchor), `spawn_variant`
(queues sibling structural jobs under parent root, config_in carries the concept →
skips LLM, creativity=focused so no re-fan-out), `_clean_bc` shared validator.
`run_structural_job` reads quality/seed from job or config_in; fan-out only for a
fresh parent. topo.py: `seed`+`asymmetry` bias sensitivities while every solve still starts
from the intact domain (Axis 2). new_job
carries quality/creativity; public_job + session /run thread them. Frontend: gate
has Quality+Creativity segmented controls (structural-only, honest variant-count ETA);
tree is a variant gallery (labels from variant_label, compliance meta, ★ best sibling);
recipe has quality select + seed row + FACE-PICKER per support/load + progressive
disclosure (primary/fine + plain labels + auto/edited tags); viewport CROSS-FADES
frames (two stacked layers, fixed camera); live compliance+material SPARKLINES;
parametric SPEC-DIFF bars; ARROW-KEY scrubber; revision COMPARE (structural wipe
divider + parametric side-by-side + metrics-delta row, compliance is the only
better/worse-colored delta); gate GAP-CHIPS (engine+units hard floor, inferred-size
warning). Verified: fanout of "small table" → qwen proposed Four-legs/Pedestal/
Cantilever, each solved, gallery shows distinct compliance (0.02 vs 0.05 vs 0.24).
Test jobs left in workspace: `fanout-test`, `-v2`, `-v3` (deletable).
Adversarial-reviewed (4-finder + skeptic-verify workflow); 3 confirmed defects FIXED:
(1) renderCompare null-deref when comparing a preview-less failed revision — compare
now gated on finalFrame(j) + null-guarded; (2) unguarded int(seed) from a crafted
/revise config — seed coerced to int-or-None in run_structural_job; (3) renderFilm
null-deref when a placeholder film card later gains a preview (workspace jobs) —
null-guarded `im`. XSS audit of all new innerHTML builders: clean (escAttr held).

## DONE 2026-07-06 (session 3): aesthetic STYLE system (slices S1–S5)
A style = a named design-language preset (`styles.json`, 12 presets, 2 axes
round↔faceted / skin-solid↔skeletal, spine Bones→Evolved→Gothic) that biases the
DETERMINISTIC stack. LLM picks it once at the gate (synthesize_spec returns
`style`), overridable; rides the recipe as `job.config.style` → LLM-free restyle.
- **S1 palette/data**: `load_styles`+`_norm_style` (normalizes every bundle → axis/
  palette/edge/structural always present, drops non-dicts), `GET /api/styles`,
  `job.config.style`; topo.py `render_iso(color,edge,edge_lw)` recolors the surface.
- **S2 picker**: `renderStyleMap` (2-axis SVG, palette dots, engine-affinity dimming,
  spine dashed) at the gate + recipe (restyle → `/revise`). Picking pre-fills Quality.
- **S3 structural stamp** (topo.py): style clamps volfrac into its band, sets penal,
  enforces `symmetrize()` (mirror = reflect x; axial/mirror-axial = 4-fold, mean-
  preserving so volfrac holds), anisotropic z_bias filter for verticality.
- **S4 template edges**: `apply_style_edges` injects corner_fillet (all templates) +
  `edge_chamfer` (added to plate_holes/flange_disc only — NOT hand-edited box_enclosure).
- **S5 mesh finish** (topo.py): `taubin_smooth` (shrink-free λ/μ) on the FINAL mesh for
  smooth/g2-polish (voxel staircase → organic bone; watertight preserved), faceted =
  edge_hex wireframe. Verified: g2-polish visibly smooths, bones-demo end-to-end.
- **PRECEDENCE (post-review)**: non-editable levers (penal/symmetry/palette/mesh_finish)
  always from the style; volfrac + corner_fillet/edge_chamfer are recipe-editable →
  the style stamps them only on a FRESH job or a RESTYLE (revise sets `cfg.restyle`
  when the style key changed), a plain param-edit revision keeps the user's values.
  `apply_style = (not config_in) or config_in.restyle`.
- Adversarial-reviewed (3-finder + skeptic); 7 confirmed defects ALL FIXED: renderStyleMap/
  updateJob/`style["palette"]` crashes on malformed styles (→ `_norm_style` + client guards),
  structural skey ignored job.style (→ `(override).get('style') or job.get('style')`),
  apply_style_edges not re-applied after stack reconfigure (→ re-stamp), volfrac/edge
  "recipe overrides" contract (→ the restyle-flag precedence above). topo-math: clean.
- Demo job `bones-demo` left in the workspace (a real bone-like styled result).
- NOT built (S6): feature_language (ribs/flutes/pockets/taper) + lattice/gyroid/tracery
  (need a cell/void primitive) — deferred per Nick's "defer tail" choice.
  [UPDATE session 4: lattice/gyroid/ribs/tracery NOW BUILT as a real fine-grid TPMS
  primitive; see "DONE 2026-07-06 (session 4)" below.]

## DONE 2026-07-06 (session 3b): S6 feature language + solver speedup
- **S6 feature_language** (per-style geometry ops, gated like edges on fresh/restyle):
  FLUTES (Neo-Deco: N vertical rim scallops on flange_disc, `flutes` param=16),
  POCKETS (CNC Billet: 2×2 rectangular lightening cuts on plate_holes, `pockets`=1),
  both SUBTRACTIVE so they stay in the bbox envelope (no spec-gate conflict);
  facets already done via S5 mesh_finish. `apply_style_edges` now also injects
  feature params via the same template-aware `setp` (no-op if the template lacks
  the param). Verified: builds valid, verification passes. DEFERRED (honest):
  LATTICE/gyroid — `gyroid_infill` (labyrinth-solid g>0 form) is BEST-EFFORT with a
  watertight guard (topo.py marching-cubes the lattice, keeps it only if watertight
  else ships the solid); on REAL optimized structures it always falls back (they're
  too thin/branchy → non-manifold), so it's effectively dormant infrastructure —
  a printable visible gyroid needs a real TPMS mesh primitive. RIBS deferred
  (additive → grows past the bbox gate). TAPER emerges naturally in structural.
  TRACERY deferred (ornamental-void primitive).
  Adversarial-reviewed (3-finder + skeptic); 6 confirmed defects FIXED: FLUTES were
  fragile — reworked from cylinders to rim-aware RECTANGULAR grooves so they (a) can't
  sever a thin rim into disconnected solids that pass is_valid [was HIGH: broken part
  shipped as valid], (b) aren't miscounted as bolt-hole cylinders by check-spec [was
  failing verification forever], (c) can't empty the disc on extreme counts; added a
  `0 < bore_d < od` guard (pre-existing oversized-bore crash); topo.py empty-field
  guard before marching_cubes (+ watertight=False on empty); server now LOGS the honest
  lattice→solid fallback (lattice_applied was a dead flag). Refuted: CholmodOOM (not
  reachable), non-lattice watertightness (unchanged), gyroid island-drop (skin keeps
  connectivity). Verified: thin-rim flange→1 solid, flutes not counted as cylinders (6
  bolts), oversized bore + flutes=48 both build, empty case no longer crashes.
- **Solver speedup** (~9x): CHOLMOD (scikit-sparse + brew suite-sparse) — see memory
  for install + rebuild recipe. topo.py `cho_factor(Kf).solve(b)` fresh each iter,
  falls back to spsolve(MMD_AT_PLUS_A)→CG. Multithreaded, exact, Fine no longer needs CG.

## DONE 2026-07-06 (session 4): real TPMS lattice primitive + Slice 14 pixel-map callouts
- **Real TPMS lattice** (topo.py) — replaced the dormant coarse-grid `gyroid_infill`
  (which evaluated `g>0` on the ~8k SIMP grid → thin walls pinched to single voxels →
  non-manifold → ALWAYS fell back to solid) with `tpms_lattice()`, evaluated on its
  OWN fine grid via `_fine_grid()` (vpc≈10 voxels/cell, uniformly coarsened only to a
  `max_vox`=10M budget; bails if a cell can't hold a ≥~3-voxel wall). Watertight BY
  CONSTRUCTION — the fine grid guarantees no single-voxel pinch. `_tpms_field()` does
  gyroid / schwarz_p / diamond (Neovius dropped: its steep gradient can't reliably
  close a sheet); modes `sheet` (|g|<iso double-wall) / `network` (g>0 ligaments);
  plus non-TPMS `ribs` (2-axis stiffening walls) and `tracery` (cubic edge struts) so
  gothic/maker-honest are unblocked too. Keeps a solid skin (struts anchor into a
  shell → one connected body) + largest-component filter. main() marches the FINE body
  and keeps the lattice mesh ONLY if `is_watertight` (so `lattice_applied` can never be
  a false positive); else ships the plain solid. Lattice mesh is NOT taubin-smoothed
  (would erase thin walls). Config: `lattice`, `lattice_kind`, `lattice_mode`,
  `lattice_cell`. server.py maps style feature tokens → kind (tracery>lattice→gyroid>
  ribs) and now logs the applied lattice stats (was a dead flag). VERIFIED: unit test
  (thin truss + block, all kinds watertight), full solve, and LIVE via `/revise`
  (bones-demo→lattice-gyroid, no LLM): `lattice_applied=True watertight=True`, gyroid
  sheet cell 37.9mm, 248cm³, shown in-browser with the Lattice/Gyroid palette.
- **Slice 14 pixel-map callouts** (topo.py render_iso + index.html) — UNBLOCKED. Camera
  is fixed, so `_pixel_map()` projects world(mm)→NORMALIZED (u,v)∈[0,1] top-left image
  fractions (recipe: `proj3d.proj_transform` → `ax.transData` → `fig.transFigure`inv;
  empirically verified — red overlay Xs land on matplotlib's own 3D scatter markers).
  Resolution/CSS-scale-independent. render_iso writes `final.map.json` beside the final
  PNG (`{space,origin,camera,domain_mm, bbox:[8 corners{world,uv}], dims:[L/W/H
  {world,uv,mid_uv,mm}]}`); result.json gains `final_map`. Client: a `⊹ dims` toggle on
  structural-done jobs draws an SVG overlay (`#jCallouts`) — faint bbox wireframe + L/W/H
  yellow leaders with mm labels — mapping the sidecar uv onto the letterboxed
  (object-fit:contain) image rect (repaints on resize; hidden in compare / mid-iter
  scrub). CONTRACT for future callouts (loads, faces): read the same sidecar; anything
  needing arbitrary points can carry the 4×4 proj matrix too (not needed yet). VERIFIED
  live in-browser (screenshot: L 160 / W 60 / H 90 leaders correctly outside the part).

## DONE 2026-07-06 (session 5): 3D viewer + surface constraints + upload→redesign
Three composed features — upload a part → it's the design domain → mark faces to preserve →
lightweight the rest → inspect by rotating in 3D. All offline, no new deps.
- **Phase 1 — hand-rolled WebGL viewer** (index.html `GLV` IIFE, ~150 lines, raw WebGL, zero
  deps). `◲ rotate` toggle on any finished job with an STL. Loads result.stl, lit two-sided
  (camera headlight + fixed fill + rim), orbit/zoom(wheel)/pan(shift-drag). Auto-fits bbox;
  tints to the style palette but FLOORS brightness (dark resin styles stayed legible). Minimal
  column-major mat4 (persp/look/mul). Releases the GL context via WEBGL_lose_context on canvas
  remount (no context leak); falls back to the PNG if WebGL is unavailable. Mutually exclusive
  with compare/dims (`.gl-on` class hides the other viewport layers).
- **Phase 2 — keep-regions** (structural only). Recipe gains a KEEP REGIONS list: each region
  = {box (normalized, same face-picker+box6 UI as supports/loads), rule}. Rules: 🧱 keep-solid
  (SIMP passive → density pinned 1), ⬛ keep-void (pinned 0), 📐 preserve (pinned 1 + kept flat
  by freezing those verts in taubin_smooth + un-latticed). topo.py: elements_in_box → solid/
  void/preserve masks forced each iter after symmetrize; tpms_lattice `protect=` keeps regions
  solid; **forced regions are NEVER dropped by the largest-component filter** (kept + honest
  `components`/`raw_components` count; server logs "in N disconnected pieces — a keep region is
  isolated" vs "trimmed islands"). server.py: `clean_regions`, sanitize_bc carries regions,
  threaded to cfg + job.config (rides the LLM-free /revise flow). Client: S.editRegions live
  list (add/remove), syncRegionsFromDOM on revise.
- **Phase 3 — upload → design domain.** `⤒ Upload part → redesign` in the rail → POST
  `/api/upload?name=&project=` (raw body). STL used directly; STEP converted via the build123d
  CAD venv (`convert_to_stl` = `export_stl(import_step(src))`). `stl_bounds` (stdlib) → domain_mm;
  creates a structural job with a provisional cantilever load case + `domain_stl`. The automatic
  upload baseline is the unchanged converted source mesh at 100% material: it bypasses SIMP,
  voxel remeshing, thresholding, and smoothing until the user identifies functional faces and
  explicitly lowers the material budget on a revision. Balanced is retained as the first revision's
  default quality. User-edited structural recipes accept material budgets through 100%.
  topo.py `read_stl` (binary+ascii, 2M-tri cap) + `voxelize_mesh` (numpy ray-parity, ~14ms on the
  test part, holes captured; sample points jittered 0.5007/0.4993 to dodge axis-aligned-edge
  double-counts) → `domain_mask` forces void OUTSIDE the part so the solver redesigns inside
  the user's envelope. `domain_stl` threaded through runs + revisions; **guarded to CAD_ROOT**
  (a revise can't point it at an arbitrary file). SolidWorks native .sldprt is NOT readable
  offline → **STL/STEP only** (export SLDPRT→STEP; IGES dropped — import_step is STEP-only).
  Verified: STL, STEP, and a browser upload all voxelize to the identical 0.89-occupancy domain
  and solve watertight; revisions keep the domain.
- **Reviewed twice** (cloud Codex gpt-5.5, then a 4-lens Claude verification workflow, then Codex
  again on the fixes). 9 confirmed defects FIXED & verified: STL 2M-tri cap (voxelize DoS);
  non-watertight upload → honest warn (was a silent full-bbox fallback); **volfrac made
  DOMAIN-RELATIVE** via an `active` mask (an uploaded part with occupancy < volfrac was an
  impossible target measured over the bbox → falsely "failed"); lattice `protect` no longer drops
  isolated keep-regions; `clean_regions` non-dict guard; `GLV.hide()` frees the context ONLY for a
  DETACHED canvas (loseContext on a live canvas broke same-job re-open) + `mount()` never throws
  (try/catch + isContextLost → PNG fallback); **floating-BC warn** (a default support/load on a
  bbox face a rounded/recessed part doesn't reach = fixed/loaded pure void → meaningless
  ~1e8-compliance result that otherwise reported as a clean pass); IGES dropped; `/api/upload`
  Content-Length parse guarded.

## DONE 2026-07-07 (session 5b): 3D click-to-pick + workspace cleanup
- **3D click-to-pick** (index.html GLV, commit 52a0866 + fixes 23dbc18): a pick toolbar (rotate /
  keep-solid / preserve / keep-void) in the WebGL view. A click (vs a drag) unprojects the pixel
  through the inverse MVP (`invMat`+`mulV`), ray-casts (Möller–Trumbore over the CPU-kept `geomV`)
  to the nearest triangle, snaps the hit to the nearest bbox face (`FACE_BOX`), and appends a
  keep-region to the recipe's live `S.editRegions` (shown with the face highlighted, editable,
  applied on the LLM-free revision). Makes upload→constrain→redesign click-driven. Codex-reviewed:
  3 fixes — sync GLV tool on stage rebuild, reset geometry on `mount()` (no stale-mesh pick during
  load), exclude pan gestures (`!drag.pan`) from picking. Verified live (top→solid, floor→preserve).
- **3D load vectors:** the same toolbar now has a Load mode with numeric X/Y/Z direction fields.
  Clicking the model replaces the lone provisional/default load, then adds or updates further
  loads by face. The exact flood-picked triangle patch is carried through recipe state and server
  sanitization; `topo.py` rasterizes it to loaded domain elements/nodes instead of relying on a
  fragile thin AABB. Load patches highlight warm red; direction magnitude is normalized. Recipe
  editing exposes the same numeric vector and supports removing loads. Verified live with an exact
  52-triangle face patch and `[1,2,-3]`, plus a solver run using triangle-defined load nodes.
- **Cleanup:** trashed the demo/test jobs to ~/.Trash (upl-*, t-ok, smoke-frame, bones-demo-r6/r7),
  killed a 37h-wedged agentcad process.
- **Mesh detail (sub-voxel smoothing):** live previews march the CONTINUOUS filtered density;
  final meshes use the volume-aware connected binary projection UPSAMPLED `mesh_detail`×
  (trilinear + a light Gaussian σ≈0.45 coarse-voxel). The old final x≥0.5 marching level erased
  valid low-budget paths whose useful densities were all below 0.5. The Gaussian removes
  grid-aligned RIBBING (trilinear zoom is only C0
  across coarse cells → marching-cubes creases; the box sensitivity filter adds grid stepping) —
  confirmed via an A/B render (concentric grid bands → clean), structure/compliance unchanged. Surface smoothness is
  DECOUPLED from the coarse solve grid (a fine mesh from a coarse solve; structure unchanged, only
  the surface). Restricted to the kept component (dilated 1 for the boundary layer), memory-capped
  (~15M voxels), faceted styles stay 1× (crystalline look). `mesh_detail` 1/2/3 = coarse/smooth/fine
  (default 2), a recipe knob independent of quality, threaded server→topo→job.config. topo.py
  `density_mesh(detail=)` + the final block; the "side x-ray" inset was a red herring — it's just
  `x.max(axis=1)` imshow'd, never fed forward. Verified: detail 1→3 on a cantilever went from heavy
  staircase to smooth, all watertight (20k→188k tris), same compliance/topology.
  Repo (branch main), local-only (no remote).

## DONE 2026-07-09: progressive, load-path-first generative solve
- **No iteration-1 demolition:** SIMP starts from the intact uploaded domain at 100% and uses
  smooth continuation for both free-region material (100%→requested) and penalization (1→style
  target) over the first ~45% of the run. Uploaded iteration 1 renders the exact welded source
  surface. Creativity seeds bias sensitivity rather than corrupting the intact start.
- **Independent vector load cases:** every authored 3D load is a separate RHS solved against the
  same stiffness matrix; element strain energies are summed. Opposing or angled loads can no
  longer cancel by being added into one force vector. Exact clicked triangle patches still define
  the loaded nodes.
- **Zones budgeted correctly:** solid, preserve, support, and load interface pads are passive
  material excluded from the free-region volume constraint. Loads/supports get a one-cell
  non-design transfer band. Preserve face patches remain a surface shell instead of being
  accidentally dilated into a huge passive slab. UI/result metrics distinguish total material,
  free-region material, passive fraction, and independent load-case count.
- **One preview/export surface contract:** once continuation reaches its target, live frames use
  the same production surface pipeline as the final STL. A stable last iterate wins when it is
  within 2% compliance of the historical best; only a materially degraded tail reverts to the
  best state, which late frames already display as `best N`. If the continuous preview is already
  one load-connected component, it is exported unchanged (`continuous-density`). Otherwise the
  visible late frames and final both use the same `projected` repair: volume-ranked density plus a
  density-weighted load-to-support path and connected high-density growth. Thus finalization never
  introduces an unseen mesh conversion. Structural continuity may explicitly exceed a requested
  target; it is logged rather than silently sacrificing the load path.
- **Verified on the uploaded 13135A59 bracket:** Balanced 25×26×25 at 10% free material starts
  at the exact original, finishes at 10.17% free / 34.84% total (the broad provisional faces are
  passive), one component, watertight; six voxels closed the projected load path. A second run
  with two non-collinear loads plus solid+preserve zones converged to 20.0% free material, two
  independent cases, one component, watertight, with final compliance 66.93.
- **Final-delta regression (2026-07-10):** replayed the exact reported r3 config (25×26×25,
  20%, 90 iterations). Iteration 90 is selected over best iteration 78 because its compliance is
  only 0.54% higher; final mode is `continuous-density`, one component/watertight, and the
  iteration-90/final geometry crops are pixel-identical (zero changed pixels). Replayed the 10%
  disconnected case too: late frames visibly show projected best iteration 25, final uses the
  same projection, and their geometry crops are also pixel-identical.

## PENDING — leftover
Parametric "measured lock rows" from the guided finalize flow are only partially served
(labels help); the session finalize path itself is unchanged. Consider: force a lighter
quality during Wild fan-out so 6 solves don't cost 80 min at Fine (currently variants
inherit the chosen quality). TPMS follow-ons if wanted: graded cell size (varies with a
field), network-mode volume-fraction targeting, and a matplotlib-independent thumbnail
so the lattice reads brighter than the translucent-resin palette.

## (historical roadmap — now built)
### A. QUALITY knob (agreed)
resolution + iterations + volfrac fidelity. Draft (~8k elems, ~2min, chunky) vs Fine
(~30–60k elems, ~10–20min, lean/bone-like). Expose resolution/volfrac/max_iters as recipe
+ confirm-gate controls. topo.py knobs: `TOPO_TARGET_ELEMS` (server.py:321, currently 8000),
max_iters (currently 50), penal 3.0, rmin 1.5, early-stop change<0.02. Finer mesh is the #1
lever for organic/bone-like; low volfrac (0.10–0.20) makes it leaner. CPU solver ceiling:
pure numpy/scipy, no GPU FEA — 30–60k elems is the practical max (CG solver for ndof>60k).

### B. CREATIVITY knob (agreed — design settled)
Two orthogonal axes from Quality (Quality=depth of ONE result; Creativity=breadth of variants).
Creativity = generate a GALLERY of divergent variants:
- **Axis 1 (interpretation):** LLM proposes 2–3 DISTINCT concepts of the brief (e.g. table →
  pedestal / four-leg / cantilever) = different load cases.
- **Axis 2 (solution diversity):** exploit SIMP non-convexity — seeded sensitivity bias
  (random/asymmetric SEED), jittered load/support boxes, varied penal/rmin → different valid
  local optima for the SAME concept.
Levels: Focused(1 deterministic) → Explore(2–3 concepts) → Wild(concepts × seeds = 6–9).
Present variants as siblings in the tree (reuse revision-tree UI) with each variant's
COMPLIANCE shown (creativity trades against optimality — show the cost). Fan-out in the job
runner. Needs: topo.py accept a `seed`/`asymmetry` param (np.random seeded, reproducible) +
  sensitivity perturbation; server fan-out that spawns N variant jobs; confirm-gate
"Explore" control (2 sliders: Quality, Creativity). NOTE: creativity ≠ bonier (that's Quality).

### C. Leftover proposal slices 7–14 (some partially done in the rebuild)
- **7 Adaptive interview**: one-screen gate, DETERMINISTIC gap-check (empty-field + unit-token
  + engine-keyword — NOT model confidence), confirm-not-ask chips, vagueness→preset chips,
  hard floor always-confirm units+engine. (Currently: brief→synthesize→confirm gate exists;
  the 3 fixed questions are already skipped; gap-chips NOT built.)
- **8 Progressive disclosure**: "Show all fields" (parametric in-rail; structural promoted to
  center stage — 20–40 fields don't fit the dock), ranked Primary/Fine-control groups,
  ·auto/·edited tags, authored plain-language label array. (NOT built — recipe shows all params flat.)
- **9 Parametric iteration treatment**: ~1s poll for active parametric (currently 2s), spec-diff
  mini-bars from check-spec/measure, clean 4-view cross-fade. (stage-rail pips DONE.)
- **10 Structural iteration treatment**: fixed-camera CROSS-FADE (fade≪poll; two stacked <img>),
  live compliance + vol-fraction sparklines (append 1 pt/poll), "seeding" label for empty early
  frames. (i/50 bar + GPU ETA DONE; cross-fade + sparklines NOT.)
- **11 Filmstrip scrubber + pin/live**: arrow-key ←/→ nav. (Play button + ●LIVE pin DONE.)
- **12 Revision compare**: metrics-delta row (both engines) + draggable wipe divider (STRUCTURAL
  only — fixed camera registers; parametric = side-by-side). (NOT built.)
- **13 Direct manipulation**: 2D unit-cube FACE-PICKER for structural supports/loads (writes
  canonical boxes, no camera math); self-describing measured lock rows for parametric final pass. (NOT built.)
- **14 DEFERRED/BLOCKED**: on-PNG clickable dimension callouts + click-the-hole locking — needs
  the renderer to emit a fixed camera + pixel-space feature map (parametric renderer auto-fits
  today). Do NOT hardcode pixel rects. Ship only after a render contract lands.

## Solver speed (2026-07-06)
topo.py's FEA solve was ~96% of each iteration (single-threaded spsolve). Now
CHOLMOD (scikit-sparse + `brew install suite-sparse`) via `cho_factor(Kf).solve(b)`
fresh each iter → ~9x faster end-to-end (draft 12-iter 19s→3.6s), multithreaded,
exact; Fine (>60k dof) no longer needs CG. Falls back to spsolve(MMD_AT_PLUS_A)→CG
if scikit-sparse absent (rebuild the venv with it: SUITESPARSE_INCLUDE_DIR/
LIBRARY_DIR env vars — see memory). Do NOT touch Nick's SPLIDAR monitor; nCAD's
process tree hangs off Claude.app only because the server is launched via preview.

## Constraints / gotchas (do not violate)
- index.html: vanilla JS ONLY, no libs/CDN/fonts, self-contained, offline. Inline 3D = server
  PNG frames (no three.js). Use Canvas/SVG for any generated graphics.
- One GPU, serial. LLM calls 10–40s; everything else fast+deterministic. Revisions LLM-free.
- Fixed camera in topo.py render_iso: elev=24 azim=-58, fixed axis limits → frames pixel-register
  (this is what makes cross-fade growth + wipe-compare possible).
- Don't clobber user input on poll (see rendering discipline). escAttr for attributes.
- Tone-as-data: header tagline rotates a lines[] array (playful, deadpan).
- Palette: bg#0e1116 panel#151b23 line#28313d ink#dbe2ec cool#4fc3f7(parametric/pulse)
  warm#ff8a5c(structural/load) ok#57d38a warn#e8c268 bad#ef6b6b; mono=ui-monospace.

## Verification method
preview_* tools (start "nCAD", eval/snapshot/screenshot — screenshots go mobile-width unless
preview_resize to 1280×860). Real runs: parametric ~12s, structural ~2min. Adversarial review
via Codex: `codex exec --profile circuitslug -s read-only --skip-git-repo-check -` (heredoc),
triage findings against the diff. Local model = qwen via LM Studio; if "no chat model loaded",
`~/.lmstudio/bin/lms load qwen3.6-35b-a3b-qwable-holo3-qwopus-qx64-hi-mlx -y`.

## User (Nick) preferences observed
Wants real engineering output + to SEE it work; playful tone; minimal friction; the LLM to
CONFIGURE a hardened deterministic stack, never be the hot-loop processor. Values honesty about
tradeoffs (optimality vs creativity, speed vs fidelity). Iterates hands-on on UX specifics.
