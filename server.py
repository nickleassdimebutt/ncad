#!/usr/bin/env python3
"""genloop — local generative-design loop for the AgentCAD stack.

The browser posts a spec; a single worker thread runs the loop:
  1. codex-hermes (local LLM) writes/edits design.py in the project dir
  2. an AST guard rejects imports/exec in the model-written script
  3. agentcad builds it (deterministic; the model never runs commands)
  4. check-spec + bbox comparison verify against the user's numbers
  5. failures become a structured feedback block for the next iteration

Stdlib only. Binds 127.0.0.1. Serves files only from under ~/Documents/CAD.
Jobs persist as <project>/job.json and reload on restart.
Run:  ~/Documents/CAD/.venv/bin/python ~/Documents/CAD/genloop/server.py
"""
import ast
import copy
import json
import os
import re
import struct
import subprocess
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CAD_ROOT = (Path.home() / "Documents" / "CAD").resolve()
APP_DIR = Path(__file__).resolve().parent
AGENTCAD = str(CAD_ROOT / ".venv" / "bin" / "agentcad")
CAD_PYTHON = str(CAD_ROOT / ".venv" / "bin" / "python")   # build123d venv (STEP/IGES import)
UPLOAD_MAX = 150 * 1024 * 1024                            # Phase 3: max upload size
MAX_UPLOAD_TRIS = 2_000_000                               # matches topo.MAX_STL_TRIS (voxel cost)
# Generation is ONE direct chat call to LM Studio — we only need file
# content, so the multi-turn codex machinery (slow, wedge-prone under a
# server) is deliberately not used here. codex-hermes remains the tool for
# interactive/agentic sessions.
LMS_API = "http://127.0.0.1:1234"
HOST, PORT = "127.0.0.1", 8765
GEN_TIMEOUT_S = 5 * 60
BUILD_TIMEOUT_S = 5 * 60
JOB_WALL_CLOCK_S = 2 * 60 * 60
MAX_ITERS_CAP = 8

JOBS = {}          # job_id -> job dict
JOB_ORDER = []     # queue of job_ids for the single worker
ACTIVE_JOB = {"id": None}   # the job the worker is actually running (incl. init)
LOCK = threading.Lock()
WAKE = threading.Event()
ACTIVE = {"job_id": None, "proc": None}   # for cancel


# ---------------------------------------------------------------- job model

def new_job(payload):
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(payload.get("name", ""))[:60]).strip("-").lower()
    if not name or name in ("genloop", ".venv"):
        raise ValueError("project name is required (letters/digits/dashes)")
    features = []
    for f in payload.get("features", []):
        if not f.get("diameter_mm") or not f.get("count"):
            continue
        feat = {
            "name": re.sub(r"[^a-zA-Z0-9_-]+", "_", str(f.get("name") or "feature"))[:40],
            "type": "cylinder",
            "diameter_mm": float(f["diameter_mm"]),
            "count": int(f["count"]),
        }
        if f.get("tolerance_mm"):
            feat["diameter_tolerance_mm"] = float(f["tolerance_mm"])
        features.append(feat)
    bbox = {a: float(v) for a, v in (payload.get("bbox") or {}).items()
            if a in ("x", "y", "z") and v}
    job = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "brief": str(payload.get("brief") or "").strip()[:4000],
        "bbox": bbox,
        "bbox_tol": float(payload.get("bbox_tol") or 0.5),
        "features": features,
        "objective": str(payload.get("objective") or "").strip()[:400],
        "max_iters": max(1, min(MAX_ITERS_CAP, int(payload.get("max_iters") or 3))),
        "model": payload.get("model") if payload.get("model") in ("local-qwen", "local-gemma") else "auto",
        "generator": payload.get("generator") if payload.get("generator") in ("stack", "structural", "llm-direct", "mock") else "stack",
        # structural knobs (quality = mesh/iters depth; creativity = variant breadth)
        "quality": payload.get("quality") if payload.get("quality") in QUALITY_PRESETS else DEFAULT_QUALITY,
        "creativity": payload.get("creativity") if payload.get("creativity") in CREATIVITY_PRESETS else DEFAULT_CREATIVITY,
        # aesthetic style preset (bundle of geometry-biasing rules); None = no style
        "style": payload.get("style") if payload.get("style") in STYLES else None,
        "state": "queued",   # queued|configuring|generating|building|verifying|done|failed|canceled|error
        "cancel": False,
        # user-provided configuration (revisions) — skips the LLM configurator
        "config_in": payload.get("config") if isinstance(payload.get("config"), dict) else None,
        "config": None,      # the resolved recipe, exposed for editing in the UI
        "iterations": [],
        "log": [],
        "created": time.time(),
        "updated": time.time(),
    }
    if not job["brief"] and not job["bbox"] and not job["features"]:
        raise ValueError("give at least a brief, a bounding box, or one feature")
    # revision lineage by explicit id, not by parsing the name — a real project
    # literally named "foo-r2" must not be mis-nested under "foo"
    root = payload.get("root")
    job["root"] = str(root) if isinstance(root, str) and root else job["id"]
    return job


def touch(job):
    job["updated"] = time.time()


def save_job(job):
    try:
        (CAD_ROOT / job["name"]).mkdir(parents=True, exist_ok=True)
        (CAD_ROOT / job["name"] / "job.json").write_text(json.dumps(job, indent=1))
    except OSError as exc:
        print(f"warn: could not persist job {job['id']}: {exc}", flush=True)


REQUIRED_JOB_KEYS = {"id", "name", "state", "iterations", "log", "created", "updated",
                     "brief", "bbox", "bbox_tol", "features", "objective",
                     "max_iters", "model", "generator", "cancel"}


def load_jobs():
    for jf in CAD_ROOT.glob("*/job.json"):
        try:
            job = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(job, dict) or not REQUIRED_JOB_KEYS.issubset(job):
            continue    # foreign/hand-made job.json (e.g. written by a TUI model)
        if job.get("state") in ("queued", "configuring", "generating", "building", "verifying", "optimizing"):
            job["state"] = "failed"
            job["log"].append("[restart] server restarted mid-run; job marked failed")
        JOBS[job["id"]] = job


def log(job, msg):
    with LOCK:
        job["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(job["log"]) > 400:
            del job["log"][:100]
        touch(job)
    print(f"{job['name']}: {msg}", flush=True)


def set_state(job, state):
    with LOCK:
        job["state"] = state
        touch(job)
    save_job(job)


# ---------------------------------------------------------- safety guards

FORBIDDEN_CALLS = {"open", "exec", "eval", "__import__", "compile", "input", "breakpoint"}


def ast_guard(src):
    """Model-written design.py is untrusted. build123d is pre-injected, so a
    legitimate script needs no imports at all — reject them plus exec-ish
    calls. Not a sandbox, just a tall fence for a local solo tool."""
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return f"design.py has a syntax error: line {exc.lineno}: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return ("design.py must not import anything — every build123d name "
                    "is already available without imports")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in FORBIDDEN_CALLS:
            return f"design.py must not call {node.func.id}()"
        if isinstance(node, ast.Attribute) and node.attr in ("system", "popen", "spawn"):
            return f"design.py must not use .{node.attr}"
    return None


# ------------------------------------------------------------ loop plumbing

def last_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        return None


def build_prompt(job, feedback, iter_no, prev_src):
    req = []
    if job["bbox"]:
        tgt = ", ".join(f"{a}={v}" for a, v in job["bbox"].items())
        req.append(f"- overall bounding box (mm, each within ±{job['bbox_tol']}): {tgt}")
    for f in job["features"]:
        tol = f.get("diameter_tolerance_mm", 0.1)
        req.append(f"- feature '{f['name']}': exactly {f['count']} cylindrical hole(s), "
                   f"diameter {f['diameter_mm']} mm (±{tol}), axis +z")
    parts = [
        "TASK: output the complete Python source of design.py. Code only.",
        "",
        "CONSTRAINTS for design.py:",
        "- plain Python, build123d algebra mode; every build123d name is pre-injected — NO import statements",
        "- no file/network/os access, no randomness, no CLI parsing, no markdown fences",
        "- all key dimensions as named variables at the top, in mm",
        "- exactly ONE solid; final variable named part; FINAL line exactly: show_object(part)",
        "- booleans: part = Box(x,y,z) - Cylinder(r,h); hole patterns:",
        "  for loc in PolarLocations(radius, n): part -= loc * Cylinder(r, h)",
        "  for loc in GridLocations(sx, sy, nx, ny): part -= loc * Cylinder(r, h)",
        "",
        f"BRIEF: {job['brief'] or '(none — satisfy the numeric requirements)'}",
        "REQUIREMENTS (all must be met):",
        *(req or ["- (none beyond the brief)"]),
    ]
    if job["objective"]:
        parts.append(f"SOFT OBJECTIVE (only after requirements pass): {job['objective']}")
    if feedback:
        parts += [
            "",
            f"PREVIOUS ATTEMPT (iteration {iter_no - 1}) FAILED.",
            "CURRENT design.py:",
            "```python", (prev_src or "").strip()[:3000], "```",
            "PROBLEMS TO FIX (change only what these require, keep the rest):",
            feedback,
        ]
    return "\n".join(parts)


def mock_design(job):
    x = job["bbox"].get("x", 60.0)
    y = job["bbox"].get("y", 40.0)
    z = job["bbox"].get("z", 5.0)
    src = [f"px, py, pz = {x}, {y}, {z}", "part = Box(px, py, pz)"]
    for f in job["features"]:
        n, d = f["count"], f["diameter_mm"]
        pitch = round(x * 0.8 / max(n, 1), 3)
        src.append(f"for loc in GridLocations({pitch}, 1, {n}, 1):")
        src.append(f"    part -= loc * Cylinder({d} / 2, pz * 2)")
    src.append("show_object(part)")
    return "\n".join(src) + "\n"


def kill_tree(proc):
    try:
        os.killpg(proc.pid, 15)
        time.sleep(1.5)
        os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError):
        pass


def run_cmd(job, argv, cwd, timeout, env=None):
    """Popen in its own process group, registered so POST /cancel can kill it."""
    proc = subprocess.Popen(argv, cwd=cwd, env=env, text=True, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with LOCK:
        ACTIVE["job_id"], ACTIVE["proc"] = job["id"], proc
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, err = proc.communicate()
        return -9, out, (err or "") + "\n(timed out)"
    finally:
        with LOCK:
            ACTIVE["job_id"], ACTIVE["proc"] = None, None


# --------------------------------------------------- direct LLM generation

def loaded_model(pref):
    """Which chat model is loaded in LM Studio; honor a local-qwen/gemma pref."""
    with urllib.request.urlopen(f"{LMS_API}/api/v0/models", timeout=10) as r:
        data = json.load(r)
    loaded = [m["id"] for m in data.get("data", [])
              if m.get("state") == "loaded" and m.get("type") != "embeddings"]
    want = {"local-qwen": "qwen", "local-gemma": "gemma"}.get(pref)
    for mid in loaded:
        if not want or want in mid.lower():
            return mid
    return None if want else (loaded[0] if loaded else None)


def extract_code(text):
    """Model reply → python source: drop <think> blocks, prefer the longest
    fenced code block, else take the raw text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text + "\n"


def chat(job, system, user, max_tokens=6000):
    """One chat completion against LM Studio. Returns (text, model_id)."""
    model = loaded_model(job["model"])
    if not model:
        raise RuntimeError("no chat model is loaded in LM Studio — load one (`lms load`) and resubmit"
                           if job["model"] == "auto"
                           else f"requested model '{job['model']}' is not loaded in LM Studio")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{LMS_API}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT_S) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"], model


def generate_direct(job, prompt):
    text, model = chat(job,
                       "You write build123d CAD scripts. Reply with ONLY the complete "
                       "Python source of design.py — no markdown fences, no explanations.",
                       prompt)
    return extract_code(text), model


# --------------------------------------------- structural (topology opt.)

TOPO_PY = APP_DIR / "topo.py"
TOPO_PYTHON = APP_DIR / ".venv" / "bin" / "python"

# QUALITY = depth of ONE result. A finer voxel mesh is the #1 lever toward
# organic / bone-like geometry (coarse meshes can only make chunky blobs); more
# iterations let compliance converge. Draft stays on the fast direct solver
# (ndof<=60k → spsolve); Fine crosses into the iterative CG path (slower).
QUALITY_PRESETS = {
    "draft":    {"target_elems": 8000,  "max_iters": 50,  "label": "Draft"},
    "balanced": {"target_elems": 16000, "max_iters": 90,  "label": "Balanced"},
    "fine":     {"target_elems": 32000, "max_iters": 130, "label": "Fine"},
}
DEFAULT_QUALITY = "draft"

# CREATIVITY = breadth of variants, NOT depth. Focused is one deterministic
# solve. Explore/Wild fan out divergent siblings — distinct concept load cases
# (Axis 1, from the model) crossed with perturbed SIMP seeds (Axis 2, exploiting
# the non-convexity of topology optimization) — so different valid local optima
# surface side by side. Creativity trades optimality for variety; the tree shows
# each variant's compliance so the cost is visible.
CREATIVITY_PRESETS = {
    "focused": {"concepts": 1, "seeds": 1, "label": "Focused"},
    "explore": {"concepts": 3, "seeds": 1, "label": "Explore"},
    "wild":    {"concepts": 3, "seeds": 2, "label": "Wild"},
}
DEFAULT_CREATIVITY = "focused"


def quality_preset(q):
    return QUALITY_PRESETS.get(str(q or ""), QUALITY_PRESETS[DEFAULT_QUALITY])


def creativity_preset(c):
    return CREATIVITY_PRESETS.get(str(c or ""), CREATIVITY_PRESETS[DEFAULT_CREATIVITY])


def topo_resolution(domain, target_elems=8000):
    lx, ly, lz = domain
    scale = (target_elems / (lx * ly * lz)) ** (1 / 3)
    return [max(6, round(v * scale)) for v in (lx, ly, lz)]


def heuristic_structural(job):
    """No-LLM fallback. Picks a load-case ARCHETYPE from the brief so the fall-
    back isn't always a cantilever (which is why a 'table' came out as an
    upside-down shelf). z is vertical (up); gravity is -z."""
    b = (job.get("brief") or "").lower()

    def has(*words):
        return any(re.search(rf"\b{re.escape(w)}\b", b) for w in words)

    if has("table", "desk", "stool", "platform", "stand", "pedestal", "tabletop",
           "riser", "bench", "podium", "leg", "legs", "feet", "foot", "tray"):
        # stands on the floor (z-min); weight presses down on the top surface
        return {"supports": [{"box": [0, 0, 0, 1, 1, 0.04]}],
                "loads": [{"box": [0.1, 0.1, 0.96, 0.9, 0.9, 1], "dir": [0, 0, -1]}],
                "volfrac": 0.3}
    if has("bridge", "beam", "span", "gantry", "joist", "lintel", "walkway"):
        # supported at both ends of the base; load down across the middle of the top
        return {"supports": [{"box": [0, 0, 0, 0.06, 1, 0.12]},
                             {"box": [0.94, 0, 0, 1, 1, 0.12]}],
                "loads": [{"box": [0.3, 0, 0.96, 0.7, 1, 1], "dir": [0, 0, -1]}],
                "volfrac": 0.3}
    if has("column", "post", "pillar", "strut", "spacer", "standoff"):
        # fixed at the base, axial load pressing straight down from the top
        return {"supports": [{"box": [0, 0, 0, 1, 1, 0.04]}],
                "loads": [{"box": [0, 0, 0.96, 1, 1, 1], "dir": [0, 0, -1]}],
                "volfrac": 0.3}
    if has("hook", "hanger", "hang"):
        # fixed at the top, load pulling down at the bottom
        return {"supports": [{"box": [0, 0, 0.9, 1, 1, 1]}],
                "loads": [{"box": [0.4, 0.4, 0, 0.6, 0.6, 0.06], "dir": [0, 0, -1]}],
                "volfrac": 0.3}
    # default: cantilever bracket/shelf — fixed to a wall, load hanging off the far end
    return {"supports": [{"box": [0, 0, 0, 0.001, 1, 1]}],
            "loads": [{"box": [0.98, 0, 0, 1, 1, 0.06], "dir": [0, 0, -1]}],
            "volfrac": 0.3}


def _clean_bc(cfg):
    """Validate + clamp a raw {supports,loads,volfrac} load case from the model.
    Raises ValueError on anything malformed. Shared by the single-config and the
    multi-concept (creativity) parsers so both enforce the same shape."""
    if not isinstance(cfg, dict) or not cfg.get("supports") or not cfg.get("loads"):
        raise ValueError("missing supports/loads")
    out = {"supports": [], "loads": []}
    for s in cfg["supports"]:
        box = [max(0.0, min(1.0, float(v))) for v in s["box"]]
        if len(box) != 6:
            raise ValueError("support box needs 6 numbers")
        out["supports"].append({"box": box})
    for l in cfg["loads"]:
        box = [max(0.0, min(1.0, float(v))) for v in l["box"]]
        if len(box) != 6:
            raise ValueError("load box needs 6 numbers")
        d = [float(v) for v in l["dir"]][:3]
        if len(d) != 3 or not any(d):
            raise ValueError("load dir needs 3 non-zero numbers")
        out["loads"].append({"box": box, "dir": d})
    out["volfrac"] = max(0.15, min(0.5, float(cfg.get("volfrac", 0.3))))
    if cfg.get("name"):
        out["_name"] = re.sub(r"[^a-zA-Z0-9 _-]+", "", str(cfg["name"]))[:24].strip() or None
    return out


def llm_structural_config(job, feedback=None):
    """One LLM call: plain-English load case → supports/loads/volfrac JSON.
    Geometry axes are normalized 0..1 boxes over the design domain."""
    user = "\n".join([
        "A topology-optimization solver needs boundary conditions for a rectangular",
        f"design domain of {job['bbox']['x']} x {job['bbox']['y']} x {job['bbox']['z']} mm (x, y, z).",
        "AXIS CONVENTION: x = horizontal width, y = horizontal depth, z = VERTICAL.",
        '"Down" / gravity / a shelf load is dir [0,0,-1]. A sideways push along the',
        "width is [1,0,0] or [-1,0,0]; along the depth is [0,1,0] or [0,-1,0].",
        "Regions are axis-aligned boxes in NORMALIZED coordinates [x0,y0,z0,x1,y1,z1],",
        "each 0..1 across the domain. z-min is the FLOOR, z-max is the TOP.",
        "COMMON ARCHETYPES — pick the closest to the brief and adapt:",
        '  table / stool / platform (stands on the floor): supports = floor [0,0,0, 1,1,0.04];',
        '     load = top surface [0.1,0.1,0.96, 0.9,0.9,1] dir [0,0,-1]',
        '  shelf / bracket (bolted to a wall): supports = wall face [0,0,0, 0.001,1,1];',
        '     load = far edge [0.98,0,0, 1,1,0.06] dir [0,0,-1]',
        '  bridge / beam (held at both ends): supports = [0,0,0, 0.06,1,0.12] AND [0.94,0,0, 1,1,0.12];',
        '     load = middle of the top [0.3,0,0.96, 0.7,1,1] dir [0,0,-1]',
        '  column / post (stands, pressed from above): supports = floor [0,0,0, 1,1,0.04];',
        '     load = whole top [0,0,0.96, 1,1,1] dir [0,0,-1]',
        "",
        f"DESIGN BRIEF: {job['brief'] or '(none — assume a cantilever shelf)'}",
        *([f"SOFT OBJECTIVE: {job['objective']}"] if job["objective"] else []),
        *([f"PREVIOUS CONFIGURATION FAILED: {feedback}"] if feedback else []),
        "",
        "Output the JSON DIRECTLY as the very first characters of your reply — do NOT",
        "wrap it in a <think> block or any prose. Reply with ONLY this JSON:",
        '{"supports": [{"box": [..6 numbers..]}], "loads": [{"box": [..6 numbers..], "dir": [dx,dy,dz]}],',
        ' "volfrac": <0.15-0.5 fraction of material to keep>}',
        "Supports are where the part is held/stands. Loads are where force is applied",
        "(dir is the force direction). If the brief names a material budget or weight",
        "reduction, set volfrac accordingly (default 0.3).",
    ])
    last_exc = None
    for attempt in (1, 2):
        text, model = chat(job, "You configure structural FEA boundary conditions. "
                           "Output ONLY valid JSON, no <think> block, no prose.", user, max_tokens=3000)
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        # fall back to the raw text if the model buried the JSON in an (unclosed)
        # think block, so a reasoning dump can't blank the whole reply
        m = re.search(r"\{.*\}", stripped, re.S) or re.search(r"\{.*\}", text, re.S)
        try:
            return _clean_bc(json.loads(m.group()) if m else {}), model
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            last_exc = exc or ValueError(f"unparseable reply: {text[:120]!r}")
    raise last_exc


def llm_structural_variants(job, n):
    """CREATIVITY Axis 1. Returns (concepts, model) — up to n DISTINCT structural
    interpretations of the brief (different plausible load cases). One array call;
    robustly falls back to the single-config path so it ALWAYS returns >=1 concept
    (seed diversity, Axis 2, then fills any shortfall to the requested count)."""
    def single():
        try:
            bc, model = llm_structural_config(job)
        except Exception:
            bc, model = heuristic_structural(job), None
        bc.setdefault("_name", "primary")
        return [bc], model
    if n <= 1:
        return single()
    user = "\n".join([
        f"A topology-optimization solver needs boundary conditions for a {job['bbox']['x']}",
        f"x {job['bbox']['y']} x {job['bbox']['z']} mm design domain (x width, y depth, z VERTICAL).",
        "Regions are axis-aligned boxes in NORMALIZED coords [x0,y0,z0,x1,y1,z1] (0..1).",
        "z-min is the FLOOR, z-max the TOP; gravity/down is dir [0,0,-1].",
        f"DESIGN BRIEF: {job['brief'] or '(none — a load-bearing bracket)'}",
        *([f"SOFT OBJECTIVE: {job['objective']}"] if job["objective"] else []),
        "",
        f"Propose {n} GENUINELY DIFFERENT ways to hold and load this part — different",
        "support layouts (e.g. central pedestal vs four corner legs vs one fixed wall),",
        "not the same case twice. Each is a distinct engineering interpretation.",
        "Output ONLY a JSON ARRAY (first character '['), no <think> block, no prose:",
        '[{"name":"<short>","supports":[{"box":[..6..]}],"loads":[{"box":[..6..],"dir":[dx,dy,dz]}],"volfrac":<0.15-0.5>}, ...]',
    ])
    for attempt in (1, 2):
        text, model = chat(job, "You propose diverse structural FEA load cases. "
                           "Output ONLY a JSON array, no <think> block, no prose.",
                           user, max_tokens=4000)
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(r"\[.*\]", stripped, re.S) or re.search(r"\[.*\]", text, re.S)
        try:
            arr = json.loads(m.group()) if m else []
            out, names = [], set()
            for c in arr[:n]:
                try:
                    bc = _clean_bc(c)
                except (ValueError, KeyError, TypeError):
                    continue
                nm = bc.get("_name") or f"concept{len(out) + 1}"
                if nm in names:
                    nm = f"{nm}{len(out) + 1}"
                bc["_name"], names = nm, names | {nm}
                out.append(bc)
            if out:
                return out, model
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return single()      # array parse failed — one concept; seeds provide the breadth


def variant_combos(concepts, cp):
    """(bc, seed, label) list of length concepts*seeds. The first appearance of
    each concept is the clean deterministic anchor (seed=None); repeats get a
    reproducible seed so topo.py perturbs the SIMP start into another local
    optimum (Axis 2). Index 0 is always the pristine primary."""
    total = max(1, cp["concepts"] * cp["seeds"])
    combos, seen, k = [], set(), 0
    while len(combos) < total and k < 16:
        c = concepts[k % len(concepts)]
        name = c.get("_name") or f"v{len(combos) + 1}"
        if name not in seen:
            seed, label = None, name
            seen.add(name)
        else:
            seed, label = 4100 + len(combos), f"{name}·s{len(combos)}"
        combos.append((c, seed, label[:24]))
        k += 1
    return combos


def spawn_variant(parent, domain, bc, seed, label, quality, style=None):
    """Queue a sibling structural job carrying a pre-resolved load case (skips the
    LLM, like a revision) under the parent's root so the tree nests it."""
    cfg = {"engine": "structural", "domain_mm": [float(v) for v in domain],
           "volfrac": bc.get("volfrac", 0.3), "quality": quality,
           "supports": bc["supports"], "loads": bc["loads"],
           "seed": seed, "variant_label": label}
    if style in STYLES:
        cfg["style"] = style
    base = re.sub(r"-v\d+$", "", parent["name"])
    with LOCK:
        taken = {j["name"] for j in JOBS.values()}
    n = 2
    while f"{base}-v{n}" in taken or (CAD_ROOT / f"{base}-v{n}").exists():
        n += 1
    try:
        submit_internal({"name": f"{base}-v{n}", "brief": parent["brief"],
                         "bbox": dict(zip("xyz", [float(v) for v in domain])),
                         "objective": parent["objective"], "generator": "structural",
                         "creativity": "focused", "config": cfg, "style": style,
                         "root": parent.get("root") or parent["id"]})
        log(parent, f"queued variant '{label}'")
    except Exception as exc:
        log(parent, f"could not spawn variant '{label}': {exc}")


def clean_regions(cfg):
    """Phase 2 keep-regions: [{box:[6 normalized], rule:'solid'|'void'|'preserve'}].
    Drops malformed / zero-volume boxes; caps the count."""
    out = []
    for r in (cfg.get("regions") or [])[:24]:
        if not isinstance(r, dict):
            continue
        try:
            box = [max(0.0, min(1.0, float(v))) for v in (r.get("box") or [])][:6]
        except (TypeError, ValueError):
            continue
        rule = r.get("rule") if r.get("rule") in ("solid", "void", "preserve") else "solid"
        if len(box) == 6 and box[3] > box[0] and box[4] > box[1] and box[5] > box[2]:
            out.append({"box": box, "rule": rule})
    return out


def sanitize_bc(cfg):
    """Clamp a user- or LLM-provided load case into a safe shape."""
    out = {"supports": [], "loads": [], "volfrac": 0.3}
    for s in (cfg.get("supports") or [])[:8]:
        box = [max(0.0, min(1.0, float(v))) for v in (s.get("box") or [])][:6]
        if len(box) == 6:
            out["supports"].append({"box": box})
    for l in (cfg.get("loads") or [])[:8]:
        box = [max(0.0, min(1.0, float(v))) for v in (l.get("box") or [])][:6]
        d = [float(v) for v in (l.get("dir") or [])][:3]
        if len(box) == 6 and len(d) == 3 and any(d):
            out["loads"].append({"box": box, "dir": d})
    try:
        out["volfrac"] = max(0.1, min(0.6, float(cfg.get("volfrac", 0.3))))
    except (TypeError, ValueError):
        pass
    out["regions"] = clean_regions(cfg)               # Phase 2 keep-regions
    if not out["supports"] or not out["loads"]:
        raise ValueError("a revision needs at least one support and one load")
    return out


def run_structural_job(job, proj):
    """Structural generative design: LLM sets the load case ONCE, then the
    deterministic SIMP solver evolves the structure; every solver iteration
    streams into the UI live. Revisions carry a user-edited load case and
    skip the LLM entirely."""
    if len(job["bbox"]) != 3:
        log(job, "structural jobs need all three bbox axes (the design domain)")
        set_state(job, "error")
        return
    override = job.get("config_in")
    # QUALITY (mesh + iters) comes from the revision/variant config if present,
    # else the job's own knob. SEED perturbs the SIMP start (creativity Axis 2).
    qkey = ((override or {}).get("quality") if override else job.get("quality"))
    qkey = qkey if qkey in QUALITY_PRESETS else DEFAULT_QUALITY
    quality = QUALITY_PRESETS[qkey]
    # STYLE (aesthetic preset) comes from the revision config if present, else the
    # job's own knob. The bundle biases geometry deterministically (S3/S5) + colors
    # the render (S1). None = no style.
    skey = (override or {}).get("style") or job.get("style")     # config_in wins, else job knob
    skey = skey if skey in STYLES else None
    style = STYLES.get(skey)
    # the style STAMPS its levers on a fresh job or an explicit restyle; a plain
    # param-edit revision keeps the user's recipe values (recipe still overrides).
    apply_style = (not override) or bool(override.get("restyle"))
    seed, label = None, ""
    if override:
        try:
            bc = sanitize_bc(override)
            seed = override.get("seed")
            label = str(override.get("variant_label") or "")[:24]
            log(job, "using edited load case" + (f" — variant '{label}'" if label else " — revision") + " (no LLM)")
        except (ValueError, TypeError) as exc:
            log(job, f"invalid config: {exc}")
            set_state(job, "error")
            return
    else:
        set_state(job, "configuring")
        cp = creativity_preset(job.get("creativity"))
        if cp["concepts"] * cp["seeds"] > 1:      # CREATIVITY: fan out siblings
            domain0 = [job["bbox"]["x"], job["bbox"]["y"], job["bbox"]["z"]]
            concepts, model = llm_structural_variants(job, cp["concepts"])
            combos = variant_combos(concepts, cp)
            log(job, f"creativity '{job.get('creativity')}': {len(combos)} variants "
                     f"({model or 'heuristic'} proposed {len(concepts)} concept(s))")
            for c, s, lab in combos[1:]:
                spawn_variant(job, domain0, c, s, lab, job.get("quality"), job.get("style"))
            bc, seed, label = combos[0]        # this job runs the pristine anchor
        else:
            try:
                bc, model = llm_structural_config(job)
                log(job, f"{model} configured load case: {bc}")
            except RuntimeError as exc:
                log(job, f"{exc}")
                log(job, "falling back to heuristic cantilever load case")
                bc = heuristic_structural(job)
            except Exception as exc:
                log(job, f"LLM structural config failed ({exc!r}); heuristic cantilever")
                bc = heuristic_structural(job)

    # a revision's config_in is user-controllable — coerce seed defensively so a
    # non-integer value degrades to "no seed" rather than crashing the worker
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None
    domain = list(override.get("domain_mm", [])[:3]) if override else []
    try:
        domain = [float(v) for v in domain]
    except (TypeError, ValueError):
        domain = []
    if len(domain) != 3 or not all(5 <= v <= 2000 for v in domain):
        domain = [job["bbox"]["x"], job["bbox"]["y"], job["bbox"]["z"]]
    # S3: the style STAMPS its structural levers onto the solve — pull material
    # into its volfrac band, set SIMP penalization, enforce symmetry, bias verticality.
    vf, penal, symmetry, z_bias, mesh_finish = bc.get("volfrac", 0.3), 3.0, "none", 1.0, "as-is"
    if style:
        st = style.get("structural") or {}
        band = st.get("volfrac")
        # volfrac is BOTH style-derived and recipe-editable → only the style clamps
        # it on a fresh job / restyle; a plain revision's edited volfrac is kept.
        if apply_style and isinstance(band, (list, tuple)) and len(band) == 2:
            try:
                lo, hi = sorted((float(band[0]), float(band[1])))
                vf = min(max(vf, lo), hi)
            except (TypeError, ValueError):
                pass
        try:
            penal = float(st.get("penal") or 3.0)
        except (TypeError, ValueError):
            penal = 3.0
        symmetry = st.get("symmetry") if st.get("symmetry") in ("mirror", "axial", "mirror-axial") else "none"
        try:
            z_bias = float(st.get("z_bias", 1.0))
        except (TypeError, ValueError):
            z_bias = 1.0
        mesh_finish = st.get("mesh_finish") if st.get("mesh_finish") in ("faceted", "smooth", "g2-polish") else "as-is"
    cfg = {
        "domain_mm": domain,
        "resolution": topo_resolution(domain, quality["target_elems"]),
        "volfrac": max(0.05, min(0.95, vf)),
        "max_iters": quality["max_iters"],
        "penal": max(1.5, min(6.0, penal)), "rmin": 1.5,
        "supports": bc["supports"],
        "loads": bc["loads"],
    }
    if seed is not None:
        cfg["seed"], cfg["asymmetry"] = seed, 0.6
    if symmetry != "none":
        cfg["symmetry"] = symmetry
    if abs(z_bias - 1.0) > 1e-6:
        cfg["z_bias"] = max(0.5, min(2.5, z_bias))
    if mesh_finish != "as-is":
        cfg["mesh_finish"] = mesh_finish           # S5: smooth vs faceted final mesh
    features = (style.get("feature_language") or []) if style else []
    # S6 structural infill: map the style's feature tokens to a fine-grid lattice kind.
    # tracery (gothic openwork) wins over a plain lattice; ribs are the coarse fallback.
    lat_kind = next((k for tok, k in (("tracery", "tracery"), ("lattice", "gyroid"),
                                      ("ribs", "ribs")) if tok in features), None)
    if lat_kind:
        cfg["lattice"] = True                      # topo.py evaluates it on its own fine grid
        cfg["lattice_kind"] = lat_kind
    if style:
        cfg["palette"] = style.get("palette")      # S1: recolor the render (normalized → always present)
    regions = bc.get("regions") or []              # Phase 2 keep-regions (from an edited recipe)
    if regions:
        cfg["regions"] = regions
    # Phase 3: an uploaded part defines the design domain — carry its STL path through
    # every run/revision so the solver keeps redesigning inside that envelope.
    dstl = ((override or {}).get("domain_stl")
            or (job.get("config") or {}).get("domain_stl") or job.get("domain_stl"))
    if dstl:                                        # only accept a domain STL inside the workspace
        try:
            dp = Path(dstl).resolve()
            if dp.is_file() and dp.is_relative_to(CAD_ROOT):
                cfg["domain_stl"] = str(dp)
        except (OSError, ValueError):
            pass
    with LOCK:
        job["config"] = {"engine": "structural", "domain_mm": domain,
                         "volfrac": cfg["volfrac"], "quality": qkey,
                         "supports": bc["supports"], "loads": bc["loads"]}
        if regions:
            job["config"]["regions"] = regions
        if dstl and Path(dstl).exists():
            job["config"]["domain_stl"] = dstl
        if skey:
            job["config"]["style"] = skey
        if seed is not None:
            job["config"]["seed"] = seed
        if label:
            job["config"]["variant_label"] = label
    (proj / "topo-config.json").write_text(json.dumps(cfg, indent=1))
    outdir = proj / "topo"
    outdir.mkdir(exist_ok=True)
    set_state(job, "optimizing")
    log(job, f"solver: {cfg['resolution']} voxels · volfrac {cfg['volfrac']:.3f} · "
             f"{quality['label']} ({cfg['max_iters']} iters) · penal {cfg['penal']}"
             + (f" · {skey}" if skey else "")
             + (f" · {symmetry}" if symmetry != "none" else "")
             + (f" · z×{z_bias:g}" if abs(z_bias - 1.0) > 1e-6 else "")
             + (f" · seed {seed}" if seed is not None else ""))

    proc = subprocess.Popen(
        [str(TOPO_PYTHON), str(TOPO_PY), str(proj / "topo-config.json"), str(outdir)],
        cwd=proj, text=True, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with LOCK:
        ACTIVE["job_id"], ACTIVE["proc"] = job["id"], proc
    final = None
    try:
        for line in proc.stdout:                        # streamed live
            if job["cancel"]:
                kill_tree(proc)
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("done"):
                final = msg
                continue
            if msg.get("warn"):                         # solver notes (e.g. domain voxelization)
                log(job, "note: " + str(msg["warn"])[:200])
                continue
            if "iter" not in msg:                       # ignore any other non-iteration line safely
                continue
            it = {"iteration": msg["iter"], "label": f"topo-{msg['iter']}", "passed": False,
                  "metrics": {"compliance": msg["compliance"], "vol_fraction": msg["vol"],
                              "change": msg["change"]},
                  "preview": f"{job['name']}/topo/{msg['preview']}",
                  "step": None, "stl": None, "viewer": None, "error": None, "feedback": None}
            with LOCK:
                job["iterations"].append(it)
                touch(job)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
    finally:
        with LOCK:
            ACTIVE["job_id"], ACTIVE["proc"] = None, None
    if canceled(job):
        return
    if not final:
        err = (proc.stderr.read() or "")[-400:] if proc.stderr else ""
        log(job, f"solver did not finish: {err}")
        set_state(job, "failed")
        return

    problems = []
    if abs(final["vol_fraction"] - cfg["volfrac"]) > 0.05:
        problems.append(f"volume fraction {final['vol_fraction']:.3f} vs target {cfg['volfrac']}")
    if final["components"] > 1:
        log(job, f"warning: result is in {final['components']} disconnected pieces — a keep "
                 "region is likely isolated from the load path; move it or bridge to it")
    elif final.get("kept_largest"):
        log(job, "note: trimmed disconnected optimization islands; kept the connected body")
    it = {"iteration": final["iterations"] + 1, "label": "final", "passed": not problems,
          "metrics": {"compliance": final["compliance"], "vol_fraction": final["vol_fraction"],
                      "mesh_volume_mm3": final["mesh_volume_mm3"],
                      "components": final["components"]},
          "preview": f"{job['name']}/topo/final.png",
          "stl": f"{job['name']}/topo/result.stl",
          "step": None, "viewer": None, "error": None,
          "feedback": "\n".join(f"- {p}" for p in problems) or None}
    record_iter(job, it)
    li = final.get("lattice")
    if final.get("lattice_applied") and li:
        log(job, f"lattice: {li['kind']}"
                 + (f" {li['mode']}" if li.get('mode') else "")
                 + f" · cell {li['cell_mm']}mm · {li['eff_vpc']} vox/cell "
                 f"· solid frac {li['solid_fraction']}")
    elif cfg.get("lattice") and not final.get("lattice_applied"):   # honest fallback
        log(job, f"note: {cfg.get('lattice_kind', 'lattice')} infill couldn't be made "
                 "watertight on this structure — shipped solid instead")
    if problems:
        log(job, "finished with warnings: " + "; ".join(problems))
        set_state(job, "failed")
    else:
        vol = final.get("mesh_volume_mm3")
        log(job, f"optimized structure ready — {final['iterations']} iterations"
                 + (f", {vol / 1000:.1f} cm³ of material" if vol else " (mesh not watertight — volume unverified)"))
        set_state(job, "done")


# ------------------------------------------------- template stack (default)

TEMPLATES = {}      # name -> {"description", "bbox_map", "params", "source"}


def load_templates():
    tdir = APP_DIR / "templates"
    manifest = json.loads((tdir / "manifest.json").read_text())
    for name, meta in manifest.items():
        src = tdir / f"{name}.py"
        if src.exists():
            meta["source"] = src.read_text()
            TEMPLATES[name] = meta
    print(f"nCAD: {len(TEMPLATES)} templates loaded: {', '.join(TEMPLATES)}", flush=True)


# ------------------------------------------------------- aesthetic styles
# A style is a bundle of geometry-biasing rules the DETERMINISTIC stack applies
# on top of whatever the engine produces. The LLM picks one ONCE at the gate
# (overridable); it rides the recipe as job.config.style so a revision restyles
# with no LLM call. See styles.json for the schema + the 12 presets.
STYLES = {}          # key -> bundle
STYLE_META = {}      # axes, spine, docs


def _norm_style(b):
    """styles.json is hand-editable 'inert data' — normalize each bundle so a
    missing axis / palette / edge / structural block can never crash a consumer
    (the frontend picker, the render palette, apply_style_edges). Returns None for
    anything not a dict."""
    if not isinstance(b, dict):
        return None
    ax = b.get("axis") if isinstance(b.get("axis"), dict) else {}
    def num(v, d):
        try:
            return float(v)
        except (TypeError, ValueError):
            return d
    b["axis"] = {"form": num(ax.get("form"), 0.5), "mass": num(ax.get("mass"), 0.5)}
    pal = b.get("palette") if isinstance(b.get("palette"), dict) else {}
    b["palette"] = {"render_hex": pal.get("render_hex") or "#7ba7c9",
                    "edge_hex": pal.get("edge_hex") or "#8493a5"}
    if not isinstance(b.get("edge"), dict):
        b["edge"] = {}
    if not isinstance(b.get("structural"), dict):
        b["structural"] = {}
    b.setdefault("label", "style")
    b.setdefault("blurb", "")
    b.setdefault("finish", "")
    b.setdefault("engine_affinity", "either")
    if not isinstance(b.get("feature_language"), list):
        b["feature_language"] = []
    return b


def load_styles():
    global STYLE_META
    try:
        data = json.loads((APP_DIR / "styles.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"nCAD: styles.json not loaded ({exc!r})", flush=True)
        return
    for k, v in (data.get("styles") or {}).items():
        nv = _norm_style(v)
        if nv:
            STYLES[k] = nv
    STYLE_META = {k: data.get(k) for k in ("axes", "spine", "spine_doc", "version")}
    print(f"nCAD: {len(STYLES)} styles loaded", flush=True)


def style_bundle(key):
    """The validated bundle for a style key, or None."""
    return STYLES.get(key) if key in STYLES else None


def clamp_params(tname, params):
    """Keep only known params, clamped into their manifest bounds."""
    spec = TEMPLATES[tname]["params"]
    out = {}
    for k, v in (params or {}).items():
        if k in spec:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            out[k] = max(spec[k]["min"], min(spec[k]["max"], v))
    return out


def heuristic_config(job):
    """Deterministic fallback configurator — the stack still works with no
    LLM at all: pick a template by brief keywords, derive params from spec."""
    brief = job["brief"].lower()

    def has(*words):
        return any(re.search(rf"\b{w}\b", brief) for w in words)

    bbox = job["bbox"]
    square_xy = "x" in bbox and "y" in bbox and abs(bbox["x"] - bbox["y"]) < 0.01
    if has("enclosure", "box", "case", "housing", "shell", "tray", "bin"):
        tname = "box_enclosure"
    elif has("flange", "disc", "disk", "hub", "washer") or \
            (has("round", "circular", "cylindrical") and square_xy):
        tname = "flange_disc"
    else:
        tname = "plate_holes"
    params = {}
    bmap = TEMPLATES[tname]["bbox_map"]
    for axis, want in job["bbox"].items():
        p = bmap.get(axis)
        if p and not p.startswith("__"):
            params[p] = want
    feats = sorted(job["features"], key=lambda f: -f["count"])
    if tname == "flange_disc":
        for f in feats:
            if f["count"] == 1:
                params["bore_d"] = f["diameter_mm"]
            else:
                params["bolt_d"] = f["diameter_mm"]
                params["bolt_n"] = f["count"]
    else:
        for f in feats:
            if f["count"] == 1:
                params["bore_d" if tname == "plate_holes" else "hole_d"] = f["diameter_mm"]
            else:
                params["hole_d"] = f["diameter_mm"]
                params["hole_n"] = f["count"]
    return tname, clamp_params(tname, params)


def llm_config(job, feedback=None):
    """ONE LLM call: pick a template and parameter values from the brief.
    The LLM does logic/configuration — never geometry code in the fast path."""
    catalog = []
    for name, t in TEMPLATES.items():
        ps = ", ".join(f"{k} (default {v['default']}{', ' + v['note'] if 'note' in v else ''})"
                       for k, v in t["params"].items())
        catalog.append(f"- {name}: {t['description']}\n  bbox axes map to: {t['bbox_map']}\n  params: {ps}")
    req = [f"bounding box target (±{job['bbox_tol']}mm): " + ", ".join(f"{a}={v}" for a, v in job["bbox"].items())] if job["bbox"] else []
    req += [f"feature '{f['name']}': {f['count']}x cylinder dia {f['diameter_mm']}mm" for f in job["features"]]
    user = "\n".join([
        "TEMPLATE CATALOG:", *catalog, "",
        f"DESIGN BRIEF: {job['brief'] or '(none)'}",
        "REQUIREMENTS:", *(req or ["(none)"]),
        *([f"SOFT OBJECTIVE: {job['objective']}"] if job["objective"] else []),
        *([f"PREVIOUS CONFIGURATION FAILED: {feedback}"] if feedback else []),
        "",
        'Pick the best template and set its parameters so every requirement passes.',
        'Set bbox-mapped params exactly to the bbox targets. Use plausible engineering',
        'values for anything the brief implies (e.g. bolt circle diameter between bore and rim).',
        'Reply with ONLY this JSON, nothing else:',
        '{"template": "<name>", "params": {"<param>": <number>, ...}}',
    ])
    last_exc = None
    for attempt in (1, 2):                       # one retry on malformed JSON
        text, model = chat(job, "You configure parametric CAD templates. Reply with ONLY valid JSON.",
                           user, max_tokens=2000)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            cfg = json.loads(m.group()) if m else {}
            tname = cfg.get("template")
            if tname not in TEMPLATES:
                raise ValueError(f"unknown template {tname!r} in reply: {text[:120]!r}")
            return tname, clamp_params(tname, cfg.get("params")), model
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
    raise last_exc


def correct_params(job, tname, params, metrics, spec_result):
    """Computed corrections — no LLM. Returns (new_params, notes) or None if
    nothing correctable remains."""
    t = TEMPLATES[tname]
    new = dict(params)
    notes = []
    dims = (metrics or {}).get("dimensions", {})
    demands = {}                                        # param -> {axis: target}
    for axis, want in job["bbox"].items():
        p = t["bbox_map"].get(axis)
        if p:
            demands.setdefault(p, {})[axis] = want
    for p, axes in demands.items():
        if len(set(axes.values())) > 1:
            # two axes demand different values from ONE param (e.g. a disc od
            # asked to be 120 x 80) — this template cannot satisfy the spec;
            # escalate to reconfiguration instead of oscillating
            return "conflict", (f"template '{tname}' maps axes {sorted(axes)} to the single "
                                f"parameter '{p}' but the spec needs {axes} — pick a different template")
    # one update per PARAM (a disc od may serve both x and y; correcting per
    # axis would double-scale) — targets per param are identical here, the
    # conflict case returned above
    for p, axes in demands.items():
        axis, want = next(iter(axes.items()))
        got = dims.get(axis)
        if got is None or abs(got - want) <= job["bbox_tol"]:
            continue
        if p == "__total_height":                       # flange: thickness + hub_h
            new["thickness"] = max(1.0, want - new.get("hub_h", 0.0))
            notes.append(f"set thickness={new['thickness']} for bbox {axis}={want}")
        elif not p.startswith("__"):
            cur = new.get(p, t["params"][p]["default"])
            new[p] = round(cur * (want / got), 4) if got else want
            notes.append(f"scaled {p} -> {new[p]} for bbox {axis} {got}->{want}")
    for f in (spec_result or {}).get("matched_features", []) + (spec_result or {}).get("missing_features", []):
        tgt_d = f.get("diameter_mm") or f.get("target", {}).get("diameter_mm")
        tgt_n = f.get("count") or f.get("target", {}).get("count")
        if tgt_d is None:
            continue
        # count decides the param family (1 → center bore, N → pattern);
        # diameter closeness only breaks ties within the family
        if tgt_n == 1:
            fam = ("bore_d",)
        else:
            fam = ("bolt_d", "hole_d")
        cands = [p for p in fam if p in t["params"]] or \
                [p for p in ("bore_d", "bolt_d", "hole_d") if p in t["params"]]
        if not cands:
            continue
        best = min(cands, key=lambda p: abs(new.get(p, 0) - tgt_d))
        if abs(new.get(best, 0) - tgt_d) > 0.05:
            new[best] = tgt_d
            notes.append(f"set {best}={tgt_d}")
        npar = {"bolt_d": "bolt_n", "hole_d": "hole_n"}.get(best)
        if npar and tgt_n and tgt_n > 1 and new.get(npar) != tgt_n:
            new[npar] = tgt_n
            notes.append(f"set {npar}={tgt_n}")
    new = clamp_params(tname, new)
    return (new, notes) if (notes and new != dict(params)) else None


def verify(job, proj, step_rel, metrics):
    """Returns (passed, feedback_text, spec_result_json)."""
    problems = []
    spec = None
    dims = metrics.get("dimensions", {})
    for axis, want in job["bbox"].items():
        got = dims.get(axis)
        if got is None or abs(got - want) > job["bbox_tol"]:
            problems.append(f"- bounding box {axis}: required {want} ±{job['bbox_tol']}, measured {got}")
    if not metrics.get("is_valid", False):
        problems.append("- geometry is not a valid solid (self-intersection or open shell)")
    if job["features"]:
        rc, out, err = run_cmd(job, [AGENTCAD, "check-spec", step_rel, "spec.json"],
                               proj, BUILD_TIMEOUT_S)
        spec = last_json(out)
        if not spec:
            problems.append(f"- check-spec produced no result: {(err or '')[:200]}")
        elif not spec.get("passed"):
            for f in spec.get("matched_features", []):
                if not f.get("passed"):
                    t, m = f.get("target", {}), f.get("measured", {})
                    problems.append(f"- feature '{f['name']}': required {t.get('count')}x "
                                    f"dia {t.get('diameter_mm')}, measured {m.get('count')}x "
                                    f"dia {m.get('diameter_mm')}")
            for f in spec.get("missing_features", []):
                tgt = f.get("target", f)
                problems.append(f"- feature '{f.get('name') or tgt.get('name')}' not found "
                                f"(need {tgt.get('count')}x dia {tgt.get('diameter_mm')})")
    return (not problems), "\n".join(problems), spec


def canceled(job):
    if job["cancel"]:
        set_state(job, "canceled")
        log(job, "canceled by user")
        return True
    return False


def build_once(job, proj, label, params=None):
    """One deterministic agentcad build. Returns (result_json, raw_tail)."""
    argv = [AGENTCAD, "run", "design.py", "--output", label, "--preview"]
    if params:
        argv += ["--params", ",".join(f"{k}={v}" for k, v in sorted(params.items()))]
    rc, out, err = run_cmd(job, argv, proj, BUILD_TIMEOUT_S)
    return last_json(out), ((out or "")[-500:] or (err or "")[-500:])


def record_iter(job, it):
    with LOCK:
        job["iterations"].append(it)
        touch(job)
    save_job(job)


STYLE_CHAMFER_POLICIES = {"edge-break", "top-edges", "facet", "fine-arris"}


def apply_style_edges(style, tname, params):
    """S4 edge language + S6 feature language: stamp the style's edge policy and
    feature_language onto the template params the template actually exposes (each
    setp() no-ops when the template lacks that param), clamped to manifest bounds.
    Non-invasive — clearing the style restores full manual control of these params."""
    if not style:
        return
    edge = style.get("edge") or {}
    spec = TEMPLATES[tname]["params"]

    def setp(name, val):
        if name in spec and val is not None:
            params[name] = max(spec[name]["min"], min(spec[name]["max"], float(val)))

    if edge.get("fillet_policy") == "none":
        setp("corner_fillet", 0.0)
    elif edge.get("corner_radius_mm") is not None:
        setp("corner_fillet", edge["corner_radius_mm"])
    if edge.get("chamfer_policy") in STYLE_CHAMFER_POLICIES:
        setp("edge_chamfer", edge.get("chamfer_mm") or 0.6)
    else:
        setp("edge_chamfer", 0.0)
    # S6 feature language (subtractive → stays in the bbox envelope; additive
    # features like ribs are deferred — they'd grow the part past its spec gate)
    feats = style.get("feature_language") or []
    setp("flutes", 16 if "flutes" in feats else 0)      # flange_disc rim reeding
    setp("pockets", 1 if "pockets" in feats else 0)     # plate lightening pockets


def run_stack_job(job, proj):
    """Default engine: LLM configures ONCE (logic only), then a hardened
    deterministic loop drives a parametric template at ~2 s/iteration."""
    override = job.get("config_in") or {}
    if override.get("template") in TEMPLATES:
        tname = override["template"]
        params = clamp_params(tname, override.get("params") or {})
        log(job, f"using your edited parameters (revision — no LLM): {tname} {params}")
    else:
        set_state(job, "configuring")
        try:
            tname, params, model = llm_config(job)
            log(job, f"{model} configured: {tname} {params}")
        except RuntimeError as exc:          # no model loaded — stay autonomous
            log(job, f"{exc}")
            log(job, "falling back to the deterministic heuristic configurator")
            tname, params = heuristic_config(job)
        except Exception as exc:
            log(job, f"LLM configuration failed ({exc!r}); using heuristic configurator")
            tname, params = heuristic_config(job)
            log(job, f"heuristic configured: {tname} {params}")
    skey0 = job.get("style") or override.get("style")
    style0 = STYLES.get(skey0) if skey0 in STYLES else None
    # stamp edges on a fresh job / restyle; a plain revision keeps the user's
    # edited corner_fillet / edge_chamfer (recipe still overrides).
    style_edges = bool(style0) and ((not override) or bool(override.get("restyle")))
    if style_edges:                           # S4: stamp the style's edge language
        apply_style_edges(style0, tname, params)
        log(job, f"style '{skey0}' → fillet {params.get('corner_fillet','-')}, "
                 f"chamfer {params.get('edge_chamfer','-')}"
                 + (f", flutes {int(params['flutes'])}" if params.get('flutes') else "")
                 + (f", pockets" if params.get('pockets') else ""))
    (proj / "design.py").write_text(TEMPLATES[tname]["source"])

    repairs = 0
    seen = {tuple(sorted(params.items()))}
    for i in range(1, 11):                   # fast loop: ~2 s per iteration
        if canceled(job):
            return
        with LOCK:                            # expose the live recipe for editing
            job["config"] = {"engine": "stack", "template": tname, "params": dict(params)}
            skey = job.get("style") or (job.get("config_in") or {}).get("style")
            if skey in STYLES:
                job["config"]["style"] = skey
        set_state(job, "building")
        log(job, f"iteration {i}: {tname} params {params or '(defaults)'}")
        result, raw = build_once(job, proj, f"iter{i}", params)
        it = {"iteration": i, "label": f"iter{i}", "passed": False, "metrics": None,
              "preview": None, "step": None, "viewer": None, "error": None,
              "feedback": None, "params": dict(params)}
        if not result or result.get("status") != "success":
            it["error"] = str((result or {}).get("error") or raw)[-400:]
            record_iter(job, it)
            log(job, f"template build failed (unexpected): {it['error'][:150]}")
            set_state(job, "error")
            return
        step_rel = result["outputs"]["step"]
        it["metrics"] = result["metrics"]
        it["step"] = f"{job['name']}/{step_rel}"
        it["preview"] = f"{job['name']}/{result['preview']}" if result.get("preview") else None
        it["viewer"] = f"{job['name']}/{result['viewer']}" if result.get("viewer") else None
        set_state(job, "verifying")
        passed, feedback, spec_result = verify(job, proj, step_rel, result["metrics"])
        it["passed"], it["feedback"] = passed, feedback
        record_iter(job, it)
        if passed:
            log(job, f"iteration {i} PASSED — {step_rel}")
            set_state(job, "done")
            return
        fix = correct_params(job, tname, params, result["metrics"], spec_result)
        conflict_msg = None
        if fix and fix[0] == "conflict":
            conflict_msg = fix[1]
            log(job, f"structural mismatch: {conflict_msg}")
            fix = None
        elif fix and tuple(sorted(fix[0].items())) in seen:
            log(job, "correction revisits already-tried parameters — escalating")
            fix = None
        if fix:
            params, notes = fix
            seen.add(tuple(sorted(params.items())))
            log(job, f"computed correction: {'; '.join(notes)}")
            continue
        if repairs < 1:                      # last resort: ONE reconfiguration
            repairs += 1
            set_state(job, "configuring")
            why = conflict_msg or f"verification failed: {feedback.replace(chr(10), ' | ')[:200]}"
            log(job, f"no computable correction — reconfiguring: {why}")
            try:
                tname, params, model = llm_config(job, feedback=why)
                log(job, f"{model} reconfigured: {tname} {params}")
            except Exception as exc:
                log(job, f"LLM reconfigure failed ({exc!r}); heuristic retry")
                tname, params = heuristic_config(job)
                log(job, f"heuristic reconfigured: {tname} {params}")
            if style_edges:                  # re-stamp edges onto the new template
                apply_style_edges(style0, tname, params)
            (proj / "design.py").write_text(TEMPLATES[tname]["source"])
            seen = {tuple(sorted(params.items()))}
            continue
        break
    set_state(job, "failed")
    log(job, "stopped without passing spec")


def run_job(job):
    proj = CAD_ROOT / job["name"]
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "spec.json").write_text(json.dumps({"features": job["features"]}, indent=2))
    if not (proj / "agentcad.json").exists():
        run_cmd(job, [AGENTCAD, "init", "--name", job["name"], "--runtime", "build123d"],
                proj, BUILD_TIMEOUT_S)
    if job["generator"] == "structural":
        return run_structural_job(job, proj)
    if job["generator"] == "stack":
        return run_stack_job(job, proj)

    started = time.time()
    feedback = ""
    for i in range(1, job["max_iters"] + 1):
        if canceled(job):
            return
        if time.time() - started > JOB_WALL_CLOCK_S:
            log(job, "job wall-clock limit reached")
            break

        prev_src = (proj / "design.py").read_text() if (proj / "design.py").exists() else ""
        set_state(job, "generating")
        log(job, f"iteration {i}/{job['max_iters']}: generating "
                 f"({'mock' if job['generator'] == 'mock' else 'hermes/' + job['model']})")
        prompt = build_prompt(job, feedback, i, prev_src)
        (proj / f"prompt_iter{i}.txt").write_text(prompt)

        if job["generator"] == "mock":
            (proj / "design.py").write_text(mock_design(job))
        else:
            try:
                code, model_used = generate_direct(job, prompt)
                (proj / "design.py").write_text(code)
                log(job, f"model {model_used} returned {len(code)} chars")
            except RuntimeError as exc:      # infra: no model loaded — abort, don't burn iterations
                log(job, str(exc))
                set_state(job, "error")
                return
            except Exception as exc:
                log(job, f"generation error: {exc!r}")
                feedback = feedback or "- generation failed to return code; output the full design.py"
                continue
        if canceled(job):
            return
        if not (proj / "design.py").exists():
            feedback = "- no design.py was produced; output the full file"
            log(job, "design.py missing after generation")
            continue

        src = (proj / "design.py").read_text()
        (proj / f"design_iter{i}.py").write_text(src)     # snapshot every attempt
        guard = ast_guard(src)
        if guard:
            feedback = f"- {guard}"
            log(job, f"AST guard rejected design.py: {guard}")
            continue

        set_state(job, "building")
        label = f"iter{i}"
        rc, out, err = run_cmd(job, [AGENTCAD, "run", "design.py", "--output", label, "--preview"],
                               proj, BUILD_TIMEOUT_S)
        result = last_json(out)
        it = {"iteration": i, "label": label, "passed": False, "metrics": None,
              "preview": None, "step": None, "viewer": None, "error": None, "feedback": None}
        if not result or result.get("status") != "success":
            msg = str((result or {}).get("error") or (out or "")[-500:] or (err or "")[-500:])
            it["error"] = msg[-500:]
            feedback = f"- design.py failed to build:\n{it['error']}"
            log(job, f"build failed: {it['error'][:200]}")
            with LOCK:
                job["iterations"].append(it)
                touch(job)
            save_job(job)
            continue

        step_rel = result["outputs"]["step"]
        it["metrics"] = result["metrics"]
        it["step"] = f"{job['name']}/{step_rel}"
        if result.get("preview"):
            it["preview"] = f"{job['name']}/{result['preview']}"
        if result.get("viewer"):
            it["viewer"] = f"{job['name']}/{result['viewer']}"

        set_state(job, "verifying")
        passed, feedback, _spec = verify(job, proj, step_rel, result["metrics"])
        it["passed"], it["feedback"] = passed, feedback
        with LOCK:
            job["iterations"].append(it)
            touch(job)
        save_job(job)
        if passed:
            log(job, f"iteration {i} PASSED — {step_rel}")
            set_state(job, "done")
            return
        log(job, f"iteration {i} failed verification: {feedback.replace(chr(10), ' | ')[:300]}")

    set_state(job, "failed")
    log(job, "stopped without passing spec")


def worker():
    while True:
        WAKE.wait()
        with LOCK:
            job_id = JOB_ORDER.pop(0) if JOB_ORDER else None
            if not JOB_ORDER:
                WAKE.clear()
        if not job_id:
            continue
        job = JOBS[job_id]
        if job["cancel"] or job["state"] == "canceled":
            continue
        with LOCK:
            job["started"] = time.time()      # for the GPU strip's elapsed clock
            ACTIVE_JOB["id"] = job_id          # honest heartbeat even during init
        try:
            run_job(job)
        except Exception as exc:   # keep the worker alive
            set_state(job, "error")
            log(job, f"internal error: {exc!r}")
        finally:
            with LOCK:
                ACTIVE_JOB["id"] = None


# ----------------------------------------------- guided design sessions

SESSIONS = {}   # session_id -> dict; in-memory (a session is one sitting)

INTERVIEW_QUESTIONS = [
    {"key": "interfaces", "q": "What are the important interfacing surfaces? "
     "(what does this part touch, mount to, or mate with — and where)"},
    {"key": "important", "q": "What aspects or features are most important? "
     "(dimensions that must hold, load-bearing areas, alignment, finish)"},
    {"key": "unimportant", "q": "What is NOT important? "
     "(areas free to change, looks, exact wall thickness, etc.)"},
]


def synthesize_spec(session):
    """One LLM call: request + interview answers → a working spec the stack
    can execute. The LLM interprets; the numbers become hard gates."""
    qa = "\n".join(f"Q: {q['q']}\nA: {session['answers'].get(q['key'], '(no answer)')}"
                   for q in INTERVIEW_QUESTIONS)
    user = "\n".join([
        "Turn this design request and interview into a machine spec.",
        f"REQUEST: {session['request']}",
        "INTERVIEW:", qa, "",
        "AXIS CONVENTION: x = horizontal width, y = horizontal depth, z = VERTICAL",
        "(heights/tall go on z; wall protrusion is horizontal).",
        "Reply with ONLY this JSON (all lengths in mm — convert inches: 1 in = 25.4 mm):",
        '{"brief": "<one paragraph the CAD engine will read: what the part is, where it',
        ' mounts, where loads act — include everything relevant from the interview>",',
        ' "bbox": {"x": <mm>, "y": <mm>, "z": <mm>},',
        ' "features": [{"name": "<snake_case>", "diameter_mm": <n>, "count": <n>}, ...],',
        ' "objective": "<soft goals: low mass, printability, ...>",',
        ' "engine": "structural" or "stack",',
        ' "style": "<one style KEY from the list, or null>"}',
        'Use "structural" when the request is about carrying loads with minimal',
        'material (organic/optimized shapes); use "stack" for conventional',
        'parametric parts (plates, flanges, enclosures, brackets with exact faces).',
        'Screw/bolt holes go in "features". bbox is the overall size envelope.',
        "STYLE is the aesthetic design language — pick the KEY whose vibe best fits",
        "the request (or null if none clearly fits). Options (key — vibe, engine):",
        *[f"  {k} — {b.get('blurb','')} ({b.get('engine_affinity')})"
          for k, b in STYLES.items()],
    ])
    fake_job = {"model": "auto"}
    text, model = chat(fake_job, "You turn engineering conversations into CAD specs. "
                                 "Reply with ONLY valid JSON.", user, max_tokens=2500)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    spec = json.loads(m.group()) if m else None
    if not isinstance(spec, dict) or not str(spec.get("brief") or "").strip():
        raise ValueError(f"spec synthesis failed: {text[:150]!r}")
    # coerce to a safe shape before it reaches new_job
    clean = {"brief": str(spec["brief"])[:4000], "bbox": {}, "features": [],
             "objective": str(spec.get("objective") or "")[:400],
             "engine": spec.get("engine") if spec.get("engine") in ("structural", "stack") else "stack",
             "style": spec.get("style") if spec.get("style") in STYLES else None}
    for a, v in (spec.get("bbox") or {}).items() if isinstance(spec.get("bbox"), dict) else []:
        try:
            v = float(v)
            if a in ("x", "y", "z") and 0.5 <= v <= 2000:
                clean["bbox"][a] = v
        except (TypeError, ValueError):
            pass
    for f in (spec.get("features") or [])[:12] if isinstance(spec.get("features"), list) else []:
        try:
            d, c = float(f["diameter_mm"]), int(f["count"])
            if 0.1 <= d <= 500 and 1 <= c <= 64:
                clean["features"].append({"name": str(f.get("name") or "feature")[:40],
                                          "diameter_mm": d, "count": c})
        except (TypeError, ValueError, KeyError):
            pass
    return clean, model


def detected_features(session):
    """Measure the rough pass's real geometry so the user can lock features."""
    job = JOBS.get(session.get("rough_job_id"))
    if not job:
        return []
    step = next((it.get("step") or it.get("stl") for it in reversed(job["iterations"])
                 if it.get("step") or it.get("stl")), None)
    if not step or not step.endswith(".step"):
        return []
    # plain subprocess — run_cmd would clobber the worker's ACTIVE cancel slot;
    # short timeout so a session GET can't hang the browser poll for minutes
    cp = subprocess.run([AGENTCAD, "measure", str(CAD_ROOT / step)],
                        cwd=CAD_ROOT / job["name"], capture_output=True, text=True,
                        timeout=60)
    res = last_json(cp.stdout) or {}
    feats = []
    for f in res.get("cylindrical_features", []):
        if 0.5 <= f.get("diameter_mm", 0) <= 200:
            feats.append({"name": f"{f['count']}x hole dia {f['diameter_mm']} mm (axis {f.get('axis')})",
                          "diameter_mm": f["diameter_mm"], "count": f["count"]})
    dims = (res.get("metrics") or {}).get("dimensions")
    if dims:
        feats.insert(0, {"name": f"overall size {dims['x']} x {dims['y']} x {dims['z']} mm",
                         "bbox": dims})
    return feats


def stl_bounds(path):
    """Min/max XYZ of a binary or ASCII STL (stdlib only). Returns (x0,y0,z0,x1,y1,z1)."""
    data = Path(path).read_bytes()
    lo = [1e18, 1e18, 1e18]; hi = [-1e18, -1e18, -1e18]
    def acc(x, y, z):
        for d, v in enumerate((x, y, z)):
            if v < lo[d]: lo[d] = v
            if v > hi[d]: hi[d] = v
    if data[:5].lower() == b"solid" and b"facet" in data[:2048].lower():        # ASCII
        for ln in data.decode("ascii", "ignore").split("\n"):
            t = ln.split()
            if len(t) >= 4 and t[0] == "vertex":
                acc(float(t[1]), float(t[2]), float(t[3]))
    else:                                                                       # binary
        n = struct.unpack("<I", data[80:84])[0]
        if n > MAX_UPLOAD_TRIS:
            raise ValueError(f"mesh too complex ({n} triangles, cap {MAX_UPLOAD_TRIS}) — decimate it")
        for i in range(n):
            base = 84 + i * 50 + 12
            for k in range(3):
                x, y, z = struct.unpack("<3f", data[base + k * 12:base + k * 12 + 12])
                acc(x, y, z)
    if hi[0] < lo[0]:
        raise ValueError("no triangles found")
    return tuple(lo + hi)


def convert_to_stl(src, dst):
    """STEP/IGES → STL via the build123d (CAD) venv. Raises on failure."""
    prog = ("import sys; from build123d import import_step, export_stl; "
            "export_stl(import_step(sys.argv[1]), sys.argv[2])")
    r = subprocess.run([CAD_PYTHON, "-c", prog, str(src), str(dst)],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not Path(dst).exists():
        raise RuntimeError((r.stderr or "conversion failed")[-300:])


def submit_internal(payload):
    """Create + queue a job from server-side code (same path as POST)."""
    job = new_job(payload)
    with LOCK:
        for jid, other in list(JOBS.items()):
            if other["name"] == job["name"]:
                if other["state"] in ("queued", "configuring", "generating",
                                      "building", "verifying", "optimizing"):
                    raise ValueError(f"project '{job['name']}' already has a running job")
                del JOBS[jid]
        JOBS[job["id"]] = job
        JOB_ORDER.append(job["id"])
        WAKE.set()
    save_job(job)
    return job


# -------------------------------------------------- workspace watcher

def workspace():
    """Read-only view of ALL agentcad output in ~/Documents/CAD — including
    work done outside genloop (Hermes TUI sessions, manual runs). Each
    version dir already carries meta.json + preview.png."""
    projects = {}
    for meta_file in CAD_ROOT.glob("*/v*_*/meta.json"):
        pdir = meta_file.parent.parent
        if pdir.name in ("genloop", ".venv"):
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        vdir = meta_file.parent
        entry = {
            "version": meta.get("version"),
            "label": meta.get("label") or vdir.name,
            "created": meta.get("created"),
            "metrics": meta.get("metrics"),
            "preview": f"{pdir.name}/{vdir.name}/preview.png"
            if (vdir / "preview.png").exists() else None,
            "step": f"{pdir.name}/{meta.get('outputs', {}).get('step', '')}"
            if meta.get("outputs", {}).get("step") else None,
            "viewer": f"{pdir.name}/{vdir.name}/viewer.html"
            if (vdir / "viewer.html").exists() else None,
            "mtime": meta_file.stat().st_mtime,
        }
        projects.setdefault(pdir.name, []).append(entry)
    out = []
    for name, versions in projects.items():
        versions.sort(key=lambda v: (v["version"] or 0))
        out.append({"name": name, "updated": max(v["mtime"] for v in versions),
                    "versions": versions})
    out.sort(key=lambda p: -p["updated"])
    return out


# ------------------------------------------------------------------- HTTP

def public_job(job):
    with LOCK:
        snap = copy.deepcopy({k: job.get(k) for k in
                              ("id", "name", "state", "iterations", "brief", "bbox", "bbox_tol",
                               "features", "objective", "max_iters", "model", "generator",
                               "quality", "creativity", "style",
                               "created", "updated", "started", "root", "config")})
        snap["log"] = job["log"][-80:]
    return snap


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _local(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def log_message(self, *a):
        pass

    def do_GET(self):
        if not self._local():
            return self._send(403, {"error": "local only"})
        if self.path in ("/", "/index.html"):
            return self._send(200, (APP_DIR / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        if self.path == "/api/jobs":
            with LOCK:
                order = sorted(JOBS.values(), key=lambda j: -j["created"])
            return self._send(200, [public_job(j) for j in order])
        if self.path == "/api/workspace":
            return self._send(200, workspace())
        if self.path == "/api/styles":
            return self._send(200, {"styles": STYLES, **STYLE_META})
        if self.path == "/api/status":
            # the honest heartbeat of the one serial GPU: what's running + the queue.
            # ACTIVE_JOB is set by the worker the moment it dequeues, so the strip
            # stays honest during the brief 'queued'-but-initializing window too.
            with LOCK:
                aj = JOBS.get(ACTIVE_JOB["id"]) if ACTIVE_JOB["id"] else None
                active = ({"id": aj["id"], "name": aj["name"], "state": aj["state"],
                           "generator": aj["generator"], "started": aj.get("started")}
                          if aj else None)
                queue = list(JOB_ORDER)
            return self._send(200, {"now": time.time(), "active": active, "queue": queue})
        m = re.match(r"^/api/sessions/([0-9a-f]+)$", self.path)
        if m:
            with LOCK:
                s = SESSIONS.get(m.group(1))
                if not s:
                    return self._send(404, {"error": "no such session"})
                rough = JOBS.get(s.get("rough_job_id"))
                rough_state = rough["state"] if rough else None
                need_detect = (rough and rough_state in ("done", "failed")
                               and "detected" not in s)
            if need_detect:
                try:
                    det = detected_features(s)
                except Exception:
                    det = []
                with LOCK:
                    s["detected"] = det
            with LOCK:
                out = copy.deepcopy(s)
            out["detected"] = out.get("detected", [])
            out["rough_state"] = rough_state
            return self._send(200, out)
        m = re.match(r"^/api/jobs/([0-9a-f]+)$", self.path)
        if m:
            job = JOBS.get(m.group(1))
            return self._send(200, public_job(job)) if job else self._send(404, {"error": "no such job"})
        if self.path.startswith("/files/"):
            rel = self.path[len("/files/"):].split("?")[0]
            try:
                target = (CAD_ROOT / rel).resolve()
            except (OSError, ValueError):
                return self._send(404, {"error": "not found"})
            if not target.is_relative_to(CAD_ROOT) or not target.is_file():
                return self._send(404, {"error": "not found"})
            ext = target.suffix.lower().lstrip(".")
            ctype = {"png": "image/png", "step": "application/step", "stl": "model/stl",
                     "html": "text/html", "glb": "model/gltf-binary", "py": "text/plain",
                     "json": "application/json", "txt": "text/plain"}.get(ext)
            if not ctype:
                return self._send(404, {"error": "unsupported type"})
            extra = {"Cache-Control": "no-store"}
            if ext in ("step", "stl", "py"):
                extra["Content-Disposition"] = f'attachment; filename="{target.name}"'
            return self._send(200, target.read_bytes(), ctype, extra)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._local():
            return self._send(403, {"error": "local only"})

        # ---- Phase 3: upload a part → design domain for a structural redesign
        if self.path.split("?")[0] == "/api/upload":
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            fname = q.get("name", ["part.stl"])[0]
            ext = Path(fname).suffix.lower()
            proj = (re.sub(r"[^a-zA-Z0-9_-]+", "-", q.get("project", [Path(fname).stem])[0])
                    [:40].strip("-").lower() or "upload")
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                return self._send(400, {"error": "bad Content-Length header"})
            if length <= 0 or length > UPLOAD_MAX:
                return self._send(400, {"error": "empty or oversized upload (max 150 MB)"})
            if ext not in (".stl", ".step", ".stp"):
                return self._send(400, {"error": f"unsupported format '{ext}' — use STL or STEP "
                                        "(export SolidWorks/other parts as STEP)"})
            raw = self.rfile.read(length)
            try:
                projdir = CAD_ROOT / proj
                projdir.mkdir(parents=True, exist_ok=True)
                (projdir / ("source" + ext)).write_bytes(raw)
                stl = projdir / "domain.stl"
                if ext == ".stl":
                    stl.write_bytes(raw)
                else:
                    convert_to_stl(projdir / ("source" + ext), stl)   # STEP/IGES → STL
                b = stl_bounds(stl)
            except Exception as exc:
                return self._send(400, {"error": f"could not read {ext or 'file'}: {str(exc)[:200]}"})
            dims = [round(b[3] - b[0], 3), round(b[4] - b[1], 3), round(b[5] - b[2], 3)]
            if not all(1 <= d <= 2000 for d in dims):
                return self._send(400, {"error": f"part size out of range: {dims} mm (expect 1–2000)"})
            cfg = {"engine": "structural", "domain_mm": dims,
                   "supports": [{"box": [0, 0, 0, 0.03, 1, 1]}],       # default cantilever;
                   "loads": [{"box": [0.97, 0, 0, 1, 1, 1], "dir": [0, 0, -1]}],   # user refines
                   "domain_stl": str(stl)}
            try:
                job = submit_internal({
                    "name": proj, "generator": "structural",
                    "brief": f"redesign uploaded part ({fname}) inside its own envelope",
                    "bbox": {"x": dims[0], "y": dims[1], "z": dims[2]},
                    "quality": q.get("quality", ["draft"])[0], "config": cfg})
            except ValueError as exc:
                return self._send(409, {"error": str(exc)})
            return self._send(201, public_job(job))

        # ---- guided design sessions
        if self.path == "/api/sessions":
            try:
                p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
                request_txt = str(p.get("request") or "").strip()[:4000]
                name = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(p.get("name") or "guided"))[:40].strip("-").lower() or "guided"
                if not request_txt:
                    raise ValueError("describe the part first")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})
            sid = uuid.uuid4().hex[:12]
            with LOCK:
                SESSIONS[sid] = {"id": sid, "name": name, "request": request_txt,
                                 "stage": "interview", "answers": {}, "spec": None,
                                 "rough_job_id": None, "final_job_id": None}
            return self._send(201, {"id": sid, "questions": INTERVIEW_QUESTIONS})
        m = re.match(r"^/api/sessions/([0-9a-f]+)/(answers|run|finalize)$", self.path)
        if m:
            with LOCK:
                s = SESSIONS.get(m.group(1))
                if not s:
                    return self._send(404, {"error": "no such session"})
            try:
                p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
                if m.group(2) == "answers":
                    # synthesize ONLY — do not run yet. The drafted spec goes to
                    # the confirm gate where the engine + fields are confirmed,
                    # then /run submits. Re-answering (edit brief) re-synthesizes.
                    with LOCK:
                        if s["stage"] not in ("interview", "drafted"):
                            return self._send(409, {"error": "this session already has a running job"})
                        s["stage"] = "synthesizing"
                        s["spec"] = None      # invalidate any prior draft during re-synth
                        s["answers"] = {k: str(v)[:2000] for k, v in (p.get("answers") or {}).items()}
                    try:
                        spec, model = synthesize_spec(s)      # slow; outside LOCK
                    except Exception:
                        with LOCK:
                            s["stage"] = "interview"          # let the user retry
                        raise
                    with LOCK:
                        s["spec"], s["stage"] = spec, "drafted"
                    return self._send(200, {"spec": spec, "model": model})

                if m.group(2) == "run":
                    # the confirm gate hit Run: apply the confirmed engine + any
                    # edited fields, then submit the job (LLM configures it).
                    with LOCK:
                        # idempotent: a double-tap on Run returns the same job
                        if s["stage"] == "running" and s.get("rough_job_id"):
                            ej = JOBS.get(s["rough_job_id"])
                            if ej:
                                return self._send(200, public_job(ej))
                        if s["stage"] != "drafted" or not s.get("spec"):
                            return self._send(409, {"error": "no confirmed draft to run"})
                        base = copy.deepcopy(s["spec"])
                        s["stage"] = "submitting"      # claim atomically vs races
                    engine = p.get("engine") if p.get("engine") in ("structural", "stack") else base["engine"]
                    bbox = {}
                    for a in ("x", "y", "z"):
                        v = (p.get("bbox") or {}).get(a, base["bbox"].get(a))
                        try:
                            v = float(v)
                            if 0.5 <= v <= 2000:
                                bbox[a] = v
                        except (TypeError, ValueError):
                            pass
                    src_feats = p.get("features") if isinstance(p.get("features"), list) else base["features"]
                    feats = []
                    for f in (src_feats or [])[:12]:
                        try:
                            d, c = float(f["diameter_mm"]), int(f["count"])
                            if 0.1 <= d <= 500 and 1 <= c <= 64:
                                feats.append({"name": str(f.get("name") or "feature")[:40],
                                              "diameter_mm": d, "count": c})
                        except (TypeError, ValueError, KeyError):
                            pass
                    objective = str(p.get("objective") if p.get("objective") is not None else base["objective"])[:400]
                    quality = p.get("quality") if p.get("quality") in QUALITY_PRESETS else DEFAULT_QUALITY
                    creativity = p.get("creativity") if p.get("creativity") in CREATIVITY_PRESETS else DEFAULT_CREATIVITY
                    style = p.get("style") if p.get("style") in STYLES else None
                    try:
                        job = submit_internal({"name": s["name"], "brief": base["brief"], "bbox": bbox,
                                               "features": feats, "objective": objective, "generator": engine,
                                               "quality": quality, "creativity": creativity, "style": style})
                    except Exception:
                        with LOCK:
                            s["stage"] = "drafted"    # release the claim so Run can retry
                        raise
                    with LOCK:
                        s["spec"] = {**base, "engine": engine, "bbox": bbox,
                                     "features": feats, "objective": objective}
                        s["rough_job_id"], s["stage"] = job["id"], "running"
                    return self._send(200, public_job(job))

                # ---- finalize: locks become hard gates, relationships steer config
                with LOCK:
                    if s["stage"] == "final" and s.get("final_job_id"):
                        return self._send(200, {"job_id": s["final_job_id"]})   # idempotent
                    rough = JOBS.get(s.get("rough_job_id"))
                    if s["stage"] != "rough" or not s.get("spec") or not rough \
                            or rough["state"] not in ("done", "failed"):
                        return self._send(409, {"error": "rough pass is not finished yet"})
                    spec = copy.deepcopy(s["spec"])
                    rough_gen = rough["generator"]
                locks, rel = [], str(p.get("relationships") or "").strip()[:2000]
                for l in (p.get("locks") or [])[:16]:
                    if not isinstance(l, dict):
                        continue
                    try:
                        d, c = float(l.get("diameter_mm") or 0), int(l.get("count") or 0)
                    except (TypeError, ValueError):
                        continue
                    entry = {"name": str(l.get("name") or "locked")[:80]}
                    if 0.1 <= d <= 500 and 1 <= c <= 64:
                        entry.update(diameter_mm=d, count=c)
                    locks.append(entry)
                locked_feats = [{"name": f"locked_{i}", "diameter_mm": l["diameter_mm"],
                                 "count": l["count"], "tolerance_mm": 0.05}
                                for i, l in enumerate(locks) if "diameter_mm" in l]
                brief = spec["brief"]
                if locks:
                    brief += " LOCKED FEATURES (must not change): " + \
                        "; ".join(l["name"] for l in locks) + "."
                if rel:
                    brief += f" IMPORTANT RELATIONSHIPS BETWEEN FEATURES: {rel}"
                job = submit_internal({
                    "name": s["name"] + "-final", "brief": brief, "bbox": spec["bbox"],
                    "features": locked_feats or spec["features"],
                    "objective": spec["objective"], "generator": rough_gen,
                })
                with LOCK:
                    s["final_job_id"], s["stage"] = job["id"], "final"
                return self._send(200, {"job_id": job["id"]})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:
                return self._send(500, {"error": f"{exc!r}"})

        # ---- revisions: user edited the recipe; rerun deterministically
        m = re.match(r"^/api/jobs/([0-9a-f]+)/revise$", self.path)
        if m:
            src = JOBS.get(m.group(1))
            if not src:
                return self._send(404, {"error": "no such job"})
            try:
                p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
                cfg = p.get("config")
                if not isinstance(cfg, dict):
                    raise ValueError("config object required")
                # a genuine RESTYLE (style key changed) re-stamps the style's levers;
                # a plain param edit keeps the user's values (recipe still overrides)
                if cfg.get("style") != (src.get("config") or {}).get("style"):
                    cfg["restyle"] = True
                base = re.sub(r"-r\d+$", "", src["name"])
                with LOCK:
                    taken = {j["name"] for j in JOBS.values()}
                n = 2
                while f"{base}-r{n}" in taken or (CAD_ROOT / f"{base}-r{n}").exists():
                    n += 1
                # the edited recipe IS the new spec — derive the verification
                # gates from it, or the old gates would "correct" the edits away
                bbox, features = src["bbox"], src["features"]
                if src["generator"] == "structural" and isinstance(cfg.get("domain_mm"), list) \
                        and len(cfg["domain_mm"]) == 3:
                    bbox = dict(zip("xyz", [float(v) for v in cfg["domain_mm"]]))
                if src["generator"] == "stack" and cfg.get("template") in TEMPLATES:
                    t = TEMPLATES[cfg["template"]]
                    prm = {k: v["default"] for k, v in t["params"].items()}
                    prm.update({k: float(v) for k, v in (cfg.get("params") or {}).items()
                                if k in t["params"]})
                    bbox = {}
                    for axis, pname in t["bbox_map"].items():
                        if pname == "__total_height":
                            bbox[axis] = prm.get("thickness", 0) + prm.get("hub_h", 0)
                        elif not pname.startswith("__"):
                            bbox[axis] = prm[pname]
                    features = []
                    if prm.get("bore_d", 0) > 0:
                        features.append({"name": "bore", "diameter_mm": prm["bore_d"], "count": 1})
                    if prm.get("bolt_d", 0) > 0:
                        features.append({"name": "bolts", "diameter_mm": prm["bolt_d"],
                                         "count": int(prm.get("bolt_n", 1))})
                    if prm.get("hole_d", 0) > 0:
                        cnt = int(prm.get("grid_nx", 1) * prm.get("grid_ny", 1)) \
                            if cfg["template"] == "plate_holes" and prm.get("pattern", 1) < 1 \
                            else int(prm.get("hole_n", 1))
                        features.append({"name": "holes", "diameter_mm": prm["hole_d"], "count": cnt})
                job = submit_internal({
                    "name": f"{base}-r{n}", "brief": src["brief"], "bbox": bbox,
                    "features": features, "objective": src["objective"],
                    "generator": src["generator"], "config": cfg,
                    "style": cfg.get("style"),      # restyle shows immediately, not just once it runs
                    "root": src.get("root") or src["id"],
                })
                return self._send(201, public_job(job))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)})

        m = re.match(r"^/api/jobs/([0-9a-f]+)/cancel$", self.path)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self._send(404, {"error": "no such job"})
            with LOCK:
                job["cancel"] = True
                if job["state"] == "queued":
                    job["state"] = "canceled"
                    if job["id"] in JOB_ORDER:
                        JOB_ORDER.remove(job["id"])
                proc = ACTIVE["proc"] if ACTIVE["job_id"] == job["id"] else None
            if proc:
                kill_tree(proc)
            save_job(job)
            return self._send(200, public_job(job))
        if self.path != "/api/jobs":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            job = new_job(json.loads(self.rfile.read(length) or b"{}"))
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"error": str(exc)})
        with LOCK:
            for jid, other in list(JOBS.items()):
                if other["name"] == job["name"]:
                    if other["state"] in ("queued", "configuring", "generating", "building", "verifying", "optimizing"):
                        return self._send(409, {"error": f"project '{job['name']}' already has a running job — cancel it first"})
                    del JOBS[jid]   # same project dir; keep only the newest job
            JOBS[job["id"]] = job
            JOB_ORDER.append(job["id"])
            WAKE.set()
        save_job(job)
        return self._send(201, public_job(job))


if __name__ == "__main__":
    load_templates()
    load_styles()
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()
    print(f"nCAD: http://{HOST}:{PORT}  (projects under {CAD_ROOT})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
