# nCAD

nCAD is a local generative-CAD web app. The app source lives in this repo; generated
CAD projects live in a separate workspace that defaults to `~/Documents/CAD`.

## Install

From the repo root:

```sh
bin/bootstrap
```

This creates two disposable Python environments with `uv`:

- Engine venv: `~/Documents/CAD/.venv` by default, for AgentCAD/build123d.
- Solver venv: `<repo>/.venv` by default, for the topology optimizer.

To recreate both environments from scratch:

```sh
bin/bootstrap --recreate
```

`scikit-sparse` is optional. If it cannot be installed, the topology solver keeps
working with scipy's sparse solvers.

## Launch

macOS/Linux:

```sh
bin/ncad
```

Windows PowerShell:

```powershell
.\bin\ncad.ps1
```

The launcher checks the environments, starts the server if needed, and opens
`http://127.0.0.1:8765/`.

## Configuration

All configuration is optional:

- `NCAD_APP_DIR`: source repo path. Defaults to the launcher/server location.
- `NCAD_CAD_ROOT`: generated projects/workspace root. Defaults to `~/Documents/CAD`.
- `NCAD_ENGINE_VENV`: engine venv. Defaults to `$NCAD_CAD_ROOT/.venv`.
- `NCAD_SOLVER_VENV`: solver venv. Defaults to `$NCAD_APP_DIR/.venv`.
- `NCAD_HOST`: server host. Defaults to `127.0.0.1`.
- `NCAD_PORT`: server port. Defaults to `8765`.
- `NCAD_LMS_API`: LM Studio API root. Defaults to `http://127.0.0.1:1234`.
- `NCAD_AUTO_START_LMS`: start LM Studio's local API server automatically for
  local `NCAD_LMS_API` targets. Defaults to `1`; set to `0` to disable.
- `NCAD_LMS_CLI`: path to the `lms` CLI if it is not on `PATH`. Defaults to
  `lms`, then `~/.lmstudio/bin/lms` when present.
- `NCAD_LOG`: launcher log path. Defaults to `<repo>/server.log`.

Existing projects under `~/Documents/CAD` remain visible unless `NCAD_CAD_ROOT`
is pointed somewhere else.
