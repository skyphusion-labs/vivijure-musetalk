# AGENTS.md

## Cursor Cloud specific instructions

This is a thin RunPod GPU serverless wrapper (`handler.py`). The heavy ML deps in
`requirements.txt` are STAGED in the Dockerfile (order/flags matter; a flat
`pip install -r requirements.txt` will NOT reproduce a working env) and import only
on the card, so a full run needs a RunPod GPU pod, not this CPU VM.

CPU-testable CI gate (`.github/workflows/ci.yml`) -- the unit tests stub the heavy
deps via `sys.modules`, so they run without a GPU:

- Set up a venv (`.venv`; `python3.12-venv` is installed by the environment update
  script, which also creates the venv and installs these tools):
  `python3 -m venv .venv && .venv/bin/pip install ruff==0.15.20 pytest==9.1.1`.
- `.venv/bin/ruff check --select E9,F .`
- `.venv/bin/python -m py_compile handler.py`
- `.venv/bin/python -m pytest -q tests/` (SoftDegrade envelope routing; deps stubbed)

Verified in this environment: ruff clean, py_compile OK, 20 tests passed.
