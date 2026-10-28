# mi_detection

Multi-modal (ECG image + signal) MI vs. Normal detection on PTB-XL, phase by phase.
The plan and phase status live in `../plan.md`; numbers per phase live in `results/phase_log.md`.

## Setup

The T7 drive is **exFAT**, which doesn't support symlinks, so the Python environment
lives on each machine's own disk, not in this folder. Point uv at it before any `uv` command.

**macOS (zsh):**
```bash
cd "/Volumes/T7 Shield/MI Project/mi_detection"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/mi_detection"
uv sync
uv run pytest -q                 # Phase 0 gate
```

**Windows (PowerShell):**
```powershell
cd "E:\MI Project\mi_detection"      # use the T7's drive letter
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\mi_detection"
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```
On Windows, PyPI's `torch` is CPU-only. A CUDA build needs a PyTorch index entry in
`pyproject.toml` that matches your GPU/driver (still to do; see plan.md Phase 0).

## Run order

| Phase | Command |
|---|---|
| 0 | `uv run python -m src.verify_data` (dataset checksums) · `uv run pytest -q` |
| 1+ | added as each phase lands |

## Rules

- Never write inside `../ptb-xl-…-1.0.3/`. All outputs go to `cache/`, `runs/`, `results/`.
- Every setting lives in `src/config.py`.
- Test fold 10 is touched once, in Phase 8.
