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
On Windows, `pyproject.toml` pulls torch from the PyTorch cu130 index (`[tool.uv.sources]`,
marker `sys_platform == 'win32'`), so `uv sync` installs the CUDA build; macOS keeps PyPI torch (MPS).

## Run order

| Phase | Command |
|---|---|
| 0 | `uv run python -m src.verify_data` (dataset checksums) · `uv run pytest -q` |
| 1+ | added as each phase lands |

## Rules

- Never write inside `../ptb-xl-…-1.0.3/`. All outputs go to `cache/`, `runs/`, `results/`.
- Every setting lives in `src/config.py`.
- Test fold 10 is touched once, in Phase 8.

## Running the full pipeline on a CUDA GPU (lab desktop)

1. `uv sync` installs torch 2.14.1+cu130 (needs a driver with CUDA >= 13.0, see `nvidia-smi`):
   ```powershell
   cd "E:\MI Project\mi_detection"
   $env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\mi_detection"
   uv sync
   uv run python -c "import torch; print(torch.cuda.is_available())"         # must print True
   ```
   Training keeps all images/signals resident on the GPU (~2.7 GB) and runs the CNN convs in bf16.
   RTX 3060, per epoch: B3 ~9 s, fusion ~10 s, B2 ~2.6 s, B1 ~2.3 s.
2. `cache/` (signals + both image sets) is already built on the drive. To rebuild: `uv run python -m src.signals` then `uv run python -m src.images`.
3. Run everything: open `notebooks/MI_detection_end_to_end.ipynb` (or
   `uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/MI_detection_end_to_end.ipynb`).
   Training is resumable: finished runs in `runs/<name>/` are reloaded (`fit_cached`), the rest train.
4. The earlier Mac (MPS) runs (B3/fusion at bs 64, 8 epochs) are kept in `runs_mps/`. `runs/` holds the all-GPU
   rerun with B3/fusion at `bs=32` (paper), up to 30 epochs, patience 5.
   `notebooks/MI_detection_train_only.ipynb` is sections 1-6 only (all training, no test scoring).
5. Test fold 10 is scored once per model via `results/test_evaluation_log.json`; delete that file only for a deliberate full rerun.
