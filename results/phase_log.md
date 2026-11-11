# Phase log

One section per phase: what was done, the numbers, and anything that changed from `../plan.md`.

## Phase 0: Setup & project scaffold (2026-10-28)

**Environment (Mac, Apple M1):**
- uv 0.12.17, Python 3.12, torch 2.14.1 (MPS backend available, CUDA n/a on this machine)
- The environment is at `~/.venvs/mi_detection` via `UV_PROJECT_ENVIRONMENT`. The T7 is exFAT (no symlinks), so `.venv` can't live on the drive.
- Dependencies: torch, numpy, pandas, pyarrow, scipy, wfdb, matplotlib, pillow, opencv-python-headless, scikit-learn, tqdm; dev: pytest. Pinned in `uv.lock`.

**Scaffold:**
- `src/config.py`: paths, seed 42, label rule (likelihood ≥ 0), folds 1–8 / 9 / 10, `paper` + `ours` profiles.
- `src/utils.py`: `seed_everything`, `get_device` (CUDA > MPS > CPU).
- `src/verify_data.py`: SHA256 check against the dataset's own `SHA256SUMS.txt`.
- `tests/test_setup.py`: 6 tests (dataset present, outputs never inside raw data, split complete and disjoint, profiles, seeding repeats, device matmul). **6/6 pass.**

**Raw data integrity:** _pending (checksum run in progress)_

**Open items:**
- Windows CUDA build of torch: needs the GPU model / driver version of the training PC.
- git repo initialised; no commit made yet (waiting for the user's go-ahead).
