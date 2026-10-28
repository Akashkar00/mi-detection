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

**Raw data integrity:** all **87,203** files match the dataset's `SHA256SUMS.txt` (0 missing, 0 mismatched; 96 s). Summary in `results/data_checksums.json`.

**Open items:**
- Windows CUDA build of torch: needs the GPU model / driver version of the training PC.
- GitHub repo: private `Akashkar00/mi-detection` (push pending; see below).

**Gate:** ✅ on this Mac (tests 6/6, MPS works, data verified). The Windows CUDA check stays open until the GPU is known.
