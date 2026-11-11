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

## Phase 1: Data audit & label lock (2026-11-11)

**Code:** `src/labels.py` (label rule), `src/audit_labels.py` (audit; run `uv run python -m src.audit_labels`). Writes `cache/labels.parquet`.

**Counts reproduce the plan exactly (all checks pass):**
- 21,799 ECGs / 18,869 patients; 44 diagnostic codes in 5 superclasses.
- Binary set 14,538: MI 5,469 (37.6 %), Normal 9,069.
- Train (folds 1-8) 11,622 / 4,379 MI; val (fold 9) 1,454 / 540; test (fold 10) 1,462 / 550.
- Patient overlap train/val/test: 0.
- MI subclasses (ECGs): IMI 3,271, AMI 3,078, LMI 201, PMI 17. MI with another class too: 2,937.

**Findings to carry forward:**
- **Age confound:** MI mean age 66-70 vs Normal 51-52 in every split. A model can partly learn "old" instead of "infarct". Report an age-matched check in Phase 7/9.
- **Sex skew in train:** 36 % male among MI vs 56 % among Normal (val/test are closer to balanced). Another shortcut to watch.
- `infarction_stadium1` is "unknown" for 3,372 of 5,469 MI records; do not use it as a feature.
- Noise flags: static_noise 13-19 %, baseline_drift 5-12 %, burst_noise 2-3 %; electrodes_problems and pacemaker are near zero after filtering. Noisy records stay in training; Phase 7 reports with/without.
- 151 binary-set records have `age = 300` (means > 89); mapped to 90 for the summary only.
- Device mix is wide (CS100, CS-12, AT-6 variants); check per-device performance in Phase 9.

**Decisions (unchanged from the plan):** likelihood >= 0 is the main rule (likelihood = 100 is ablation A6); noisy records kept.

**Gate:** rule and split are in `src/config.py` and frozen. Changing them means restarting from Phase 2.
