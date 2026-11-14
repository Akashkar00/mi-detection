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

## Phase 0 addendum: Windows CUDA (2026-11-14)

- Lab desktop: RTX 3060 12 GB, driver CUDA 13.3, i7-14700, 32 GB RAM.
- `pyproject.toml`: torch from the PyTorch cu130 index on `sys_platform == 'win32'` only (macOS keeps PyPI/MPS). `uv sync` gives torch 2.14.1+cu130; `torch.cuda.is_available()` is True. Phase 0 gate passes on the training machine.
- Speed-ups in `src/train.py` / `src/models.py`: images + signals read into RAM once and kept on the GPU (~2.7 GB), batches indexed there; loss summed on device (no per-step sync); bf16 autocast in the PaperCNN convs only. A bf16 LSTM converged worse (B1 val 0.56-0.77 vs 0.88-0.91 in fp32 over the first epochs), so signal models and fusion heads stay fp32. B1/B2 give bit-identical histories to fp32.
- Per epoch on the 3060, old loop -> new: B3 17.2 -> 9.4 s, fusion 18.4 -> 10.3 s, B2 4.1 -> 2.6 s, B1 3.2 -> 2.3 s. Full training (26 runs) took ~25 min.
- Early-epoch val AUROC of B1 is seed/device-sensitive (Mac 0.93, CUDA 0.88, CPU 0.77 at epoch 0, same seed); all converge to ~0.95.

## Phases 2-9: all-GPU rerun (2026-11-14)

**Changes vs the Mac runs:** Mac/MPS runs moved to `runs_mps/` (kept). B3 and fusion now trained at bs 32 (paper), max 30 epochs, patience 5 (Mac: bs 64, 8 epochs, patience 3). Training cells also saved as `notebooks/MI_detection_train_only.ipynb` (sections 1-6).

**Phase 4, unimodal baselines (val, fold 9):**

| Model | Val AUROC | Note |
|---|---|---|
| B1 LSTM (paper, 100 steps), 3 seeds | 0.9520 ± 0.0005 | lr 1e-3 |
| B2 ResNet1D (1000 steps), 3 seeds | 0.9741 ± 0.0014 | lr 1e-3 |
| B3 CNN (paper image) | 0.9665 [0.957, 0.975] | used all 30 epochs, flat at ~0.966 (Mac 8 ep: 0.949) |

**Phase 5, fusion (val):** Attention R1 0.9509, Attention R2 0.9496, Concat 0.9650, Late B1+B3 0.9644.

**Phase 6, tuning (val, lr in {3e-4, 1e-3, 3e-3}, seed 42):** B1 0.9532 / 0.9515 / 0.9558, B2 0.9716 / 0.9729 / 0.9736, so both frozen at lr 3e-3. Image/fusion models keep lr 1e-3 (not tuned; limitation). Final 3-seed runs: B1 0.9540 ± 0.0013, B2 0.9731 ± 0.0016; seed ensembles B1 0.9579, B2 0.9762.

**Phase 7, ablations (val):**
- Paired ΔAUROC: R1 - B1 -0.0069 [-0.012, -0.002]; R1 - B2 -0.0252 [-0.034, -0.018]; R1 - Concat -0.0141 [-0.022, -0.006]; R1 - Late -0.0161 [-0.022, -0.011]; R1 - R2 +0.0013 [-0.003, 0.005] (n.s.); B2 - B1 +0.0183 [0.012, 0.026]; B1 - B3 -0.0087 [-0.017, -0.001].
- A3 LSTM input length: 100 0.9558 · 250 0.9696 · 500 0.9731 · 1000 0.9712. The L1000 run diverged at epoch 11 (loss 0.17 -> 0.44, val 0.65) at lr 3e-3; best checkpoint (epoch 10) kept.
- A7 noisy vs clean val records (n 355 / 1099): B1 0.954 / 0.960, B2 0.977 / 0.976, B3 0.963 / 0.968, R1 0.942 / 0.955.

**Phase 8, test (fold 10, 1,462 ECGs; scored once, Youden threshold from val; 7 entries in `results/test_evaluation_log.json`, git hash "none" since no commit exists yet):**

| Model | AUROC [95% CI] | AUPRC | Sens | Spec | Brier |
|---|---|---|---|---|---|
| B2 ResNet1D | 0.9698 [0.961, 0.978] | 0.961 | 0.824 | 0.965 | 0.063 |
| B3 CNN | 0.9608 [0.951, 0.970] | 0.951 | 0.853 | 0.930 | 0.083 |
| Late B1+B3 | 0.9595 [0.949, 0.969] | 0.952 | 0.864 | 0.935 | 0.073 |
| Concat | 0.9522 [0.941, 0.963] | 0.941 | 0.836 | 0.931 | 0.087 |
| B1 LSTM | 0.9482 [0.936, 0.959] | 0.940 | 0.858 | 0.910 | 0.077 |
| Attn R2 | 0.9362 [0.921, 0.949] | 0.928 | 0.826 | 0.934 | 0.085 |
| Attn R1 | 0.9361 [0.922, 0.948] | 0.925 | 0.815 | 0.911 | 0.088 |

- Paired ΔAUROC (test): R1 - B1 -0.0121 [-0.018, -0.006]; R1 - B2 -0.0337 [-0.045, -0.024]; R1 - Concat -0.0161 [-0.026, -0.006]; B2 - B1 +0.0217 [0.013, 0.031]; B2 - B3 +0.0091 [0.001, 0.016].
- Subgroups (B1 / B2 / R1): male 0.943 / 0.963 / 0.934; female 0.952 / 0.976 / 0.938; age <50 0.912 / 0.964 / 0.904; 50-65 0.890 / 0.934 / 0.859; 65-80 0.954 / 0.973 / 0.935; 80+ 0.938 / 0.953 / 0.939; clean 0.953 / 0.974 / 0.941; noisy 0.932 / 0.957 / 0.926. IMI vs Normal 0.941 / 0.968 / 0.929; AMI vs Normal 0.972 / 0.981 / 0.963.
- The 50-65 band is hardest for every model. The age confound from Phase 1 still needs the age-matched check.

**Phase 9, explainability & errors:**
- R1 attention stays near 0.5/0.5 (img<-img 0.598 Normal / 0.504 MI), so fusion attention adds little over concat.
- Figures: `gradcam.png`, `gradcam_leads.png`, `saliency.png`, `test_curves.png`, `fusion_curves.png`, `baseline_curves.png`.
- Errors of the best model (B2): 97 FN; subclass IMI 61, AMI 30, LMI 5; stadium unknown 68, II-III 20, III 8, I 1; noisy share 0.29 vs 0.26 for all test MI. 50 errors in `results/error_review.csv`.

**Headline:** the paper's attention fusion does not beat its own LSTM branch, and a plain 1D ResNet on the 1000-step signal is best (test AUROC 0.970).
