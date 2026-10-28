"""Single source of truth: paths, label rule, folds, seeds, input profiles.

Phase 1 freezes the label rule and split below. Changing them later means
re-running everything from Phase 2 and noting it in results/phase_log.md.
"""
from pathlib import Path

# ---------- paths ----------
PROJECT_DIR = Path(__file__).resolve().parents[1]          # .../MI Project/mi_detection
ROOT = PROJECT_DIR.parent / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CACHE_DIR = PROJECT_DIR / "cache"                           # derived data, rebuildable, git-ignored
RUNS_DIR = PROJECT_DIR / "runs"                             # one folder per training run
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIGS_DIR = PROJECT_DIR / "configs"                       # frozen final configs (Phase 6)

# ---------- reproducibility ----------
SEED = 42
N_SEEDS = 3                                                 # seeds per model for mean ± sd

# ---------- labels (frozen in Phase 1) ----------
LIKELIHOOD_MIN = 0          # MI if any MI code has likelihood >= this; 100 is an ablation (A6)
POSITIVE_CLASS = "MI"       # any MI diagnostic code
NEGATIVE_CLASS = "NORM"     # superclass set must be exactly {NORM}

# ---------- split (PTB-XL recommendation; folds are patient-wise) ----------
TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10              # touched once, in Phase 8, through src.evaluate's guard

# ---------- signal ----------
FS = 100                    # records100 = 100 Hz -> (1000, 12); 500 Hz is ablation A4
N_LEADS = 12
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
BANDPASS_HZ = (0.5, 40.0)   # our choice; the paper describes no filtering

# ---------- image ----------
IMG_SIZE = 224              # our choice; the paper gives no size
Y_LIMIT_MV = 3.0            # fixed y-range per lead so amplitude means the same in every image

# ---------- input profiles ----------
# "paper": as close to the paper as it states; "ours": our improved defaults.
PROFILES = {
    "paper": {"seq_len": 100, "image_axes": True, "image_format": "jpg"},
    "ours": {"seq_len": 1000, "image_axes": False, "image_format": "png"},
}

# ---------- training defaults (paper Tables 1-2) ----------
BATCH_SIZE = 32
LR = 1e-3
OPTIMIZER = "adam"
MAX_EPOCHS = 50


def profile(name: str) -> dict:
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]
