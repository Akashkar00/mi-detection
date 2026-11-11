"""Phase 1: build the binary MI-vs-Normal label table from PTB-XL metadata.

MI     = any MI diagnostic code with likelihood >= config.LIKELIHOOD_MIN
Normal = diagnostic superclass set is exactly {NORM}
Everything else is dropped.
"""
import ast

import pandas as pd

from src import config

NOISE_FLAGS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "pacemaker")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    db = pd.read_csv(config.ROOT / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(config.ROOT / "scp_statements.csv", index_col=0)
    return db, scp[scp["diagnostic"] == 1]


def build_labels(likelihood_min: float = config.LIKELIHOOD_MIN) -> pd.DataFrame:
    db, diag = load_raw()
    superclass = diag["diagnostic_class"].to_dict()
    subclass = diag["diagnostic_subclass"].to_dict()

    def classes(codes: dict) -> set:
        return {superclass[c] for c in codes if c in superclass}

    def mi_codes(codes: dict) -> list:
        return [c for c, p in codes.items() if superclass.get(c) == "MI" and p >= likelihood_min]

    db["superclasses"] = db["scp_codes"].apply(classes)
    db["mi_codes"] = db["scp_codes"].apply(mi_codes)
    is_mi = db["mi_codes"].str.len() > 0
    is_norm = db["superclasses"].apply(lambda s: s == {config.NEGATIVE_CLASS})
    # an ECG with an MI code below the likelihood cut is neither class: drop it
    keep = is_mi | (is_norm & ~db["scp_codes"].apply(lambda c: any(superclass.get(k) == "MI" for k in c)))

    out = db[keep].copy()
    out["y"] = is_mi[keep].astype(int)
    out["mi_subclass"] = out["mi_codes"].apply(lambda cs: sorted({subclass[c] for c in cs}))
    out["other_classes"] = out["superclasses"].apply(lambda s: sorted(s - {"MI"}))
    out["patient_id"] = out["patient_id"].astype(int)
    out["fold"] = out["strat_fold"].astype(int)
    cols = ["patient_id", "fold", "y", "mi_subclass", "other_classes", "age", "sex", "device",
            "infarction_stadium1", *NOISE_FLAGS, "filename_lr", "filename_hr"]
    return out[cols]


def split_of(fold: int) -> str:
    if fold in config.TRAIN_FOLDS:
        return "train"
    return "val" if fold == config.VAL_FOLD else "test"
