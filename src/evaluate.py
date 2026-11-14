"""Metrics, bootstrap CIs, paired bootstrap, and the guarded test-fold evaluation (Phase 8)."""
import json
import subprocess
import time

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from src import config

TEST_LOG = config.RESULTS_DIR / "test_evaluation_log.json"


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    sens = tp / max(tp + fn, 1)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "f1": 2 * prec * sens / max(prec + sens, 1e-9), "sensitivity": sens,
            "specificity": tn / max(tn + fp, 1), "brier": brier_score_loss(y, p), "threshold": thr}


def bootstrap_ci(y, p, fn=roc_auc_score, n=1000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_auc(y, pa, pb, n=1000, seed=0):
    """AUROC(a) - AUROC(b) with paired bootstrap 95% CI."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def log_test_run(model: str, result: dict, rerun_reason: str | None = None) -> None:
    """Test fold may be scored once per model; a second run needs a reason and is logged."""
    log = json.loads(TEST_LOG.read_text()) if TEST_LOG.exists() else []
    if any(r["model"] == model for r in log) and not rerun_reason:
        raise RuntimeError(f"test fold already evaluated for {model}; pass rerun_reason to repeat")
    try:
        gh = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_DIR, text=True).strip()
    except Exception:
        gh = "none"
    log.append({"model": model, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": gh,
                "rerun_reason": rerun_reason, **result})
    TEST_LOG.write_text(json.dumps(log, indent=1))
