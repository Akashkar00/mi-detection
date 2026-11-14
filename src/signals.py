"""Phase 2: load PTB-XL 100 Hz records, band-pass, cache, normalise with train-only stats."""
from multiprocessing import Pool

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

from src import config
from src.labels import build_labels, split_of

SIG_PATH = config.CACHE_DIR / "signals_mv_1000.npy"      # (N, 1000, 12) band-passed, mV
NORM_PATH = config.CACHE_DIR / "norm_stats.npz"


def _load(fname: str) -> np.ndarray:
    sig, _ = wfdb.rdsamp(str(config.ROOT / fname))
    b, a = butter(3, [config.BANDPASS_HZ[0], config.BANDPASS_HZ[1]], btype="band", fs=config.FS)
    return filtfilt(b, a, sig, axis=0).astype(np.float32)


def build_signals() -> np.ndarray:
    labels = build_labels()
    with Pool(8) as p:
        arrs = p.map(_load, labels["filename_lr"].tolist(), chunksize=64)
    x = np.stack(arrs)
    assert x.shape == (len(labels), 1000, 12) and np.isfinite(x).all()
    config.CACHE_DIR.mkdir(exist_ok=True)
    np.save(SIG_PATH, x)
    train = x[(labels["fold"].isin(config.TRAIN_FOLDS)).values]
    np.savez(NORM_PATH, mean=train.mean((0, 1)), std=train.std((0, 1)))   # train folds only
    return x


def load_signals(seq_len: int = 1000, normalise: bool = True) -> np.ndarray:
    x = np.load(SIG_PATH)
    if normalise:
        s = np.load(NORM_PATH)
        x = (x - s["mean"]) / s["std"]
    if seq_len != 1000:                                   # block-mean downsample (100 -> 10 Hz for "paper")
        x = x.reshape(len(x), seq_len, 1000 // seq_len, 12).mean(2)
    return x.astype(np.float32)


if __name__ == "__main__":
    x = build_signals()
    print(x.shape, x.dtype, float(x.mean()), float(x.std()))
