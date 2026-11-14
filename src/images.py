"""Phase 3A/3B: render band-passed mV signals as 12-row ECG images, cache as uint8.

Deterministic cv2 renderer (no matplotlib state). Drawn at 2x then INTER_AREA down to 224
so thin traces stay visible. Cached inverted: trace bright (255), paper dark (0).
 - "ours":  no tick text, lossless
 - "paper": lead labels + time axis ticks drawn, JPEG round trip (quality 90)
"""
from multiprocessing import Pool

import cv2
import numpy as np

from src import config
from src.signals import SIG_PATH

S = config.IMG_SIZE * 2
ROW = S // 12


def render(sig: np.ndarray, axes: bool, jpeg: bool) -> np.ndarray:
    """sig (1000, 12) mV -> (224, 224) uint8, trace bright on dark."""
    canvas = np.full((S, S), 255, np.uint8)
    xs = np.linspace(0, S - 1, sig.shape[0]).astype(np.float32)
    for k in range(12):
        c = k * ROW + ROW / 2
        y = c - np.clip(sig[:, k], -config.Y_LIMIT_MV, config.Y_LIMIT_MV) / config.Y_LIMIT_MV * (ROW / 2)
        pts = np.stack([xs, y.astype(np.float32)], 1).round().astype(np.int32)
        cv2.polylines(canvas, [pts], False, 0, 2, cv2.LINE_AA)
        if axes:
            cv2.putText(canvas, config.LEADS[k], (2, int(c - ROW / 2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    if axes:
        for t in range(0, 11, 2):
            x = min(int(t / 10 * (S - 1)), S - 14)
            cv2.putText(canvas, str(t), (x, S - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    if jpeg:
        canvas = cv2.imdecode(cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(canvas, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    return 255 - small


def _job(args):
    return render(*args)


def img_path(profile: str):
    return config.CACHE_DIR / f"images_{profile}.npy"


def build_images(profile: str) -> np.ndarray:
    p = config.profile(profile)
    x = np.load(SIG_PATH, mmap_mode="r")
    jobs = [(np.asarray(x[i]), p["image_axes"], p["image_format"] == "jpg") for i in range(len(x))]
    with Pool(8) as pool:
        imgs = np.stack(pool.map(_job, jobs, chunksize=64))
    np.save(img_path(profile), imgs)
    return imgs


def load_images(profile: str) -> np.ndarray:
    return np.load(img_path(profile), mmap_mode="r")


if __name__ == "__main__":
    for prof in config.PROFILES:
        a = build_images(prof)
        print(prof, a.shape, a.dtype, a.mean())
