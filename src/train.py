"""One training entry point. Predictions for val AND test are saved per run, but test is
only ever scored via evaluate.log_test_run in Phase 8."""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src import config
from src.images import load_images
from src.labels import build_labels, split_of
from src.models import build
from src.signals import load_signals
from src.utils import get_device, seed_everything

_DATA, _IMG, _SIG = {}, {}, {}


def get_data(profile="ours", seq_len=None, img_profile=None):
    """Tensors for all records in label order + split masks. Pixel stats fit on train only.
    Arrays are read into RAM once (not memmapped) and a copy is kept on the training device,
    so batches are indexed there with no per-step disk reads or host->device copies."""
    seq_len = seq_len or config.profile(profile)["seq_len"]
    img_profile = img_profile or profile
    key = (seq_len, img_profile)
    if key not in _DATA:
        dev = get_device()
        lab = build_labels()
        split = lab["fold"].map(split_of).values
        if img_profile not in _IMG:                            # shared across seq_lens (A3 ablation)
            img = torch.from_numpy(np.array(load_images(img_profile)))                 # uint8 (N,224,224)
            f = img[torch.from_numpy(split == "train")].float() / 255
            _IMG[img_profile] = (img, (float(f.mean()), float(f.std())), img.to(dev))
            del f
        if seq_len not in _SIG:
            sig = torch.from_numpy(load_signals(seq_len))
            _SIG[seq_len] = (sig, sig.to(dev))
        img, stats, img_dev = _IMG[img_profile]
        y = lab["y"].values.astype(np.float32)
        _DATA[key] = dict(lab=lab, split=split, y=y, img=img, img_stats=stats, sig=_SIG[seq_len][0],
                          img_dev=img_dev, sig_dev=_SIG[seq_len][1], y_dev=torch.from_numpy(y).to(dev))
    return _DATA[key]


def _batch(d, idx, dev, aug=False):
    on = d["img_dev"].device.type == torch.device(dev).type     # use the device-resident copy when it matches
    src_img, src_sig = (d["img_dev"], d["sig_dev"]) if on else (d["img"], d["sig"])
    idx = idx.to(src_img.device)
    img = ((src_img[idx].float() / 255 - d["img_stats"][0]) / d["img_stats"][1]).unsqueeze(1)
    if aug:                                                    # small shift only: no flips (they destroy ST/time)
        dx, dy = np.random.randint(-4, 5, 2)
        img = torch.roll(img, (int(dy), int(dx)), (2, 3))
        img = img + 0.05 * torch.randn_like(img)
    return img.to(dev), src_sig[idx].to(dev)


@torch.no_grad()
def predict(model, d, mask, dev, bs=512):
    model.eval()
    ids = np.where(mask)[0]
    out = []
    for i in range(0, len(ids), bs):
        ii = torch.from_numpy(ids[i:i + bs])
        out.append(torch.sigmoid(model(*_batch(d, ii, dev))).cpu())
    return torch.cat(out).numpy()


def fit(name, seed=42, lr=1e-3, wd=0.0, bs=32, max_epochs=config.MAX_EPOCHS, patience=5, dropout=0.0,
        profile="ours", seq_len=None, img_profile=None, augment=True, pos_weight=None,
        run_name=None, train_mask=None, verbose=True):
    """Train, early-stop on val AUROC, save run folder, return dict with val/test probs."""
    seed_everything(seed)
    dev = get_device()
    d = get_data(profile, seq_len, img_profile)
    model = build(name, dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tr_mask = (d["split"] == "train") if train_mask is None else train_mask
    tr_ids, y = np.where(tr_mask)[0], d["y_dev"].to(dev)
    val_mask = d["split"] == "val"
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=dev)
    best, best_state, bad, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        perm = np.random.permutation(tr_ids)
        tot = torch.zeros((), device=dev)                      # summed on device: no per-step sync
        for i in range(0, len(perm), bs):
            ii = torch.from_numpy(perm[i:i + bs])
            logit = model(*_batch(d, ii, dev, aug=augment and name != "B1_lstm" and name != "B2_resnet1d"))
            loss = F.binary_cross_entropy_with_logits(logit, y[ii.to(dev)], pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.detach() * len(ii)
        tot = tot.item()
        pv = predict(model, d, val_mask, dev)
        auc = roc_auc_score(d["y"][val_mask], pv)
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_auroc": auc})
        if verbose:
            print(f"{run_name or name} ep{ep:02d} loss {tot/len(perm):.4f} val_auc {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    res = {"name": name, "run": run_name or name, "seed": seed, "val_auroc": best, "epochs": len(hist),
           "history": hist, "p_val": predict(model, d, val_mask, dev),
           "p_test": predict(model, d, d["split"] == "test", dev),
           "cfg": dict(lr=lr, wd=wd, bs=bs, dropout=dropout, profile=profile, seq_len=seq_len,
                       img_profile=img_profile, augment=augment, pos_weight=pos_weight)}
    run_dir = config.RUNS_DIR / (run_name or f"{name}_s{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("p_")}, indent=1))
    np.savez(run_dir / "preds.npz", p_val=res["p_val"], p_test=res["p_test"])
    torch.save(best_state, run_dir / "checkpoint.pt")
    res["model"] = model
    return res


def fit_cached(name, run_name, force=False, **kw):
    """Resume-friendly wrapper: reuse runs/<run_name> if finished, else train."""
    run_dir = config.RUNS_DIR / run_name
    if not force and (run_dir / "preds.npz").exists():
        m = json.loads((run_dir / "metrics.json").read_text())
        p = np.load(run_dir / "preds.npz")
        print(f"[cached] {run_name} val_auroc={m['val_auroc']:.4f}")
        model = build(name, m["cfg"]["dropout"])
        model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location="cpu"))
        return {**m, "p_val": p["p_val"], "p_test": p["p_test"], "model": model}
    return fit(name, run_name=run_name, **kw)
