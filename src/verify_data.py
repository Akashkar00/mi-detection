"""Check the raw PTB-XL files against the dataset's own SHA256SUMS.txt (read-only).

Usage: python -m src.verify_data
Writes nothing inside the dataset folder; the summary goes to results/data_checksums.json.
"""
import hashlib
import json
import time

from tqdm import tqdm

from src.config import RESULTS_DIR, ROOT


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    entries = []
    for line in (ROOT / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries.append((digest, name.strip()))
    missing, mismatched = [], []
    t0 = time.time()
    for digest, name in tqdm(entries, desc="sha256", unit="file"):
        path = ROOT / name
        if not path.exists():
            missing.append(name)
        elif sha256(path) != digest:
            mismatched.append(name)
    summary = {
        "checked": len(entries),
        "missing": missing,
        "mismatched": mismatched,
        "ok": not missing and not mismatched,
        "seconds": round(time.time() - t0, 1),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_checksums.json").write_text(json.dumps(summary, indent=2))
    print(f"checked {summary['checked']} files: {len(missing)} missing, {len(mismatched)} mismatched -> ok={summary['ok']}")


if __name__ == "__main__":
    main()
