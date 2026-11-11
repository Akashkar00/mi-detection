"""Phase 1 audit: reproduce plan counts, check leakage, describe splits, write cache/labels.parquet.

Run from mi_detection/:  uv run python -m src.audit_labels
"""
import pandas as pd

from src import config
from src.labels import NOISE_FLAGS, build_labels, load_raw, split_of

EXPECTED = {"n_ecg": 14538, "mi": 5469, "norm": 9069,
            "train": (11622, 4379), "val": (1454, 540), "test": (1462, 550)}


def main() -> None:
    db, diag = load_raw()
    print(f"records {len(db):,} / patients {db['patient_id'].nunique():,}")
    print(f"diagnostic codes {len(diag)} -> superclasses {sorted(diag['diagnostic_class'].unique())}")

    df = build_labels()
    df["split"] = df["fold"].map(split_of)
    mi, norm = int(df["y"].sum()), int((df["y"] == 0).sum())
    print(f"\nbinary set {len(df):,}  MI {mi:,}  Normal {norm:,}  MI% {100 * mi / len(df):.1f}")
    checks = {"n_ecg": len(df) == EXPECTED["n_ecg"], "mi": mi == EXPECTED["mi"], "norm": norm == EXPECTED["norm"]}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        print(f"  {s:5s} {len(d):6,}  MI {int(d['y'].sum()):5,}  ({100 * d['y'].mean():.1f}%)")
        checks[s] = (len(d), int(d["y"].sum())) == EXPECTED[s]

    pats = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    leaks = {f"{a}/{b}": len(pats[a] & pats[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    print("\npatient overlap:", leaks)
    checks["no_leak"] = not any(leaks.values())

    mi_df = df[df["y"] == 1]
    print("\nMI subclass mix (ECGs):", mi_df["mi_subclass"].explode().value_counts().to_dict())
    print("MI with other classes:", int((mi_df["other_classes"].str.len() > 0).sum()))
    print("age 300 (>89):", int((df["age"] == 300).sum()))

    print("\nper split, by label:")
    d = df.assign(age=df["age"].where(df["age"] != 300, 90))
    g = d.groupby(["split", "y"])
    summary = pd.DataFrame({"n": g.size(), "age_mean": g["age"].mean().round(1),
                            "male_%": (100 * g["sex"].apply(lambda s: (s == 1).mean())).round(1)})
    for f in NOISE_FLAGS:
        summary[f + "_%"] = (100 * g[f].apply(lambda s: s.notna().mean())).round(1)
    print(summary.to_string())
    print("\ninfarction_stadium1 (MI):", mi_df["infarction_stadium1"].value_counts(dropna=False).to_dict())
    print("device:", df["device"].value_counts().to_dict())

    config.CACHE_DIR.mkdir(exist_ok=True)
    out = df.assign(mi_subclass=df["mi_subclass"].apply(",".join), other_classes=df["other_classes"].apply(",".join))
    out.to_parquet(config.CACHE_DIR / "labels.parquet")
    print("\nchecks:", checks)
    print("ALL PASS" if all(checks.values()) else "MISMATCH: stop and investigate")


if __name__ == "__main__":
    main()
