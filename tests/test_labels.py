"""Phase 1 gate: label rule, counts and split integrity."""
import pytest

from src import config
from src.labels import build_labels, split_of


@pytest.fixture(scope="module")
def df():
    return build_labels()


def test_binary_counts(df):
    assert len(df) == 14538
    assert df["y"].sum() == 5469
    assert (df["y"] == 0).sum() == 9069


def test_split_counts(df):
    d = df.assign(split=df["fold"].map(split_of))
    got = {s: (len(g), int(g["y"].sum())) for s, g in d.groupby("split")}
    assert got == {"train": (11622, 4379), "val": (1454, 540), "test": (1462, 550)}


def test_no_patient_leakage(df):
    d = df.assign(split=df["fold"].map(split_of))
    p = {s: set(g["patient_id"]) for s, g in d.groupby("split")}
    assert not (p["train"] & p["val"] or p["train"] & p["test"] or p["val"] & p["test"])


def test_normals_have_no_other_class(df):
    norm = df[df["y"] == 0]
    assert all(norm["other_classes"].apply(lambda c: c == ["NORM"]))
    assert all(norm["mi_subclass"].str.len() == 0)


def test_mi_all_have_mi_subclass(df):
    assert all(df.loc[df["y"] == 1, "mi_subclass"].str.len() > 0)


def test_known_label_noise_dropped(df):
    assert 17 not in df.index  # report says inferior infarct, but no MI code


def test_likelihood_100_is_stricter():
    assert build_labels(100)["y"].sum() == 2975  # plan: ablation A6


def test_folds_cover_config(df):
    assert set(df["fold"]) == set(config.TRAIN_FOLDS) | {config.VAL_FOLD, config.TEST_FOLD}
