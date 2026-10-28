"""Phase 0 gate: config loads, paths are sane, raw data is never a write target."""
import torch

from src import config
from src.utils import get_device, seed_everything


def test_dataset_present():
    assert config.ROOT.is_dir()
    for name in ("ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt", "records100", "records500"):
        assert (config.ROOT / name).exists(), name


def test_outputs_never_inside_raw_data():
    for d in (config.CACHE_DIR, config.RUNS_DIR, config.RESULTS_DIR, config.CONFIGS_DIR):
        assert config.ROOT not in d.resolve().parents and d.resolve() != config.ROOT


def test_split_is_disjoint_and_complete():
    folds = set(config.TRAIN_FOLDS) | {config.VAL_FOLD, config.TEST_FOLD}
    assert folds == set(range(1, 11))
    assert config.VAL_FOLD not in config.TRAIN_FOLDS and config.TEST_FOLD not in config.TRAIN_FOLDS


def test_profiles():
    assert config.profile("paper")["seq_len"] == 100
    assert config.profile("ours")["seq_len"] == 1000


def test_seeding_repeats():
    seed_everything(config.SEED)
    a = torch.rand(3)
    seed_everything(config.SEED)
    assert torch.equal(a, torch.rand(3))


def test_device_runs_a_matmul():
    dev = get_device()
    x = torch.ones(4, 4, device=dev)
    assert float((x @ x).sum().cpu()) == 64.0
