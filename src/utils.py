"""Small shared helpers: seeding and device choice."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU, CUDA, MPS) so runs repeat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                                 # also seeds CUDA and MPS
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """CUDA (Windows/Linux GPU) > MPS (Apple silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
