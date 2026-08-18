"""Seed control.

Every run records its seed in the manifest, and three seeds per condition is
pre-registered. A run whose seed is not reproducible is not evidence.
"""

from __future__ import annotations

import os
import random

__all__ = ["seed_everything", "PREREGISTERED_SEEDS"]

#: Locked in configs/prereg.lock.yaml. Do not add a fourth after seeing results.
PREREGISTERED_SEEDS = (42, 1337, 2024)


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed python, numpy and torch. Returns the seed, for logging.

    ``deterministic=True`` also pins cuDNN, which costs throughput. The pilot set
    ``cudnn.benchmark = True`` and ``deterministic = False``, so its runs were not
    bit-reproducible -- acceptable, but it has to be recorded rather than assumed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    return seed
