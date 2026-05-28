"""Seed plumbing (Ref: Sec. V.C — seeds 42 / 123 / 7)."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

_LOG = logging.getLogger(__name__)


def set_seed(seed: int, deterministic_cudnn: bool = True) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic_cudnn
    torch.backends.cudnn.deterministic = deterministic_cudnn
    _LOG.info("set_seed=%d deterministic_cudnn=%s", seed, deterministic_cudnn)
    return seed
