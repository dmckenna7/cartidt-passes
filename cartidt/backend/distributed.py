"""Minimal DDP/torchrun wiring (Ref: Sec. V.C — 4 × A100 80GB)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int


def init_distributed() -> DistInfo:
    rank_env = os.environ.get("RANK")
    if rank_env is None:
        return DistInfo(rank=0, local_rank=0, world_size=1)
    rank = int(rank_env)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    _LOG.info(
        "DDP initialised rank=%d local_rank=%d world_size=%d backend=%s",
        rank,
        local_rank,
        world_size,
        backend,
    )
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size)


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
