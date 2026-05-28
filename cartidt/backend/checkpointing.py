"""Atomic checkpoint IO (Ref: kickoff R4)."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

_LOG = logging.getLogger(__name__)


def atomic_save(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if Path(tmp_path).exists():
            with contextlib.suppress(FileNotFoundError):
                Path(tmp_path).unlink()
    _LOG.info("checkpoint saved atomically to %s", path)
    return path


def restore_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location)
    _LOG.info("checkpoint restored from %s", path)
    return payload
