from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_config(repo_root: Path) -> OmegaConf:
    return OmegaConf.load(repo_root / "configs" / "_unittest.yaml")


@pytest.fixture(scope="session", autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)
