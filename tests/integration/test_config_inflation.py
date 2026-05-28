from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
EXPECTED_KEYS = {"data", "model", "train"}


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_every_config_loads_with_expected_sections(config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    missing = EXPECTED_KEYS - set(cfg.keys())
    assert not missing, f"{config_path.name} missing sections: {missing}"
    if (
        "ablation" in config_path.name
        and "transfer" not in config_path.name
        and "no_joint" not in config_path.name
    ):
        return
    assert int(cfg.model.num_seg_classes) >= 2
    assert int(cfg.train.epochs) > 0


@pytest.mark.parametrize("ablation", ["transfer", "gnn", "component"])
def test_every_ablation_family_present(ablation: str) -> None:
    matches = list(CONFIG_DIR.glob(f"ablation_{ablation}_*.yaml"))
    assert matches, f"no ablation configs for family {ablation}"
