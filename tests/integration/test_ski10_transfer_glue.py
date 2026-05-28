from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "transfer_ski10.yaml"


def test_transfer_ski10_loads_and_is_inference_only() -> None:
    cfg = OmegaConf.load(CONFIG_PATH)
    assert cfg.data.name.lower() == "ski10"
    assert int(cfg.train.epochs) == 0
