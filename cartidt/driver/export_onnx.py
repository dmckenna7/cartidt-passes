"""CLI entry — export the seg head to ONNX (the GNN + evidence head stays in Python)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cartidt.backend.checkpointing import restore_checkpoint
from cartidt.link import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
_LOG = logging.getLogger("cartidt.export")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CartiDT segmentation graph to ONNX")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./cartidt_seg.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args(argv)


class _SegWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        seg_logits, _alpha = self.model(volume)
        return seg_logits


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    payload = restore_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    target = tuple(int(x) for x in cfg.data.target_shape)
    if len(target) != 3:
        raise ValueError("data.target_shape must be 3D")
    dummy = torch.zeros(1, 1, target[0], target[1], target[2], device=device)
    wrapper = _SegWrapper(model)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(args.out),
        opset_version=int(args.opset),
        input_names=["volume"],
        output_names=["seg_logits"],
        dynamic_axes={"volume": {0: "batch"}, "seg_logits": {0: "batch"}},
    )
    _LOG.info("ONNX file written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
