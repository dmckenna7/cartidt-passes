"""CLI entry — infer (Ref: Sec. V.C, single forward pass)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

from cartidt.backend.checkpointing import restore_checkpoint
from cartidt.frontend.conditioning.intensity_norm import zscore_per_volume
from cartidt.link import build_model
from cartidt.passes.evidence.decompose import aleatoric, epistemic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
_LOG = logging.getLogger("cartidt.infer")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CartiDT inference")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./infer"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    payload = restore_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    arr = np.asarray(nib.load(str(args.volume)).get_fdata(), dtype=np.float32)
    vol = torch.from_numpy(np.ascontiguousarray(arr))
    while vol.ndim < 5:
        vol = vol.unsqueeze(0)
    vol = zscore_per_volume(vol).to(device)
    with torch.no_grad():
        seg_logits, alpha = model(vol)
        seg = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
        u_ale = aleatoric(alpha).cpu().numpy()
        u_epi = epistemic(alpha).cpu().numpy()
        grade = alpha.argmax(dim=-1).cpu().numpy().astype(np.int16)
    args.out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(args.out / "segmentation.nii.gz"))
    np.savez(args.out / "alpha.npz", alpha=alpha.cpu().numpy(), u_ale=u_ale, u_epi=u_epi, grade=grade)
    _LOG.info("artefacts saved to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
