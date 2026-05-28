"""CLI entry — evaluate (Ref: Sec. V.D)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cartidt.backend.checkpointing import restore_checkpoint
from cartidt.backend.reproducibility import set_seed
from cartidt.diagnostics.calibration import (
    area_under_risk_coverage,
    expected_calibration_error,
    fpr_at_threshold,
)
from cartidt.diagnostics.ordinal import auroc_any_damage, quadratic_weighted_kappa
from cartidt.diagnostics.overlap import average_symmetric_surface_distance, dice_similarity_coefficient
from cartidt.frontend.catalog import build_loaders
from cartidt.link import build_model
from cartidt.passes.evidence.decompose import epistemic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
_LOG = logging.getLogger("cartidt.evaluate")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CartiDT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./eval"))
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
    set_seed(int(cfg.train.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    payload = restore_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    loaders = build_loaders(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    label_ids = list(range(1, int(cfg.model.num_seg_classes)))
    dsc_per_label: dict[int, list[float]] = {label: [] for label in label_ids}
    assd_per_label: dict[int, list[float]] = {label: [] for label in label_ids}
    alpha_all: list[torch.Tensor] = []
    grade_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loaders.test:
            batch = batch.to(device)
            seg_logits, alpha = model(batch.volume)
            prediction = seg_logits.argmax(dim=1)
            for b in range(batch.batch_size):
                spacing = batch.meta[b].voxel_spacing_mm
                for label in label_ids:
                    dsc_per_label[label].append(
                        dice_similarity_coefficient(prediction[b], batch.segmentation[b], label)
                    )
                    assd_per_label[label].append(
                        average_symmetric_surface_distance(
                            prediction[b], batch.segmentation[b], label, voxel_spacing_mm=spacing
                        )
                    )
            alpha_all.append(alpha.detach().cpu())
            grade_all.append(batch.grade.detach().cpu())
    for label, values in dsc_per_label.items():
        metrics[f"dsc_label_{label}"] = float(sum(values) / max(len(values), 1))
    for label, values in assd_per_label.items():
        cleaned = [v for v in values if v == v]
        metrics[f"assd_mm_label_{label}"] = float(sum(cleaned) / len(cleaned)) if cleaned else float("nan")
    if alpha_all:
        alpha_cat = torch.cat(alpha_all, dim=0)
        grade_cat = torch.cat(grade_all, dim=0)
        u_epi = epistemic(alpha_cat)
        metrics["auroc"] = auroc_any_damage(alpha_cat, grade_cat)
        metrics["kappa_w"] = quadratic_weighted_kappa(alpha_cat, grade_cat)
        prediction_any = (alpha_cat.argmax(dim=-1) >= 1).long()
        target_any = (grade_cat >= 1).long()
        metrics["ece"] = expected_calibration_error(
            confidence=1.0 - u_epi.clamp(0.0, 1.0),
            correctness=(alpha_cat.argmax(dim=-1) == grade_cat).float(),
        )
        metrics["aurc"] = area_under_risk_coverage(
            uncertainty=u_epi, correctness=(alpha_cat.argmax(dim=-1) == grade_cat).float()
        )
        metrics["fpr_at_15"] = fpr_at_threshold(
            uncertainty=u_epi,
            target_any_damage=target_any,
            prediction_any_damage=prediction_any,
            abstention_rate=0.15,
        )
    with (args.out / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    _LOG.info("metrics written to %s", args.out / "metrics.json")
    for key, value in metrics.items():
        _LOG.info("%s = %.4f", key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
