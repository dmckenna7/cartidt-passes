"""Baseline registry — descriptive only (Ref: Sec. IV.B, Sec. V.A-B).

We only retain the descriptive metadata; baseline numbers are taken either from
the IWOAI 2019 leaderboard or from a † re-run with the original authors' code
when available. No baseline model code lives in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaselineEntry:
    name: str
    family: str
    reference: str
    reproduced_by_us: bool


BASELINE_REGISTRY: tuple[BaselineEntry, ...] = (
    BaselineEntry("3D nnU-Net", "self-configuring", "Isensee et al. 2021", True),
    BaselineEntry("3D V-Net", "encoder-decoder", "Milletari et al. 2016", False),
    BaselineEntry("Attention U-Net", "U-Net w/ attention", "Oktay et al. 2018", False),
    BaselineEntry("TransUNet", "CNN-Transformer hybrid", "Chen et al. 2021", False),
    BaselineEntry("MedSAM", "SAM medical fine-tune", "Ma et al. 2024", True),
    BaselineEntry("SAM-Med3D", "SAM 3D", "Wang et al. 2024", True),
    BaselineEntry("SAMRI-2", "memory-based VFM", "Ferreira et al. 2026", False),
    BaselineEntry("Prob. U-Net", "probabilistic", "Kohl et al. 2018", False),
)
