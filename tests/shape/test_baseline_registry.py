from __future__ import annotations

from cartidt.diagnostics.reference_models import BASELINE_REGISTRY


def test_baseline_registry_has_eight_entries() -> None:
    assert len(BASELINE_REGISTRY) == 8
    names = {b.name for b in BASELINE_REGISTRY}
    for required in ("3D nnU-Net", "SAMRI-2", "SAM-Med3D", "Prob. U-Net"):
        assert required in names
