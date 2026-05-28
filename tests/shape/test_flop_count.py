from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cartidt.diagnostics.cost import count_params, profile_per_volume


def test_count_params_reports_in_millions() -> None:
    layer = nn.Linear(100, 100)
    p = count_params(layer)
    assert p == pytest.approx(0.0101, rel=0.05)


def test_profile_runs_on_cpu() -> None:
    model = nn.Conv3d(1, 4, kernel_size=3, padding=1)
    vol = torch.zeros(1, 1, 4, 8, 8)
    result = profile_per_volume(model, vol, iterations=2)
    assert result.seconds_per_volume > 0.0
    assert result.peak_memory_gb == 0.0
