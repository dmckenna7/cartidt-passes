from __future__ import annotations

import torch
import torch.nn as nn

from cartidt.passes.encode.lora import LoRAProjection


def test_lora_layer_freezes_base_weights() -> None:
    base = nn.Linear(64, 64, bias=False)
    wrapped = LoRAProjection(in_features=64, out_features=64, base=base, rank=16, alpha=32)
    assert wrapped.base.weight.requires_grad is False
    assert wrapped.a.requires_grad is True
    assert wrapped.b.requires_grad is True


def test_lora_layer_starts_as_identity() -> None:
    torch.manual_seed(0)
    base = nn.Linear(32, 32, bias=False)
    nn.init.normal_(base.weight)
    wrapped = LoRAProjection(in_features=32, out_features=32, base=base, rank=8, alpha=16)
    x = torch.randn(4, 32)
    expected = base(x)
    got = wrapped(x)
    assert torch.allclose(got, expected, atol=1e-6)


def test_lora_total_param_count_matches_rank() -> None:
    base = nn.Linear(1024, 1024, bias=False)
    wrapped = LoRAProjection(in_features=1024, out_features=1024, base=base, rank=16, alpha=32)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    assert trainable == 2 * 16 * 1024
