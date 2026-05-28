from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cartidt.backend.segmentation_loss import SoftDiceCELoss
from cartidt.passes.evidence.edl import EDLLoss


class _Tiny3DDecoder(nn.Module):
    def __init__(self, in_ch: int = 1, classes: int = 4) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.GELU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.GELU(),
        )
        self.head = nn.Conv3d(16, classes, kernel_size=1)
        self.grade_proj = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.body(x)
        seg = self.head(feats)
        node = feats.mean(dim=(2, 3, 4))
        evidence = torch.nn.functional.softplus(self.grade_proj(node)) + 1.0
        return seg, evidence.unsqueeze(1)


@pytest.mark.slow
def test_single_batch_overfit_drives_seg_loss_below_threshold() -> None:
    torch.manual_seed(0)
    model = _Tiny3DDecoder(in_ch=1, classes=4)
    volume = torch.randn(1, 1, 8, 16, 16)
    target = torch.zeros(1, 8, 16, 16, dtype=torch.long)
    target[..., :8] = 1
    target[..., 8:12] = 2
    target[..., 12:] = 3
    grade = torch.tensor([[2]])
    dice_ce = SoftDiceCELoss(num_classes=4)
    edl = EDLLoss(num_grades=4, anneal_epochs=10)
    optimiser = torch.optim.AdamW(model.parameters(), lr=5e-3)
    final_seg = float("inf")
    for _step in range(200):
        seg_logits, alpha = model(volume)
        loss = dice_ce(seg_logits, target) + 0.1 * edl(alpha, grade, epoch=10.0)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        final_seg = float(dice_ce(seg_logits.detach(), target).item())
    assert final_seg < 0.05, f"single-batch overfit failed; L_seg={final_seg:.4f}"
