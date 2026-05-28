from __future__ import annotations

import numpy as np
import torch
from scipy.stats import dirichlet

from cartidt.passes.evidence.dirichlet import dirichlet_log_pdf


def test_dirichlet_log_pdf_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    alpha_np = rng.uniform(1.0, 5.0, size=(4,))
    pi_np = rng.dirichlet(alpha_np)
    expected = dirichlet.logpdf(pi_np, alpha_np)
    got = dirichlet_log_pdf(torch.from_numpy(pi_np).unsqueeze(0), torch.from_numpy(alpha_np).unsqueeze(0))
    assert float(got.item()) == float(np.float64(expected))
