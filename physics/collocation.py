from __future__ import annotations

import torch


def full_collocation_mask(residual: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(residual)
