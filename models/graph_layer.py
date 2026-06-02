from __future__ import annotations

import torch
from torch import nn


class GraphMixing(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, H], adj: [N, N]
        neigh = torch.einsum("nm,btmh->btnh", adj, h)
        out = self.self_linear(h) + self.neigh_linear(neigh)
        out = self.act(out)
        return self.norm(out + h)
