from __future__ import annotations

import torch
from torch import nn

from .encoder_tcn import TemporalConvEncoder
from .graph_layer import GraphMixing


class BaseTCNGraph(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = TemporalConvEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.graph = GraphMixing(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_obs, mask], dim=-1)
        h = self.encoder(z)
        h = self.graph(h, adj)
        return self.head(h)
