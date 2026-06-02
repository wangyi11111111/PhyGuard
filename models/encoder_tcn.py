from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.blocks = nn.ModuleList([TemporalBlock(hidden_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C]
        out = x.permute(0, 3, 1, 2)
        out = self.input_proj(out)
        for block in self.blocks:
            out = block(out)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        return out
