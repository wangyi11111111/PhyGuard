from __future__ import annotations

import torch
from torch import nn


class GraphGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gates = nn.Linear(input_dim + hidden_dim * 2, hidden_dim * 2)
        self.candidate = nn.Linear(input_dim + hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("nm,bmh->bnh", adj, h)
        gate_input = torch.cat([x, h, neigh], dim=-1)
        reset, update = torch.sigmoid(self.gates(gate_input)).chunk(2, dim=-1)
        candidate_input = torch.cat([x, reset * h, neigh], dim=-1)
        h_tilde = torch.tanh(self.candidate(candidate_input))
        h_next = update * h + (1.0 - update) * h_tilde
        return self.norm(h_next)


class GRINLite(nn.Module):
    """A compact GRIN-style bidirectional graph recurrent imputer.

    This is intentionally small for debug experiments. It is not a byte-for-byte
    reproduction of the official GRIN implementation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = int(output_dim)
        step_input_dim = input_dim * 2
        self.fwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.bwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.fwd_head = nn.Linear(hidden_dim, output_dim)
        self.bwd_head = nn.Linear(hidden_dim, output_dim)
        self.out = nn.Linear(output_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _direction(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor, reverse: bool, return_hidden: bool = False):
        if reverse:
            x_obs = torch.flip(x_obs, dims=[1])
            mask = torch.flip(mask, dims=[1])
            cell = self.bwd_cell
            head = self.bwd_head
        else:
            cell = self.fwd_cell
            head = self.fwd_head

        batch, steps, nodes, _ = x_obs.shape
        h = x_obs.new_zeros(batch, nodes, cell.hidden_dim)
        previous = x_obs.new_zeros(batch, nodes, self.output_dim)
        preds = []
        hidden_states = []
        for t in range(steps):
            current = mask[:, t] * x_obs[:, t] + (1.0 - mask[:, t]) * previous
            step_input = torch.cat([current, mask[:, t]], dim=-1)
            h = cell(step_input, h, adj)
            h = self.dropout(h)
            previous = head(h)
            preds.append(previous)
            hidden_states.append(h)
        pred = torch.stack(preds, dim=1)
        hidden = torch.stack(hidden_states, dim=1)
        if reverse:
            pred = torch.flip(pred, dims=[1])
            hidden = torch.flip(hidden, dims=[1])
        if return_hidden:
            return pred, hidden
        return pred

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        fwd = self._direction(x_obs, mask, adj, reverse=False)
        bwd = self._direction(x_obs, mask, adj, reverse=True)
        return self.out(torch.cat([fwd, bwd], dim=-1))
