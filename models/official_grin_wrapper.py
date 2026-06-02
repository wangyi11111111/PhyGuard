from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torch import nn

from .litetrust_pinn import _node_failure_signal


DEFAULT_OFFICIAL_GRIN_ROOT = Path("C:/Users/21329/grin_official_cache")


def _load_official_grinet(official_root: str | Path | None = None):
    root = Path(official_root) if official_root is not None else DEFAULT_OFFICIAL_GRIN_ROOT
    if not root.exists():
        raise FileNotFoundError(f"official GRIN repo not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    utils_pkg = "lib.utils"
    if utils_pkg not in sys.modules:
        module = types.ModuleType(utils_pkg)
        module.__path__ = [str(root / "lib" / "utils")]
        sys.modules[utils_pkg] = module
    from lib.nn.models.grin import GRINet

    return GRINet


class OfficialGRINWrapper(nn.Module):
    """Thin adapter around the official GRINet implementation."""

    def __init__(
        self,
        adj,
        input_dim: int,
        hidden_dim: int = 32,
        ff_dim: int = 32,
        dropout: float = 0.0,
        official_root: str | Path | None = None,
    ):
        super().__init__()
        grinet_cls = _load_official_grinet(official_root)
        self.model = grinet_cls(
            adj=adj,
            d_in=int(input_dim),
            d_hidden=int(hidden_dim),
            d_ff=int(ff_dim),
            ff_dropout=float(dropout),
            n_layers=1,
            kernel_size=2,
            decoder_order=1,
            d_u=0,
            d_emb=8,
            layer_norm=False,
            merge="mlp",
            impute_only_holes=True,
        )

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask > 0.5
        output = self.model(x_obs, mask_bool)
        if isinstance(output, tuple):
            output = output[0]
        return output


class OfficialGRINLiteTrustCorrection(nn.Module):
    """Official GRIN prediction plus lightweight physics correction/router."""

    def __init__(
        self,
        adj,
        input_dim: int,
        hidden_dim: int = 32,
        ff_dim: int = 32,
        dropout: float = 0.0,
        correction_clip: float = 0.5,
        flow_only_correction: bool = False,
        correction_mode: str = "mixed",
        official_root: str | Path | None = None,
    ):
        super().__init__()
        if correction_mode not in {"mixed", "generic", "physics", "gated"}:
            raise ValueError("correction_mode must be one of: mixed, generic, physics, gated")
        self.grin = OfficialGRINWrapper(
            adj=adj,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            official_root=official_root,
        )
        self.correction_clip = float(correction_clip)
        self.flow_only_correction = bool(flow_only_correction)
        self.correction_mode = correction_mode
        self.input_dim = int(input_dim)
        self.fd_gain = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        feature_dim = int(input_dim) + 1 + 6
        correction_hidden = max(hidden_dim // 2, 16)
        self.generic_head = nn.Sequential(
            nn.Linear(int(input_dim) + 6, correction_hidden),
            nn.GELU(),
            nn.Linear(correction_hidden, input_dim),
        )
        self.physics_head = nn.Sequential(
            nn.Linear(feature_dim, correction_hidden),
            nn.GELU(),
            nn.Linear(correction_hidden, input_dim),
        )
        self.physics_projection_head = nn.Sequential(
            nn.Linear(8, 12),
            nn.GELU(),
            nn.Linear(12, 3),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(feature_dim + 2 * input_dim, correction_hidden),
            nn.GELU(),
            nn.Linear(correction_hidden, input_dim),
        )
        self.region_gate_head = nn.Sequential(
            nn.Linear(7, 8),
            nn.GELU(),
            nn.Linear(8, input_dim),
        )
        self.phys_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        residual_signed: torch.Tensor,
        extra_feature: torch.Tensor,
    ) -> dict:
        mu_data = self.grin(x_obs, mask)
        residual = residual_signed.detach()
        correction_input = torch.cat([mu_data.detach(), residual, extra_feature.detach()], dim=-1)
        generic_input = torch.cat([mu_data.detach(), extra_feature.detach()], dim=-1)
        generic_delta = torch.tanh(self.generic_head(generic_input)) * self.correction_clip
        learned_phys_delta = torch.tanh(self.physics_head(correction_input)) * self.correction_clip

        obs = mask.mean(dim=-1, keepdim=True)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        residual_rank = extra_feature[..., 3:4] if extra_feature.shape[-1] >= 4 else residual.abs().clamp(0.0, 1.0)
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        neighbor_missing = extra_feature[..., 5:6] if extra_feature.shape[-1] >= 6 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        temporal = extra_feature[..., 0:1] if extra_feature.shape[-1] >= 1 else torch.zeros_like(obs)
        spatial = extra_feature[..., 1:2] if extra_feature.shape[-1] >= 2 else torch.zeros_like(obs)
        low_residual = 1.0 - residual_rank
        random_missing = torch.clamp(local_missing - node_failure, min=0.0, max=1.0)
        region_evidence = torch.cat(
            [obs, local_missing, node_failure, neighbor_missing, temporal, spatial, residual_rank],
            dim=-1,
        )

        fd_delta = torch.zeros_like(mu_data)
        fd_delta[..., 0:1] = -torch.clamp(residual, min=-2.0, max=2.0)
        neigh = torch.einsum("nm,btmc->btnc", adj, mu_data.detach())
        graph_delta = neigh - mu_data.detach()
        temporal_delta = torch.zeros_like(mu_data)
        if mu_data.shape[1] > 1:
            temporal_delta[:, 1:-1] = 0.5 * (mu_data[:, :-2].detach() + mu_data[:, 2:].detach()) - mu_data[:, 1:-1].detach()
            temporal_delta[:, 0] = mu_data[:, 1].detach() - mu_data[:, 0].detach()
            temporal_delta[:, -1] = mu_data[:, -2].detach() - mu_data[:, -1].detach()

        graph_component = torch.zeros_like(mu_data)
        if mu_data.shape[-1] > 1:
            graph_component[..., 1:2] = 0.15 * graph_delta[..., 1:2]
        if mu_data.shape[-1] > 2:
            graph_component[..., 2:3] = 0.25 * graph_delta[..., 2:3]

        temporal_component = torch.zeros_like(mu_data)
        if mu_data.shape[-1] > 2:
            temporal_component[..., 2:3] = 0.15 * temporal_delta[..., 2:3]

        projection_input = torch.cat([region_evidence.detach(), residual.detach().abs()], dim=-1)
        projection_logits = self.physics_projection_head(projection_input)
        component_strength = 0.1 + 0.9 * torch.sigmoid(projection_logits)
        fd_strength = component_strength[..., 0:1]
        graph_strength = component_strength[..., 1:2]
        temporal_strength = component_strength[..., 2:3]
        explicit_phys_delta = fd_strength * fd_delta + graph_strength * graph_component + temporal_strength * temporal_component
        delta_phys = learned_phys_delta + explicit_phys_delta + self.fd_gain * fd_delta
        delta_phys = torch.clamp(delta_phys, min=-self.correction_clip, max=self.correction_clip)
        if self.flow_only_correction:
            delta_phys = delta_phys.clone()
            delta_phys[..., 1:] = 0.0

        prior_score = (
            1.3 * random_missing
            + 0.7 * low_residual
            - 1.5 * node_failure
            - 0.5 * temporal
            - 0.4 * residual_rank
            + self.phys_bias
        )
        learned_gate = self.gate_head(torch.cat([correction_input, generic_delta.detach(), delta_phys.detach()], dim=-1))
        region_gate = self.region_gate_head(region_evidence.detach())
        phys_weight = torch.sigmoid(prior_score + learned_gate + region_gate).clamp(0.0, 0.8)
        if self.correction_mode == "generic":
            final_delta = generic_delta
            generic_weight = torch.ones_like(phys_weight)
            phys_weight = torch.zeros_like(phys_weight)
        elif self.correction_mode == "physics":
            final_delta = delta_phys
            generic_weight = torch.zeros_like(phys_weight)
            phys_weight = torch.ones_like(phys_weight)
        elif self.correction_mode == "gated":
            generic_weight = 1.0 - phys_weight
            final_delta = generic_weight * generic_delta + phys_weight * delta_phys
        else:
            final_delta = delta_phys
            generic_weight = torch.zeros_like(phys_weight)
        mu = mu_data + final_delta
        x_generic = mu_data + generic_delta
        x_phys = mu_data + delta_phys
        return {
            "mu": mu,
            "mu_data": mu_data,
            "x_generic": x_generic,
            "x_phys": x_phys,
            "delta_phys": delta_phys,
            "generic_delta": generic_delta,
            "explicit_phys_delta": explicit_phys_delta,
            "physics_projection_strength": component_strength,
            "fd_projection_strength": fd_strength,
            "graph_projection_strength": graph_strength,
            "temporal_projection_strength": temporal_strength,
            "final_delta": final_delta,
            "phys_weight": phys_weight,
            "generic_weight": generic_weight,
            "region_gate": region_gate,
            "region_evidence": region_evidence,
            "prior_phys_weight": torch.sigmoid(prior_score).clamp(0.0, 0.8),
            "trust": phys_weight,
        }
