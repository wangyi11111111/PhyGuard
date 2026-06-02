from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .encoder_tcn import TemporalConvEncoder
from .grin_baseline import GraphGRUCell
from .graph_layer import GraphMixing
from physics.traffic_residuals import graph_flow_residual_full


def _batch_rank(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    order = torch.argsort(flat, dim=1)
    ranks = torch.zeros_like(flat)
    rank_values = torch.linspace(0.0, 1.0, steps=flat.shape[1], device=x.device, dtype=x.dtype)
    ranks.scatter_(1, order, rank_values.expand(flat.shape[0], -1))
    return ranks.reshape_as(x)


def _temporal_change_score(x_obs: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
    change = torch.zeros_like(x_obs[..., :1])
    valid_pairs = obs_mask[:, 1:] * obs_mask[:, :-1]
    diff = torch.abs(x_obs[:, 1:] - x_obs[:, :-1]) * valid_pairs
    denom = valid_pairs.sum(dim=-1, keepdim=True).clamp_min(1.0)
    change[:, 1:] = diff.sum(dim=-1, keepdim=True) / denom
    return _batch_rank(change.detach())


def _spatial_deviation_score(x_obs: torch.Tensor, obs_mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    neigh = torch.einsum("nm,btmc->btnc", adj, x_obs)
    diff = torch.abs(x_obs - neigh) * obs_mask
    denom = obs_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    deviation = diff.sum(dim=-1, keepdim=True) / denom
    return _batch_rank(deviation.detach())


def _trust_extra_features(x_obs: torch.Tensor, obs_mask: torch.Tensor, adj: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    temporal_feature = _temporal_change_score(x_obs.detach(), obs_mask.detach())
    spatial_feature = _spatial_deviation_score(x_obs.detach(), obs_mask.detach(), adj)
    missing_feature = 1.0 - obs_mask.detach().mean(dim=-1, keepdim=True)
    residual_rank = _batch_rank(residual.detach().abs())
    node_missing = 1.0 - obs_mask.detach().mean(dim=(1, 3))[:, None, :, None]
    node_missing = node_missing.expand(-1, x_obs.shape[1], -1, -1)
    neighbor_obs = torch.einsum("nm,btmc->btnc", adj, obs_mask.detach()).mean(dim=-1, keepdim=True)
    neighbor_missing = 1.0 - neighbor_obs
    return torch.cat(
        [temporal_feature, spatial_feature, missing_feature, residual_rank, node_missing, neighbor_missing],
        dim=-1,
    )


def _node_failure_signal(node_missing: torch.Tensor, local_missing: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    temperature = max(float(temperature), 1e-3)
    return torch.sigmoid((node_missing - local_missing) / temperature)


def _temporal_smooth_delta(x: torch.Tensor) -> torch.Tensor:
    delta = torch.zeros_like(x)
    if x.shape[1] > 1:
        delta[:, 1:-1] = 0.5 * (x[:, :-2].detach() + x[:, 2:].detach()) - x[:, 1:-1].detach()
        delta[:, 0] = x[:, 1].detach() - x[:, 0].detach()
        delta[:, -1] = x[:, -2].detach() - x[:, -1].detach()
    return delta


class PhysicsTrustGate(nn.Module):
    def __init__(self, hidden_dim: int, w_min: float = 0.0, extra_feature_dim: int = 1):
        super().__init__()
        self.w_min = float(w_min)
        self.extra_feature_dim = int(extra_feature_dim)
        gate_hidden = max(hidden_dim // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3 + self.extra_feature_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        residual_abs: torch.Tensor,
        log_var: torch.Tensor | None,
        obs_mask: torch.Tensor,
        extra_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if log_var is None:
            log_var = torch.zeros_like(residual_abs)
        if extra_feature is None:
            extra_feature = torch.zeros(*residual_abs.shape[:-1], self.extra_feature_dim, dtype=residual_abs.dtype, device=residual_abs.device)
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        mask_feature = obs_mask.mean(dim=-1, keepdim=True)
        gate_input = torch.cat(
            [
                h,
                residual_abs.detach(),
                log_var.detach(),
                mask_feature,
                extra_feature.detach(),
            ],
            dim=-1,
        )
        trust = torch.sigmoid(self.net(gate_input))
        if self.w_min > 0.0:
            trust = self.w_min + (1.0 - self.w_min) * trust
        return trust


class CorrectionExpertRouter(nn.Module):
    def __init__(self, hidden_dim: int, extra_feature_dim: int = 1):
        super().__init__()
        self.extra_feature_dim = int(extra_feature_dim)
        gate_hidden = max(hidden_dim // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3 + self.extra_feature_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, 0.0)
        with torch.no_grad():
            final.bias[0] = 1.0
            final.bias[1] = -0.2
            final.bias[2] = -0.2

    def forward(
        self,
        h: torch.Tensor,
        residual_abs: torch.Tensor,
        log_var: torch.Tensor | None,
        obs_mask: torch.Tensor,
        extra_feature: torch.Tensor,
        graph_available: bool = True,
    ) -> torch.Tensor:
        if log_var is None:
            log_var = torch.zeros_like(residual_abs)
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        mask_feature = obs_mask.mean(dim=-1, keepdim=True)
        route_input = torch.cat([h, residual_abs.detach(), log_var.detach(), mask_feature, extra_feature.detach()], dim=-1)
        logits = self.net(route_input)
        if extra_feature.shape[-1] >= 4:
            local_missing = extra_feature[..., 2:3]
            residual_rank = extra_feature[..., 3:4]
            logits[..., 2:3] = logits[..., 2:3] + 0.8 * local_missing + 0.5 * residual_rank
            logits[..., 0:1] = logits[..., 0:1] + 0.5 * mask_feature - 0.5 * local_missing
        if extra_feature.shape[-1] >= 5:
            node_missing = extra_feature[..., 4:5]
            failed_sensor = _node_failure_signal(node_missing, local_missing)
            logits[..., 0:1] = logits[..., 0:1] - 0.5 * failed_sensor
            logits[..., 1:2] = logits[..., 1:2] + 2.0 * failed_sensor
            logits[..., 2:3] = logits[..., 2:3] - 0.3 * failed_sensor
        if not graph_available:
            logits[..., 1] = -12.0
        return torch.softmax(logits, dim=-1)


class ExpertRiskRouter(nn.Module):
    def __init__(self, hidden_dim: int, extra_feature_dim: int = 1, temperature: float = 0.5):
        super().__init__()
        self.extra_feature_dim = int(extra_feature_dim)
        self.temperature = float(temperature)
        gate_hidden = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3 + self.extra_feature_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )

    def forward(
        self,
        h: torch.Tensor,
        residual_abs: torch.Tensor,
        log_var: torch.Tensor | None,
        obs_mask: torch.Tensor,
        extra_feature: torch.Tensor,
        use_phys_expert: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if log_var is None:
            log_var = torch.zeros_like(residual_abs)
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        mask_feature = obs_mask.mean(dim=-1, keepdim=True)
        risk_input = torch.cat([h, residual_abs.detach(), log_var.detach(), mask_feature, extra_feature.detach()], dim=-1)
        risks = torch.nn.functional.softplus(self.net(risk_input)) + 1e-4
        logits = -risks / max(self.temperature, 1e-3)
        if not use_phys_expert:
            logits[..., 2] = -12.0
            risks[..., 2] = risks[..., 2].detach() + 10.0
        weights = torch.softmax(logits, dim=-1)
        return risks, weights


class PhysicsFormRouter(nn.Module):
    """Select among explicit physics correction forms rather than a single residual."""

    def __init__(self, hidden_dim: int, extra_feature_dim: int = 6, temperature: float = 0.7):
        super().__init__()
        self.extra_feature_dim = int(extra_feature_dim)
        self.temperature = float(temperature)
        gate_hidden = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 6 + self.extra_feature_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 4),
        )
        self.bias = nn.Parameter(torch.tensor([0.2, 0.1, 0.0, 0.3], dtype=torch.float32))

    def forward(
        self,
        h: torch.Tensor,
        residual_abs: torch.Tensor,
        residual_rank: torch.Tensor,
        candidate_mags: torch.Tensor,
        extra_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        if candidate_mags.shape[-1] != 4:
            raise ValueError(f"candidate_mags must have 4 forms, got {candidate_mags.shape[-1]}.")
        obs = extra_feature[..., 0:1] if extra_feature.shape[-1] >= 1 else torch.zeros_like(residual_abs[..., :1])
        spatial = extra_feature[..., 1:2] if extra_feature.shape[-1] >= 2 else torch.zeros_like(obs)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        neighbor_missing = extra_feature[..., 5:6] if extra_feature.shape[-1] >= 6 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        random_missing = torch.clamp(local_missing - node_failure, min=0.0, max=1.0)
        route_input = torch.cat([h, residual_abs.detach(), residual_rank.detach(), candidate_mags.detach(), extra_feature.detach()], dim=-1)
        logits = self.net(route_input) + self.bias.view(1, 1, 1, 4)
        form_prior = torch.cat(
            [
                0.5 * node_failure + 0.2 * neighbor_missing + 0.2 * residual_rank - 0.1 * random_missing,
                0.4 * node_failure + 0.4 * spatial + 0.2 * neighbor_missing,
                0.3 * node_failure + 0.3 * spatial + 0.3 * residual_rank + 0.1 * neighbor_missing,
                0.6 * random_missing + 0.2 * spatial + 0.2 * residual_rank - 0.1 * node_failure,
            ],
            dim=-1,
        )
        logits = logits + form_prior
        weights = torch.softmax(logits / max(self.temperature, 1e-3), dim=-1)
        entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-6)), dim=-1, keepdim=True)
        confidence = 1.0 - entropy / torch.log(torch.tensor(4.0, dtype=weights.dtype, device=weights.device))
        return weights, logits, confidence


class LightweightReliabilityRouter(nn.Module):
    """Small interpretable router over data, graph, and physics experts."""

    def __init__(self, use_phys_expert: bool = True, temperature: float = 0.7):
        super().__init__()
        self.use_phys_expert = bool(use_phys_expert)
        self.temperature = float(temperature)
        self.bias = nn.Parameter(torch.tensor([0.4, -0.6, 0.1], dtype=torch.float32))
        coeff = torch.tensor(
            [
                [1.0, -0.5, -0.4, -0.5, -1.5, 0.0, -0.2, -0.3],
                [-0.2, -0.2, 0.0, 0.0, 3.0, 1.2, -0.2, 0.0],
                [-0.2, 1.3, -1.2, -0.8, -2.0, 0.1, 1.3, -0.4],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("coeff", coeff)

    def forward(
        self,
        residual_abs: torch.Tensor,
        log_var: torch.Tensor | None,
        obs_mask: torch.Tensor,
        extra_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if log_var is None:
            log_var = torch.zeros_like(residual_abs)
        obs = obs_mask.mean(dim=-1, keepdim=True)
        temporal = extra_feature[..., 0:1] if extra_feature.shape[-1] >= 1 else torch.zeros_like(obs)
        spatial = extra_feature[..., 1:2] if extra_feature.shape[-1] >= 2 else torch.zeros_like(obs)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        residual_rank = extra_feature[..., 3:4] if extra_feature.shape[-1] >= 4 else residual_abs.detach().clamp(0.0, 1.0)
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        neighbor_missing = extra_feature[..., 5:6] if extra_feature.shape[-1] >= 6 else local_missing
        neighbor_observed = 1.0 - neighbor_missing
        uncertainty = torch.sigmoid(log_var.detach())
        evidence = torch.cat(
            [
                obs,
                local_missing,
                residual_rank,
                temporal,
                node_failure,
                neighbor_observed,
                1.0 - residual_rank,
                uncertainty,
            ],
            dim=-1,
        )
        scores = torch.matmul(evidence, self.coeff.t()) + self.bias
        if not self.use_phys_expert:
            scores[..., 2] = -12.0
        weights = torch.softmax(scores / max(self.temperature, 1e-3), dim=-1)
        return scores, weights


class PhysicsProjectionController(nn.Module):
    """Controls how strongly the FD projection is applied."""

    def __init__(self, extra_feature_dim: int = 1):
        super().__init__()
        self.extra_feature_dim = int(extra_feature_dim)
        self.bias = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        coeff = torch.tensor([1.2, -1.4, -0.8, -1.5, 1.0, -0.5], dtype=torch.float32)
        self.register_buffer("coeff", coeff)

    def forward(self, residual_abs: torch.Tensor, obs_mask: torch.Tensor, extra_feature: torch.Tensor) -> torch.Tensor:
        obs = obs_mask.mean(dim=-1, keepdim=True)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        residual_rank = extra_feature[..., 3:4] if extra_feature.shape[-1] >= 4 else residual_abs.detach().clamp(0.0, 1.0)
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        temporal = extra_feature[..., 0:1] if extra_feature.shape[-1] >= 1 else torch.zeros_like(obs)
        low_residual = 1.0 - residual_rank
        random_missing = torch.clamp(local_missing - node_failure, min=0.0, max=1.0)
        evidence = torch.cat([random_missing, residual_rank, temporal, node_failure, low_residual, obs], dim=-1)
        score = torch.sum(evidence * self.coeff.view(1, 1, 1, -1), dim=-1, keepdim=True) + self.bias
        return torch.sigmoid(score)


class PhysicsValidityGate(nn.Module):
    """Estimates whether the physics expert is likely useful for reconstruction."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        coeff = torch.tensor([1.4, -1.0, -0.8, -2.0, 0.8, -0.4], dtype=torch.float32)
        self.register_buffer("coeff", coeff)

    def forward(self, residual_abs: torch.Tensor, obs_mask: torch.Tensor, extra_feature: torch.Tensor) -> torch.Tensor:
        obs = obs_mask.mean(dim=-1, keepdim=True)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        residual_rank = extra_feature[..., 3:4] if extra_feature.shape[-1] >= 4 else residual_abs.detach().clamp(0.0, 1.0)
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        temporal = extra_feature[..., 0:1] if extra_feature.shape[-1] >= 1 else torch.zeros_like(obs)
        low_residual = 1.0 - residual_rank
        random_missing = torch.clamp(local_missing - node_failure, min=0.0, max=1.0)
        evidence = torch.cat([random_missing, residual_rank, temporal, node_failure, low_residual, obs], dim=-1)
        score = torch.sum(evidence * self.coeff.view(1, 1, 1, -1), dim=-1, keepdim=True) + self.bias
        return torch.sigmoid(score)


class SpatialConservationPhysicsExpert(nn.Module):
    """Lightweight spatial physics correction driven by neighbor FD residuals."""

    def __init__(self, hidden_dim: int, output_dim: int, extra_feature_dim: int = 1, correction_clip: float = 1.0):
        super().__init__()
        self.output_dim = int(output_dim)
        self.extra_feature_dim = int(extra_feature_dim)
        self.correction_clip = float(correction_clip)
        expert_hidden = max(hidden_dim // 2, 16)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim + output_dim + 2 + extra_feature_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, output_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(5, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self.fd_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(
        self,
        h: torch.Tensor,
        mu_data: torch.Tensor,
        x_obs: torch.Tensor,
        obs_mask: torch.Tensor,
        adj: torch.Tensor,
        residual_signed: torch.Tensor,
        extra_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs_mask.mean(dim=-1, keepdim=True)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        neighbor_missing = extra_feature[..., 5:6] if extra_feature.shape[-1] >= 6 else local_missing
        neighbor_observed = 1.0 - neighbor_missing
        residual = residual_signed.detach()
        neigh_residual = torch.einsum("nm,btmc->btnc", adj, residual)
        context = obs_mask * x_obs + (1.0 - obs_mask) * mu_data.detach()
        neigh_context = torch.einsum("nm,btmc->btnc", adj, context)
        context_gap = neigh_context - mu_data.detach()
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        delta_input = torch.cat([h, context_gap, residual, neigh_residual, extra_feature.detach()], dim=-1)
        learned_delta = torch.tanh(self.delta_head(delta_input)) * self.correction_clip
        fd_delta = torch.zeros_like(mu_data)
        fd_delta[..., 0:1] = -torch.clamp(neigh_residual, min=-1.0, max=1.0)
        gate_input = torch.cat(
            [
                node_failure,
                neighbor_observed,
                local_missing,
                residual.abs().clamp(0.0, 1.0),
                neigh_residual.abs().clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_head(gate_input)) * node_failure * neighbor_observed
        spatial_delta = gate * (learned_delta + self.fd_gain * fd_delta)
        return spatial_delta, gate


class DirectionalConservationPhysicsExpert(nn.Module):
    """Direction-aware conservation correction for node-level sensor failure."""

    def __init__(self, hidden_dim: int, output_dim: int, extra_feature_dim: int = 1, correction_clip: float = 1.0):
        super().__init__()
        self.output_dim = int(output_dim)
        self.extra_feature_dim = int(extra_feature_dim)
        self.correction_clip = float(correction_clip)
        expert_hidden = max(hidden_dim // 2, 16)
        # Inputs: hidden state, directional physics features, local prediction, and trust features.
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim + 7 + output_dim + extra_feature_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, output_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(7, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self.balance_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @staticmethod
    def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
        return matrix / torch.clamp(matrix.sum(dim=-1, keepdim=True), min=1e-6)

    def _directional_matrices(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = adj.shape[0]
        offdiag = adj * (1.0 - torch.eye(nodes, dtype=adj.dtype, device=adj.device))
        asym = offdiag - offdiag.transpose(0, 1)
        upstream = torch.relu(asym)
        downstream = torch.relu(-asym)
        if bool((upstream.sum() + downstream.sum()).detach().cpu() <= 1e-6):
            idx = torch.arange(nodes, device=adj.device)
            weak_up = (idx.view(1, -1) < idx.view(-1, 1)).to(adj.dtype)
            weak_down = (idx.view(1, -1) > idx.view(-1, 1)).to(adj.dtype)
            upstream = offdiag * weak_up
            downstream = offdiag * weak_down
        return self._normalize_rows(upstream), self._normalize_rows(downstream)

    def forward(
        self,
        h: torch.Tensor,
        mu_data: torch.Tensor,
        x_obs: torch.Tensor,
        obs_mask: torch.Tensor,
        adj: torch.Tensor,
        residual_signed: torch.Tensor,
        extra_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs_mask.mean(dim=-1, keepdim=True)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        neighbor_missing = extra_feature[..., 5:6] if extra_feature.shape[-1] >= 6 else local_missing
        neighbor_observed = 1.0 - neighbor_missing
        upstream, downstream = self._directional_matrices(adj)
        context = obs_mask * x_obs + (1.0 - obs_mask) * mu_data.detach()
        q = context[..., 0:1]
        rho = context[..., 1:2]
        speed = context[..., 2:3]
        pred_q = mu_data[..., 0:1].detach()
        pred_rho = mu_data[..., 1:2].detach()
        pred_speed = mu_data[..., 2:3].detach()
        q_in = torch.einsum("nm,btmc->btnc", upstream, q)
        q_out = torch.einsum("nm,btmc->btnc", downstream, q)
        rho_up = torch.einsum("nm,btmc->btnc", upstream, rho)
        speed_up = torch.einsum("nm,btmc->btnc", upstream, speed)
        speed_down = torch.einsum("nm,btmc->btnc", downstream, speed)
        flow_balance_target = 0.5 * (q_in + q_out)
        balance_residual = pred_q - flow_balance_target
        density_balance = q_in - q_out
        density_residual = pred_rho - (rho_up + 0.1 * density_balance)
        speed_residual = pred_speed - 0.5 * (speed_up + speed_down)
        fd_residual = residual_signed.detach()
        direction_features = torch.cat(
            [
                balance_residual,
                density_residual,
                speed_residual,
                q_in - pred_q,
                q_out - pred_q,
                density_balance,
                fd_residual,
            ],
            dim=-1,
        )
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        delta_input = torch.cat([h, direction_features.detach(), mu_data.detach(), extra_feature.detach()], dim=-1)
        learned_delta = torch.tanh(self.delta_head(delta_input)) * self.correction_clip
        conservation_delta = torch.zeros_like(mu_data)
        conservation_delta[..., 0:1] = -torch.clamp(balance_residual, min=-1.0, max=1.0)
        conservation_delta[..., 1:2] = -0.25 * torch.clamp(density_residual, min=-1.0, max=1.0)
        conservation_delta[..., 2:3] = -0.25 * torch.clamp(speed_residual, min=-1.0, max=1.0)
        gate_input = torch.cat(
            [
                node_failure,
                neighbor_observed,
                balance_residual.abs().clamp(0.0, 1.0),
                density_residual.abs().clamp(0.0, 1.0),
                speed_residual.abs().clamp(0.0, 1.0),
                fd_residual.abs().clamp(0.0, 1.0),
                local_missing,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_head(gate_input)) * node_failure * neighbor_observed
        delta = gate * (learned_delta + self.balance_gain * conservation_delta)
        conservation_residual = torch.mean(torch.abs(direction_features), dim=-1, keepdim=True)
        return delta, gate, conservation_residual


class LiteTrustPINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        w_min: float = 0.0,
        use_uncertainty: bool = False,
        extra_feature_dim: int = 1,
    ):
        super().__init__()
        self.use_uncertainty = bool(use_uncertainty)
        self.encoder = TemporalConvEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.graph = GraphMixing(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        self.trust_gate = PhysicsTrustGate(hidden_dim, w_min=w_min, extra_feature_dim=extra_feature_dim)

    def encode(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_obs, mask], dim=-1)
        h = self.encoder(z)
        return self.graph(h, adj)

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

    def predict_log_var(self, h: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.logvar_head(h), min=-6.0, max=3.0)

    def trust_from_residual(
        self,
        h: torch.Tensor,
        residual_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.trust_gate(h, residual_abs, log_var, obs_mask, extra_feature=extra_feature)

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        residual_abs: torch.Tensor | None = None,
        residual_signed: torch.Tensor | None = None,
        physics_delta: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> dict:
        h = self.encode(x_obs, mask, adj)
        mu = self.reconstruct(h)
        output = {"mu": mu, "h": h}
        if self.use_uncertainty:
            output["log_var"] = self.predict_log_var(h)
        if residual_abs is not None:
            output["trust"] = self.trust_from_residual(h, residual_abs, mask, log_var=log_var, extra_feature=extra_feature)
        return output


class LiteTrustGRIN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        w_min: float = 0.0,
        use_uncertainty: bool = True,
        extra_feature_dim: int = 4,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.use_uncertainty = bool(use_uncertainty)
        step_input_dim = int(input_dim) * 2
        self.fwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.bwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.fwd_head = nn.Linear(hidden_dim, output_dim)
        self.bwd_head = nn.Linear(hidden_dim, output_dim)
        self.head = nn.Linear(output_dim * 2, output_dim)
        self.hidden_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        self.trust_gate = PhysicsTrustGate(hidden_dim, w_min=w_min, extra_feature_dim=extra_feature_dim)
        self.dropout = nn.Dropout(dropout)

    def _direction(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor, reverse: bool):
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
            h = self.dropout(cell(step_input, h, adj))
            previous = head(h)
            preds.append(previous)
            hidden_states.append(h)
        pred = torch.stack(preds, dim=1)
        hidden = torch.stack(hidden_states, dim=1)
        if reverse:
            pred = torch.flip(pred, dims=[1])
            hidden = torch.flip(hidden, dims=[1])
        return pred, hidden

    def predict_log_var(self, h: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.logvar_head(h), min=-6.0, max=3.0)

    def trust_from_residual(
        self,
        h: torch.Tensor,
        residual_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.trust_gate(h, residual_abs, log_var, obs_mask, extra_feature=extra_feature)

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        residual_abs: torch.Tensor | None = None,
        residual_signed: torch.Tensor | None = None,
        physics_delta: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> dict:
        fwd_pred, fwd_h = self._direction(x_obs, mask, adj, reverse=False)
        bwd_pred, bwd_h = self._direction(x_obs, mask, adj, reverse=True)
        h = torch.tanh(self.hidden_fuse(torch.cat([fwd_h, bwd_h], dim=-1)))
        mu = self.head(torch.cat([fwd_pred, bwd_pred], dim=-1))
        output = {"mu": mu, "h": h}
        if self.use_uncertainty:
            output["log_var"] = self.predict_log_var(h)
        if residual_abs is not None:
            output["trust"] = self.trust_from_residual(h, residual_abs, mask, log_var=log_var, extra_feature=extra_feature)
        return output


class LiteTrustGRINCorrection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        w_min: float = 0.0,
        use_uncertainty: bool = True,
        extra_feature_dim: int = 4,
        use_graph_delta: bool = True,
        use_phys_expert: bool = True,
        failure_routing: str = "soft_prior",
        failed_graph_prior: float = 0.85,
        failed_phys_prior: float = 0.05,
        correction_clip: float = 1.0,
    ):
        super().__init__()
        if failure_routing not in {"learned", "soft_prior", "hard"}:
            raise ValueError("failure_routing must be one of: learned, soft_prior, hard")
        self.output_dim = int(output_dim)
        self.use_uncertainty = bool(use_uncertainty)
        self.use_graph_delta = bool(use_graph_delta)
        self.use_phys_expert = bool(use_phys_expert)
        self.failure_routing = failure_routing
        self.failed_graph_prior = float(failed_graph_prior)
        self.failed_phys_prior = float(failed_phys_prior)
        self.correction_clip = float(correction_clip)
        step_input_dim = int(input_dim) * 2
        self.fwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.bwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.fwd_head = nn.Linear(hidden_dim, output_dim)
        self.bwd_head = nn.Linear(hidden_dim, output_dim)
        self.data_head = nn.Linear(output_dim * 2, output_dim)
        self.hidden_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        correction_hidden = max(hidden_dim // 2, 16)
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 + extra_feature_dim, correction_hidden),
            nn.GELU(),
            nn.Linear(correction_hidden, output_dim),
        )
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        self.trust_gate = PhysicsTrustGate(hidden_dim, w_min=w_min, extra_feature_dim=extra_feature_dim)
        self.expert_router = CorrectionExpertRouter(hidden_dim, extra_feature_dim=extra_feature_dim)
        self.fd_projection_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def _direction(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor, reverse: bool):
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
            h = self.dropout(cell(step_input, h, adj))
            previous = head(h)
            preds.append(previous)
            hidden_states.append(h)
        pred = torch.stack(preds, dim=1)
        hidden = torch.stack(hidden_states, dim=1)
        if reverse:
            pred = torch.flip(pred, dims=[1])
            hidden = torch.flip(hidden, dims=[1])
        return pred, hidden

    def predict_log_var(self, h: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.logvar_head(h), min=-6.0, max=3.0)

    def trust_from_residual(
        self,
        h: torch.Tensor,
        residual_abs: torch.Tensor,
        obs_mask: torch.Tensor,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.trust_gate(h, residual_abs, log_var, obs_mask, extra_feature=extra_feature)

    def correction_from_features(
        self,
        h: torch.Tensor,
        residual_signal: torch.Tensor,
        obs_mask: torch.Tensor,
        log_var: torch.Tensor | None,
        extra_feature: torch.Tensor,
    ) -> torch.Tensor:
        if log_var is None:
            log_var = torch.zeros_like(residual_signal)
        gate_mask = obs_mask.mean(dim=-1, keepdim=True)
        signed_residual = residual_signal.detach()
        features = torch.cat([h, signed_residual, log_var.detach(), gate_mask, extra_feature.detach()], dim=-1)
        delta = self.correction_head(features)
        fd_delta = torch.zeros_like(delta)
        fd_delta[..., 0:1] = -torch.clamp(self.fd_projection_gain, min=0.0, max=1.0) * signed_residual
        delta = delta + fd_delta
        if self.correction_clip > 0.0:
            delta = self.correction_clip * torch.tanh(delta / self.correction_clip)
        return delta

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        residual_abs: torch.Tensor | None = None,
        residual_signed: torch.Tensor | None = None,
        physics_delta: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> dict:
        fwd_pred, fwd_h = self._direction(x_obs, mask, adj, reverse=False)
        bwd_pred, bwd_h = self._direction(x_obs, mask, adj, reverse=True)
        h = torch.tanh(self.hidden_fuse(torch.cat([fwd_h, bwd_h], dim=-1)))
        mu_data = self.data_head(torch.cat([fwd_pred, bwd_pred], dim=-1))
        output = {"mu": mu_data, "mu_data": mu_data, "h": h}
        if self.use_uncertainty:
            output["log_var"] = self.predict_log_var(h)
        if residual_abs is not None and extra_feature is not None:
            gate_log_var = log_var
            residual_for_trust = residual_abs.detach().abs()
            residual_for_correction = residual_signed if residual_signed is not None else residual_abs
            local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - mask.detach().mean(dim=-1, keepdim=True)
            trust = self.trust_from_residual(h, residual_for_trust, mask, log_var=gate_log_var, extra_feature=extra_feature)
            learned_delta_phys = self.correction_from_features(h, residual_for_correction, mask, gate_log_var, extra_feature)
            graph_delta = torch.zeros_like(learned_delta_phys)
            failed_sensor = torch.zeros_like(residual_for_trust)
            if self.use_graph_delta and extra_feature.shape[-1] >= 5:
                node_missing = extra_feature[..., 4:5]
                failed_sensor = _node_failure_signal(node_missing, local_missing)
                graph_context = mask * x_obs + (1.0 - mask) * mu_data
                neigh_context = torch.einsum("nm,btmc->btnc", adj, graph_context)
                graph_delta = failed_sensor * (neigh_context - mu_data)
            delta_phys = learned_delta_phys
            expert_weights = self.expert_router(
                h,
                residual_for_trust,
                gate_log_var,
                mask,
                extra_feature,
                graph_available=self.use_graph_delta,
            )
            data_weight = expert_weights[..., 0:1]
            graph_weight = expert_weights[..., 1:2]
            phys_weight = expert_weights[..., 2:3]
            if not self.use_phys_expert:
                phys_weight = torch.zeros_like(phys_weight)
                denom = torch.clamp(data_weight + graph_weight, min=1e-6)
                data_weight = data_weight / denom
                graph_weight = graph_weight / denom
            if self.use_graph_delta and self.failure_routing == "hard":
                data_weight = (1.0 - failed_sensor) * data_weight
                graph_weight = failed_sensor + (1.0 - failed_sensor) * graph_weight
                phys_weight = (1.0 - failed_sensor) * phys_weight
            elif self.use_graph_delta and self.failure_routing == "soft_prior":
                failed_phys_prior = self.failed_phys_prior if self.use_phys_expert else 0.0
                failed_graph_prior = self.failed_graph_prior
                failed_data_prior = max(0.0, 1.0 - failed_graph_prior - failed_phys_prior)
                data_weight = (1.0 - failed_sensor) * data_weight + failed_sensor * failed_data_prior
                graph_weight = (1.0 - failed_sensor) * graph_weight + failed_sensor * failed_graph_prior
                phys_weight = (1.0 - failed_sensor) * phys_weight + failed_sensor * failed_phys_prior
                denom = torch.clamp(data_weight + graph_weight + phys_weight, min=1e-6)
                data_weight = data_weight / denom
                graph_weight = graph_weight / denom
                phys_weight = phys_weight / denom
            output["raw_physics_gate"] = trust
            output["delta_phys"] = delta_phys
            output["graph_delta"] = graph_delta
            output["expert_weights"] = expert_weights
            output["data_weight"] = data_weight
            output["graph_weight"] = graph_weight
            output["phys_weight"] = phys_weight
            output["correction_trust"] = graph_weight + phys_weight
            output["trust"] = phys_weight
            output["mu"] = mu_data + graph_weight * graph_delta + phys_weight * delta_phys
        return output


class LiteTrustGRINRiskRouter(LiteTrustGRINCorrection):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        w_min: float = 0.0,
        use_uncertainty: bool = True,
        extra_feature_dim: int = 4,
        use_phys_expert: bool = True,
        correction_clip: float = 1.0,
        risk_temperature: float = 0.5,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            w_min=w_min,
            use_uncertainty=use_uncertainty,
            extra_feature_dim=extra_feature_dim,
            use_graph_delta=True,
            use_phys_expert=use_phys_expert,
            failure_routing="learned",
            correction_clip=correction_clip,
        )
        self.risk_router = ExpertRiskRouter(hidden_dim, extra_feature_dim=extra_feature_dim, temperature=risk_temperature)

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        residual_abs: torch.Tensor | None = None,
        residual_signed: torch.Tensor | None = None,
        physics_delta: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> dict:
        fwd_pred, fwd_h = self._direction(x_obs, mask, adj, reverse=False)
        bwd_pred, bwd_h = self._direction(x_obs, mask, adj, reverse=True)
        h = torch.tanh(self.hidden_fuse(torch.cat([fwd_h, bwd_h], dim=-1)))
        mu_data = self.data_head(torch.cat([fwd_pred, bwd_pred], dim=-1))
        output = {"mu": mu_data, "mu_data": mu_data, "h": h}
        if self.use_uncertainty:
            output["log_var"] = self.predict_log_var(h)
        if residual_abs is not None and extra_feature is not None:
            gate_log_var = log_var
            residual_for_trust = residual_abs.detach().abs()
            residual_for_correction = residual_signed if residual_signed is not None else residual_abs
            learned_delta_phys = self.correction_from_features(h, residual_for_correction, mask, gate_log_var, extra_feature)
            delta_phys = physics_delta if physics_delta is not None else learned_delta_phys
            if extra_feature.shape[-1] >= 5:
                node_missing = extra_feature[..., 4:5]
                local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - mask.detach().mean(dim=-1, keepdim=True)
                graph_applicability = _node_failure_signal(node_missing, local_missing)
            else:
                graph_applicability = 1.0 - mask.detach().mean(dim=-1, keepdim=True)
            graph_context = mask * x_obs + (1.0 - mask) * mu_data
            neigh_context = torch.einsum("nm,btmc->btnc", adj, graph_context)
            graph_delta = graph_applicability * (neigh_context - mu_data)
            risk_pred, expert_weights = self.risk_router(
                h,
                residual_for_trust,
                gate_log_var,
                mask,
                extra_feature,
                use_phys_expert=self.use_phys_expert,
            )
            data_weight = expert_weights[..., 0:1]
            graph_weight = expert_weights[..., 1:2]
            phys_weight = expert_weights[..., 2:3]
            x_graph = mu_data + graph_delta
            x_phys = mu_data + phys_weight.new_tensor(1.0) * delta_phys
            output["delta_phys"] = delta_phys
            output["learned_delta_phys"] = learned_delta_phys
            output["graph_delta"] = graph_delta
            output["x_graph"] = x_graph
            output["x_phys"] = x_phys
            output["risk_pred"] = risk_pred
            output["expert_weights"] = expert_weights
            output["data_weight"] = data_weight
            output["graph_weight"] = graph_weight
            output["phys_weight"] = phys_weight
            output["correction_trust"] = graph_weight + phys_weight
            output["trust"] = phys_weight
            output["mu"] = data_weight * mu_data + graph_weight * x_graph + phys_weight * x_phys
        return output


class BidirectionalGraphExpert(nn.Module):
    """Compact GRIN-style expert for random/incident recovery."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = int(output_dim)
        step_input_dim = int(input_dim) * 2
        self.fwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.bwd_cell = GraphGRUCell(step_input_dim, hidden_dim)
        self.fwd_head = nn.Linear(hidden_dim, output_dim)
        self.bwd_head = nn.Linear(hidden_dim, output_dim)
        self.hidden_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(output_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _direction(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor, reverse: bool):
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
            h = self.dropout(cell(step_input, h, adj))
            previous = head(h)
            preds.append(previous)
            hidden_states.append(h)
        pred = torch.stack(preds, dim=1)
        hidden = torch.stack(hidden_states, dim=1)
        if reverse:
            pred = torch.flip(pred, dims=[1])
            hidden = torch.flip(hidden, dims=[1])
        return pred, hidden

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fwd_pred, fwd_h = self._direction(x_obs, mask, adj, reverse=False)
        bwd_pred, bwd_h = self._direction(x_obs, mask, adj, reverse=True)
        h = torch.tanh(self.hidden_fuse(torch.cat([fwd_h, bwd_h], dim=-1)))
        mu = self.out(torch.cat([fwd_pred, bwd_pred], dim=-1))
        return mu, h


class TemporalSAITSLite(nn.Module):
    """Tiny SAITS-style temporal expert for sensor-failure recovery."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_nodes: int | None = None,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.max_nodes = max_nodes
        token_dim = int(input_dim) * 2 + 1
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.node_embedding = nn.Embedding(max_nodes, hidden_dim) if max_nodes is not None else None
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=max(hidden_dim * 2, 32),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes, channels = x_obs.shape
        time = torch.linspace(0.0, 1.0, steps, device=x_obs.device, dtype=x_obs.dtype).view(1, steps, 1, 1)
        time = time.expand(batch, steps, nodes, 1)
        tokens = torch.cat([x_obs, mask, time], dim=-1).permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels * 2 + 1)
        h = self.input_proj(tokens) + self.time_proj(time.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 1))
        if self.node_embedding is not None:
            if nodes > self.max_nodes:
                raise ValueError(f"TemporalSAITSLite max_nodes={self.max_nodes} is smaller than input nodes={nodes}.")
            node_ids = torch.arange(nodes, device=x_obs.device)
            node_h = self.node_embedding(node_ids).view(1, nodes, 1, -1).expand(batch, nodes, steps, -1)
            h = h + node_h.reshape(batch * nodes, steps, -1)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        pred = self.head(h).reshape(batch, nodes, steps, self.output_dim).permute(0, 2, 1, 3)
        hidden = h.reshape(batch, nodes, steps, -1).permute(0, 2, 1, 3)
        return pred, hidden


class TemporalAnchorPhysicsGuarded(nn.Module):
    """Temporal self-attention anchor with physics/graph utility correction.

    The temporal branch is the primary reconstruction path. Physics is not a
    competing answer generator here; it provides graph and residual features
    that decide whether a local correction should be applied to the anchor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        correction_clip: float = 0.75,
        extra_feature_dim: int = 6,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.correction_clip = float(correction_clip)
        self.extra_feature_dim = int(extra_feature_dim)
        self.anchor = TemporalSAITSLite(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_nodes=512,
        )
        self.spatial_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=max(hidden_dim * 2, 32),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(1)
            ]
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_head = nn.Linear(hidden_dim, output_dim)
        self.anchor_fuse_head = nn.Sequential(
            nn.Linear(hidden_dim + output_dim * 2 + 3, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )
        corr_hidden = max(hidden_dim // 2, 16)
        corr_in = hidden_dim + output_dim * 4 + extra_feature_dim
        self.delta_head = nn.Sequential(
            nn.Linear(corr_in, corr_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(corr_hidden, output_dim),
        )
        gate_in = hidden_dim + output_dim * 3 + extra_feature_dim
        self.gate_head = nn.Sequential(
            nn.Linear(gate_in, corr_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(corr_hidden, 1),
        )

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> dict:
        neighbor_obs = torch.einsum("nm,btmc->btnc", adj, x_obs)
        neighbor_mask = torch.einsum("nm,btmc->btnc", adj, mask).clamp(0.0, 1.0)
        anchor_input = mask * x_obs + (1.0 - mask) * neighbor_obs
        anchor_mask = torch.clamp(mask + (1.0 - mask) * neighbor_mask, 0.0, 1.0)
        x_anchor, h = self.anchor(anchor_input, anchor_mask)
        batch, steps, nodes, hidden = h.shape
        spatial_h = h.reshape(batch * steps, nodes, hidden)
        for layer in self.spatial_layers:
            spatial_h = layer(spatial_h)
        spatial_h = self.spatial_norm(spatial_h).reshape(batch, steps, nodes, hidden)
        x_spatial = self.spatial_head(spatial_h)
        local_missing = 1.0 - mask.mean(dim=-1, keepdim=True)
        node_missing = 1.0 - mask.mean(dim=(1, 3))[:, None, :, None]
        node_missing = node_missing.expand(-1, steps, -1, -1)
        neighbor_missing = 1.0 - neighbor_mask.mean(dim=-1, keepdim=True)
        anchor_fuse_input = torch.cat(
            [
                spatial_h,
                torch.abs(x_spatial - x_anchor),
                x_anchor,
                local_missing,
                node_missing,
                neighbor_missing,
            ],
            dim=-1,
        )
        anchor_fuse = torch.sigmoid(self.anchor_fuse_head(anchor_fuse_input))
        x_anchor = (1.0 - anchor_fuse) * x_anchor + anchor_fuse * x_spatial
        h = 0.5 * (h + spatial_h)
        context = mask * x_obs + (1.0 - mask) * x_anchor
        neigh = torch.einsum("nm,btmc->btnc", adj, context)
        graph_delta = neigh - x_anchor
        temporal_delta = _temporal_smooth_delta(x_anchor)
        if self.output_dim == 1:
            residual = graph_flow_residual_full(x_anchor[..., 0:1], adj)
        else:
            residual = torch.zeros_like(x_anchor[..., :1])
        extra = _trust_extra_features(x_obs, mask, adj, residual)
        delta_input = torch.cat(
            [
                h,
                x_anchor,
                graph_delta,
                temporal_delta,
                residual.expand_as(x_anchor),
                extra,
            ],
            dim=-1,
        )
        raw_delta = torch.tanh(self.delta_head(delta_input)) * self.correction_clip
        gate_input = torch.cat(
            [
                h,
                graph_delta.detach().abs(),
                temporal_delta.detach().abs(),
                residual.detach().abs().expand_as(x_anchor),
                extra.detach(),
            ],
            dim=-1,
        )
        correction_gate = torch.sigmoid(self.gate_head(gate_input))
        x_corrected = x_anchor + correction_gate * raw_delta
        mu = mask * x_obs + (1.0 - mask) * x_corrected
        return {
            "mu": mu,
            "x_anchor": x_anchor,
            "x_spatial_anchor": x_spatial,
            "anchor_fuse": anchor_fuse,
            "anchor_input": anchor_input,
            "anchor_mask": anchor_mask,
            "x_corrected": x_corrected,
            "x_graph": neigh,
            "graph_delta": graph_delta,
            "temporal_delta": temporal_delta,
            "delta": raw_delta,
            "correction_gate": correction_gate,
            "residual_anchor": residual,
            "extra_feature": extra,
            "h": h,
        }


class MaskAwareGraphRepair(nn.Module):
    """Internal MagiNet-inspired graph repair without using MagiNet outputs.

    It extracts the useful idea from MagiNet: missing-aware tokenization plus a
    learned repair graph. The graph is used as an internal propagation operator,
    not as an external model candidate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_nodes: int = 512,
        dropout: float = 0.1,
        top_k: int = 16,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.max_nodes = int(max_nodes)
        self.top_k = int(top_k)
        self.input_proj = nn.Linear(input_dim * 2 + 1, hidden_dim)
        self.node_embedding = nn.Embedding(max_nodes, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        nn.init.uniform_(self.missing_token, -0.02, 0.02)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.graph_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gtu3 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 3), padding=(0, 1))
        self.gtu5 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 5), padding=(0, 2))
        self.gtu7 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 7), padding=(0, 3))
        self.temporal_mix = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> dict:
        batch, steps, nodes, channels = x_obs.shape
        if nodes > self.max_nodes:
            raise ValueError(f"MaskAwareGraphRepair max_nodes={self.max_nodes} is smaller than input nodes={nodes}.")
        time = torch.linspace(0.0, 1.0, steps, device=x_obs.device, dtype=x_obs.dtype).view(1, steps, 1, 1)
        time = time.expand(batch, steps, nodes, 1)
        token_input = torch.cat([x_obs, mask, time], dim=-1)
        h = self.input_proj(token_input)
        h = mask.mean(dim=-1, keepdim=True) * h + (1.0 - mask.mean(dim=-1, keepdim=True)) * self.missing_token
        node_ids = torch.arange(nodes, device=x_obs.device)
        h = h + self.node_embedding(node_ids).view(1, 1, nodes, -1)
        h = h + self.time_proj(time)
        source_reliability = mask.mean(dim=-1)
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        logits = torch.einsum("btih,btjh->btij", q, k) / max(q.shape[-1] ** 0.5, 1.0)
        graph_prior = adj.view(1, 1, nodes, nodes)
        reliability_prior = source_reliability[:, :, None, :]
        graph_logits = torch.relu(logits) + 0.5 * graph_prior + reliability_prior
        k = min(max(self.top_k, 1), nodes)
        top_values, top_indices = torch.topk(graph_logits, k=k, dim=-1)
        sparse_logits = torch.full_like(graph_logits, -1e9)
        sparse_logits.scatter_(-1, top_indices, top_values)
        learned_graph = torch.softmax(sparse_logits, dim=-1)
        dyn_context = torch.einsum("btij,btjh->btih", learned_graph, v)
        static_context = torch.einsum("nm,btmh->btnh", adj, h)
        neighbor_obs = torch.einsum("nm,btmc->btnc", adj, mask).mean(dim=-1, keepdim=True)
        local_missing = 1.0 - mask.mean(dim=-1, keepdim=True)
        node_missing = 1.0 - mask.mean(dim=(1, 3))[:, None, :, None]
        node_missing = node_missing.expand(-1, steps, -1, -1)
        graph_conf = learned_graph.max(dim=-1, keepdim=True).values
        fuse_input = torch.cat([h, dyn_context, static_context, local_missing, node_missing, 1.0 - neighbor_obs, graph_conf], dim=-1)
        h_graph = self.graph_fuse(fuse_input)
        x_time = h_graph.permute(0, 3, 2, 1)
        def _gtu(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
            z = conv(x)
            a, b = torch.chunk(z, 2, dim=1)
            return torch.tanh(a) * torch.sigmoid(b)

        h_time = self.temporal_mix(torch.cat([_gtu(self.gtu3, x_time), _gtu(self.gtu5, x_time), _gtu(self.gtu7, x_time)], dim=1))
        h_graph = h_time.permute(0, 3, 2, 1)
        h_graph = self.norm(h_graph + h)
        pred = self.head(h_graph)
        mu = mask * x_obs + (1.0 - mask) * pred
        return {
            "mu": mu,
            "x_graph_repair": pred,
            "learned_graph": learned_graph,
            "graph_confidence": graph_conf,
            "h": h_graph,
        }


def _cheb_like_supports(adj: torch.Tensor, k_order: int) -> torch.Tensor:
    nodes = adj.shape[0]
    eye = torch.eye(nodes, device=adj.device, dtype=adj.dtype)
    supports = [eye]
    if k_order > 1:
        supports.append(adj)
    for _ in range(2, k_order):
        supports.append(torch.clamp(2.0 * adj @ supports[-1] - supports[-2], min=-1.0, max=1.0))
    return torch.stack(supports[:k_order], dim=0)


class MagiStyleRepairBlock(nn.Module):
    """Near-MagiNet block: residual temporal attention, dynamic graph, graph conv, and GTU."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, k_order: int = 3, dropout: float = 0.1, top_k: int = 16):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.k_order = int(k_order)
        self.top_k = int(top_k)
        self.t_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.t_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.t_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.t_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.s_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.s_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for _ in range(k_order)])
        self.mask = nn.ParameterList([nn.Parameter(torch.empty(1, 1, 1)) for _ in range(k_order)])
        for param in self.theta:
            nn.init.xavier_uniform_(param)
        for param in self.mask:
            nn.init.uniform_(param, -0.02, 0.02)
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.gtu3 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 3), padding=(0, 1))
        self.gtu5 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 5), padding=(0, 2))
        self.gtu7 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 7), padding=(0, 3))
        self.time_mix = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _temporal_attention(self, h: torch.Tensor, obs_mask: torch.Tensor, res_att: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, nodes, hidden = h.shape
        q = self.t_q(h).permute(0, 2, 1, 3).reshape(batch, nodes, steps, self.num_heads, self.head_dim).transpose(2, 3)
        k = self.t_k(h).permute(0, 2, 1, 3).reshape(batch, nodes, steps, self.num_heads, self.head_dim).transpose(2, 3)
        v = self.t_v(h).permute(0, 2, 1, 3).reshape(batch, nodes, steps, self.num_heads, self.head_dim).transpose(2, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / max(self.head_dim ** 0.5, 1.0)
        if res_att is not None:
            scores = scores + res_att
        key_conf = obs_mask.mean(dim=-1).permute(0, 2, 1)[:, :, None, None, :]
        scores = scores + 0.5 * key_conf
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(2, 3).reshape(batch, nodes, steps, hidden).permute(0, 2, 1, 3)
        return self.t_norm(h + self.t_out(context)), scores.detach()

    def _dynamic_graph(self, h_temporal: torch.Tensor, obs_mask: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _steps, nodes, _hidden = h_temporal.shape
        node_h = h_temporal.mean(dim=1)
        sim = torch.relu(torch.matmul(node_h, node_h.transpose(-1, -2)) / max(self.hidden_dim ** 0.5, 1.0))
        source_reliability = obs_mask.mean(dim=(1, 3))[:, None, :]
        graph_logits = sim + 0.5 * adj.view(1, nodes, nodes) + source_reliability
        k = min(max(self.top_k, 1), nodes)
        top_values, top_indices = torch.topk(graph_logits, k=k, dim=-1)
        sparse_logits = torch.full_like(graph_logits, -1e9)
        sparse_logits.scatter_(-1, top_indices, top_values)
        adj_mgl = torch.softmax(sparse_logits, dim=-1)
        q = self.s_q(node_h).reshape(batch, nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k_node = self.s_k(node_h).reshape(batch, nodes, self.num_heads, self.head_dim).transpose(1, 2)
        spatial_attn = torch.matmul(q, k_node.transpose(-1, -2)) / max(self.head_dim ** 0.5, 1.0)
        return spatial_attn, adj_mgl

    def _graph_conv(self, h: torch.Tensor, spatial_attn: torch.Tensor, adj_mgl: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(h)
        for k in range(self.k_order):
            support = supports[k].view(1, support_nodes := supports.shape[-1], support_nodes)
            head_attn = spatial_attn[:, k % self.num_heads]
            weights = torch.softmax(head_attn + self.mask[k] * adj_mgl + 0.5 * support, dim=-1)
            aggregated = torch.einsum("bij,btjh->btih", weights, h)
            out = out + aggregated.matmul(self.theta[k])
        return self.graph_norm(F.gelu(out) + h)

    def _gtu(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        z = conv(x)
        a, b = torch.chunk(z, 2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(
        self,
        h: torch.Tensor,
        obs_mask: torch.Tensor,
        adj: torch.Tensor,
        supports: torch.Tensor,
        res_att: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_temporal, res_att = self._temporal_attention(h, obs_mask, res_att)
        spatial_attn, adj_mgl = self._dynamic_graph(h_temporal, obs_mask, adj)
        h_graph = self._graph_conv(h_temporal, spatial_attn, adj_mgl, supports)
        x_time = h_graph.permute(0, 3, 2, 1)
        h_time = self.time_mix(torch.cat([self._gtu(self.gtu3, x_time), self._gtu(self.gtu5, x_time), self._gtu(self.gtu7, x_time)], dim=1))
        h_out = self.out_norm(h_graph + h_time.permute(0, 3, 2, 1))
        return h_out, res_att, adj_mgl


class MaskAwareGraphRepairV2(nn.Module):
    """MagiNet-like internal graph repair backbone for LiteTrust."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        num_heads: int = 4,
        k_order: int = 3,
        max_nodes: int = 512,
        dropout: float = 0.1,
        top_k: int = 16,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.max_nodes = int(max_nodes)
        self.k_order = int(k_order)
        self.input_proj = nn.Linear(input_dim + 1, hidden_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        nn.init.uniform_(self.missing_token, -0.02, 0.02)
        self.node_embedding = nn.Embedding(max_nodes, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                MagiStyleRepairBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    k_order=k_order,
                    dropout=dropout,
                    top_k=top_k,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.final_head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> dict:
        batch, steps, nodes, _channels = x_obs.shape
        if nodes > self.max_nodes:
            raise ValueError(f"MaskAwareGraphRepairV2 max_nodes={self.max_nodes} is smaller than input nodes={nodes}.")
        time = torch.linspace(0.0, 1.0, steps, device=x_obs.device, dtype=x_obs.dtype).view(1, steps, 1, 1)
        time = time.expand(batch, steps, nodes, 1)
        h = self.input_proj(torch.cat([x_obs, mask.mean(dim=-1, keepdim=True)], dim=-1))
        obs = mask.mean(dim=-1, keepdim=True)
        h = obs * h + (1.0 - obs) * self.missing_token
        h = h + self.time_proj(time)
        node_ids = torch.arange(nodes, device=x_obs.device)
        h = h + self.node_embedding(node_ids).view(1, 1, nodes, -1)
        supports = _cheb_like_supports(adj, self.k_order)
        res_att = None
        learned_graph = None
        for block in self.blocks:
            h, res_att, learned_graph = block(h, mask, adj, supports, res_att)
        rnn_in = h.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        rnn_out, _ = self.final_rnn(rnn_in)
        pred = self.final_head(rnn_out).reshape(batch, nodes, steps, self.output_dim).permute(0, 2, 1, 3)
        mu = mask * x_obs + (1.0 - mask) * pred
        graph_conf = learned_graph.max(dim=-1, keepdim=True).values[:, None].expand(-1, steps, -1, -1) if learned_graph is not None else torch.zeros_like(mu)
        return {
            "mu": mu,
            "x_graph_repair": pred,
            "learned_graph": learned_graph,
            "graph_confidence": graph_conf,
            "h": h,
        }


class PhysicsPromotedExpert(nn.Module):
    """Lightweight physics correction that acts as a third expert."""

    def __init__(self, hidden_dim: int, output_dim: int, extra_feature_dim: int = 6, correction_clip: float = 0.75):
        super().__init__()
        self.output_dim = int(output_dim)
        self.extra_feature_dim = int(extra_feature_dim)
        self.correction_clip = float(correction_clip)
        expert_hidden = max(hidden_dim // 2, 16)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim + output_dim * 2 + 3 + extra_feature_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, output_dim),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(7, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self.fd_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(
        self,
        h: torch.Tensor,
        mu_global: torch.Tensor,
        mu_sensor: torch.Tensor,
        x_obs: torch.Tensor,
        obs_mask: torch.Tensor,
        adj: torch.Tensor,
        residual_signed: torch.Tensor,
        extra_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        obs = obs_mask.mean(dim=-1, keepdim=True)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        residual_rank = extra_feature[..., 3:4] if extra_feature.shape[-1] >= 4 else residual_signed.detach().abs().clamp(0.0, 1.0)
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        neighbor_missing = extra_feature[..., 5:6] if extra_feature.shape[-1] >= 6 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        neighbor_observed = 1.0 - neighbor_missing
        residual_scalar = residual_signed.detach().abs().mean(dim=-1, keepdim=True)
        graph_context = obs_mask * x_obs + (1.0 - obs_mask) * mu_global.detach()
        neigh_context = torch.einsum("nm,btmc->btnc", adj, graph_context)
        graph_gap = neigh_context - mu_global.detach()
        sensor_gap = mu_sensor.detach() - mu_global.detach()
        graph_gap_scalar = graph_gap.detach().abs().mean(dim=-1, keepdim=True)
        sensor_gap_scalar = sensor_gap.detach().abs().mean(dim=-1, keepdim=True)
        if self.output_dim == 1:
            explicit_delta = torch.zeros_like(mu_global)
            explicit_delta[..., 0:1] = -torch.clamp(residual_signed.detach(), min=-1.0, max=1.0)
        else:
            explicit_delta = torch.zeros_like(mu_global)
            explicit_delta[..., 0:1] = -torch.clamp(residual_signed.detach()[..., 0:1], min=-1.0, max=1.0)
            explicit_delta[..., 1:] = 0.1 * graph_gap[..., 1:]
        delta_input = torch.cat(
            [
                h,
                mu_global.detach(),
                mu_sensor.detach(),
                residual_scalar,
                graph_gap_scalar,
                sensor_gap_scalar,
                extra_feature.detach(),
            ],
            dim=-1,
        )
        learned_delta = torch.tanh(self.delta_head(delta_input)) * self.correction_clip
        gate_input = torch.cat(
            [
                node_failure,
                neighbor_observed,
                local_missing,
                residual_rank,
                residual_scalar,
                graph_gap_scalar,
                sensor_gap_scalar,
            ],
            dim=-1,
        )
        physics_trust = torch.sigmoid(self.gate_head(gate_input)) * torch.clamp(1.0 - 0.6 * node_failure, min=0.0, max=1.0)
        delta = physics_trust * (learned_delta + self.fd_gain * explicit_delta)
        return delta, physics_trust


class ScenarioAwareFusionRouter(nn.Module):
    """Route between global, sensor-failure, and physics candidates."""

    def __init__(self, hidden_dim: int, extra_feature_dim: int = 6, temperature: float = 0.7):
        super().__init__()
        self.extra_feature_dim = int(extra_feature_dim)
        self.temperature = float(temperature)
        gate_hidden = max(hidden_dim // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 9 + extra_feature_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )
        self.bias = nn.Parameter(torch.tensor([0.6, 0.7, -0.2], dtype=torch.float32))

    def forward(
        self,
        h: torch.Tensor,
        mu_global: torch.Tensor,
        mu_sensor: torch.Tensor,
        mu_phys: torch.Tensor,
        residual_abs: torch.Tensor,
        log_var: torch.Tensor | None,
        obs_mask: torch.Tensor,
        extra_feature: torch.Tensor,
        physics_trust: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if log_var is None:
            log_var = torch.zeros_like(residual_abs)
        if extra_feature.shape[-1] != self.extra_feature_dim:
            raise ValueError(f"extra_feature dim must be {self.extra_feature_dim}, got {extra_feature.shape[-1]}.")
        residual_scalar = residual_abs.mean(dim=-1, keepdim=True)
        log_var_scalar = log_var.mean(dim=-1, keepdim=True)
        obs = obs_mask.mean(dim=-1, keepdim=True)
        local_missing = extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - obs
        temporal = extra_feature[..., 0:1] if extra_feature.shape[-1] >= 1 else torch.zeros_like(obs)
        spatial = extra_feature[..., 1:2] if extra_feature.shape[-1] >= 2 else torch.zeros_like(obs)
        residual_rank = extra_feature[..., 3:4] if extra_feature.shape[-1] >= 4 else residual_abs.detach().clamp(0.0, 1.0)
        node_missing = extra_feature[..., 4:5] if extra_feature.shape[-1] >= 5 else local_missing
        neighbor_missing = extra_feature[..., 5:6] if extra_feature.shape[-1] >= 6 else local_missing
        node_failure = _node_failure_signal(node_missing, local_missing)
        random_missing = torch.clamp(local_missing - node_failure, min=0.0, max=1.0)
        incident_score = 0.5 * temporal + 0.5 * spatial + 0.2 * residual_rank
        gap_gs = torch.abs(mu_global.detach() - mu_sensor.detach()).mean(dim=-1, keepdim=True)
        gap_gp = torch.abs(mu_global.detach() - mu_phys.detach()).mean(dim=-1, keepdim=True)
        gap_sp = torch.abs(mu_sensor.detach() - mu_phys.detach()).mean(dim=-1, keepdim=True)
        route_input = torch.cat(
            [
                h,
                residual_scalar.detach(),
                log_var_scalar.detach(),
                obs,
                local_missing,
                node_missing,
                neighbor_missing,
                gap_gs,
                gap_gp,
                gap_sp,
                extra_feature.detach(),
            ],
            dim=-1,
        )
        logits = self.net(route_input)
        global_prior = 1.3 * random_missing + 1.1 * incident_score + 0.4 * obs - 0.6 * node_failure
        sensor_prior = 2.6 * node_failure + 1.0 * neighbor_missing + 0.6 * local_missing - 0.2 * incident_score
        physics_prior = 1.1 * obs + 0.9 * residual_rank + 0.3 * gap_gs - 1.1 * node_failure - 0.7 * local_missing
        prior = torch.cat([global_prior, sensor_prior, physics_prior], dim=-1)
        logits = logits + prior + self.bias.view(1, 1, 1, 3)
        logits = logits + torch.cat([gap_gs, gap_sp, gap_gp], dim=-1)
        weights = torch.softmax(logits / max(self.temperature, 1e-3), dim=-1)
        return weights, prior, logits


class LiteTrustFusion(nn.Module):
    """Three-expert fusion: global GRIN-like, sensor-failure SAITS-like, and physics-promoted correction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        w_min: float = 0.0,
        use_uncertainty: bool = True,
        extra_feature_dim: int = 6,
        sensor_layers: int = 1,
        sensor_heads: int = 2,
        router_temperature: float = 0.7,
        correction_clip: float = 0.75,
    ):
        super().__init__()
        self.use_uncertainty = bool(use_uncertainty)
        self.output_dim = int(output_dim)
        self.global_expert = BidirectionalGraphExpert(input_dim, hidden_dim, output_dim, dropout=dropout)
        self.sensor_expert = TemporalSAITSLite(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=sensor_layers,
            num_heads=sensor_heads,
            dropout=dropout,
        )
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        self.physics_expert = PhysicsPromotedExpert(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            extra_feature_dim=extra_feature_dim,
            correction_clip=correction_clip,
        )
        self.trust_gate = PhysicsTrustGate(hidden_dim, w_min=w_min, extra_feature_dim=extra_feature_dim)
        self.router = ScenarioAwareFusionRouter(hidden_dim, extra_feature_dim=extra_feature_dim, temperature=router_temperature)

    def predict_log_var(self, h: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.logvar_head(h), min=-6.0, max=3.0)

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        residual_abs: torch.Tensor | None = None,
        residual_signed: torch.Tensor | None = None,
        physics_delta: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> dict:
        supplied_extra_feature = extra_feature
        mu_global, h = self.global_expert(x_obs, mask, adj)
        mu_sensor, sensor_h = self.sensor_expert(x_obs, mask)
        output = {
            "mu_data": mu_global,
            "x_global": mu_global,
            "x_sensor": mu_sensor,
            "h": h,
            "sensor_h": sensor_h,
        }
        if self.use_uncertainty:
            output["log_var"] = self.predict_log_var(h)
        gate_log_var = log_var if log_var is not None else output.get("log_var", None)
        if residual_abs is None:
            if self.output_dim == 1:
                residual_abs = graph_flow_residual_full(mu_global[..., 0:1], adj)
            else:
                residual_abs = torch.zeros_like(mu_global)
        if residual_signed is None:
            residual_signed = residual_abs
        if supplied_extra_feature is None:
            extra_feature = _trust_extra_features(x_obs, mask, adj, residual_abs if residual_abs.shape[-1] == 1 else residual_abs[..., :1])
        else:
            extra_feature = supplied_extra_feature
        delta_phys, physics_trust = self.physics_expert(
            h,
            mu_global,
            mu_sensor,
            x_obs,
            mask,
            adj,
            residual_signed,
            extra_feature,
        )
        verifier_trust = self.trust_gate(h, residual_abs, gate_log_var, mask, extra_feature=extra_feature)
        physics_trust = physics_trust * verifier_trust
        if physics_delta is not None:
            delta_phys = physics_delta
        mu_phys = mu_global + delta_phys
        weights, prior, router_logits = self.router(
            h,
            mu_global,
            mu_sensor,
            mu_phys,
            residual_abs.detach().abs(),
            gate_log_var,
            mask,
            extra_feature,
            physics_trust,
        )
        global_weight = weights[..., 0:1]
        sensor_weight = weights[..., 1:2]
        phys_weight = weights[..., 2:3]
        mu = global_weight * mu_global + sensor_weight * mu_sensor + phys_weight * mu_phys
        output.update(
            {
                "mu": mu,
                "x_phys": mu_phys,
                "delta_phys": delta_phys,
                "physics_trust": physics_trust,
                "trust": physics_trust,
                "expert_weights": weights,
                "global_weight": global_weight,
                "sensor_weight": sensor_weight,
                "phys_weight": phys_weight,
                "region_gate": weights,
                "router_prior": prior,
                "router_logits": router_logits,
                "correction_trust": sensor_weight + phys_weight,
            }
        )
        return output


class LiteTrustGRINReliabilityRouter(LiteTrustGRINCorrection):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        w_min: float = 0.0,
        use_uncertainty: bool = True,
        extra_feature_dim: int = 4,
        use_phys_expert: bool = True,
        correction_clip: float = 1.0,
        router_temperature: float = 0.7,
        use_projection_controller: bool = False,
        use_validity_gate: bool = False,
        use_spatial_physics: bool = False,
        use_directional_physics: bool = False,
        directional_shift_max: float = 0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            w_min=w_min,
            use_uncertainty=use_uncertainty,
            extra_feature_dim=extra_feature_dim,
            use_graph_delta=True,
            use_phys_expert=use_phys_expert,
            failure_routing="learned",
            correction_clip=correction_clip,
        )
        self.use_projection_controller = bool(use_projection_controller)
        self.use_validity_gate = bool(use_validity_gate)
        self.use_spatial_physics = bool(use_spatial_physics)
        self.use_directional_physics = bool(use_directional_physics)
        self.directional_shift_max = float(directional_shift_max)
        self.reliability_router = LightweightReliabilityRouter(use_phys_expert=use_phys_expert, temperature=router_temperature)
        self.physics_controller = PhysicsProjectionController(extra_feature_dim=extra_feature_dim)
        self.physics_validity = PhysicsValidityGate()
        self.physics_form_router = PhysicsFormRouter(hidden_dim=hidden_dim, extra_feature_dim=extra_feature_dim)
        self.spatial_physics = SpatialConservationPhysicsExpert(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            extra_feature_dim=extra_feature_dim,
            correction_clip=correction_clip,
        )
        self.directional_physics = DirectionalConservationPhysicsExpert(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            extra_feature_dim=extra_feature_dim,
            correction_clip=correction_clip,
        )

    def forward(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        residual_abs: torch.Tensor | None = None,
        residual_signed: torch.Tensor | None = None,
        physics_delta: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
        extra_feature: torch.Tensor | None = None,
    ) -> dict:
        fwd_pred, fwd_h = self._direction(x_obs, mask, adj, reverse=False)
        bwd_pred, bwd_h = self._direction(x_obs, mask, adj, reverse=True)
        h = torch.tanh(self.hidden_fuse(torch.cat([fwd_h, bwd_h], dim=-1)))
        mu_data = self.data_head(torch.cat([fwd_pred, bwd_pred], dim=-1))
        output = {"mu": mu_data, "mu_data": mu_data, "h": h}
        if self.use_uncertainty:
            output["log_var"] = self.predict_log_var(h)
        if residual_abs is not None and extra_feature is not None:
            gate_log_var = log_var
            residual_for_trust = residual_abs.detach().abs()
            residual_for_correction = residual_signed if residual_signed is not None else residual_abs
            learned_delta_phys = self.correction_from_features(h, residual_for_correction, mask, gate_log_var, extra_feature)
            projection_delta = torch.zeros_like(learned_delta_phys)
            if residual_signed is not None:
                projection_delta[..., 0:1] = -torch.clamp(residual_signed.detach(), min=-1.0, max=1.0)
            if self.use_projection_controller:
                projection_gamma = self.physics_controller(residual_for_trust, mask, extra_feature)
            else:
                projection_gamma = torch.zeros_like(residual_for_trust)
            if physics_delta is not None:
                projection_delta = physics_delta
                projection_gamma = torch.ones_like(projection_gamma)
            spatial_phys_delta = torch.zeros_like(learned_delta_phys)
            spatial_phys_gate = torch.zeros_like(residual_for_trust)
            directional_phys_delta = torch.zeros_like(learned_delta_phys)
            directional_phys_gate = torch.zeros_like(residual_for_trust)
            directional_conservation_residual = torch.zeros_like(residual_for_trust)
            if self.use_spatial_physics and self.use_phys_expert and residual_signed is not None:
                spatial_phys_delta, spatial_phys_gate = self.spatial_physics(
                    h,
                    mu_data,
                    x_obs,
                    mask,
                    adj,
                    residual_signed,
                    extra_feature,
                )
            if self.use_directional_physics and self.use_phys_expert and residual_signed is not None:
                directional_phys_delta, directional_phys_gate, directional_conservation_residual = self.directional_physics(
                    h,
                    mu_data,
                    x_obs,
                    mask,
                    adj,
                    residual_signed,
                    extra_feature,
                )
            temporal_phys_delta = 0.5 * _temporal_smooth_delta(mu_data)
            candidate_deltas = torch.stack(
                [
                    projection_delta,
                    spatial_phys_delta,
                    directional_phys_delta,
                    temporal_phys_delta,
                ],
                dim=-1,
            )
            candidate_mags = candidate_deltas.abs().mean(dim=-2)
            residual_scalar = residual_for_trust.abs().mean(dim=-1, keepdim=True)
            residual_rank = extra_feature[..., 3:4] if extra_feature.shape[-1] >= 4 else residual_scalar.clamp(0.0, 1.0)
            physics_form_weights, physics_form_logits, physics_form_confidence = self.physics_form_router(
                h,
                residual_scalar,
                residual_rank,
                candidate_mags,
                extra_feature,
            )
            physics_form_delta = torch.sum(candidate_deltas * physics_form_weights.unsqueeze(-2), dim=-1)
            delta_phys = physics_form_confidence * (learned_delta_phys + projection_gamma * physics_form_delta)
            if extra_feature.shape[-1] >= 5:
                node_missing = extra_feature[..., 4:5]
                graph_applicability = _node_failure_signal(node_missing, extra_feature[..., 2:3] if extra_feature.shape[-1] >= 3 else 1.0 - mask.detach().mean(dim=-1, keepdim=True))
            else:
                graph_applicability = 1.0 - mask.detach().mean(dim=-1, keepdim=True)
            graph_context = mask * x_obs + (1.0 - mask) * mu_data
            neigh_context = torch.einsum("nm,btmc->btnc", adj, graph_context)
            graph_delta = graph_applicability * (neigh_context - mu_data)
            reliability_scores, expert_weights = self.reliability_router(residual_abs.detach().abs(), gate_log_var, mask, extra_feature)
            data_weight = expert_weights[..., 0:1]
            graph_weight = expert_weights[..., 1:2]
            phys_weight = expert_weights[..., 2:3]
            graph_inactive = 1.0 - graph_applicability
            if self.use_phys_expert:
                phys_weight = phys_weight + 0.75 * graph_inactive * graph_weight
                data_weight = data_weight + 0.25 * graph_inactive * graph_weight
            else:
                data_weight = data_weight + graph_inactive * graph_weight
            graph_weight = graph_applicability * graph_weight
            denom = torch.clamp(data_weight + graph_weight + phys_weight, min=1e-6)
            data_weight = data_weight / denom
            graph_weight = graph_weight / denom
            phys_weight = phys_weight / denom
            if self.use_validity_gate:
                physics_validity = self.physics_validity(residual_for_trust, mask, extra_feature)
                validity_mix = 0.5 + 0.5 * physics_validity
                invalid_phys = (1.0 - validity_mix) * phys_weight
                phys_weight = validity_mix * phys_weight
                data_weight = data_weight + invalid_phys
                denom = torch.clamp(data_weight + graph_weight + phys_weight, min=1e-6)
                data_weight = data_weight / denom
                graph_weight = graph_weight / denom
                phys_weight = phys_weight / denom
            else:
                physics_validity = torch.ones_like(phys_weight)
            if self.use_directional_physics and self.use_phys_expert and self.directional_shift_max > 0.0:
                directional_shift = torch.minimum(
                    torch.clamp(self.directional_shift_max * directional_phys_gate.detach() * graph_weight, min=0.0),
                    graph_weight,
                )
                graph_weight = graph_weight - directional_shift
                phys_weight = phys_weight + directional_shift
                denom = torch.clamp(data_weight + graph_weight + phys_weight, min=1e-6)
                data_weight = data_weight / denom
                graph_weight = graph_weight / denom
                phys_weight = phys_weight / denom
            else:
                directional_shift = torch.zeros_like(phys_weight)
            form_mix = 0.5 + 0.5 * physics_form_confidence
            phys_weight = phys_weight * form_mix
            data_weight = data_weight + (1.0 - form_mix) * phys_weight
            denom = torch.clamp(data_weight + graph_weight + phys_weight, min=1e-6)
            data_weight = data_weight / denom
            graph_weight = graph_weight / denom
            phys_weight = phys_weight / denom
            x_graph = mu_data + graph_delta
            x_phys = mu_data + delta_phys
            output["delta_phys"] = delta_phys
            output["learned_delta_phys"] = learned_delta_phys
            output["projection_delta"] = projection_delta
            output["projection_gamma"] = projection_gamma
            output["spatial_phys_delta"] = spatial_phys_delta
            output["spatial_phys_gate"] = spatial_phys_gate
            output["directional_phys_delta"] = directional_phys_delta
            output["directional_phys_gate"] = directional_phys_gate
            output["directional_shift"] = directional_shift
            output["directional_conservation_residual"] = directional_conservation_residual
            output["temporal_phys_delta"] = temporal_phys_delta
            output["physics_form_candidates"] = candidate_deltas
            output["physics_form_weights"] = physics_form_weights
            output["physics_form_logits"] = physics_form_logits
            output["physics_form_confidence"] = physics_form_confidence
            output["physics_validity"] = physics_validity
            output["graph_delta"] = graph_delta
            output["x_graph"] = x_graph
            output["x_phys"] = x_phys
            output["reliability_scores"] = reliability_scores
            output["expert_weights"] = expert_weights
            output["data_weight"] = data_weight
            output["graph_weight"] = graph_weight
            output["phys_weight"] = phys_weight
            output["physics_trust"] = phys_weight
            output["correction_trust"] = graph_weight + phys_weight * physics_form_confidence
            output["trust"] = phys_weight
            output["mu"] = data_weight * mu_data + graph_weight * x_graph + phys_weight * x_phys
        return output
