from __future__ import annotations

import torch


def _safe_scale(residual: torch.Tensor, clip_value: float = 10.0) -> torch.Tensor:
    scale = residual.detach().abs().mean().clamp_min(1e-6)
    return torch.clamp(residual / scale, min=-clip_value, max=clip_value)


def _residual_scale(residual: torch.Tensor, normalizer, flow_idx: int, clip_value: float = 10.0) -> torch.Tensor:
    if normalizer is None:
        return _safe_scale(residual, clip_value=clip_value)
    scale = torch.as_tensor(normalizer.std[..., flow_idx : flow_idx + 1], dtype=residual.dtype, device=residual.device)
    return torch.clamp(residual / scale.clamp_min(1e-6), min=-clip_value, max=clip_value)


def _inverse_channel(x: torch.Tensor, normalizer, channel_idx: int) -> torch.Tensor:
    if normalizer is None:
        return x
    mean = torch.as_tensor(normalizer.mean[..., channel_idx : channel_idx + 1], dtype=x.dtype, device=x.device)
    std = torch.as_tensor(normalizer.std[..., channel_idx : channel_idx + 1], dtype=x.dtype, device=x.device)
    return x * std + mean


def _fd_scale(normalizer, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if normalizer is None or getattr(normalizer, "fd_scale", None) is None:
        return torch.as_tensor(1.0, dtype=dtype, device=device)
    return torch.as_tensor(normalizer.fd_scale, dtype=dtype, device=device)


CHANNEL_ORDERS = {
    "flow_occupancy_speed": {"flow": 0, "occupancy": 1, "speed": 2},
    "flow_speed_occupancy": {"flow": 0, "speed": 1, "occupancy": 2},
}


def channel_indices(channel_order: str = "flow_occupancy_speed") -> dict[str, int]:
    if channel_order not in CHANNEL_ORDERS:
        supported = ", ".join(sorted(CHANNEL_ORDERS))
        raise ValueError(f"Unsupported channel_order {channel_order!r}. Supported: {supported}.")
    return CHANNEL_ORDERS[channel_order]


def fundamental_residual(
    flow_hat: torch.Tensor,
    speed_hat: torch.Tensor,
    occ_hat: torch.Tensor,
    normalizer=None,
    flow_idx: int = 0,
    speed_idx: int = 2,
    occ_idx: int = 1,
) -> torch.Tensor:
    flow_hat = _inverse_channel(flow_hat, normalizer, channel_idx=flow_idx)
    speed_hat = _inverse_channel(speed_hat, normalizer, channel_idx=speed_idx)
    occ_hat = _inverse_channel(occ_hat, normalizer, channel_idx=occ_idx)
    residual = flow_hat - _fd_scale(normalizer, flow_hat.dtype, flow_hat.device) * occ_hat * speed_hat
    return _residual_scale(residual, normalizer, flow_idx=flow_idx)


def fundamental_residual_from_prediction(
    pred: torch.Tensor,
    normalizer=None,
    channel_order: str = "flow_occupancy_speed",
) -> torch.Tensor:
    indices = channel_indices(channel_order)
    flow_idx = indices["flow"]
    speed_idx = indices["speed"]
    occ_idx = indices["occupancy"]
    return fundamental_residual(
        pred[..., flow_idx : flow_idx + 1],
        pred[..., speed_idx : speed_idx + 1],
        pred[..., occ_idx : occ_idx + 1],
        normalizer=normalizer,
        flow_idx=flow_idx,
        speed_idx=speed_idx,
        occ_idx=occ_idx,
    )


def graph_speed_residual(speed_hat: torch.Tensor, adj: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    current = speed_hat[:, :-1]
    future = speed_hat[:, 1:]
    neigh = torch.einsum("nm,btm->btn", adj, current.squeeze(-1)).unsqueeze(-1)
    residual = future - current + eta * (current - neigh)
    return _safe_scale(residual)


def temporal_smooth_residual(pred: torch.Tensor) -> torch.Tensor:
    residual = torch.zeros_like(pred)
    if pred.shape[1] > 1:
        residual[:, 1:-1] = 0.5 * (pred[:, :-2].detach() + pred[:, 2:].detach()) - pred[:, 1:-1].detach()
        residual[:, 0] = pred[:, 1].detach() - pred[:, 0].detach()
        residual[:, -1] = pred[:, -2].detach() - pred[:, -1].detach()
    return _safe_scale(residual)


def spatial_neighbor_residual(pred: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    neigh = torch.einsum("nm,btmc->btnc", adj, pred.detach())
    residual = pred - neigh
    return _safe_scale(residual)


def graph_flow_residual(flow_hat: torch.Tensor, adj: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    current = flow_hat[:, :-1]
    future = flow_hat[:, 1:]
    neigh = torch.einsum("nm,btm->btn", adj, current.squeeze(-1)).unsqueeze(-1)
    residual = future - current + eta * (current - neigh)
    return _safe_scale(residual)


def graph_flow_residual_full(flow_hat: torch.Tensor, adj: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    residual = graph_flow_residual(flow_hat, adj, eta=eta)
    pad = torch.zeros_like(flow_hat[:, :1])
    return torch.cat([pad, residual], dim=1)


def residual_bank_from_prediction(
    pred: torch.Tensor,
    adj: torch.Tensor,
    normalizer=None,
    channel_order: str = "flow_occupancy_speed",
) -> dict[str, torch.Tensor]:
    if pred.shape[-1] <= 2:
        primary = graph_speed_residual(pred[..., 0:1], adj)
    else:
        primary = fundamental_residual_from_prediction(pred, normalizer=normalizer, channel_order=channel_order)
    return {
        "primary": primary,
        "temporal": temporal_smooth_residual(pred),
        "spatial": spatial_neighbor_residual(pred, adj),
    }


def graph_conservation_residual(density_hat: torch.Tensor, flow_hat: torch.Tensor, adj: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    neigh_flow = torch.einsum("nm,btm->btn", adj, flow_hat.squeeze(-1)).unsqueeze(-1)
    residual = density_hat[:, 1:] - density_hat[:, :-1] + kappa * (flow_hat[:, :-1] - neigh_flow[:, :-1])
    return _safe_scale(residual)
