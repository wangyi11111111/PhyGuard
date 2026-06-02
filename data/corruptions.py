from __future__ import annotations

import numpy as np


def add_gaussian_noise(x: np.ndarray, noise_std: float, seed: int | None = None) -> np.ndarray:
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if noise_std == 0.0:
        return x.copy()
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)


def incident_perturbation(
    x: np.ndarray,
    adj: np.ndarray,
    drop_ratio: float,
    duration: int,
    region_size: int,
    seed: int | None = None,
    return_mask: bool = False,
    flow_drop_ratio: float | None = None,
    speed_drop_ratio: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if x.ndim != 4:
        raise ValueError("x must have shape [B, T, N, C].")
    if adj.shape != (x.shape[2], x.shape[2]):
        raise ValueError("adj must have shape [N, N].")
    if not 0.0 <= drop_ratio <= 1.0:
        raise ValueError("drop_ratio must be in [0, 1].")
    if flow_drop_ratio is not None and not 0.0 <= flow_drop_ratio <= 1.0:
        raise ValueError("flow_drop_ratio must be in [0, 1].")
    if speed_drop_ratio is not None and not 0.0 <= speed_drop_ratio <= 1.0:
        raise ValueError("speed_drop_ratio must be in [0, 1].")
    if duration <= 0:
        raise ValueError("duration must be positive.")
    if region_size <= 0:
        raise ValueError("region_size must be positive.")

    rng = np.random.default_rng(seed)
    perturbed = x.copy()
    incident_mask = np.zeros_like(x, dtype=np.float32)
    center = int(rng.integers(0, x.shape[2]))
    scores = np.asarray(adj[center], dtype=np.float32).copy()
    scores[center] = np.inf
    region = np.argsort(scores)[::-1][: min(x.shape[2], region_size)]
    duration = min(duration, x.shape[1])
    start = int(rng.integers(0, max(1, x.shape[1] - duration + 1)))
    end = start + duration

    # Channel convention for toy/PEMS-like data: 0=flow, 1=occupancy, 2=speed.
    flow_drop = drop_ratio if flow_drop_ratio is None else flow_drop_ratio
    speed_drop = drop_ratio if speed_drop_ratio is None else speed_drop_ratio
    perturbed[:, start:end, region, 0] *= 1.0 - flow_drop
    if x.shape[-1] > 2:
        perturbed[:, start:end, region, 2] *= 1.0 - speed_drop
    incident_mask[:, start:end, region, :] = 1.0
    if return_mask:
        return perturbed, incident_mask
    return perturbed
