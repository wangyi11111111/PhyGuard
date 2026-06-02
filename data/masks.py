from __future__ import annotations

import numpy as np


def _validate_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 4:
        raise ValueError("mask shape must be [B, T, N, C].")
    if any(dim <= 0 for dim in shape):
        raise ValueError("all mask dimensions must be positive.")


def _validate_rate(name: str, value: float, allow_one: bool = False) -> None:
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not 0.0 <= value and upper_ok:
        raise ValueError(f"{name} must be non-negative.")
    if not upper_ok:
        upper = "[0, 1]" if allow_one else "[0, 1)"
        raise ValueError(f"{name} must be in {upper}.")


def random_missing_mask(shape: tuple[int, ...], missing_rate: float, seed: int | None = None) -> np.ndarray:
    _validate_shape(shape)
    _validate_rate("missing_rate", missing_rate)
    rng = np.random.default_rng(seed)
    keep_prob = 1.0 - missing_rate
    mask = rng.binomial(1, keep_prob, size=shape).astype(np.float32)
    return mask


def sensor_failure_mask(shape: tuple[int, ...], fail_rate: float, seed: int | None = None) -> np.ndarray:
    _validate_shape(shape)
    _validate_rate("fail_rate", fail_rate, allow_one=True)
    rng = np.random.default_rng(seed)
    mask = np.ones(shape, dtype=np.float32)
    num_nodes = shape[2]
    fail_nodes = int(round(num_nodes * fail_rate))
    if fail_rate > 0.0:
        fail_nodes = max(1, fail_nodes)
    if fail_nodes == 0:
        return mask
    chosen = rng.choice(num_nodes, size=fail_nodes, replace=False)
    mask[:, :, chosen, :] = 0.0
    return mask


def block_missing_mask(shape: tuple[int, ...], adj: np.ndarray, block_size: int, seed: int | None = None) -> np.ndarray:
    _validate_shape(shape)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if adj.shape != (shape[2], shape[2]):
        raise ValueError("adj must have shape [N, N].")
    rng = np.random.default_rng(seed)
    mask = np.ones(shape, dtype=np.float32)
    center = int(rng.integers(0, shape[2]))
    # Pick the center plus strongest connected neighbors. This simulates a local spatial outage.
    scores = np.asarray(adj[center], dtype=np.float32).copy()
    scores[center] = np.inf
    order = np.argsort(scores)[::-1]
    nodes = order[: min(shape[2], block_size)]
    mask[:, :, nodes, :] = 0.0
    return mask


def temporal_missing_mask(shape: tuple[int, ...], missing_rate: float, duration: int, seed: int | None = None) -> np.ndarray:
    _validate_shape(shape)
    _validate_rate("missing_rate", missing_rate, allow_one=True)
    if duration <= 0:
        raise ValueError("duration must be positive.")
    rng = np.random.default_rng(seed)
    mask = np.ones(shape, dtype=np.float32)
    total_steps = shape[1]
    if missing_rate == 0.0:
        return mask

    target_steps = min(total_steps, max(1, int(round(total_steps * missing_rate))))
    duration = min(duration, total_steps)
    dropped = np.zeros(total_steps, dtype=bool)

    # Add contiguous blocks until the requested temporal missing ratio is reached closely.
    max_attempts = total_steps * 4
    attempts = 0
    while dropped.sum() < target_steps and attempts < max_attempts:
        start = int(rng.integers(0, total_steps - duration + 1))
        dropped[start : start + duration] = True
        attempts += 1

    if dropped.sum() < target_steps:
        remaining = np.flatnonzero(~dropped)
        extra = remaining[: target_steps - dropped.sum()]
        dropped[extra] = True

    mask[:, dropped, :, :] = 0.0
    return mask
