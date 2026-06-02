from __future__ import annotations

import numpy as np


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    denom = np.clip(mask.sum(), 1.0, None)
    return float((values * mask).sum() / denom)


def compute_metrics(pred: np.ndarray, target: np.ndarray, target_mask: np.ndarray) -> dict:
    error = pred - target
    abs_error = np.abs(error)
    sq_error = error ** 2
    denom = np.clip(np.abs(target), 1e-6, None)
    ape = abs_error / denom

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(sq_error.mean()))
    mape = float(ape.mean())
    masked_mae = _masked_mean(abs_error, target_mask)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "masked_mae": masked_mae,
    }
