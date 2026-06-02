from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray
    fd_scale: np.ndarray | None = None

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mean = x.mean(axis=(0, 1, 2), keepdims=True)
        std = x.std(axis=(0, 1, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        fd_scale = None
        if x.shape[-1] >= 3:
            flow = x[..., 0:1]
            occupancy = x[..., 1:2]
            speed = x[..., 2:3]
            denom = np.sum(occupancy * speed)
            if abs(float(denom)) > 1e-6:
                fd_scale = np.asarray([[[[np.sum(flow) / denom]]]], dtype=np.float32)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), fd_scale=fd_scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return x * std + mean
        return (x * self.std + self.mean).astype(np.float32)
