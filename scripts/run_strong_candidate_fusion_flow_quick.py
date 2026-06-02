from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from models.litetrust_pinn import _node_failure_signal
from scripts.run_conflict_test import _batch_rank
from scripts.run_five_baselines_flow_quick import (
    MAGI_ROOT,
    _load_flow_splits,
    _prepare_magi_files,
    _run_pypots_model,
    _scenario_data,
)
from scripts.train import resolve_device

try:
    from pypots.imputation import SAITS
except Exception as exc:  # pragma: no cover
    SAITS = None
    PYPOTS_IMPORT_ERROR = exc
else:
    PYPOTS_IMPORT_ERROR = None


class MaskAwareGraphCore(nn.Module):
    """Internal mask-aware graph reconstruction core.

    This keeps the MagiNet-side contract used by the LiteTrust pipeline:
    input is observed values plus masks, output is a dense reconstruction.
    It is intentionally implemented in local PyTorch so the framework can be
    evaluated without importing the external MagiNet repository.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.temporal = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # features: [B,T,N,F], adj: [N,N]
        h = self.input_proj(features)
        for conv, norm in zip(self.temporal, self.norms):
            residual = h
            y = conv(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            y = self.dropout(F.gelu(y))
            h = norm(residual + y)
        neigh = torch.einsum("nm,btmc->btnc", adj, h)
        h = F.gelu(self.self_proj(h) + self.neigh_proj(neigh))
        return self.head(h)


def _normalize_adj_np(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    denom = np.clip(adj.sum(axis=1, keepdims=True), 1e-6, None)
    return (adj / denom).astype(np.float32)


def _core_features(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, t, n, _ = obs.shape
    time = np.linspace(0.0, 1.0, t, dtype=np.float32)[None, :, None, None]
    time = np.repeat(np.repeat(time, b, axis=0), n, axis=2)
    node_missing = 1.0 - mask.mean(axis=(1, 3), keepdims=True)
    node_missing = np.repeat(node_missing, t, axis=1).astype(np.float32)
    return np.concatenate([obs, mask, time, node_missing], axis=-1).astype(np.float32)


def _run_internal_mask_aware_core_all_splits(train, val, test, adj: np.ndarray, device: torch.device, epochs: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    observed = train_mask > 0.5
    mean = float(train_obs[observed].mean()) if np.any(observed) else 0.0
    std = float(train_obs[observed].std()) if np.any(observed) else 1.0
    std = max(std, 1e-6)

    def norm_x(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / std).astype(np.float32)

    train_feat = _core_features(norm_x(train_obs), train_mask)
    val_feat = _core_features(norm_x(val_obs), val_mask)
    test_feat = _core_features(norm_x(test_obs), test_mask)
    train_y = norm_x(train_full)
    val_y = norm_x(val_full)
    adj_t = torch.tensor(_normalize_adj_np(adj), dtype=torch.float32, device=device)
    model = MaskAwareGraphCore(train_feat.shape[-1], hidden_dim=64, layers=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    batch_size = 16
    generator = torch.Generator().manual_seed(seed + 3107)

    x_train = torch.tensor(train_feat, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    m_train = torch.tensor(1.0 - train_mask, dtype=torch.float32)
    x_val = torch.tensor(val_feat, dtype=torch.float32, device=device)
    y_val = torch.tensor(val_y, dtype=torch.float32, device=device)
    m_val = torch.tensor(1.0 - val_mask, dtype=torch.float32, device=device)
    for _epoch in range(max(1, epochs)):
        order = torch.randperm(x_train.shape[0], generator=generator)
        model.train()
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            mb = m_train[idx].to(device)
            pred = model(xb, adj_t)
            obs_weight = 0.2 * (1.0 - mb)
            miss_weight = mb
            weight = obs_weight + miss_weight
            loss = (F.smooth_l1_loss(pred, yb, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val, adj_t)
            val_loss = (torch.abs(val_pred - y_val) * m_val).sum() / m_val.sum().clamp_min(1.0)
        if float(val_loss) < best_val:
            best_val = float(val_loss)
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(feat: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, feat.shape[0], batch_size):
                xb = torch.tensor(feat[start : start + batch_size], dtype=torch.float32, device=device)
                pred = model(xb, adj_t).cpu().numpy() * std + mean
                preds.append(pred.astype(np.float32))
        dense = np.concatenate(preds, axis=0)
        return (mask * obs + (1.0 - mask) * dense).astype(np.float32)

    return (
        predict(train_feat, train_obs, train_mask),
        predict(val_feat, val_obs, val_mask),
        predict(test_feat, test_obs, test_mask),
    )


class CandidateRouter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.utility_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.local_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.utility_bias = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.local_bias = nn.Parameter(torch.tensor([0.6, 0.1, 0.4], dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = features[..., 7:8]
        node_missing = features[..., 8:9]
        neighbor_missing = features[..., 9:10]
        temporal = features[..., 10:11]
        spatial = features[..., 11:12]
        residual = features[..., 12:13]
        reliability_gap = torch.clamp(node_missing - neighbor_missing, min=0.0, max=1.0)
        utility_prior = (
            1.1 * missing
            + 0.9 * node_missing
            + 0.7 * temporal
            + 0.5 * spatial
            + 0.3 * residual
            - 0.2 * neighbor_missing
        )
        utility = torch.sigmoid(self.utility_head(features) + utility_prior + self.utility_bias)
        repair_prior = 1.8 * node_missing + 0.6 * reliability_gap + 0.4 * neighbor_missing + 0.5 * residual + 0.2 * temporal - 0.1 * spatial
        local_prior = torch.cat(
            [
                0.5 * temporal - 0.2 * spatial + 0.3 * node_missing,
                0.6 * residual - 0.6 * node_missing - 0.3 * neighbor_missing,
                repair_prior,
            ],
            dim=-1,
        )
        local_logits = self.local_head(features) + self.local_bias.view(1, 1, 1, 3) + local_prior
        local_probs = torch.softmax(local_logits / 0.5, dim=-1)
        local_mass = torch.clamp(utility, 0.0, 1.0)
        weights = torch.cat([(1.0 - local_mass), local_mass * local_probs], dim=-1)
        unreliable_node = _node_failure_signal(node_missing, neighbor_missing, temperature=0.2)
        local_safe = torch.cat(
            [
                torch.zeros_like(unreliable_node),
                0.55 * torch.ones_like(unreliable_node),
                0.05 * torch.ones_like(unreliable_node),
                0.40 * torch.ones_like(unreliable_node),
            ],
            dim=-1,
        )
        weights = (1.0 - unreliable_node) * weights + unreliable_node * local_safe
        return weights, utility, local_probs


def _load_maginet_symbols():
    old_argv = sys.argv[:]
    sys.argv = ["main.py"]
    root = str(MAGI_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    cwd = Path.cwd()
    try:
        import os

        os.chdir(MAGI_ROOT)
        main_mod = importlib.import_module("main")
        load_mod = importlib.import_module("load_data")
        utils_mod = importlib.import_module("utils")
        model_mod = importlib.import_module("models.model")
    finally:
        os.chdir(cwd)
        sys.argv = old_argv
    original_scaled_laplacian = model_mod.scaled_Laplacian

    def robust_scaled_laplacian(adj_mx):
        try:
            return original_scaled_laplacian(adj_mx)
        except Exception:
            degree = np.diag(np.sum(adj_mx, axis=1))
            laplacian = degree - adj_mx
            eigvals = np.linalg.eigvals(laplacian)
            lambda_max = float(np.max(np.real(eigvals)))
            if not np.isfinite(lambda_max) or abs(lambda_max) < 1e-8:
                lambda_max = 1.0
            return (2 * laplacian) / lambda_max - np.identity(adj_mx.shape[0])

    model_mod.scaled_Laplacian = robust_scaled_laplacian
    return main_mod.MagiNet, main_mod.seed_torch, load_mod.generate_miss_loader, utils_mod.weight_matrix, utils_mod.unnormalization


def _predict_maginet_split(model, loader, device, mean, std) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x, m, _y in loader:
            x = x.to(device)
            m = m.to(device)
            x_hat = model(x, m).detach().cpu().numpy()
            unnorm_x = _predict_maginet_split.unnormalization(x[:, :, :1, :].detach().cpu().numpy(), mean, std)
            unnorm_x_hat = _predict_maginet_split.unnormalization(x_hat, mean, std)
            mask = m.detach().cpu().numpy()
            filled = unnorm_x_hat * (1.0 - mask[:, :, :1, :]) + unnorm_x * mask[:, :, :1, :]
            preds.append(filled)
    pred = np.concatenate(preds, axis=0)  # [B,N,1,T]
    return np.transpose(pred, (0, 3, 1, 2)).astype(np.float32)


def _run_maginet_all_splits(scenario: str, train, val, test, adj: np.ndarray, device: torch.device, epochs: int):
    import os

    if os.environ.get("LITETRUST_USE_INTERNAL_MAGI_CORE", "0") == "1":
        return _run_internal_mask_aware_core_all_splits(train, val, test, adj, device, epochs, seed=1)
    if not MAGI_ROOT.exists():
        raise FileNotFoundError(f"MagiNet repo not found: {MAGI_ROOT}")
    _prepare_magi_files(MAGI_ROOT, scenario, train, val, test, adj, epochs)
    MagiNet, seed_torch, generate_miss_loader, weight_matrix, unnormalization = _load_maginet_symbols()
    _predict_maginet_split.unnormalization = unnormalization
    cwd = Path.cwd()

    os.chdir(MAGI_ROOT)
    try:
        seed_torch(1)
        train_loader, valid_loader, test_loader, mean, std, A = generate_miss_loader(
            "PEMS08_debug",
            scenario,
            0.5,
            int(train[0].shape[1]),
            16,
            16,
            16,
        )
        train_eval_loader = DataLoader(train_loader.dataset, batch_size=16, shuffle=False)
        adj_mx = weight_matrix(A)
        model = MagiNet(
            device=device,
            num_nodes=int(train[0].shape[2]),
            seqlen=int(train[0].shape[1]),
            in_channels=1,
            hidden_dim=16,
            st_block=2,
            K=3,
            d_model=64,
            n_heads=2,
            adj_mx=adj_mx,
            learnable=True,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.SmoothL1Loss()
        best_state = None
        best_val = float("inf")
        for _epoch in range(epochs):
            model.train()
            for x, m, y in train_loader:
                x = x.to(device)
                m = m.to(device)
                y = y.to(device)
                loss = loss_fn(model(x, m), y[:, :, :1, :])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            val_maes = []
            with torch.no_grad():
                for x, m, y in valid_loader:
                    x = x.to(device)
                    m = m.to(device)
                    y = y[:, :, :1, :].to(device)
                    x_hat = model(x, m)
                    val_maes.append(float(torch.mean(torch.abs(x_hat - y)).cpu()))
            val_mae = float(np.mean(val_maes))
            if val_mae < best_val:
                best_val = val_mae
                best_state = deepcopy(model.state_dict())
        if best_state is not None:
            model.load_state_dict(best_state)
        return (
            _predict_maginet_split(model, train_eval_loader, device, mean, std),
            _predict_maginet_split(model, valid_loader, device, mean, std),
            _predict_maginet_split(model, test_loader, device, mean, std),
        )
    finally:
        os.chdir(cwd)


def _run_saits_all_splits(train, val, test, device: torch.device, epochs: int):
    if SAITS is None:
        raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    model = SAITS(
        n_steps=train_full.shape[1],
        n_features=train_full.shape[2],
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_k=16,
        d_v=16,
        d_ffn=64,
        dropout=0.1,
        attn_dropout=0.1,
        batch_size=16,
        epochs=epochs,
        patience=None,
        device=device,
        verbose=False,
    )
    train_dict = {"X": np.where(train_mask > 0.5, train_obs, np.nan).reshape(train_obs.shape[0], train_obs.shape[1], train_obs.shape[2])}
    val_dict = {
        "X": np.where(val_mask > 0.5, val_obs, np.nan).reshape(val_obs.shape[0], val_obs.shape[1], val_obs.shape[2]),
        "X_ori": val_full.reshape(val_full.shape[0], val_full.shape[1], val_full.shape[2]),
    }
    model.fit(train_dict, val_dict)

    def predict(split):
        full, obs, mask = split
        payload = {
            "X": np.where(mask > 0.5, obs, np.nan).reshape(obs.shape[0], obs.shape[1], obs.shape[2]),
            "X_ori": full.reshape(full.shape[0], full.shape[1], full.shape[2]),
        }
        pred = model.predict(payload)
        pred = pred["imputation"] if isinstance(pred, dict) else pred
        return np.asarray(pred, dtype=np.float32).reshape(full.shape)

    return predict(train), predict(val), predict(test)


def _physics_candidate(x_magi: np.ndarray, obs: np.ndarray, mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    x = torch.tensor(x_magi, dtype=torch.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    adj_t = torch.tensor(adj, dtype=torch.float32)
    context = mask_t * obs_t + (1.0 - mask_t) * x
    neigh = torch.einsum("nm,btmc->btnc", adj_t, context)
    temporal = torch.zeros_like(x)
    temporal[:, 1:-1] = 0.5 * (x[:, :-2] + x[:, 2:]) - x[:, 1:-1]
    temporal[:, 0] = x[:, 1] - x[:, 0]
    temporal[:, -1] = x[:, -2] - x[:, -1]
    pred = x + 0.15 * (neigh - x) + 0.10 * temporal
    pred = mask_t * obs_t + (1.0 - mask_t) * pred
    return pred.numpy().astype(np.float32)


def _feature_tensor(x_magi, x_saits, x_phys, x_repair, obs, mask, adj):
    device = torch.device("cpu")
    magi = torch.tensor(x_magi, dtype=torch.float32, device=device)
    saits = torch.tensor(x_saits, dtype=torch.float32, device=device)
    phys = torch.tensor(x_phys, dtype=torch.float32, device=device)
    repair = torch.tensor(x_repair, dtype=torch.float32, device=device)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    missing = 1.0 - mask_t
    node_missing = 1.0 - mask_t.mean(dim=(1, 3))[:, None, :, None]
    node_missing = node_missing.expand(-1, mask_t.shape[1], -1, -1)
    neighbor_obs = torch.einsum("nm,btmc->btnc", adj_t, mask_t).mean(dim=-1, keepdim=True)
    neighbor_missing = 1.0 - neighbor_obs
    temporal = torch.zeros_like(magi)
    temporal[:, 1:] = torch.abs(obs_t[:, 1:] - obs_t[:, :-1]) * mask_t[:, 1:] * mask_t[:, :-1]
    temporal = _batch_rank(temporal)
    neigh_obs = torch.einsum("nm,btmc->btnc", adj_t, obs_t)
    spatial = _batch_rank(torch.abs(obs_t - neigh_obs) * mask_t)
    residual = torch.zeros_like(magi)
    residual[:, 1:] = magi[:, 1:] - magi[:, :-1] + (magi[:, :-1] - torch.einsum("nm,btmc->btnc", adj_t, magi[:, :-1]))
    residual_rank = _batch_rank(torch.abs(residual))
    return torch.cat(
        [
            magi,
            saits,
            phys,
            repair,
            torch.abs(magi - saits),
            torch.abs(magi - phys),
            torch.abs(saits - phys),
            missing,
            node_missing,
            neighbor_missing,
            temporal,
            spatial,
            residual_rank,
        ],
        dim=-1,
    )


class ReliabilityRepairNet(nn.Module):
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(features), dim=-1)


def _reliability_features(x_magi, x_saits, x_phys, obs, mask, adj) -> torch.Tensor:
    magi = torch.tensor(x_magi, dtype=torch.float32)
    saits = torch.tensor(x_saits, dtype=torch.float32)
    phys = torch.tensor(x_phys, dtype=torch.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    adj_t = torch.tensor(adj, dtype=torch.float32)
    context = mask_t * obs_t + (1.0 - mask_t) * saits
    neigh = torch.einsum("nm,btmc->btnc", adj_t, context)
    temporal = torch.zeros_like(saits)
    temporal[:, 1:-1] = 0.5 * (saits[:, :-2] + saits[:, 2:]) - saits[:, 1:-1]
    temporal[:, 0] = saits[:, 1] - saits[:, 0]
    temporal[:, -1] = saits[:, -2] - saits[:, -1]
    missing = 1.0 - mask_t
    node_missing = 1.0 - mask_t.mean(dim=(1, 3))[:, None, :, None]
    node_missing = node_missing.expand(-1, mask_t.shape[1], -1, -1)
    neighbor_obs = torch.einsum("nm,btmc->btnc", adj_t, mask_t).mean(dim=-1, keepdim=True)
    neighbor_missing = 1.0 - neighbor_obs
    return torch.cat(
        [
            saits,
            magi,
            phys,
            neigh,
            temporal,
            torch.abs(saits - neigh),
            torch.abs(saits - magi),
            missing,
            node_missing,
            neighbor_missing,
        ],
        dim=-1,
    )


def _local_utility_targets(candidates: torch.Tensor, target: torch.Tensor, temperature: float = 0.08) -> tuple[torch.Tensor, torch.Tensor]:
    magi_err = torch.abs(candidates[..., 0:1] - target)
    local_err = torch.abs(candidates[..., 1:] - target)
    best_local_err = torch.min(local_err, dim=-1, keepdim=True).values
    utility_target = torch.sigmoid((magi_err - best_local_err) / max(temperature, 1e-4)).detach()
    local_target = torch.softmax(-local_err.detach() / max(temperature, 1e-4), dim=-1)
    return utility_target, local_target


def _masked_mae_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    denom = np.clip(mask.sum(), 1.0, None)
    return float((np.abs(pred - target) * mask).sum() / denom)


def _train_reliability_candidate(train_pack, val_pack, test_pack, adj: np.ndarray, epochs: int = 180):
    train_full, train_obs, train_mask, train_magi, train_saits, train_phys = train_pack
    val_full, val_obs, val_mask, val_magi, val_saits, val_phys = val_pack
    test_full, test_obs, test_mask, test_magi, test_saits, test_phys = test_pack
    model = ReliabilityRepairNet(feature_dim=10, hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = _reliability_features(train_magi, train_saits, train_phys, train_obs, train_mask, adj)
    y_train = torch.tensor(train_full, dtype=torch.float32)
    saits_train = torch.tensor(train_saits, dtype=torch.float32)
    target_mask = torch.tensor(1.0 - train_mask, dtype=torch.float32)
    node_missing = x_train[..., 8:9]
    neighbor_missing = x_train[..., 9:10]
    train_weight = target_mask * (1.0 + 4.0 * _node_failure_signal(node_missing, neighbor_missing, temperature=0.2))

    x_val = _reliability_features(val_magi, val_saits, val_phys, val_obs, val_mask, adj)
    y_val = torch.tensor(val_full, dtype=torch.float32)
    saits_val = torch.tensor(val_saits, dtype=torch.float32)
    val_target_mask = torch.tensor(1.0 - val_mask, dtype=torch.float32)
    best_state = None
    best_val = float("inf")
    batch_size = 16
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0])
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            weights = model(x_train[idx])
            neigh = x_train[idx][..., 3:4]
            pred = (
                weights[..., 0:1] * saits_train[idx]
                + weights[..., 1:2] * neigh
                + weights[..., 2:3] * torch.tensor(train_phys, dtype=torch.float32)[idx]
            )
            weight = train_weight[idx]
            target = y_train[idx]
            abs_err = torch.abs(pred - target)
            saits_err = torch.abs(saits_train[idx].detach() - target)
            neigh_err = torch.abs(neigh.detach() - target)
            phys_err = torch.abs(torch.tensor(train_phys, dtype=torch.float32)[idx].detach() - target)
            data_loss = torch.sum(abs_err * weight) / weight.sum().clamp_min(1.0)
            utility_target = torch.softmax(
                -torch.cat([saits_err, neigh_err, phys_err], dim=-1).detach() / 0.05,
                dim=-1,
            )
            gate_loss = torch.sum(
                nn.functional.binary_cross_entropy(weights.clamp(1e-4, 1.0 - 1e-4), utility_target, reduction="none") * weight
            ) / weight.sum().clamp_min(1.0)
            harm_loss = torch.sum(torch.relu(abs_err - torch.minimum(saits_err, torch.minimum(neigh_err, phys_err))) * weight) / weight.sum().clamp_min(1.0)
            loss = data_loss + 0.25 * gate_loss + 0.15 * harm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_weights = model(x_val)
            val_neigh = x_val[..., 3:4]
            val_pred = (
                val_weights[..., 0:1] * saits_val
                + val_weights[..., 1:2] * val_neigh
                + val_weights[..., 2:3] * torch.tensor(val_phys, dtype=torch.float32)
            )
            val_mae = float((torch.abs(val_pred - y_val) * val_target_mask).sum() / val_target_mask.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(split_full, split_obs, split_mask, split_magi, split_saits, split_phys):
        features = _reliability_features(split_magi, split_saits, split_phys, split_obs, split_mask, adj)
        with torch.no_grad():
            weights = model(features)
            raw = (
                weights[..., 0:1] * torch.tensor(split_saits, dtype=torch.float32)
                + weights[..., 1:2] * features[..., 3:4]
                + weights[..., 2:3] * torch.tensor(split_phys, dtype=torch.float32)
            )
        raw = raw.numpy().astype(np.float32)
        split_mask_np = split_mask.astype(np.float32)
        return split_mask_np * split_obs + (1.0 - split_mask_np) * raw

    raw_train = predict(train_full, train_obs, train_mask, train_magi, train_saits, train_phys)
    raw_val = predict(val_full, val_obs, val_mask, val_magi, val_saits, val_phys)
    raw_test = predict(test_full, test_obs, test_mask, test_magi, test_saits, test_phys)
    train_node_missing = _reliability_features(train_magi, train_saits, train_phys, train_obs, train_mask, adj)[..., 8:9].numpy().astype(np.float32)
    val_node_missing = _reliability_features(val_magi, val_saits, val_phys, val_obs, val_mask, adj)[..., 8:9].numpy().astype(np.float32)
    test_node_missing = _reliability_features(test_magi, test_saits, test_phys, test_obs, test_mask, adj)[..., 8:9].numpy().astype(np.float32)
    train_neighbor_missing = _reliability_features(train_magi, train_saits, train_phys, train_obs, train_mask, adj)[..., 9:10].numpy().astype(np.float32)
    val_neighbor_missing = _reliability_features(val_magi, val_saits, val_phys, val_obs, val_mask, adj)[..., 9:10].numpy().astype(np.float32)
    test_neighbor_missing = _reliability_features(test_magi, test_saits, test_phys, test_obs, test_mask, adj)[..., 9:10].numpy().astype(np.float32)
    train_reliability = 1.0 / (1.0 + np.exp(-((train_node_missing - train_neighbor_missing) / 0.2)))
    val_reliability = 1.0 / (1.0 + np.exp(-((val_node_missing - val_neighbor_missing) / 0.2)))
    test_reliability = 1.0 / (1.0 + np.exp(-((test_node_missing - test_neighbor_missing) / 0.2)))
    best_alpha = 0.0
    best_mode = "saits"
    best_combo = (1.0, 0.0, 0.0, 0.0)
    best_blend_val = _masked_mae_np(val_saits, val_full, 1.0 - val_mask)
    for alpha in np.linspace(0.0, 1.0, 21):
        blended_val = alpha * raw_val + (1.0 - alpha) * val_saits
        mae = _masked_mae_np(blended_val, val_full, 1.0 - val_mask)
        if mae < best_blend_val:
            best_blend_val = mae
            best_alpha = float(alpha)
            best_mode = "learned_raw"

    combo_grid = np.linspace(0.0, 1.0, 11)
    for w_saits in combo_grid:
        for w_neigh in combo_grid:
            for w_phys in combo_grid:
                w_magi = 1.0 - w_saits - w_neigh - w_phys
                if w_magi < -1e-6:
                    continue
                combo_val = (
                    w_saits * val_saits
                    + w_neigh * val_neigh
                    + w_phys * val_phys
                    + max(w_magi, 0.0) * val_magi
                )
                mae = _masked_mae_np(combo_val, val_full, 1.0 - val_mask)
                if mae < best_blend_val:
                    best_blend_val = mae
                    best_mode = "convex_context"
                    best_combo = (float(w_saits), float(w_neigh), float(w_phys), float(max(w_magi, 0.0)))

    for raw_alpha in np.linspace(0.0, 1.0, 11):
        for context_alpha in np.linspace(0.0, 1.0, 11):
            context_val = context_alpha * val_neigh + (1.0 - context_alpha) * val_saits
            gated_val = (
                (1.0 - val_reliability) * (raw_alpha * raw_val + (1.0 - raw_alpha) * val_saits)
                + val_reliability * context_val
            )
            mae = _masked_mae_np(gated_val, val_full, 1.0 - val_mask)
            if mae < best_blend_val:
                best_blend_val = mae
                best_mode = "reliability_context"
                best_alpha = float(raw_alpha)
                best_combo = (float(1.0 - context_alpha), float(context_alpha), 0.0, 0.0)

    if best_mode == "convex_context":
        w_saits, w_neigh, w_phys, w_magi = best_combo
        train_repair = w_saits * train_saits + w_neigh * train_neigh + w_phys * train_phys + w_magi * train_magi
        val_repair = w_saits * val_saits + w_neigh * val_neigh + w_phys * val_phys + w_magi * val_magi
        test_repair = w_saits * test_saits + w_neigh * test_neigh + w_phys * test_phys + w_magi * test_magi
    elif best_mode == "reliability_context":
        w_saits, w_neigh, _w_phys, _w_magi = best_combo
        train_context = w_saits * train_saits + w_neigh * train_neigh
        val_context = w_saits * val_saits + w_neigh * val_neigh
        test_context = w_saits * test_saits + w_neigh * test_neigh
        train_raw_blend = best_alpha * raw_train + (1.0 - best_alpha) * train_saits
        val_raw_blend = best_alpha * raw_val + (1.0 - best_alpha) * val_saits
        test_raw_blend = best_alpha * raw_test + (1.0 - best_alpha) * test_saits
        train_repair = (1.0 - train_reliability) * train_raw_blend + train_reliability * train_context
        val_repair = (1.0 - val_reliability) * val_raw_blend + val_reliability * val_context
        test_repair = (1.0 - test_reliability) * test_raw_blend + test_reliability * test_context
    else:
        train_repair = best_alpha * raw_train + (1.0 - best_alpha) * train_saits
        val_repair = best_alpha * raw_val + (1.0 - best_alpha) * val_saits
        test_repair = best_alpha * raw_test + (1.0 - best_alpha) * test_saits
    meta = {
        "alpha": best_alpha,
        "mode": best_mode,
        "combo_saits": best_combo[0],
        "combo_neigh": best_combo[1],
        "combo_phys": best_combo[2],
        "combo_magi": best_combo[3],
        "val_mae": best_blend_val,
    }
    return train_repair.astype(np.float32), val_repair.astype(np.float32), test_repair.astype(np.float32), meta


def _train_candidate_router(train_pack, val_pack, test_pack, adj: np.ndarray, epochs: int = 300):
    train_full, train_obs, train_mask, train_magi, train_saits, train_phys, train_repair = train_pack
    val_full, val_obs, val_mask, val_magi, val_saits, val_phys, val_repair = val_pack
    test_full, test_obs, test_mask, test_magi, test_saits, test_phys, test_repair = test_pack
    model = CandidateRouter(feature_dim=13, hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = _feature_tensor(train_magi, train_saits, train_phys, train_repair, train_obs, train_mask, adj)
    y_train = torch.tensor(train_full, dtype=torch.float32)
    mask_train = torch.tensor(1.0 - train_mask, dtype=torch.float32)
    cand_train = torch.cat(
        [
            torch.tensor(train_magi, dtype=torch.float32),
            torch.tensor(train_saits, dtype=torch.float32),
            torch.tensor(train_phys, dtype=torch.float32),
            torch.tensor(train_repair, dtype=torch.float32),
        ],
        dim=-1,
    )
    utility_target_train, local_target_train = _local_utility_targets(cand_train, y_train)
    x_val = _feature_tensor(val_magi, val_saits, val_phys, val_repair, val_obs, val_mask, adj)
    y_val = torch.tensor(val_full, dtype=torch.float32)
    mask_val = torch.tensor(1.0 - val_mask, dtype=torch.float32)
    cand_val = torch.cat(
        [
            torch.tensor(val_magi, dtype=torch.float32),
            torch.tensor(val_saits, dtype=torch.float32),
            torch.tensor(val_phys, dtype=torch.float32),
            torch.tensor(val_repair, dtype=torch.float32),
        ],
        dim=-1,
    )
    utility_target_val, local_target_val = _local_utility_targets(cand_val, y_val)
    best_state = None
    best_val = float("inf")
    batch_size = 16
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0])
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            weight, utility, local_probs = model(x_train[idx])
            pred = torch.sum(weight * cand_train[idx], dim=-1, keepdim=True)
            target = y_train[idx]
            target_mask = mask_train[idx]
            utility_target = utility_target_train[idx]
            local_target = local_target_train[idx]
            err = torch.abs(cand_train[idx] - target)
            oracle = torch.softmax(-err.detach() / 0.08, dim=-1)
            gate_mask = target_mask
            data_loss = torch.sum(torch.abs(pred - target) * target_mask) / target_mask.sum().clamp_min(1.0)
            utility_loss = torch.sum(
                nn.functional.binary_cross_entropy(weight.clamp(1e-4, 1.0 - 1e-4), oracle, reduction="none") * gate_mask
            ) / gate_mask.sum().clamp_min(1.0)
            utility_gate_loss = torch.sum(
                nn.functional.binary_cross_entropy(utility.clamp(1e-4, 1.0 - 1e-4), utility_target, reduction="none") * gate_mask
            ) / gate_mask.sum().clamp_min(1.0)
            local_loss = torch.sum(
                -(local_target * torch.log(local_probs.clamp_min(1e-6))).sum(dim=-1, keepdim=True) * gate_mask
            ) / gate_mask.sum().clamp_min(1.0)
            harm = torch.relu(torch.abs(pred - target) - torch.min(err, dim=-1, keepdim=True).values)
            harm_loss = torch.sum(harm * target_mask) / target_mask.sum().clamp_min(1.0)
            loss = data_loss + 0.12 * utility_loss + 0.12 * utility_gate_loss + 0.10 * local_loss + 0.15 * harm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_weight, val_utility, val_local = model(x_val)
            val_pred = torch.sum(val_weight * cand_val, dim=-1, keepdim=True)
            val_mae = float((torch.abs(val_pred - y_val) * mask_val).sum() / mask_val.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_weights, val_utility, val_local_probs = model(x_val)
        val_router_pred = torch.sum(val_weights * cand_val, dim=-1, keepdim=True).numpy()
        x_test = _feature_tensor(test_magi, test_saits, test_phys, test_repair, test_obs, test_mask, adj)
        cand_test = torch.cat(
            [
                torch.tensor(test_magi, dtype=torch.float32),
                torch.tensor(test_saits, dtype=torch.float32),
                torch.tensor(test_phys, dtype=torch.float32),
                torch.tensor(test_repair, dtype=torch.float32),
            ],
            dim=-1,
        )
        weights, utility, local_probs = model(x_test)
        router_pred = torch.sum(weights * cand_test, dim=-1, keepdim=True).numpy()
    val_candidates = {
        "router": val_router_pred,
        "magi": val_magi,
        "saits": val_saits,
        "phys": val_phys,
        "repair": val_repair,
    }
    test_candidates = {
        "router": router_pred,
        "magi": test_magi,
        "saits": test_saits,
        "phys": test_phys,
        "repair": test_repair,
    }
    selected_output = "router"
    selected_val_mae = float("inf")
    for name, candidate in val_candidates.items():
        mae = _masked_mae_np(candidate, val_full, 1.0 - val_mask)
        if mae < selected_val_mae:
            selected_val_mae = mae
            selected_output = name
    pred = test_candidates[selected_output]
    metrics = compute_metrics(pred, test_full, 1.0 - test_mask)
    metrics.update(
        {
            "model": "StrongCandidateFusion",
            "selected_output": selected_output,
            "selected_val_mae": selected_val_mae,
            "magi_weight_mean": float(weights[..., 0:1].mean()),
            "saits_weight_mean": float(weights[..., 1:2].mean()),
            "phys_weight_mean": float(weights[..., 2:3].mean()),
            "repair_weight_mean": float(weights[..., 3:4].mean()),
            "utility_mean": float(utility.mean()),
            "local_saits_mean": float(local_probs[..., 0:1].mean()),
            "local_phys_mean": float(local_probs[..., 1:2].mean()),
            "local_repair_mean": float(local_probs[..., 2:3].mean()),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--router-epochs", type=int, default=300)
    parser.add_argument("--repair-epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--output-dir", default="C:/tmp/strong_candidate_fusion_flow_quick")
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, val_x, test_x, adj, _scaler, _metadata = _load_flow_splits(args.seed)
    adj_np = np.asarray(adj, dtype=np.float32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in args.scenarios:
        print(f"running strong candidate fusion {scenario}", file=sys.stderr, flush=True)
        train_obs, train_mask = _scenario_data(train_x, adj_np, scenario, args.seed)
        val_obs, val_mask = _scenario_data(val_x, adj_np, scenario, args.seed + 11)
        test_obs, test_mask = _scenario_data(test_x, adj_np, scenario, args.seed + 29)
        train = (train_x, train_obs, train_mask)
        val = (val_x, val_obs, val_mask)
        test = (test_x, test_obs, test_mask)
        magi_train, magi_val, magi_test = _run_maginet_all_splits(scenario, train, val, test, adj_np, device, args.epochs)
        saits_train, saits_val, saits_test = _run_saits_all_splits(train, val, test, device, args.epochs)
        phys_train = _physics_candidate(magi_train, train_obs, train_mask, adj_np)
        phys_val = _physics_candidate(magi_val, val_obs, val_mask, adj_np)
        phys_test = _physics_candidate(magi_test, test_obs, test_mask, adj_np)
        repair_train, repair_val, repair_test, repair_meta = _train_reliability_candidate(
            (train_x, train_obs, train_mask, magi_train, saits_train, phys_train),
            (val_x, val_obs, val_mask, magi_val, saits_val, phys_val),
            (test_x, test_obs, test_mask, magi_test, saits_test, phys_test),
            adj_np,
            epochs=args.repair_epochs,
        )
        rows.append({"scenario": scenario, "model": "MagiNet", **compute_metrics(magi_test, test_x, 1.0 - test_mask)})
        rows.append({"scenario": scenario, "model": "SAITS", **compute_metrics(saits_test, test_x, 1.0 - test_mask)})
        rows.append({"scenario": scenario, "model": "PhysicsFromMagi", **compute_metrics(phys_test, test_x, 1.0 - test_mask)})
        rows.append(
            {
                "scenario": scenario,
                "model": "ReliabilityRepair",
                **compute_metrics(repair_test, test_x, 1.0 - test_mask),
                "repair_alpha": repair_meta["alpha"],
                "repair_mode": repair_meta["mode"],
                "repair_combo_saits": repair_meta["combo_saits"],
                "repair_combo_neigh": repair_meta["combo_neigh"],
                "repair_combo_phys": repair_meta["combo_phys"],
                "repair_combo_magi": repair_meta["combo_magi"],
                "repair_val_mae": repair_meta["val_mae"],
            }
        )
        fusion = _train_candidate_router(
            (train_x, train_obs, train_mask, magi_train, saits_train, phys_train, repair_train),
            (val_x, val_obs, val_mask, magi_val, saits_val, phys_val, repair_val),
            (test_x, test_obs, test_mask, magi_test, saits_test, phys_test, repair_test),
            adj_np,
            epochs=args.router_epochs,
        )
        rows.append({"scenario": scenario, **fusion})

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Strong Candidate Fusion Flow Quick\n\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- expert epochs: {args.epochs}\n")
        f.write(f"- repair epochs: {args.repair_epochs}\n")
        f.write(f"- router epochs: {args.router_epochs}\n\n")
        for scenario in args.scenarios:
            f.write(f"## {scenario}\n\n")
            f.write("| Model | masked MAE | RMSE | MAPE | Magi w | SAITS w | Phys w | Repair w | Utility | Local SAITS | Local Phys | Local Repair | Repair alpha |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in [r for r in rows if r["scenario"] == scenario]:
                f.write(
                    f"| {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} | "
                    f"{row.get('magi_weight_mean', float('nan')):.6f} | "
                    f"{row.get('saits_weight_mean', float('nan')):.6f} | "
                    f"{row.get('phys_weight_mean', float('nan')):.6f} | "
                    f"{row.get('repair_weight_mean', float('nan')):.6f} | "
                    f"{row.get('utility_mean', float('nan')):.6f} | "
                    f"{row.get('local_saits_mean', float('nan')):.6f} | "
                    f"{row.get('local_phys_mean', float('nan')):.6f} | "
                    f"{row.get('local_repair_mean', float('nan')):.6f} | "
                    f"{row.get('repair_alpha', float('nan')):.6f} |\n"
                )
            f.write("\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
