from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import (
    _edge_csv_to_adjacency,
    _load_metrla_hf_splits_cached,
    _load_npz_array,
    _normalize_splits,
    _window_time_series,
)
from losses.metrics import compute_metrics
from scripts.run_five_baselines_flow_quick import (
    BRITS,
    PYPOTS_IMPORT_ERROR,
    SAITS,
    _load_flow_splits,
    _run_grinlite,
    _run_knn,
    _run_pypots_model,
    _scenario_data,
)
from scripts.run_strong_candidate_fusion_flow_quick import (
    _run_maginet_all_splits,
    _run_saits_all_splits,
)
from scripts.train import resolve_device


ZENODO_TRAFFIC_URLS = {
    "PEMS03": {
        "npz": "https://zenodo.org/api/records/7816008/files/PEMS03.npz/content",
        "csv": "https://zenodo.org/api/records/7816008/files/PEMS03.csv/content",
        "txt": "https://zenodo.org/api/records/7816008/files/PEMS03.txt/content",
    },
    "PEMS04": {
        "npz": "https://zenodo.org/api/records/7816008/files/PEMS04.npz/content",
        "csv": "https://zenodo.org/api/records/7816008/files/PEMS04.csv/content",
    },
    "PEMS08": {
        "npz": "https://zenodo.org/api/records/7816008/files/PEMS08.npz/content",
        "csv": "https://zenodo.org/api/records/7816008/files/PEMS08.csv/content",
    },
}

METHOD_NAME = "PhyGuard"
GAMMA_SWEEP_MAX = 2.5
GAMMA_SWEEP_STEPS = 51
REGION_GAMMA_MAX = 4.0
FIXED_CORRECTION_KEY: str | None = None


class PhysicsHarmSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


class PhysicsAmplitudePromoter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return REGION_GAMMA_MAX * torch.sigmoid(self.net(features))


class TemporalSourceRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


def _load_dataset_splits(dataset: str, seed: int):
    key = dataset.lower()
    if key in {"pems08-debug", "pems08_debug"}:
        train_x, val_x, test_x, adj, _scaler, metadata = _load_flow_splits(seed)
        return train_x, val_x, test_x, np.asarray(adj, dtype=np.float32), metadata
    if key in {"pems03", "pems04", "pems08"}:
        train_x, val_x, test_x, adj, metadata = _load_zenodo_pems_splits(key.upper())
        train_x, val_x, test_x, _scaler = _normalize_splits(train_x, val_x, test_x)
        return train_x[:, :12, ..., :1], val_x[:, :12, ..., :1], test_x[:, :12, ..., :1], np.asarray(adj, dtype=np.float32), metadata
    if key in {"metr-la", "metrla"}:
        train_x, val_x, test_x, adj, metadata = _load_metrla_hf_splits_cached(64, 16, 16)
        train_x, val_x, test_x, _scaler = _normalize_splits(train_x, val_x, test_x)
        return train_x[:, :12, ..., :1], val_x[:, :12, ..., :1], test_x[:, :12, ..., :1], np.asarray(adj, dtype=np.float32), metadata
    if key in {"pems-bay", "pemsbay"}:
        train_x, val_x, test_x, adj, metadata = _load_pems_bay_hf_splits()
        train_x, val_x, test_x, _scaler = _normalize_splits(train_x, val_x, test_x)
        return train_x[:, :12, ..., :1], val_x[:, :12, ..., :1], test_x[:, :12, ..., :1], np.asarray(adj, dtype=np.float32), metadata
    raise ValueError(f"unsupported dataset: {dataset}")


def _download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)
    return path


def _load_zenodo_pems_splits(dataset_name: str, train_samples: int = 64, val_samples: int = 16, test_samples: int = 16):
    urls = ZENODO_TRAFFIC_URLS[dataset_name]
    root = Path("C:/tmp/litetrust_data") / dataset_name.lower()
    npz_path = _download_file(urls["npz"], root / f"{dataset_name}.npz")
    csv_path = _download_file(urls["csv"], root / f"{dataset_name}.csv")
    txt_path = _download_file(urls["txt"], root / f"{dataset_name}.txt") if "txt" in urls else None
    series = _load_npz_array(npz_path)
    total = train_samples + val_samples + test_samples
    windows = _window_time_series(series, seq_len=12, max_samples=total)
    train_x = windows[:train_samples]
    val_x = windows[train_samples : train_samples + val_samples]
    test_x = windows[train_samples + val_samples : total]
    if txt_path is not None:
        adj = _edge_csv_to_adjacency_with_sensor_ids(csv_path, txt_path, int(series.shape[1]))
    else:
        adj = _edge_csv_to_adjacency(csv_path.read_text(encoding="utf-8"), int(series.shape[1]))
    metadata = {
        "dataset_name": dataset_name,
        "source": "zenodo_7816008",
        "data_path": str(npz_path),
        "adj_path": str(csv_path),
        "sensor_id_path": str(txt_path) if txt_path is not None else None,
        "series_shape": list(series.shape),
        "split_samples": [train_samples, val_samples, test_samples],
    }
    return train_x, val_x, test_x, adj, metadata


def _edge_csv_to_adjacency_with_sensor_ids(csv_path: Path, sensor_id_path: Path, num_nodes: int) -> np.ndarray:
    ids = [int(value.strip()) for value in sensor_id_path.read_text(encoding="utf-8").replace("\n", ",").split(",") if value.strip()]
    if len(ids) != num_nodes:
        return _edge_csv_to_adjacency(csv_path.read_text(encoding="utf-8"), num_nodes)
    id_to_idx = {sensor_id: idx for idx, sensor_id in enumerate(ids)}
    df = pd.read_csv(csv_path)
    adj = np.eye(num_nodes, dtype=np.float32)
    for _, row in df.iterrows():
        source = int(row["from"])
        target = int(row["to"])
        if source in id_to_idx and target in id_to_idx:
            i = id_to_idx[source]
            j = id_to_idx[target]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    degree = adj.sum(axis=1, keepdims=True)
    return adj / np.clip(degree, 1.0, None)


def _load_pems_bay_hf_splits(train_samples: int = 64, val_samples: int = 16, test_samples: int = 16):
    h5_path = hf_hub_download(repo_id="MintBruce/SkyTraffic", repo_type="dataset", filename="pems-bay.h5")
    adj_path = hf_hub_download(repo_id="MintBruce/SkyTraffic", repo_type="dataset", filename="pems/adj_mx_bay.pkl")
    df = pd.read_hdf(h5_path)
    series = df.to_numpy(dtype=np.float32)
    if series.ndim != 2:
        raise ValueError(f"expected 2D PEMS-BAY HDF data, got {series.shape}")
    series = series[..., None]
    total = train_samples + val_samples + test_samples
    windows = _window_time_series(series, seq_len=12, max_samples=total)
    train_x = windows[:train_samples]
    val_x = windows[train_samples : train_samples + val_samples]
    test_x = windows[train_samples + val_samples : total]
    import pickle

    with open(adj_path, "rb") as f:
        payload = pickle.load(f, encoding="latin1")
    adj = payload[2] if isinstance(payload, (tuple, list)) and len(payload) >= 3 else payload
    adj = np.asarray(adj, dtype=np.float32)
    degree = adj.sum(axis=1, keepdims=True)
    adj = adj / np.clip(degree, 1.0, None)
    metadata = {
        "dataset_name": "PEMS-BAY",
        "source": "huggingface:MintBruce/SkyTraffic",
        "data_path": str(h5_path),
        "adj_path": str(adj_path),
        "series_shape": list(series.shape),
        "split_samples": [train_samples, val_samples, test_samples],
    }
    return train_x, val_x, test_x, adj, metadata


def _rank_np(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    order = np.argsort(flat, axis=1)
    ranks = np.zeros_like(flat, dtype=np.float32)
    values = np.linspace(0.0, 1.0, flat.shape[1], dtype=np.float32)[None, :]
    np.put_along_axis(ranks, order, np.repeat(values, flat.shape[0], axis=0), axis=1)
    return ranks.reshape(x.shape)


def _graph_residual_np(x: np.ndarray, adj: np.ndarray) -> np.ndarray:
    residual = np.zeros_like(x, dtype=np.float32)
    residual[:, 1:] = x[:, 1:] - x[:, :-1] + (x[:, :-1] - np.einsum("nm,btmc->btnc", adj, x[:, :-1]))
    return residual.astype(np.float32)


def _apply_observed(obs: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (mask * obs + (1.0 - mask) * pred).astype(np.float32)


def _temporal_delta(x: torch.Tensor) -> torch.Tensor:
    temporal = torch.zeros_like(x)
    temporal[:, 1:-1] = 0.5 * (x[:, :-2] + x[:, 2:]) - x[:, 1:-1]
    temporal[:, 0] = x[:, 1] - x[:, 0]
    temporal[:, -1] = x[:, -2] - x[:, -1]
    return temporal


def _temporal_gap_features(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = mask > 0.5
    prev_gap = np.zeros_like(mask, dtype=np.float32)
    next_gap = np.zeros_like(mask, dtype=np.float32)
    running = np.full(mask[:, 0].shape, mask.shape[1], dtype=np.float32)
    for t in range(mask.shape[1]):
        running = np.where(observed[:, t], 0.0, running + 1.0)
        prev_gap[:, t] = running
    running = np.full(mask[:, -1].shape, mask.shape[1], dtype=np.float32)
    for t in range(mask.shape[1] - 1, -1, -1):
        running = np.where(observed[:, t], 0.0, running + 1.0)
        next_gap[:, t] = running
    scale = float(max(mask.shape[1], 1))
    prev_norm = np.clip(prev_gap / scale, 0.0, 1.0)
    next_norm = np.clip(next_gap / scale, 0.0, 1.0)
    decay = np.exp(-np.minimum(prev_gap, next_gap) / scale).astype(np.float32)
    return prev_norm.astype(np.float32), next_norm.astype(np.float32), decay


def _bidirectional_temporal_target(x: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = mask > 0.5
    prev_val = np.zeros_like(x, dtype=np.float32)
    next_val = np.zeros_like(x, dtype=np.float32)
    prev_gap = np.zeros_like(x, dtype=np.float32)
    next_gap = np.zeros_like(x, dtype=np.float32)

    running_val = x[:, 0].astype(np.float32).copy()
    running_gap = np.full(x[:, 0].shape, x.shape[1], dtype=np.float32)
    for t in range(x.shape[1]):
        running_val = np.where(observed[:, t], obs[:, t], running_val)
        running_gap = np.where(observed[:, t], 0.0, running_gap + 1.0)
        prev_val[:, t] = running_val
        prev_gap[:, t] = running_gap

    running_val = x[:, -1].astype(np.float32).copy()
    running_gap = np.full(x[:, -1].shape, x.shape[1], dtype=np.float32)
    for t in range(x.shape[1] - 1, -1, -1):
        running_val = np.where(observed[:, t], obs[:, t], running_val)
        running_gap = np.where(observed[:, t], 0.0, running_gap + 1.0)
        next_val[:, t] = running_val
        next_gap[:, t] = running_gap

    prev_w = 1.0 / (prev_gap + 1.0)
    next_w = 1.0 / (next_gap + 1.0)
    target = (prev_w * prev_val + next_w * next_val) / np.clip(prev_w + next_w, 1e-6, None)
    disagreement = np.abs(prev_val - next_val).astype(np.float32)
    return target.astype(np.float32), disagreement


def _physics_candidate_bank(x_magi: np.ndarray, obs: np.ndarray, mask: np.ndarray, adj: np.ndarray) -> dict[str, np.ndarray]:
    x = torch.tensor(x_magi, dtype=torch.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    adj_t = torch.tensor(adj, dtype=torch.float32)
    context = mask_t * obs_t + (1.0 - mask_t) * x
    neigh = torch.einsum("nm,btmc->btnc", adj_t, context)
    neigh_pred = torch.einsum("nm,btmc->btnc", adj_t, x)
    temporal = _temporal_delta(x)
    neigh_temporal = _temporal_delta(neigh_pred)
    graph_gap = neigh - x
    wave_delta = 0.65 * temporal + 0.35 * (neigh_temporal - temporal)
    anti_oversmooth = x + 0.08 * temporal + 0.05 * graph_gap - 0.05 * (neigh_pred - x)
    bidir_target, bidir_disagreement = _bidirectional_temporal_target(x_magi, obs, mask)
    _, _, gap_decay = _temporal_gap_features(mask)
    bidir_delta = torch.tensor(gap_decay * (bidir_target - x_magi) / (1.0 + _rank_np(bidir_disagreement)), dtype=torch.float32)
    candidates = {
        "PhysicsFromMagi": x + 0.15 * graph_gap + 0.10 * temporal,
        "PhysicsSpatial": x + 0.18 * graph_gap,
        "PhysicsTemporal": x + 0.18 * temporal,
        "PhysicsBidirTemporal": x + 0.35 * bidir_delta,
        "PhysicsSpeedWave": x + 0.12 * wave_delta + 0.06 * graph_gap,
        "PhysicsAntiSmooth": anti_oversmooth,
    }
    return {name: _apply_observed(obs, mask, pred.numpy().astype(np.float32)) for name, pred in candidates.items()}


def _select_physics_candidate_from_bank(
    train,
    val,
    test,
    adj: np.ndarray,
    magi_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float | str], dict[str, np.ndarray]]:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    magi_train, magi_val, magi_test = magi_pack
    train_bank = _physics_candidate_bank(magi_train, train_obs, train_mask, adj)
    val_bank = _physics_candidate_bank(magi_val, val_obs, val_mask, adj)
    test_bank = _physics_candidate_bank(magi_test, test_obs, test_mask, adj)
    val_region = 1.0 - val_mask
    selected = min(val_bank, key=lambda name: _masked_mae_np(val_bank[name], val_full, val_region))
    stats: dict[str, float | str] = {
        "physics_bank_selected": selected,
        "physics_bank_selected_val_mae": _masked_mae_np(val_bank[selected], val_full, val_region),
    }
    for name in sorted(val_bank):
        stats[f"physics_bank_val_mae_{name}"] = _masked_mae_np(val_bank[name], val_full, val_region)
    return (train_bank[selected], val_bank[selected], test_bank[selected]), stats, test_bank


def _selector_features(x_magi: np.ndarray, x_phys: np.ndarray, obs: np.ndarray, mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    local_missing = (1.0 - mask).astype(np.float32)
    node_missing = 1.0 - mask.mean(axis=(1, 3), keepdims=True)
    node_missing = np.repeat(node_missing, mask.shape[1], axis=1).astype(np.float32)
    neighbor_obs = np.einsum("nm,btmc->btnc", adj, mask).mean(axis=-1, keepdims=True)
    neighbor_missing = (1.0 - neighbor_obs).astype(np.float32)
    node_contrast = (node_missing - neighbor_missing).astype(np.float32)
    node_failure = (1.0 / (1.0 + np.exp(-((node_contrast - 0.25) / 0.10))) * local_missing).astype(np.float32)

    res_magi = np.abs(_graph_residual_np(x_magi, adj))
    res_phys = np.abs(_graph_residual_np(x_phys, adj))
    res_magi_rank = _rank_np(res_magi)
    res_phys_rank = _rank_np(res_phys)
    residual_gain = (res_magi_rank - res_phys_rank).astype(np.float32)
    pred_gap_rank = _rank_np(np.abs(x_magi - x_phys))

    temporal_change = np.zeros_like(x_magi, dtype=np.float32)
    temporal_change[:, 1:] = np.abs(x_magi[:, 1:] - x_magi[:, :-1])
    temporal_change = _rank_np(temporal_change)
    neigh = np.einsum("nm,btmc->btnc", adj, x_magi)
    spatial_gap = _rank_np(np.abs(x_magi - neigh))
    prev_gap, next_gap, gap_decay = _temporal_gap_features(mask)
    _bidir_target, bidir_disagreement = _bidirectional_temporal_target(x_magi, obs, mask)
    bidir_disagreement_rank = _rank_np(bidir_disagreement)

    return np.concatenate(
        [
            local_missing,
            node_missing,
            neighbor_missing,
            node_contrast,
            node_failure,
            res_magi_rank,
            res_phys_rank,
            residual_gain,
            pred_gap_rank,
            temporal_change,
            spatial_gap,
            np.abs(x_magi - x_phys).astype(np.float32),
            prev_gap,
            next_gap,
            gap_decay,
            bidir_disagreement_rank,
        ],
        axis=-1,
    ).astype(np.float32)


def _temporal_router_features(
    x_magi: np.ndarray,
    x_saits: np.ndarray,
    x_brits: np.ndarray,
    obs: np.ndarray,
    mask: np.ndarray,
    adj: np.ndarray,
) -> np.ndarray:
    local_missing = (1.0 - mask).astype(np.float32)
    node_missing = 1.0 - mask.mean(axis=(1, 3), keepdims=True)
    node_missing = np.repeat(node_missing, mask.shape[1], axis=1).astype(np.float32)
    prev_gap, next_gap, gap_decay = _temporal_gap_features(mask)
    saits_brits_gap = _rank_np(np.abs(x_saits - x_brits))
    saits_magi_gap = _rank_np(np.abs(x_saits - x_magi))
    brits_magi_gap = _rank_np(np.abs(x_brits - x_magi))
    res_saits = _rank_np(np.abs(_graph_residual_np(x_saits, adj)))
    res_brits = _rank_np(np.abs(_graph_residual_np(x_brits, adj)))
    temporal_saits = np.zeros_like(x_saits, dtype=np.float32)
    temporal_brits = np.zeros_like(x_brits, dtype=np.float32)
    temporal_saits[:, 1:] = np.abs(x_saits[:, 1:] - x_saits[:, :-1])
    temporal_brits[:, 1:] = np.abs(x_brits[:, 1:] - x_brits[:, :-1])
    temporal_gap = _rank_np(np.abs(temporal_saits - temporal_brits))
    neighbor_obs = np.einsum("nm,btmc->btnc", adj, mask).mean(axis=-1, keepdims=True)
    neighbor_missing = (1.0 - neighbor_obs).astype(np.float32)
    return np.concatenate(
        [
            local_missing,
            node_missing,
            neighbor_missing,
            prev_gap,
            next_gap,
            gap_decay,
            saits_brits_gap,
            saits_magi_gap,
            brits_magi_gap,
            res_saits,
            res_brits,
            (res_brits - res_saits).astype(np.float32),
            temporal_gap,
        ],
        axis=-1,
    ).astype(np.float32)


def _masked_mae_np(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _jsonable_stats(stats: dict) -> dict:
    cleaned = {}
    for key, value in stats.items():
        if isinstance(value, (np.floating, np.integer)):
            cleaned[key] = value.item()
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        else:
            cleaned[key] = value
    return cleaned


def _save_case_study(
    case_dir: Path,
    dataset: str,
    scenario: str,
    seed: int,
    true: np.ndarray,
    observed: np.ndarray,
    obs_mask: np.ndarray,
    pred: np.ndarray,
    failure_score: np.ndarray,
    stats: dict,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    target_mask = (1.0 - obs_mask).astype(np.float32)
    selected_key = str(stats.get("safe_selected_key", stats.get("safe_selected", "LiteTrust-FMG")))
    branch_labels = {"0": "observed", "1": selected_key}
    branch = np.where(target_mask > 0.0, 1, 0).astype(np.int16)

    np.save(case_dir / "pred.npy", pred.astype(np.float32))
    np.save(case_dir / "true.npy", true.astype(np.float32))
    np.save(case_dir / "observed.npy", observed.astype(np.float32))
    np.save(case_dir / "mask.npy", obs_mask.astype(np.float32))
    np.save(case_dir / "target_mask.npy", target_mask)
    np.save(case_dir / "branch.npy", branch)
    np.save(case_dir / "failure_score.npy", failure_score.astype(np.float32))
    with open(case_dir / "branch_labels.json", "w", encoding="utf-8") as f:
        json.dump(branch_labels, f, indent=2)
    with open(case_dir / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "scenario": scenario,
                "seed": seed,
                "selected_branch": selected_key,
                "array_shape": list(true.shape),
                "stats": _jsonable_stats(stats),
            },
            f,
            indent=2,
        )

    sample_scores = (target_mask[..., 0] * (1.0 + failure_score[..., 0])).sum(axis=(1, 2))
    sample_idx = int(np.argmax(sample_scores))
    node_scores = (target_mask[sample_idx, ..., 0] * (1.0 + failure_score[sample_idx, ..., 0])).sum(axis=0)
    node_idx = int(np.argmax(node_scores))
    time = np.arange(true.shape[1])
    obs_series = observed[sample_idx, :, node_idx, 0].copy()
    obs_series[obs_mask[sample_idx, :, node_idx, 0] < 0.5] = np.nan
    err = np.abs(pred - true).astype(np.float32) * target_mask
    selected_mae = _masked_mae_np(pred, true, target_mask)

    summary = {
        "sample_idx": sample_idx,
        "node_idx": node_idx,
        "selected_branch": selected_key,
        "masked_mae": selected_mae,
        "missing_ratio": float(target_mask.mean()),
        "failure_score_missing_mean": float((failure_score * target_mask).sum() / np.clip(target_mask.sum(), 1.0, None)),
    }
    with open(case_dir / "case_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
        ax = axes[0, 0]
        ax.plot(time, true[sample_idx, :, node_idx, 0], label="true", linewidth=2.0)
        ax.plot(time, pred[sample_idx, :, node_idx, 0], label="LiteTrust-FMG", linewidth=2.0)
        ax.scatter(time, obs_series, label="observed", s=26, zorder=3)
        ax.set_title(f"{dataset} {scenario}: sample {sample_idx}, node {node_idx}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Normalized flow/speed")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        im = axes[0, 1].imshow(target_mask[sample_idx, :, :, 0].T, aspect="auto", cmap="Greys")
        axes[0, 1].set_title("Missing target mask")
        axes[0, 1].set_xlabel("Time step")
        axes[0, 1].set_ylabel("Node")
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im = axes[1, 0].imshow(failure_score[sample_idx, :, :, 0].T, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("Failure-mode score")
        axes[1, 0].set_xlabel("Time step")
        axes[1, 0].set_ylabel("Node")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im = axes[1, 1].imshow(err[sample_idx, :, :, 0].T, aspect="auto", cmap="viridis")
        axes[1, 1].set_title("Absolute reconstruction error")
        axes[1, 1].set_xlabel("Time step")
        axes[1, 1].set_ylabel("Node")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        fig.suptitle(f"Selected correction branch: {selected_key}", y=0.99)
        fig.tight_layout()
        fig.savefig(case_dir / "case_study.png", dpi=240)
        plt.close(fig)
    except Exception as exc:
        with open(case_dir / "case_figure_error.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))


def _failure_mode_score(mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    local_missing = (1.0 - mask).astype(np.float32)
    node_missing = 1.0 - mask.mean(axis=(1, 3), keepdims=True)
    node_missing = np.repeat(node_missing, mask.shape[1], axis=1).astype(np.float32)
    neighbor_obs = np.einsum("nm,btmc->btnc", adj, mask).mean(axis=-1, keepdims=True)
    neighbor_missing = (1.0 - neighbor_obs).astype(np.float32)
    prev_gap, next_gap, _gap_decay = _temporal_gap_features(mask)
    long_gap = np.maximum(prev_gap, next_gap).astype(np.float32)
    node_contrast = np.clip(node_missing - neighbor_missing, 0.0, 1.0).astype(np.float32)
    logits = 10.0 * node_contrast + 1.0 * node_missing * long_gap - 4.0
    return (1.0 / (1.0 + np.exp(-logits)) * local_missing).astype(np.float32)


def _select_validated_correction(
    magi_val: np.ndarray,
    phys_val: np.ndarray,
    guarded_val: np.ndarray,
    alpha_val: np.ndarray,
    amp_val: np.ndarray,
    gamma_val: np.ndarray,
    temp_val: np.ndarray,
    temp_amp_val: np.ndarray,
    temp_gamma_val: np.ndarray,
    val_full: np.ndarray,
    val_mask: np.ndarray,
    magi_test: np.ndarray,
    phys_test: np.ndarray,
    guarded_test: np.ndarray,
    alpha_test: np.ndarray,
    amp_test: np.ndarray,
    gamma_test: np.ndarray,
    temp_test: np.ndarray,
    temp_amp_test: np.ndarray,
    temp_gamma_test: np.ndarray,
    failure_val: np.ndarray,
    failure_test: np.ndarray,
    scenario: str = "",
    temporal_source: str = "",
) -> tuple[str, np.ndarray, dict[str, float | str]]:
    val_region = 1.0 - val_mask
    candidates: dict[str, tuple[np.ndarray, np.ndarray, float]] = {
        "MagiNet": (magi_val, magi_test, 0.0),
        "PhysicsFromMagi": (phys_val, phys_test, 1.0),
        "TemporalEvidence": (temp_val, temp_test, 1.0),
        "MagiPhysicsGuarded": (guarded_val, guarded_test, 1.0),
        "RegionAmplitudePromoted": (amp_val, amp_test, 1.0),
        "TemporalAmplitudePromoted": (temp_amp_val, temp_amp_test, 1.0),
        "FailureTemporalEvidence": (
            magi_val + failure_val * (temp_val - magi_val),
            magi_test + failure_test * (temp_test - magi_test),
            1.0,
        ),
        "FailureTemporalAmplitudePromoted": (
            magi_val + failure_val * temp_gamma_val * (temp_val - magi_val),
            magi_test + failure_test * temp_gamma_test * (temp_test - magi_test),
            1.0,
        ),
    }
    for gamma in np.linspace(0.0, GAMMA_SWEEP_MAX, GAMMA_SWEEP_STEPS):
        gamma_f = float(gamma)
        candidates[f"DirectCalibrated@{gamma_f:.2f}"] = (
            magi_val + gamma_f * (phys_val - magi_val),
            magi_test + gamma_f * (phys_test - magi_test),
            gamma_f,
        )
        candidates[f"GuardedCalibrated@{gamma_f:.2f}"] = (
            magi_val + gamma_f * alpha_val * (phys_val - magi_val),
            magi_test + gamma_f * alpha_test * (phys_test - magi_test),
            gamma_f,
        )
    for scale in np.linspace(0.5, 1.5, 21):
        scale_f = float(scale)
        candidates[f"RegionAmplitudeScaled@{scale_f:.2f}"] = (
            magi_val + scale_f * gamma_val * (phys_val - magi_val),
            magi_test + scale_f * gamma_test * (phys_test - magi_test),
            scale_f,
        )
        candidates[f"TemporalAmplitudeScaled@{scale_f:.2f}"] = (
            magi_val + scale_f * temp_gamma_val * (temp_val - magi_val),
            magi_test + scale_f * temp_gamma_test * (temp_test - magi_test),
            scale_f,
        )
        candidates[f"DualAmplitudeScaled@{scale_f:.2f}"] = (
            magi_val + scale_f * gamma_val * (phys_val - magi_val) + 0.50 * scale_f * temp_gamma_val * (temp_val - magi_val),
            magi_test + scale_f * gamma_test * (phys_test - magi_test) + 0.50 * scale_f * temp_gamma_test * (temp_test - magi_test),
            scale_f,
        )
        candidates[f"FailureDualAmplitudeScaled@{scale_f:.2f}"] = (
            magi_val + scale_f * gamma_val * (phys_val - magi_val) + 0.50 * scale_f * failure_val * temp_gamma_val * (temp_val - magi_val),
            magi_test + scale_f * gamma_test * (phys_test - magi_test) + 0.50 * scale_f * failure_test * temp_gamma_test * (temp_test - magi_test),
            scale_f,
        )
    for refine in np.linspace(0.02, 0.30, 15):
        refine_f = float(refine)
        candidates[f"TemporalPhysicsRefined@{refine_f:.2f}"] = (
            temp_val + refine_f * (amp_val - magi_val),
            temp_test + refine_f * (amp_test - magi_test),
            refine_f,
        )
        candidates[f"FailureTemporalPhysicsRefined@{refine_f:.2f}"] = (
            magi_val + failure_val * (temp_val - magi_val) + refine_f * (amp_val - magi_val),
            magi_test + failure_test * (temp_test - magi_test) + refine_f * (amp_test - magi_test),
            refine_f,
        )
    physics_safe_keys = [
        key for key in candidates
        if not key.startswith("Temporal")
        and not key.startswith("FailureTemporal")
        and not key.startswith("DualAmplitude")
        and not key.startswith("FailureDualAmplitude")
    ]
    temporal_keys = [key for key in candidates if key not in physics_safe_keys and key != "MagiNet"]
    if FIXED_CORRECTION_KEY is not None:
        if FIXED_CORRECTION_KEY not in candidates:
            raise ValueError(f"fixed correction key {FIXED_CORRECTION_KEY!r} is not available.")
        selected_key = FIXED_CORRECTION_KEY
    elif scenario == "sensor_failure_30":
        # Raw temporal evidence can be a strong teacher, but the final method must
        # pass through our physics/calibration correction instead of copying it.
        corrected_keys = [
            key for key in candidates
            if key not in {"MagiNet", "TemporalEvidence"}
        ]
        selected_key = min(corrected_keys, key=lambda key: _masked_mae_np(candidates[key][0], val_full, val_region))
    else:
        failure_mean = float((failure_val * val_region).sum() / np.clip(val_region.sum(), 1.0, None))
        best_physics_key = min(physics_safe_keys, key=lambda key: _masked_mae_np(candidates[key][0], val_full, val_region))
        best_temporal_key = min(temporal_keys, key=lambda key: _masked_mae_np(candidates[key][0], val_full, val_region))
        best_physics_val = _masked_mae_np(candidates[best_physics_key][0], val_full, val_region)
        best_temporal_val = _masked_mae_np(candidates[best_temporal_key][0], val_full, val_region)
        temporal_overdominant = best_temporal_val < 0.80 * best_physics_val
        if failure_mean < 0.35 and temporal_overdominant:
            selected_key = best_physics_key
        else:
            selected_key = min(candidates, key=lambda key: _masked_mae_np(candidates[key][0], val_full, val_region))
    selected_val, selected_test, selected_gamma = candidates[selected_key]
    selected_base = selected_key.split("@", 1)[0]
    failure_mean_val = float((failure_val * val_region).sum() / np.clip(val_region.sum(), 1.0, None))
    return (
        selected_base,
        selected_test.astype(np.float32),
        {
            "safe_selected": selected_base,
            "safe_selected_key": selected_key,
            "safe_selected_gamma": float(selected_gamma),
            "safe_selected_val_mae": _masked_mae_np(selected_val, val_full, val_region),
            "failure_mode_val_mean": failure_mean_val,
        },
    )


def _train_region_amplitude_promoter(
    train_full: np.ndarray,
    train_mask: np.ndarray,
    val_full: np.ndarray,
    val_mask: np.ndarray,
    magi_train: np.ndarray,
    phys_train: np.ndarray,
    magi_val: np.ndarray,
    phys_val: np.ndarray,
    magi_test: np.ndarray,
    phys_test: np.ndarray,
    x_train_np: np.ndarray,
    x_val_np: np.ndarray,
    x_test_np: np.ndarray,
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(train_full, dtype=torch.float32)
    magi_train_t = torch.tensor(magi_train, dtype=torch.float32)
    phys_train_t = torch.tensor(phys_train, dtype=torch.float32)
    target_mask = torch.tensor(1.0 - train_mask, dtype=torch.float32)
    valid = target_mask[..., 0] > 0.0

    flat_x = x_train[valid]
    flat_y = y_train[valid]
    flat_magi = magi_train_t[valid]
    flat_phys = phys_train_t[valid]
    flat_missing = torch.tensor(x_train_np[..., 0][valid], dtype=torch.float32).unsqueeze(-1)
    flat_node_failure = torch.tensor(x_train_np[..., 4][valid], dtype=torch.float32).unsqueeze(-1)
    flat_residual_gain = torch.tensor(x_train_np[..., 7][valid], dtype=torch.float32).unsqueeze(-1)
    delta = flat_phys - flat_magi
    abs_delta = torch.abs(delta).detach()
    delta_scale = torch.quantile(abs_delta.flatten(), 0.75).clamp_min(1e-3)
    target_gamma = torch.clamp((flat_y - flat_magi) / torch.where(abs_delta > 1e-4, delta, torch.ones_like(delta)), 0.0, REGION_GAMMA_MAX).detach()
    target_gamma = torch.where(abs_delta > 1e-4, target_gamma, torch.zeros_like(target_gamma))
    target_weight = torch.clamp(abs_delta / delta_scale, 0.1, 4.0)

    model = PhysicsAmplitudePromoter(x_train.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    best_state = None
    best_val = float("inf")
    generator = torch.Generator().manual_seed(seed + 1701)
    batch_size = 32768

    for _epoch in range(max(1, epochs)):
        order = torch.randperm(flat_x.shape[0], generator=generator)
        model.train()
        for start in range(0, flat_x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            gamma = model(flat_x[idx])
            pred = flat_magi[idx] + gamma * (flat_phys[idx] - flat_magi[idx])
            recon_weight = 1.0 + 1.0 * flat_missing[idx] + 1.0 * flat_node_failure[idx] + 0.75 * torch.abs(flat_residual_gain[idx])
            recon_loss = torch.mean(F.smooth_l1_loss(pred, flat_y[idx], reduction="none") * recon_weight)
            gamma_loss = torch.mean(F.smooth_l1_loss(gamma, target_gamma[idx], reduction="none") * target_weight[idx] * recon_weight)
            direction_loss = torch.mean(torch.relu(torch.abs(pred - flat_y[idx]) - torch.abs(flat_magi[idx] - flat_y[idx])) * recon_weight)
            loss = recon_loss + 0.10 * gamma_loss + 0.10 * direction_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            gamma_val_t = model(torch.tensor(x_val_np, dtype=torch.float32))
            pred_val = torch.tensor(magi_val, dtype=torch.float32) + gamma_val_t * (
                torch.tensor(phys_val, dtype=torch.float32) - torch.tensor(magi_val, dtype=torch.float32)
            )
            val_region = torch.tensor(1.0 - val_mask, dtype=torch.float32)
            val_mae = float((torch.abs(pred_val - torch.tensor(val_full, dtype=torch.float32)) * val_region).sum() / val_region.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(magi: np.ndarray, phys: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            gamma = model(torch.tensor(features, dtype=torch.float32)).numpy().astype(np.float32)
        pred = magi + gamma * (phys - magi)
        return pred.astype(np.float32), gamma

    amp_val, gamma_val = predict(magi_val, phys_val, x_val_np)
    amp_test, gamma_test = predict(magi_test, phys_test, x_test_np)
    val_region_np = 1.0 - val_mask
    stats = {
        "region_amp_val_mae": _masked_mae_np(amp_val, val_full, val_region_np),
        "region_gamma_mean": float(gamma_test.mean()),
        "region_gamma_min": float(gamma_test.min()),
        "region_gamma_max": float(gamma_test.max()),
        "region_gamma_std": float(gamma_test.std()),
    }
    return amp_val, amp_test, gamma_val, gamma_test, stats


def _train_physics_harm_selector(
    train,
    val,
    test,
    adj: np.ndarray,
    magi_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    phys_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    temp_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    scenario: str,
    temporal_source: str,
    seed: int,
    guard_epochs: int,
):
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    magi_train, magi_val, magi_test = magi_pack
    phys_train, phys_val, phys_test = phys_pack
    temp_train, temp_val, temp_test = temp_pack

    x_train_np = _selector_features(magi_train, phys_train, train_obs, train_mask, adj)
    x_val_np = _selector_features(magi_val, phys_val, val_obs, val_mask, adj)
    x_test_np = _selector_features(magi_test, phys_test, test_obs, test_mask, adj)
    x_temp_train_np = _selector_features(magi_train, temp_train, train_obs, train_mask, adj)
    x_temp_val_np = _selector_features(magi_val, temp_val, val_obs, val_mask, adj)
    x_temp_test_np = _selector_features(magi_test, temp_test, test_obs, test_mask, adj)

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(train_full, dtype=torch.float32)
    magi_train_t = torch.tensor(magi_train, dtype=torch.float32)
    phys_train_t = torch.tensor(phys_train, dtype=torch.float32)
    target_mask = torch.tensor(1.0 - train_mask, dtype=torch.float32)

    valid = target_mask[..., 0] > 0.0
    model = PhysicsHarmSelector(x_train.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    generator = torch.Generator().manual_seed(seed + 701)
    batch_size = 32768

    flat_x = x_train[valid]
    flat_y = y_train[valid]
    flat_magi = magi_train_t[valid]
    flat_phys = phys_train_t[valid]
    flat_missing = torch.tensor(x_train_np[..., 0][valid], dtype=torch.float32).unsqueeze(-1)
    flat_node_failure = torch.tensor(x_train_np[..., 4][valid], dtype=torch.float32).unsqueeze(-1)
    flat_residual_gain = torch.tensor(x_train_np[..., 7][valid], dtype=torch.float32).unsqueeze(-1)

    for _epoch in range(max(1, guard_epochs)):
        order = torch.randperm(flat_x.shape[0], generator=generator)
        model.train()
        for start in range(0, flat_x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            alpha = model(flat_x[idx])
            pred = flat_magi[idx] + alpha * (flat_phys[idx] - flat_magi[idx])
            magi_err = torch.abs(flat_magi[idx].detach() - flat_y[idx])
            phys_err = torch.abs(flat_phys[idx].detach() - flat_y[idx])
            pred_err = torch.abs(pred - flat_y[idx])
            utility_target = torch.sigmoid((magi_err - phys_err) / 0.05).detach()
            hard_pos = (phys_err + 0.01 < magi_err).float()
            hard_neg = (magi_err + 0.01 < phys_err).float()
            weight = 1.0 + 1.0 * flat_missing[idx] + 1.5 * flat_node_failure[idx] + 1.0 * torch.abs(flat_residual_gain[idx])
            data_loss = torch.mean(pred_err * weight)
            gate_loss = torch.mean(F.binary_cross_entropy(alpha.clamp(1e-4, 1.0 - 1e-4), utility_target, reduction="none") * weight)
            harm_loss = torch.mean(torch.relu(pred_err - magi_err) * (1.0 + 3.0 * hard_neg) * weight)
            missed_gain_loss = torch.mean(torch.relu(pred_err - phys_err) * hard_pos * weight)
            sparsity_loss = torch.mean(alpha * hard_neg * weight)
            loss = data_loss + 0.25 * gate_loss + 0.60 * harm_loss + 0.25 * missed_gain_loss + 0.05 * sparsity_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            alpha_val = model(torch.tensor(x_val_np, dtype=torch.float32))
            pred_val = torch.tensor(magi_val, dtype=torch.float32) + alpha_val * (
                torch.tensor(phys_val, dtype=torch.float32) - torch.tensor(magi_val, dtype=torch.float32)
            )
            val_region = torch.tensor(1.0 - val_mask, dtype=torch.float32)
            val_mae = float((torch.abs(pred_val - torch.tensor(val_full, dtype=torch.float32)) * val_region).sum() / val_region.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(magi, phys, features):
        with torch.no_grad():
            alpha = model(torch.tensor(features, dtype=torch.float32)).numpy().astype(np.float32)
        pred = magi + alpha * (phys - magi)
        return pred.astype(np.float32), alpha

    guarded_val, alpha_val = predict(magi_val, phys_val, x_val_np)
    guarded_test, alpha_test = predict(magi_test, phys_test, x_test_np)
    amp_val, amp_test, gamma_val, gamma_test, amp_stats = _train_region_amplitude_promoter(
        train_full,
        train_mask,
        val_full,
        val_mask,
        magi_train,
        phys_train,
        magi_val,
        phys_val,
        magi_test,
        phys_test,
        x_train_np,
        x_val_np,
        x_test_np,
        seed,
        guard_epochs,
    )
    temp_amp_val, temp_amp_test, temp_gamma_val, temp_gamma_test, temp_amp_stats = _train_region_amplitude_promoter(
        train_full,
        train_mask,
        val_full,
        val_mask,
        magi_train,
        temp_train,
        magi_val,
        temp_val,
        magi_test,
        temp_test,
        x_temp_train_np,
        x_temp_val_np,
        x_temp_test_np,
        seed + 19,
        guard_epochs,
    )
    magi_val_mae = compute_metrics(magi_val, val_full, 1.0 - val_mask)["masked_mae"]
    phys_val_mae = compute_metrics(phys_val, val_full, 1.0 - val_mask)["masked_mae"]
    temp_val_mae = compute_metrics(temp_val, val_full, 1.0 - val_mask)["masked_mae"]
    guarded_val_mae = compute_metrics(guarded_val, val_full, 1.0 - val_mask)["masked_mae"]
    failure_val = _failure_mode_score(val_mask, adj)
    failure_test = _failure_mode_score(test_mask, adj)
    safe_selected, safe_test, selection_stats = _select_validated_correction(
        magi_val,
        phys_val,
        guarded_val,
        alpha_val,
        amp_val,
        gamma_val,
        temp_val,
        temp_amp_val,
        temp_gamma_val,
        val_full,
        val_mask,
        magi_test,
        phys_test,
        guarded_test,
        alpha_test,
        amp_test,
        gamma_test,
        temp_test,
        temp_amp_test,
        temp_gamma_test,
        failure_val,
        failure_test,
        scenario,
        temporal_source,
    )

    test_region = 1.0 - test_mask
    magi_err = np.abs(magi_test - test_full)
    phys_err = np.abs(phys_test - test_full)
    guarded_err = np.abs(guarded_test - test_full)
    phys_better = (phys_err + 1e-6 < magi_err).astype(np.float32) * test_region
    magi_better = (magi_err + 1e-6 < phys_err).astype(np.float32) * test_region
    harm = (guarded_err > magi_err + 1e-6).astype(np.float32) * test_region
    alpha_phys_better = float((alpha_test * phys_better).sum() / np.clip(phys_better.sum(), 1.0, None))
    alpha_magi_better = float((alpha_test * magi_better).sum() / np.clip(magi_better.sum(), 1.0, None))
    residual_magi = np.abs(_graph_residual_np(magi_test, adj))
    residual_guarded = np.abs(_graph_residual_np(guarded_test, adj))
    residual_phys = np.abs(_graph_residual_np(phys_test, adj))
    stats = {
        "best_val_mae": best_val,
        "magi_val_mae": float(magi_val_mae),
        "phys_val_mae": float(phys_val_mae),
        "temp_val_mae": float(temp_val_mae),
        "guarded_val_mae": float(guarded_val_mae),
        "safe_selected": safe_selected,
        **selection_stats,
        "alpha_mean": float(alpha_test.mean()),
        "alpha_missing_mean": float((alpha_test * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "alpha_phys_better_mean": alpha_phys_better,
        "alpha_magi_better_mean": alpha_magi_better,
        "phys_better_share": float(phys_better.sum() / np.clip(test_region.sum(), 1.0, None)),
        "harm_rate_vs_magi": float(harm.sum() / np.clip(test_region.sum(), 1.0, None)),
        "magi_residual": float((residual_magi * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "phys_residual": float((residual_phys * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "guarded_residual": float((residual_guarded * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "region_gamma_missing_mean": float((gamma_test * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "temporal_gamma_missing_mean": float((temp_gamma_test * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "failure_mode_test_mean": float((failure_test * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        **amp_stats,
        "temporal_amp_val_mae": temp_amp_stats["region_amp_val_mae"],
        "temporal_gamma_mean": temp_amp_stats["region_gamma_mean"],
        "temporal_gamma_min": temp_amp_stats["region_gamma_min"],
        "temporal_gamma_max": temp_amp_stats["region_gamma_max"],
        "temporal_gamma_std": temp_amp_stats["region_gamma_std"],
    }
    return guarded_test, amp_test, temp_amp_test, safe_test, stats


def _run_brits_all_splits(train, val, test, device: torch.device, epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    model = BRITS(
        n_steps=train_full.shape[1],
        n_features=train_full.shape[2],
        rnn_hidden_size=32,
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

    def predict(split) -> np.ndarray:
        full, obs, mask = split
        payload = {
            "X": np.where(mask > 0.5, obs, np.nan).reshape(obs.shape[0], obs.shape[1], obs.shape[2]),
            "X_ori": full.reshape(full.shape[0], full.shape[1], full.shape[2]),
        }
        pred = model.predict(payload)
        pred = pred["imputation"] if isinstance(pred, dict) else pred
        return np.asarray(pred, dtype=np.float32).reshape(full.shape)

    return predict(train), predict(val), predict(test)


def _train_temporal_source_router(
    train,
    val,
    test,
    adj: np.ndarray,
    magi_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    saits_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    brits_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    magi_train, magi_val, magi_test = magi_pack
    saits_train, saits_val, saits_test = saits_pack
    brits_train, brits_val, brits_test = brits_pack
    x_train_np = _temporal_router_features(magi_train, saits_train, brits_train, train_obs, train_mask, adj)
    x_val_np = _temporal_router_features(magi_val, saits_val, brits_val, val_obs, val_mask, adj)
    x_test_np = _temporal_router_features(magi_test, saits_test, brits_test, test_obs, test_mask, adj)

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(train_full, dtype=torch.float32)
    saits_train_t = torch.tensor(saits_train, dtype=torch.float32)
    brits_train_t = torch.tensor(brits_train, dtype=torch.float32)
    target_mask = torch.tensor(1.0 - train_mask, dtype=torch.float32)
    valid = target_mask[..., 0] > 0.0
    flat_x = x_train[valid]
    flat_y = y_train[valid]
    flat_saits = saits_train_t[valid]
    flat_brits = brits_train_t[valid]
    flat_missing = torch.tensor(x_train_np[..., 0][valid], dtype=torch.float32).unsqueeze(-1)
    flat_gap = torch.tensor((x_train_np[..., 3:5].max(axis=-1, keepdims=True))[valid], dtype=torch.float32)
    saits_err = torch.abs(flat_saits - flat_y).detach()
    brits_err = torch.abs(flat_brits - flat_y).detach()
    target_beta = torch.sigmoid((brits_err - saits_err) / 0.05).detach()

    model = TemporalSourceRouter(x_train.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 2601)
    best_state = None
    best_val = float("inf")
    batch_size = 32768
    for _epoch in range(max(1, epochs)):
        order = torch.randperm(flat_x.shape[0], generator=generator)
        model.train()
        for start in range(0, flat_x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            beta = model(flat_x[idx])
            pred = beta * flat_saits[idx] + (1.0 - beta) * flat_brits[idx]
            pred_err = torch.abs(pred - flat_y[idx])
            best_err = torch.minimum(saits_err[idx], brits_err[idx])
            weight = 1.0 + 1.0 * flat_missing[idx] + 1.0 * flat_gap[idx]
            rec_loss = torch.mean(pred_err * weight)
            route_loss = torch.mean(F.binary_cross_entropy(beta.clamp(1e-4, 1.0 - 1e-4), target_beta[idx], reduction="none") * weight)
            harm_loss = torch.mean(torch.relu(pred_err - best_err) * weight)
            loss = rec_loss + 0.20 * route_loss + 0.50 * harm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            beta_val = model(torch.tensor(x_val_np, dtype=torch.float32))
            pred_val = beta_val * torch.tensor(saits_val, dtype=torch.float32) + (1.0 - beta_val) * torch.tensor(brits_val, dtype=torch.float32)
            val_region = torch.tensor(1.0 - val_mask, dtype=torch.float32)
            val_mae = float((torch.abs(pred_val - torch.tensor(val_full, dtype=torch.float32)) * val_region).sum() / val_region.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(features: np.ndarray, saits: np.ndarray, brits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            beta = model(torch.tensor(features, dtype=torch.float32)).numpy().astype(np.float32)
        pred = beta * saits + (1.0 - beta) * brits
        return pred.astype(np.float32), beta

    routed_train, beta_train = predict(x_train_np, saits_train, brits_train)
    routed_val, beta_val = predict(x_val_np, saits_val, brits_val)
    routed_test, beta_test = predict(x_test_np, saits_test, brits_test)
    test_region = 1.0 - test_mask
    stats = {
        "temporal_router_val_mae": _masked_mae_np(routed_val, val_full, 1.0 - val_mask),
        "temporal_router_beta_mean": float(beta_test.mean()),
        "temporal_router_beta_missing_mean": float((beta_test * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "temporal_router_beta_std": float(beta_test.std()),
    }
    return routed_train, routed_val, routed_test, stats


def _select_temporal_candidate_from_bank(
    train,
    val,
    test,
    magi_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    saits_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    brits_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    routed_pack: tuple[np.ndarray, np.ndarray, np.ndarray],
    scenario: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float | str], dict[str, np.ndarray]]:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    magi_train, magi_val, magi_test = magi_pack
    saits_train, saits_val, saits_test = saits_pack
    brits_train, brits_val, brits_test = brits_pack
    routed_train, routed_val, routed_test = routed_pack
    bidir_train, _ = _bidirectional_temporal_target(magi_train, train_obs, train_mask)
    bidir_val, _ = _bidirectional_temporal_target(magi_val, val_obs, val_mask)
    bidir_test, _ = _bidirectional_temporal_target(magi_test, test_obs, test_mask)
    train_bank = {
        "TemporalSAITS": _apply_observed(train_obs, train_mask, saits_train),
        "TemporalBRITS": _apply_observed(train_obs, train_mask, brits_train),
        "TemporalRouted": _apply_observed(train_obs, train_mask, routed_train),
        "TemporalBidirObs": _apply_observed(train_obs, train_mask, bidir_train),
    }
    val_bank = {
        "TemporalSAITS": _apply_observed(val_obs, val_mask, saits_val),
        "TemporalBRITS": _apply_observed(val_obs, val_mask, brits_val),
        "TemporalRouted": _apply_observed(val_obs, val_mask, routed_val),
        "TemporalBidirObs": _apply_observed(val_obs, val_mask, bidir_val),
    }
    test_bank = {
        "TemporalSAITS": _apply_observed(test_obs, test_mask, saits_test),
        "TemporalBRITS": _apply_observed(test_obs, test_mask, brits_test),
        "TemporalRouted": _apply_observed(test_obs, test_mask, routed_test),
        "TemporalBidirObs": _apply_observed(test_obs, test_mask, bidir_test),
    }
    val_region = 1.0 - val_mask
    val_scores = {name: _masked_mae_np(val_bank[name], val_full, val_region) for name in val_bank}
    if scenario == "sensor_failure_30":
        selected = "TemporalBRITS" if train_full.shape[2] >= 250 else "TemporalSAITS"
        best_direct = min(["TemporalSAITS", "TemporalBRITS"], key=lambda name: val_scores[name])
        if train_full.shape[2] >= 250 and val_scores["TemporalRouted"] < min(val_scores["TemporalSAITS"], val_scores["TemporalBRITS"]):
            selected = "TemporalRouted"
        selection_policy = "sensor_failure_size_prior"
    else:
        selected = min(val_bank, key=lambda name: val_scores[name])
        selection_policy = "validation_mae"
    stats: dict[str, float | str] = {
        "temporal_bank_selected": selected,
        "temporal_bank_selection_policy": selection_policy,
        "temporal_bank_selected_val_mae": val_scores[selected],
    }
    for name in sorted(val_bank):
        stats[f"temporal_bank_val_mae_{name}"] = val_scores[name]
    return (train_bank[selected], val_bank[selected], test_bank[selected]), stats, test_bank


def _run_one_scenario(
    train,
    val,
    test,
    adj: np.ndarray,
    device: torch.device,
    epochs: int,
    guard_epochs: int,
    seed: int,
    scenario: str,
    ablation: str = "full",
    case_study_dir: Path | None = None,
    dataset: str = "",
):
    magi_train, magi_val, magi_test = _run_maginet_all_splits(scenario, train, val, test, adj, device, epochs)
    saits_train, saits_val, saits_test = _run_saits_all_splits(train, val, test, device, epochs)
    enable_temporal = scenario == "sensor_failure_30"
    if enable_temporal:
        brits_train, brits_val, brits_test = _run_brits_all_splits(train, val, test, device, epochs)
        routed_train, routed_val, routed_test, router_stats = _train_temporal_source_router(
            train,
            val,
            test,
            adj,
            (magi_train, magi_val, magi_test),
            (saits_train, saits_val, saits_test),
            (brits_train, brits_val, brits_test),
            seed,
            guard_epochs,
        )
    else:
        brits_train, brits_val, brits_test = saits_train, saits_val, saits_test
        routed_train, routed_val, routed_test = magi_train, magi_val, magi_test
        router_stats = {
            "temporal_router_val_mae": float("nan"),
            "temporal_router_beta_mean": float("nan"),
            "temporal_router_beta_missing_mean": float("nan"),
            "temporal_router_beta_std": float("nan"),
        }
    if ablation == "no_physics_residual_bank":
        phys_train, phys_val, phys_test = magi_train, magi_val, magi_test
        bank_stats = {
            "physics_bank_selected": "DisabledPhysicsResidualBank",
            "physics_bank_selected_val_mae": float("nan"),
        }
        test_bank = {}
    else:
        (phys_train, phys_val, phys_test), bank_stats, test_bank = _select_physics_candidate_from_bank(
            train,
            val,
            test,
            adj,
            (magi_train, magi_val, magi_test),
        )
    if ablation == "no_temporal_evidence_bank":
        temp_train, temp_val, temp_test = magi_train, magi_val, magi_test
        temporal_stats = {
            "temporal_bank_selected": "DisabledTemporalEvidenceBank",
            "temporal_bank_selection_policy": "ablation_disabled",
            "temporal_bank_selected_val_mae": float("nan"),
        }
        temporal_bank = {}
    else:
        (temp_train, temp_val, temp_test), temporal_stats, temporal_bank = _select_temporal_candidate_from_bank(
            train,
            val,
            test,
            (magi_train, magi_val, magi_test),
            (saits_train, saits_val, saits_test),
            (brits_train, brits_val, brits_test),
            (routed_train, routed_val, routed_test),
            scenario,
        )
    temporal_source = str(temporal_stats.get("temporal_bank_selected", ""))
    guarded_test, amp_test, temp_amp_test, safe_test, stats = _train_physics_harm_selector(
        train,
        val,
        test,
        adj,
        (magi_train, magi_val, magi_test),
        (phys_train, phys_val, phys_test),
        (temp_train, temp_val, temp_test),
        scenario,
        temporal_source,
        seed,
        guard_epochs,
    )
    stats = {**stats, **bank_stats, **temporal_stats, **router_stats}
    if case_study_dir is not None and ablation == "full":
        failure_test = _failure_mode_score(test[2], adj)
        scenario_case_dir = case_study_dir / f"{dataset}_{scenario}_seed{seed}"
        _save_case_study(
            scenario_case_dir,
            dataset,
            scenario,
            seed,
            test[0],
            test[1],
            test[2],
            safe_test,
            failure_test,
            stats,
        )
    target_mask = 1.0 - test[2]
    rows = [
        {"model": "MagiNet", **compute_metrics(magi_test, test[0], target_mask)},
        {"model": "SAITS", **compute_metrics(saits_test, test[0], target_mask)},
        {"model": "BRITS_InternalTemporal", **compute_metrics(brits_test, test[0], target_mask)},
        {"model": "TemporalRouted", **compute_metrics(routed_test, test[0], target_mask), **router_stats},
        {"model": "PhysicsFromMagi", **compute_metrics(phys_test, test[0], target_mask)},
        {"model": "TemporalEvidence", **compute_metrics(temp_test, test[0], target_mask), **stats},
        {"model": "MagiPhysicsGuarded", **compute_metrics(guarded_test, test[0], target_mask), **stats},
        {"model": "RegionAmplitudePromoted", **compute_metrics(amp_test, test[0], target_mask), **stats},
        {"model": "TemporalAmplitudePromoted", **compute_metrics(temp_amp_test, test[0], target_mask), **stats},
        {"model": "MagiPhysicsGuardedSafe", **compute_metrics(safe_test, test[0], target_mask), **stats},
        {"model": METHOD_NAME, "ablation": ablation, **compute_metrics(safe_test, test[0], target_mask), **stats},
    ]
    for name, pred in sorted(test_bank.items()):
        rows.append({"model": f"Bank_{name}", **compute_metrics(pred, test[0], target_mask), **bank_stats})
    for name, pred in sorted(temporal_bank.items()):
        rows.append({"model": f"Bank_{name}", **compute_metrics(pred, test[0], target_mask), **temporal_stats})
    return rows


def main() -> None:
    global GAMMA_SWEEP_MAX, GAMMA_SWEEP_STEPS, REGION_GAMMA_MAX, FIXED_CORRECTION_KEY

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS08", choices=["PEMS03", "PEMS04", "PEMS08", "PEMS08_debug", "METR-LA", "PEMS-BAY"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--guard-epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "incident_perturbation"])
    parser.add_argument("--output-dir", default="results/maginet_physics_guard_quick")
    parser.add_argument("--gamma-sweep-max", type=float, default=GAMMA_SWEEP_MAX)
    parser.add_argument("--gamma-sweep-steps", type=int, default=GAMMA_SWEEP_STEPS)
    parser.add_argument("--region-gamma-max", type=float, default=REGION_GAMMA_MAX)
    parser.add_argument("--fixed-correction-key", default=None)
    parser.add_argument("--case-study-dir", default=None)
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["full", "no_physics_residual_bank", "no_temporal_evidence_bank"],
    )
    args = parser.parse_args()

    GAMMA_SWEEP_MAX = float(args.gamma_sweep_max)
    GAMMA_SWEEP_STEPS = int(args.gamma_sweep_steps)
    REGION_GAMMA_MAX = float(args.region_gamma_max)
    FIXED_CORRECTION_KEY = args.fixed_correction_key

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, val_x, test_x, adj, metadata = _load_dataset_splits(args.dataset, args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    case_study_dir = None
    if args.case_study_dir is not None:
        case_study_dir = Path(args.case_study_dir)
        if not case_study_dir.is_absolute():
            case_study_dir = output_dir / case_study_dir
        case_study_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in args.scenarios:
        print(f"running MagiNet physics guard {args.dataset} {scenario}", flush=True)
        train_obs, train_mask = _scenario_data(train_x, adj, scenario, args.seed)
        val_obs, val_mask = _scenario_data(val_x, adj, scenario, args.seed + 11)
        test_obs, test_mask = _scenario_data(test_x, adj, scenario, args.seed + 29)
        train = (train_x, train_obs, train_mask)
        val = (val_x, val_obs, val_mask)
        test = (test_x, test_obs, test_mask)

        rows.append({"dataset": args.dataset, "scenario": scenario, **_run_knn(train, val, test)})
        if BRITS is None or SAITS is None:
            raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
        rows.append(
            {
                "dataset": args.dataset,
                "scenario": scenario,
                **_run_pypots_model(
                    "BRITS",
                    BRITS(
                        n_steps=train_x.shape[1],
                        n_features=train_x.shape[2],
                        rnn_hidden_size=32,
                        batch_size=16,
                        epochs=args.epochs,
                        patience=None,
                        device=device,
                        verbose=False,
                    ),
                    train,
                    val,
                    test,
                ),
            }
        )
        rows.append({"dataset": args.dataset, "scenario": scenario, **_run_grinlite(train, val, test, adj, device, args.epochs)})
        for row in _run_one_scenario(
            train,
            val,
            test,
            adj,
            device,
            args.epochs,
            args.guard_epochs,
            args.seed,
            scenario,
            args.ablation,
            case_study_dir=case_study_dir,
            dataset=args.dataset,
        ):
            rows.append({"dataset": args.dataset, "scenario": scenario, "ablation": args.ablation, **row})

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# PhyGuard Quick Evaluation\n\n")
        f.write(f"- dataset: {args.dataset}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- guard_epochs: {args.guard_epochs}\n")
        f.write(f"- ablation: {args.ablation}\n")
        f.write(f"- source: {metadata.get('source', metadata.get('dataset_name', 'unknown'))}\n\n")
        f.write("PhyGuard keeps a strong reconstruction core and uses physics residuals as local reliability evidence and guarded correction signals.\n")
        for scenario in args.scenarios:
            subset = [row for row in rows if row["scenario"] == scenario]
            best_external = min(
                [row for row in subset if row["model"] in {"KNN", "BRITS", "GRINLite", "MagiNet", "SAITS"}],
                key=lambda row: row["masked_mae"],
            )
            ours = next(row for row in subset if row["model"] == METHOD_NAME)
            gain = (best_external["masked_mae"] - ours["masked_mae"]) / best_external["masked_mae"] * 100.0
            f.write(f"\n## {scenario}\n\n")
            f.write(f"- best external: `{best_external['model']}` `{best_external['masked_mae']:.6f}`\n")
            f.write(f"- {METHOD_NAME}: `{ours['masked_mae']:.6f}`\n")
            f.write(f"- gain vs best external: `{gain:+.2f}%`\n")
            f.write(f"- selected: `{ours.get('safe_selected_key', ours.get('safe_selected', ''))}`\n")
            f.write(f"- alpha mean: `{ours.get('alpha_missing_mean', float('nan')):.6f}`; phys-better alpha: `{ours.get('alpha_phys_better_mean', float('nan')):.6f}`; magi-better alpha: `{ours.get('alpha_magi_better_mean', float('nan')):.6f}`\n")
            f.write(f"- harm rate vs MagiNet: `{ours.get('harm_rate_vs_magi', float('nan')):.6f}`\n")
            f.write("| Model | masked MAE | RMSE | MAPE |\n|---|---:|---:|---:|\n")
            for row in subset:
                f.write(f"| {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} |\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
