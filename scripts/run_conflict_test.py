from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.corruptions import incident_perturbation
from data.datasets import _generate_toy_tensor, _load_metrla_hf_splits_cached
from data.masks import random_missing_mask, sensor_failure_mask
from data.normalization import StandardScaler
from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.litetrust_pinn import LiteTrustPINN
from physics.traffic_residuals import fundamental_residual_from_prediction
from scripts.train import load_config, resolve_device


class RegionTrafficDataset(Dataset):
    def __init__(self, full_x: np.ndarray, obs_mask: np.ndarray, incident_mask: np.ndarray):
        self.full_x = torch.tensor(full_x, dtype=torch.float32)
        self.obs_mask = torch.tensor(obs_mask, dtype=torch.float32)
        self.obs_x = self.full_x * self.obs_mask
        self.incident_mask = torch.tensor(incident_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return self.full_x.shape[0]

    def __getitem__(self, idx: int) -> dict:
        full_x = self.full_x[idx]
        obs_mask = self.obs_mask[idx]
        target_mask = 1.0 - obs_mask
        return {
            "x_full": full_x,
            "x_obs": self.obs_x[idx],
            "mask": obs_mask,
            "target_mask": target_mask,
            "incident_mask": self.incident_mask[idx],
        }


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _stage7_config() -> dict:
    config = deepcopy(load_config(str(ROOT / "configs" / "default.yaml")))
    config["seed"] = 1
    config["device"] = "cpu"
    config["dataset"].update(
        {
            "name": "toy",
            "nodes": 20,
            "seq_len": 24,
            "channels": 3,
            "train_samples": 128,
            "val_samples": 32,
            "test_samples": 32,
            "missing_rate": 0.5,
        }
    )
    config["model"].update(
        {
            "input_dim": 6,
            "hidden_dim": 32,
            "output_dim": 3,
            "num_layers": 2,
            "dropout": 0.1,
        }
    )
    config["train"].update(
        {
            "epochs": 20,
            "batch_size": 16,
            "lr": 0.001,
            "weight_decay": 0.0,
            "num_workers": 0,
        }
    )
    config["physics"] = {
        "lambda_max": 0.005,
        "warmup_start_epoch": 5,
        "warmup_end_epoch": 15,
        "loss": "smooth_l1",
        "residual": "fundamental_diagram",
    }
    config["trust"] = {
        "trust_floor": 0.3,
        "beta_floor": 0.05,
        "beta_smooth": 0.001,
        "beta_variance": 0.02,
        "trust_min_std": 0.08,
        "beta_rank": 0.05,
        "rank_margin": 0.1,
        "extra_feature_dim": 4,
        "w_min": 0.0,
    }
    config["uncertainty"] = {
        "use_uncertainty_loss": True,
        "alpha_unc": 0.1,
    }
    config["conflict"] = {
        "beta_conflict": 0.05,
        "start_epoch": 5,
        "clip": 10.0,
    }
    config["method"] = {
        "physics_form_weight": 0.0,
        "physics_form_temperature": 0.08,
    }
    config["incident"] = {
        "drop_ratio": 0.5,
        "flow_drop_ratio": 0.0,
        "speed_drop_ratio": 0.5,
        "duration": 6,
        "region_size": 4,
    }
    config["results_dir"] = "results/v4_conflict_aware"
    return config


def _lambda_schedule(epoch: int, physics_cfg: dict) -> float:
    lambda_max = float(physics_cfg["lambda_max"])
    start = int(physics_cfg["warmup_start_epoch"])
    end = int(physics_cfg["warmup_end_epoch"])
    if epoch < start:
        return 0.0
    if epoch >= end:
        return lambda_max
    return lambda_max * float(epoch - start + 1) / float(max(end - start + 1, 1))


def _build_scenario_loaders(config: dict, scenario: str):
    dataset_cfg = config["dataset"]
    seed = int(config["seed"])
    if dataset_cfg.get("name") == "METR-LA" and str(dataset_cfg.get("source", "hf")).lower() == "hf":
        train_x, val_x, test_x, adj, _metadata = _load_metrla_hf_splits_cached(
            int(dataset_cfg["train_samples"]),
            int(dataset_cfg["val_samples"]),
            int(dataset_cfg["test_samples"]),
        )
    else:
        train_x, adj = _generate_toy_tensor(
            int(dataset_cfg["train_samples"]),
            int(dataset_cfg["seq_len"]),
            int(dataset_cfg["nodes"]),
            int(dataset_cfg["channels"]),
            seed,
        )
        val_x, _ = _generate_toy_tensor(
            int(dataset_cfg["val_samples"]),
            int(dataset_cfg["seq_len"]),
            int(dataset_cfg["nodes"]),
            int(dataset_cfg["channels"]),
            seed + 1,
        )
        test_x, _ = _generate_toy_tensor(
            int(dataset_cfg["test_samples"]),
            int(dataset_cfg["seq_len"]),
            int(dataset_cfg["nodes"]),
            int(dataset_cfg["channels"]),
            seed + 2,
        )
    train_incident = np.zeros_like(train_x, dtype=np.float32)
    val_incident = np.zeros_like(val_x, dtype=np.float32)
    test_incident = np.zeros_like(test_x, dtype=np.float32)
    if scenario == "incident_perturbation":
        inc_cfg = config["incident"]
        train_x, train_incident = incident_perturbation(
            train_x,
            adj,
            drop_ratio=float(inc_cfg["drop_ratio"]),
            duration=int(inc_cfg["duration"]),
            region_size=int(inc_cfg["region_size"]),
            seed=seed + 101,
            return_mask=True,
            flow_drop_ratio=float(inc_cfg.get("flow_drop_ratio", inc_cfg["drop_ratio"])),
            speed_drop_ratio=float(inc_cfg.get("speed_drop_ratio", inc_cfg["drop_ratio"])),
        )
        val_x, val_incident = incident_perturbation(
            val_x,
            adj,
            drop_ratio=float(inc_cfg["drop_ratio"]),
            duration=int(inc_cfg["duration"]),
            region_size=int(inc_cfg["region_size"]),
            seed=seed + 102,
            return_mask=True,
            flow_drop_ratio=float(inc_cfg.get("flow_drop_ratio", inc_cfg["drop_ratio"])),
            speed_drop_ratio=float(inc_cfg.get("speed_drop_ratio", inc_cfg["drop_ratio"])),
        )
        test_x, test_incident = incident_perturbation(
            test_x,
            adj,
            drop_ratio=float(inc_cfg["drop_ratio"]),
            duration=int(inc_cfg["duration"]),
            region_size=int(inc_cfg["region_size"]),
            seed=seed + 103,
            return_mask=True,
            flow_drop_ratio=float(inc_cfg.get("flow_drop_ratio", inc_cfg["drop_ratio"])),
            speed_drop_ratio=float(inc_cfg.get("speed_drop_ratio", inc_cfg["drop_ratio"])),
        )
    elif scenario not in {"random_missing_50", "sensor_failure_30"}:
        raise ValueError(f"Unsupported scenario: {scenario}")

    scaler = StandardScaler.fit(train_x)
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    missing_rate = float(dataset_cfg["missing_rate"])
    if scenario == "sensor_failure_30":
        train_mask = sensor_failure_mask(train_x.shape, 0.3, seed=seed)
        val_mask = sensor_failure_mask(val_x.shape, 0.3, seed=seed + 11)
        test_mask = sensor_failure_mask(test_x.shape, 0.3, seed=seed + 29)
    else:
        train_mask = random_missing_mask(train_x.shape, missing_rate, seed=seed)
        val_mask = random_missing_mask(val_x.shape, missing_rate, seed=seed + 11)
        test_mask = random_missing_mask(test_x.shape, missing_rate, seed=seed + 29)
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 0))
    return (
        DataLoader(RegionTrafficDataset(train_x, train_mask, train_incident), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(RegionTrafficDataset(val_x, val_mask, val_incident), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(RegionTrafficDataset(test_x, test_mask, test_incident), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        torch.tensor(adj, dtype=torch.float32),
        scaler,
    )


def _instantiate(config: dict) -> LiteTrustPINN:
    model_cfg = config["model"]
    return LiteTrustPINN(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        w_min=float(config["trust"].get("w_min", 0.0)),
        use_uncertainty=True,
        extra_feature_dim=int(config["trust"].get("extra_feature_dim", 1)),
    )


def _residual(pred: torch.Tensor, scaler) -> torch.Tensor:
    return fundamental_residual_from_prediction(pred, normalizer=scaler)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _batch_rank(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    order = torch.argsort(flat, dim=1)
    ranks = torch.zeros_like(flat)
    denom = max(flat.shape[1] - 1, 1)
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


def _anomaly_scalar(extra_feature: torch.Tensor) -> torch.Tensor:
    if extra_feature.shape[-1] == 1:
        return extra_feature
    temporal = extra_feature[..., 0:1]
    spatial = extra_feature[..., 1:2]
    missing = extra_feature[..., 2:3]
    residual_rank = extra_feature[..., 3:4]
    if extra_feature.shape[-1] >= 6:
        node_missing = extra_feature[..., 4:5]
        neighbor_missing = extra_feature[..., 5:6]
        return torch.maximum(
            torch.maximum(torch.maximum(temporal, spatial), torch.maximum(missing, residual_rank)),
            torch.maximum(node_missing, neighbor_missing),
        )
    return torch.maximum(torch.maximum(temporal, spatial), torch.maximum(missing, residual_rank))


def _trust_variance_loss(trust: torch.Tensor, config: dict) -> torch.Tensor:
    min_std = float(config["trust"].get("trust_min_std", 0.0))
    beta = float(config["trust"].get("beta_variance", 0.0))
    if beta <= 0.0 or min_std <= 0.0:
        return torch.zeros((), dtype=trust.dtype, device=trust.device)
    return beta * torch.relu(torch.as_tensor(min_std, dtype=trust.dtype, device=trust.device) - trust.std()) ** 2


def _trust_ranking_loss(trust: torch.Tensor, conflict_score: torch.Tensor, config: dict) -> torch.Tensor:
    beta = float(config["trust"].get("beta_rank", 0.0))
    if beta <= 0.0:
        return torch.zeros((), dtype=trust.dtype, device=trust.device)
    margin = float(config["trust"].get("rank_margin", 0.05))
    losses = []
    flat_trust = trust.reshape(trust.shape[0], -1)
    flat_score = conflict_score.detach().reshape(conflict_score.shape[0], -1)
    for batch_idx in range(flat_score.shape[0]):
        score = flat_score[batch_idx]
        high = score >= torch.quantile(score, 0.8)
        low = score <= torch.quantile(score, 0.2)
        if high.any() and low.any():
            high_trust = flat_trust[batch_idx][high].mean()
            low_trust = flat_trust[batch_idx][low].mean()
            losses.append(torch.relu(torch.as_tensor(margin, dtype=trust.dtype, device=trust.device) + high_trust - low_trust))
    if not losses:
        return torch.zeros((), dtype=trust.dtype, device=trust.device)
    return beta * torch.stack(losses).mean()


def _safe_region_mean(values: list[np.ndarray]) -> float | None:
    if not values:
        return None
    joined = np.concatenate([v.reshape(-1) for v in values], axis=0)
    if joined.size == 0:
        return None
    return float(joined.mean())


def _region_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float | None:
    denom = float(mask.sum().detach().cpu())
    if denom <= 0.0:
        return None
    return float(((pred - target).abs() * mask).sum().detach().cpu() / denom)


def _forward(model, x_obs, obs_mask, adj, scaler):
    output = model(x_obs, obs_mask, adj)
    pred = output["mu"]
    log_var = output["log_var"]
    residual = _residual(pred, scaler)
    gate_log_var = log_var.mean(dim=-1, keepdim=True)
    anomaly_feature = _trust_extra_features(x_obs, obs_mask, adj, residual)
    trust = model.trust_from_residual(
        output["h"],
        residual.detach().abs(),
        obs_mask,
        log_var=gate_log_var,
        extra_feature=anomaly_feature,
    )
    return pred, residual, trust, log_var, anomaly_feature


def _loss_terms(pred, target, target_mask, obs_mask, anomaly_feature, residual, trust, log_var, epoch, lambda_phys, config, use_conflict):
    data_mae = masked_mae_loss(pred, target, target_mask)
    hetero = torch.exp(-log_var) * torch.abs(target - pred) + log_var
    hetero_loss = _masked_mean(hetero, target_mask)
    data_loss = data_mae + float(config["uncertainty"]["alpha_unc"]) * hetero_loss
    physics_loss = torch.mean(trust * F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none"))
    form_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)

    trust_floor = float(config["trust"]["trust_floor"])
    floor_loss = float(config["trust"]["beta_floor"]) * torch.relu(torch.as_tensor(trust_floor, device=trust.device) - trust.mean()) ** 2
    smooth_loss = float(config["trust"]["beta_smooth"]) * torch.mean(torch.abs(trust[:, 1:] - trust[:, :-1]))
    variance_loss = _trust_variance_loss(trust, config)

    conflict_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    rank_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if use_conflict and epoch >= int(config["conflict"]["start_epoch"]):
        data_conf = torch.exp(-log_var.detach().mean(dim=-1, keepdim=True))
        obs_conf = obs_mask.detach().mean(dim=-1, keepdim=True)
        residual_score = _batch_rank(residual.detach().abs())
        anomaly_tail = torch.sigmoid((_anomaly_scalar(anomaly_feature).detach() - 0.6) / 0.15)
        conflict_score = obs_conf * data_conf * residual_score * anomaly_tail
        conflict_score = torch.clamp(conflict_score, max=float(config["conflict"]["clip"]))
        conflict_loss = float(config["conflict"]["beta_conflict"]) * torch.mean(conflict_score * trust)
        rank_loss = _trust_ranking_loss(trust, conflict_score, config)
    total = data_loss + float(lambda_phys) * physics_loss + floor_loss + smooth_loss + variance_loss + conflict_loss + rank_loss + form_loss
    return total, {
        "data_mae": data_mae,
        "hetero_loss": hetero_loss,
        "physics_loss": physics_loss,
        "floor_loss": floor_loss,
        "smooth_loss": smooth_loss,
        "variance_loss": variance_loss,
        "conflict_loss": conflict_loss,
        "rank_loss": rank_loss,
        "form_loss": form_loss,
        "residual_abs": residual.detach().abs().mean(),
    }


def _run_epoch(model, loader, adj, scaler, optimizer, device, epoch, lambda_phys, config, use_conflict):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals = []
    data_maes = []
    physics_losses = []
    conflict_losses = []
    residual_means = []
    preds = []
    targets = []
    target_masks = []
    trust_all = []
    logvar_all = []
    trust_incident = []
    trust_normal = []
    resid_incident = []
    resid_normal = []
    mae_normal = []
    mae_missing = []
    mae_incident = []

    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        x_full = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        incident_mask = batch["incident_mask"].to(device)
        incident_nt = incident_mask.max(dim=-1, keepdim=True).values
        normal_nt = 1.0 - incident_nt

        with torch.set_grad_enabled(train_mode):
            pred, residual, trust, log_var, anomaly_feature = _forward(model, x_obs, obs_mask, adj, scaler)
            total_loss, terms = _loss_terms(
                pred,
                x_full,
                target_mask,
                obs_mask,
                anomaly_feature,
                residual,
                trust,
                log_var,
                epoch,
                lambda_phys,
                config,
                use_conflict,
            )
            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        totals.append(float(total_loss.detach().cpu()))
        data_maes.append(float(terms["data_mae"].detach().cpu()))
        physics_losses.append(float(terms["physics_loss"].detach().cpu()))
        conflict_losses.append(float(terms["conflict_loss"].detach().cpu()))
        residual_means.append(float(terms["residual_abs"].detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        target_masks.append(target_mask.detach().cpu().numpy())
        trust_all.append(trust.detach().cpu().numpy())
        logvar_all.append(log_var.detach().cpu().numpy())

        incident_values = incident_nt.bool().detach().cpu().numpy()
        normal_values = normal_nt.bool().detach().cpu().numpy()
        trust_np = trust.detach().cpu().numpy()
        resid_np = residual.detach().abs().cpu().numpy()
        if incident_values.any():
            trust_incident.append(trust_np[incident_values])
            resid_incident.append(resid_np[incident_values])
        if normal_values.any():
            trust_normal.append(trust_np[normal_values])
            resid_normal.append(resid_np[normal_values])

        normal_mask = target_mask * (1.0 - incident_mask)
        incident_region_mask = target_mask * incident_mask
        mae_normal_value = _region_mae(pred, x_full, normal_mask)
        mae_missing_value = _region_mae(pred, x_full, target_mask)
        mae_incident_value = _region_mae(pred, x_full, incident_region_mask)
        if mae_normal_value is not None:
            mae_normal.append(np.asarray([mae_normal_value], dtype=np.float32))
        if mae_missing_value is not None:
            mae_missing.append(np.asarray([mae_missing_value], dtype=np.float32))
        if mae_incident_value is not None:
            mae_incident.append(np.asarray([mae_incident_value], dtype=np.float32))

    metrics = compute_metrics(
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(target_masks, axis=0),
    )
    trust_joined = np.concatenate([v.reshape(-1) for v in trust_all], axis=0)
    logvar_joined = np.concatenate([v.reshape(-1) for v in logvar_all], axis=0)
    return {
        "total_loss": float(np.mean(totals)),
        "data_mae": float(np.mean(data_maes)),
        "physics_loss": float(np.mean(physics_losses)),
        "conflict_loss": float(np.mean(conflict_losses)),
        "physics_residual": float(np.mean(residual_means)),
        "trust_mean": float(trust_joined.mean()),
        "trust_std": float(trust_joined.std()),
        "trust_min": float(trust_joined.min()),
        "trust_max": float(trust_joined.max()),
        "log_var_mean": float(logvar_joined.mean()),
        "log_var_std": float(logvar_joined.std()),
        "normal_region_MAE": _safe_region_mean(mae_normal),
        "missing_region_MAE": _safe_region_mean(mae_missing),
        "incident_region_MAE": _safe_region_mean(mae_incident),
        "trust_mean_normal": _safe_region_mean(trust_normal),
        "trust_mean_incident": _safe_region_mean(trust_incident),
        "physics_residual_normal": _safe_region_mean(resid_normal),
        "physics_residual_incident": _safe_region_mean(resid_incident),
        **metrics,
    }


def _train_variant(config: dict, scenario: str, use_conflict: bool) -> dict:
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, adj, scaler = _build_scenario_loaders(config, scenario)
    model = _instantiate(config).to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    log_rows = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        lambda_phys = _lambda_schedule(epoch, config["physics"])
        train_stats = _run_epoch(model, train_loader, adj, scaler, optimizer, device, epoch, lambda_phys, config, use_conflict)
        val_stats = _run_epoch(model, val_loader, adj, scaler, None, device, epoch, lambda_phys, config, use_conflict)
        log_rows.append(
            {
                "epoch": epoch,
                "lambda_phys": lambda_phys,
                "train_data_mae": train_stats["data_mae"],
                "train_physics_loss": train_stats["physics_loss"],
                "train_conflict_loss": train_stats["conflict_loss"],
                "val_masked_mae": val_stats["masked_mae"],
                "val_trust_mean": val_stats["trust_mean"],
                "val_trust_std": val_stats["trust_std"],
                "val_trust_mean_normal": val_stats["trust_mean_normal"],
                "val_trust_mean_incident": val_stats["trust_mean_incident"],
                "val_incident_region_MAE": val_stats["incident_region_MAE"],
            }
        )
    test_stats = _run_epoch(model, test_loader, adj, scaler, None, device, int(config["train"]["epochs"]), 0.0, config, use_conflict)
    return {
        "test": {
            **test_stats,
            "device": str(device),
            "fallback_used": False,
        },
        "final_val_masked_mae": float(log_rows[-1]["val_masked_mae"]),
        "final_val_trust_mean": float(log_rows[-1]["val_trust_mean"]),
        "final_val_trust_std": float(log_rows[-1]["val_trust_std"]),
        "final_val_trust_mean_normal": log_rows[-1]["val_trust_mean_normal"],
        "final_val_trust_mean_incident": log_rows[-1]["val_trust_mean_incident"],
        "first_train_data_mae": float(log_rows[0]["train_data_mae"]),
        "last_train_data_mae": float(log_rows[-1]["train_data_mae"]),
        "log_rows": log_rows,
    }


def _write_artifacts(summary: dict, config: dict) -> None:
    output_dir = ROOT / config["results_dir"]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in summary.items() if k != "logs"}, f, indent=2, default=_json_default)
        rows = []
        for key, values in summary["logs"].items():
            scenario, variant = key.split("::")
            for row in values:
                rows.append({"scenario": scenario, "variant": variant, **row})
        with open(output_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except OSError as exc:
        expected_files = ["metrics.json", "train_log.csv", "config.yaml"]
        artifacts_exist = all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in expected_files)
        if not artifacts_exist:
            summary["artifact_save_error"] = str(exc)


def main() -> None:
    config = _stage7_config()
    scenarios = ["random_missing_50", "incident_perturbation"]
    results = {}
    logs = {}
    for scenario in scenarios:
        v3 = _train_variant(config, scenario, use_conflict=False)
        v4 = _train_variant(config, scenario, use_conflict=True)
        results[scenario] = {
            "v3_without_conflict": {k: v for k, v in v3.items() if k != "log_rows"},
            "v4_with_conflict": {k: v for k, v in v4.items() if k != "log_rows"},
            "comparison": {
                "v4_minus_v3_masked_mae": float(v4["test"]["masked_mae"] - v3["test"]["masked_mae"]),
                "v4_minus_v3_incident_region_MAE": (
                    None
                    if v3["test"]["incident_region_MAE"] is None or v4["test"]["incident_region_MAE"] is None
                    else float(v4["test"]["incident_region_MAE"] - v3["test"]["incident_region_MAE"])
                ),
                "v4_minus_v3_trust_incident": (
                    None
                    if v3["test"]["trust_mean_incident"] is None or v4["test"]["trust_mean_incident"] is None
                    else float(v4["test"]["trust_mean_incident"] - v3["test"]["trust_mean_incident"])
                ),
            },
        }
        logs[f"{scenario}::v3_without_conflict"] = v3["log_rows"]
        logs[f"{scenario}::v4_with_conflict"] = v4["log_rows"]

    incident_v4 = results["incident_perturbation"]["v4_with_conflict"]["test"]
    incident_v3 = results["incident_perturbation"]["v3_without_conflict"]["test"]
    criteria = {
        "incident_trust_lower_than_normal": incident_v4["trust_mean_incident"] is not None
        and incident_v4["trust_mean_incident"] < incident_v4["trust_mean_normal"],
        "incident_mae_not_worse": incident_v4["incident_region_MAE"] is not None
        and incident_v3["incident_region_MAE"] is not None
        and incident_v4["incident_region_MAE"] <= incident_v3["incident_region_MAE"] + 0.01,
        "overall_mae_not_much_worse": incident_v4["masked_mae"] <= incident_v3["masked_mae"] + 0.01,
        "trust_not_collapsed": 0.1 <= incident_v4["trust_mean"] <= 0.95 and incident_v4["trust_std"] > 1e-4,
        "no_nan": bool(np.isfinite(incident_v4["masked_mae"]) and np.isfinite(incident_v4["trust_mean"])),
    }
    criteria["recommend_enter_v5_or_stage1"] = all(criteria.values())

    summary = {
        "stage": "stage_7_v4_conflict_aware",
        "config": config,
        "results": results,
        "criteria": criteria,
        "logs": logs,
    }
    _write_artifacts(summary, config)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
