from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from losses.metrics import compute_metrics
from models.litetrust_pinn import _node_failure_signal
from scripts.run_conflict_test import _trust_extra_features
from scripts.run_stage10a_pems08_real_debug import (
    _config as pems08_config,
    _instantiate,
    _lambda_schedule,
    _run_epoch,
    _scenario_loaders as pems08_loaders,
)
from scripts.run_stage2_three_dataset_quick import _build_scenario_loaders as metrla_loaders
from scripts.run_stage2_three_dataset_quick import _forward


SCENARIOS = ["random_missing_50", "sensor_failure_30"]
CANDIDATE_NAMES = [
    "LiteTrustPINN_full",
    "LiteTrustGRINCorrection",
    "absolute_node_missing_regime",
    "contrast_sensor_regime",
    "residual_verified_regime",
]


class ResidualUtilityRouter(nn.Module):
    def __init__(self, input_dim: int, num_candidates: int):
        super().__init__()
        hidden = 32
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_candidates),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _batch_rank_score(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    order = torch.argsort(flat, dim=1)
    ranks = torch.zeros_like(flat)
    values = torch.linspace(0.0, 1.0, steps=flat.shape[1], dtype=x.dtype, device=x.device)
    ranks.scatter_(1, order, values.expand(flat.shape[0], -1))
    return ranks.reshape_as(x)


def _regime_weights(
    pred_pinn: torch.Tensor,
    pred_corr: torch.Tensor,
    residual_pinn: torch.Tensor,
    residual_corr: torch.Tensor,
    extra: torch.Tensor,
) -> dict[str, torch.Tensor]:
    local_missing = extra[..., 2:3]
    node_missing = extra[..., 4:5]
    neighbor_missing = extra[..., 5:6]
    node_failure = _node_failure_signal(node_missing, local_missing)

    # Previous absolute-threshold regime. Kept only as a comparison.
    absolute_sensor = torch.sigmoid((node_missing - 0.7) / 0.12) * torch.clamp(local_missing, 0.0, 1.0)
    absolute_regime = torch.maximum(absolute_sensor, node_failure * absolute_sensor)

    # Cross-dataset sensor-failure signal: a node is unreliable when its missingness
    # is higher than the local graph neighborhood, not just when node_missing is large.
    node_contrast = node_missing - neighbor_missing
    contrast_sensor = torch.sigmoid((node_contrast - 0.25) / 0.10) * torch.clamp(local_missing, 0.0, 1.0)

    # Random/intermittent missing is allowed to use correction only when physics residual
    # prefers the correction candidate and both candidates remain close enough.
    residual_pinn_rank = _batch_rank_score(residual_pinn.detach().abs().mean(dim=-1, keepdim=True))
    residual_corr_rank = _batch_rank_score(residual_corr.detach().abs().mean(dim=-1, keepdim=True))
    residual_prefers_corr = torch.sigmoid((residual_pinn_rank - residual_corr_rank - 0.05) / 0.12)
    pred_gap_rank = _batch_rank_score(torch.abs(pred_pinn.detach() - pred_corr.detach()).mean(dim=-1, keepdim=True))
    candidate_agreement = torch.sigmoid((0.70 - pred_gap_rank) / 0.12)
    intermittent_region = torch.sigmoid((0.20 - node_contrast.abs()) / 0.10) * torch.clamp(local_missing, 0.0, 1.0)
    residual_verified = 0.25 * intermittent_region * residual_prefers_corr * candidate_agreement

    return {
        "absolute_node_missing_regime": absolute_regime,
        "contrast_sensor_regime": contrast_sensor,
        "residual_verified_regime": torch.maximum(contrast_sensor, residual_verified),
    }


def _candidate_predictions(
    pred_pinn: torch.Tensor,
    pred_corr: torch.Tensor,
    residual_pinn: torch.Tensor,
    residual_corr: torch.Tensor,
    extra: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    regimes = _regime_weights(pred_pinn, pred_corr, residual_pinn, residual_corr, extra)
    preds = {
        "LiteTrustPINN_full": pred_pinn,
        "LiteTrustGRINCorrection": pred_corr,
    }
    for name, regime in regimes.items():
        preds[name] = (1.0 - regime) * pred_pinn + regime * pred_corr
    return preds, regimes


def _router_features(
    pred_pinn: torch.Tensor,
    pred_corr: torch.Tensor,
    residual_pinn: torch.Tensor,
    residual_corr: torch.Tensor,
    extra: torch.Tensor,
    regimes: dict[str, torch.Tensor],
) -> torch.Tensor:
    residual_pinn_rank = _batch_rank_score(residual_pinn.detach().abs().mean(dim=-1, keepdim=True))
    residual_corr_rank = _batch_rank_score(residual_corr.detach().abs().mean(dim=-1, keepdim=True))
    pred_gap_rank = _batch_rank_score(torch.abs(pred_pinn.detach() - pred_corr.detach()).mean(dim=-1, keepdim=True))
    residual_gap = residual_pinn_rank - residual_corr_rank
    regime_features = torch.cat([regimes[name] for name in regimes], dim=-1)
    return torch.cat(
        [
            extra.detach(),
            residual_pinn_rank,
            residual_corr_rank,
            residual_gap,
            pred_gap_rank,
            regime_features.detach(),
        ],
        dim=-1,
    )


def _metrla_config() -> dict:
    config = deepcopy(pems08_config())
    config["dataset"].update(
        {
            "name": "METR-LA",
            "nodes": 207,
            "seq_len": 24,
            "channels": 2,
            "train_samples": 64,
            "val_samples": 16,
            "test_samples": 16,
            "missing_rate": 0.5,
            "source": "hf",
        }
    )
    config["model"].update(
        {
            "input_dim": 4,
            "hidden_dim": 32,
            "output_dim": 2,
            "num_layers": 2,
            "dropout": 0.1,
        }
    )
    config["dataset_residual"] = "graph_speed"
    return config


def _unpack_loader(result, dataset_name: str):
    if len(result) == 6:
        return result
    train_loader, val_loader, test_loader, adj, scaler = result
    return train_loader, val_loader, test_loader, adj, scaler, {
        "real_data_used": dataset_name == "METR-LA",
        "fallback_used": False if dataset_name == "METR-LA" else True,
    }


def _train_model(config: dict, model_name: str, train_loader, val_loader, adj, scaler, device: torch.device):
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model = _instantiate(config, model_name).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        lambda_phys = 0.0 if model_name == "BaseTCN" else _lambda_schedule(epoch, config["physics"])
        _run_epoch(model, model_name, train_loader, adj, scaler, optimizer, device, epoch, lambda_phys, config)
        _run_epoch(model, model_name, val_loader, adj, scaler, None, device, epoch, lambda_phys, config)
    return model


def _router_batch(
    config: dict,
    pinn_model,
    correction_model,
    batch,
    adj,
    scaler,
    device: torch.device,
):
    x_obs = batch["x_obs"].to(device)
    x_full = batch["x_full"].to(device)
    obs_mask = batch["mask"].to(device)
    target_mask = batch["target_mask"].to(device)
    with torch.no_grad():
        pred_pinn, residual_pinn, *_ = _forward(pinn_model, "LiteTrustPINN_full", x_obs, obs_mask, adj, scaler, config["dataset_residual"])
        pred_corr, residual_corr, *_ = _forward(correction_model, "LiteTrustGRINCorrection", x_obs, obs_mask, adj, scaler, config["dataset_residual"])
        extra = _trust_extra_features(x_obs, obs_mask, adj, residual_pinn)
        preds, regimes = _candidate_predictions(pred_pinn, pred_corr, residual_pinn, residual_corr, extra)
        features = _router_features(pred_pinn, pred_corr, residual_pinn, residual_corr, extra, regimes)
    return preds, regimes, features, x_full, target_mask


def _collect_router_training_data(config: dict, pinn_model, correction_model, loader, adj, scaler, device: torch.device):
    feature_parts = []
    label_parts = []
    weight_parts = []
    for batch in loader:
        preds, _regimes, features, target, target_mask = _router_batch(config, pinn_model, correction_model, batch, adj, scaler, device)
        candidate_stack = torch.stack([preds[name] for name in CANDIDATE_NAMES], dim=-1)
        valid = target_mask.mean(dim=-1, keepdim=True).squeeze(-1) > 0.0
        denom = target_mask.sum(dim=-1).clamp_min(1.0)
        errors = torch.sum(torch.abs(candidate_stack - target.unsqueeze(-1)) * target_mask.unsqueeze(-1), dim=-2) / denom.unsqueeze(-1)
        labels = torch.argmin(errors, dim=-1)
        local_missing = features[..., 2]
        node_missing = features[..., 4]
        neighbor_missing = features[..., 5]
        node_failure = _node_failure_signal(node_missing.unsqueeze(-1), local_missing.unsqueeze(-1)).squeeze(-1)
        node_contrast = torch.clamp(node_missing - neighbor_missing, min=0.0)
        sample_weight = 1.0 + 1.5 * local_missing + 3.0 * node_failure + 1.0 * node_contrast
        feature_parts.append(features[valid].detach().cpu())
        label_parts.append(labels[valid].detach().cpu())
        weight_parts.append(sample_weight[valid].detach().cpu())
    features = torch.cat(feature_parts, dim=0)
    labels = torch.cat(label_parts, dim=0).long()
    weights = torch.cat(weight_parts, dim=0).float()
    max_samples = 120_000
    if features.shape[0] > max_samples:
        generator = torch.Generator().manual_seed(int(config["seed"]))
        idx = torch.randperm(features.shape[0], generator=generator)[:max_samples]
        features = features[idx]
        labels = labels[idx]
        weights = weights[idx]
    return features, labels, weights


def _train_residual_utility_router(config: dict, pinn_model, correction_model, train_loader, adj, scaler, device: torch.device):
    features, labels, weights = _collect_router_training_data(config, pinn_model, correction_model, train_loader, adj, scaler, device)
    router = ResidualUtilityRouter(features.shape[-1], len(CANDIDATE_NAMES)).to(device)
    features = features.to(device)
    labels = labels.to(device)
    weights = weights.to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.002, weight_decay=1e-4)
    batch_size = 4096
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 17)
    router.train()
    final_loss = 0.0
    for _epoch in range(50):
        order = torch.randperm(features.shape[0], generator=generator, device=device)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, features.shape[0], batch_size):
            idx = order[start : start + batch_size]
            logits = router(features[idx])
            loss_vec = F.cross_entropy(logits, labels[idx], reduction="none")
            loss = torch.mean(loss_vec * weights[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * float(idx.numel())
            total_weight += float(idx.numel())
        final_loss = total_loss / max(total_weight, 1.0)
    with torch.no_grad():
        pred_label = torch.argmax(router(features), dim=-1)
        acc = float((pred_label == labels).float().mean().detach().cpu())
        counts = torch.bincount(labels.detach().cpu(), minlength=len(CANDIDATE_NAMES)).float()
        counts = counts / counts.sum().clamp_min(1.0)
    router.eval()
    return router, {
        "router_train_loss": final_loss,
        "router_train_acc": acc,
        "router_label_distribution": {name: float(counts[i]) for i, name in enumerate(CANDIDATE_NAMES)},
    }


def _eval_regime_adaptive(config: dict, pinn_model, correction_model, loader, adj, scaler, device: torch.device, utility_router=None) -> list[dict]:
    preds_by_model: dict[str, list[np.ndarray]] = {name: [] for name in CANDIDATE_NAMES}
    if utility_router is not None:
        preds_by_model["LearnedResidualUtilityRouter"] = []
        preds_by_model["LearnedResidualUtilityRouterHard"] = []
    targets = []
    masks = []
    regime_weights: dict[str, list[np.ndarray]] = {
        "absolute_node_missing_regime": [],
        "contrast_sensor_regime": [],
        "residual_verified_regime": [],
    }
    utility_weight_parts = []
    utility_confidence_parts = []
    utility_sensor_like_parts = []
    for batch in loader:
        preds, regimes, features, x_full, target_mask = _router_batch(config, pinn_model, correction_model, batch, adj, scaler, device)
        with torch.no_grad():
            for name in CANDIDATE_NAMES:
                preds_by_model[name].append(preds[name].detach().cpu().numpy())
            for name, regime in regimes.items():
                regime_weights[name].append(regime.detach().cpu().numpy())
            if utility_router is not None:
                logits = utility_router(features)
                weights = torch.softmax(logits, dim=-1)
                candidate_stack = torch.stack([preds[name] for name in CANDIDATE_NAMES], dim=-1)
                pred = torch.sum(candidate_stack * weights.unsqueeze(-2), dim=-1)
                preds_by_model["LearnedResidualUtilityRouter"].append(pred.detach().cpu().numpy())
                hard_idx = torch.argmax(logits, dim=-1)
                hard_weights = F.one_hot(hard_idx, num_classes=len(CANDIDATE_NAMES)).to(candidate_stack.dtype)
                hard_pred = torch.sum(candidate_stack * hard_weights.unsqueeze(-2), dim=-1)
                preds_by_model["LearnedResidualUtilityRouterHard"].append(hard_pred.detach().cpu().numpy())
                utility_weight_parts.append(weights.detach().cpu().numpy())
                utility_confidence_parts.append(torch.max(weights, dim=-1).values.detach().cpu().numpy())
                local_missing = features[..., 2:3]
                node_missing = features[..., 4:5]
                neighbor_missing = features[..., 5:6]
                sensor_like = torch.sigmoid((node_missing - neighbor_missing - 0.25) / 0.10) * torch.clamp(local_missing, 0.0, 1.0)
                utility_sensor_like_parts.append(sensor_like.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(masks, axis=0)
    rows = []
    for name, pred_parts in preds_by_model.items():
        metrics = compute_metrics(np.concatenate(pred_parts, axis=0), target_np, mask_np)
        if name in regime_weights:
            joined_regime = np.concatenate([r.reshape(-1) for r in regime_weights[name]], axis=0)
            metrics["regime_weight_mean"] = float(joined_regime.mean())
        if name == "LearnedResidualUtilityRouter" and utility_weight_parts:
            joined_weights = np.concatenate([w.reshape(-1, len(CANDIDATE_NAMES)) for w in utility_weight_parts], axis=0)
            for i, candidate_name in enumerate(CANDIDATE_NAMES):
                metrics[f"utility_weight_{candidate_name}"] = float(joined_weights[:, i].mean())
            metrics["router_confidence_mean"] = float(np.concatenate([c.reshape(-1) for c in utility_confidence_parts], axis=0).mean())
            metrics["router_sensor_like_mean"] = float(np.concatenate([s.reshape(-1) for s in utility_sensor_like_parts], axis=0).mean())
        rows.append({"model": name, **metrics})
    return rows


def _run_dataset(dataset_name: str, config: dict, loader_fn) -> list[dict]:
    rows = []
    device = torch.device("cpu")
    for scenario in SCENARIOS:
        train_loader, val_loader, test_loader, adj, scaler, metadata = _unpack_loader(loader_fn(config, scenario), dataset_name)
        adj = adj.to(device)
        pinn = _train_model(config, "LiteTrustPINN_full", train_loader, val_loader, adj, scaler, device)
        correction = _train_model(config, "LiteTrustGRINCorrection", train_loader, val_loader, adj, scaler, device)
        utility_router, router_stats = _train_residual_utility_router(config, pinn, correction, train_loader, adj, scaler, device)
        val_metric_rows = _eval_regime_adaptive(config, pinn, correction, val_loader, adj, scaler, device, utility_router=utility_router)
        selected_name = min(val_metric_rows, key=lambda item: item["masked_mae"])["model"]
        metric_rows = _eval_regime_adaptive(config, pinn, correction, test_loader, adj, scaler, device, utility_router=utility_router)
        soft_utility = next(item for item in metric_rows if item["model"] == "LearnedResidualUtilityRouter")
        hard_utility = next(item for item in metric_rows if item["model"] == "LearnedResidualUtilityRouterHard")
        selected_metric = next(item for item in metric_rows if item["model"] == selected_name)
        if soft_utility.get("router_confidence_mean", 0.0) >= 0.50:
            guarded_metric = hard_utility if soft_utility.get("router_sensor_like_mean", 0.0) >= 0.15 else soft_utility
            guarded_reason = "confident_sensor_hard" if guarded_metric is hard_utility else "confident_soft"
        else:
            guarded_metric = selected_metric
            guarded_reason = "low_confidence_validation_fallback"
        for metrics in metric_rows:
            row = {
                "dataset": dataset_name,
                "scenario": scenario,
                "real_data_used": bool(metadata.get("real_data_used", False)),
                "fallback_used": bool(metadata.get("fallback_used", False)),
                **router_stats,
                **metrics,
            }
            rows.append(row)
            if metrics["model"] == selected_name:
                selected_row = dict(row)
                selected_row["model"] = "ValidationSelectedResidualRegime"
                selected_row["selected_candidate"] = selected_name
                rows.append(selected_row)
        guarded_row = {
            "dataset": dataset_name,
            "scenario": scenario,
            "real_data_used": bool(metadata.get("real_data_used", False)),
            "fallback_used": bool(metadata.get("fallback_used", False)),
            **router_stats,
            **guarded_metric,
        }
        guarded_row["model"] = "ConfidenceGuardedUtilityRouter"
        guarded_row["guard_selected_candidate"] = guarded_metric["model"]
        guarded_row["guard_reason"] = guarded_reason
        rows.append(guarded_row)
        best = min(metric_rows, key=lambda item: item["masked_mae"])
        selected_test = next(item for item in metric_rows if item["model"] == selected_name)
        print(
            f"done {dataset_name} {scenario}: best {best['model']} {best['masked_mae']:.6f}; "
            f"guard {guarded_metric['model']} {guarded_metric['masked_mae']:.6f}; "
            f"val-selected {selected_name} {selected_test['masked_mae']:.6f}",
            flush=True,
        )
    return rows


def main() -> None:
    pems = pems08_config()
    pems["seed"] = 1
    pems["train"]["epochs"] = 10
    rows = _run_dataset("PEMS08", pems, pems08_loaders)
    try:
        metrla = _metrla_config()
        metrla["seed"] = 1
        metrla["train"]["epochs"] = 10
        rows += _run_dataset("METR-LA", metrla, metrla_loaders)
    except Exception as exc:
        rows.append({"dataset": "METR-LA", "error": str(exc)})
    payload = json.dumps(rows, indent=2)
    print(payload)
    try:
        out_file = Path("C:/tmp/regime_adaptive_litetrust_quick_summary.json")
        out_file.write_text(payload, encoding="utf-8")
    except OSError as exc:
        print(f"warning: failed to save summary.json: {exc}", flush=True)


if __name__ == "__main__":
    main()
