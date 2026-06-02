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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.base_model import BaseTCNGraph
from models.litetrust_pinn import LiteTrustGRIN, LiteTrustGRINCorrection, LiteTrustPINN
from physics.traffic_residuals import channel_indices, fundamental_residual_from_prediction, graph_speed_residual
from scripts.run_conflict_test import (
    _batch_rank,
    _build_scenario_loaders,
    _anomaly_scalar,
    _json_default,
    _lambda_schedule,
    _masked_mean,
    _safe_region_mean,
    _spatial_deviation_score,
    _stage7_config,
    _temporal_change_score,
    _trust_extra_features,
    _trust_ranking_loss,
    _trust_variance_loss,
)
from scripts.train import resolve_device


DATASETS = {
    "PEMS08": {"nodes": 20, "residual": "fundamental"},
    "PEMS04": {"nodes": 24, "residual": "fundamental"},
    "METR-LA": {"nodes": 207, "residual": "graph_speed"},
}
SCENARIOS = ["random_missing_50", "sensor_failure_30"]
MODELS = ["BaseTCN", "FixedPhysics", "LiteTrustPINN_full", "LiteTrustGRIN", "LiteTrustGRINCorrection"]


def _stage10_config(dataset_name: str) -> dict:
    config = deepcopy(_stage7_config())
    config["device"] = "cpu"
    config["results_dir"] = "results/stage2_three_dataset_quick"
    config["dataset"].update(
        {
            "name": dataset_name,
            "nodes": DATASETS[dataset_name]["nodes"],
            "seq_len": 24,
            "channels": 2 if dataset_name == "METR-LA" else 3,
            "train_samples": 16,
            "val_samples": 8,
            "test_samples": 8,
            "missing_rate": 0.5,
            "source": "hf" if dataset_name == "METR-LA" else "toy",
        }
    )
    config["model"].update(
        {
            "input_dim": 4 if dataset_name == "METR-LA" else 6,
            "hidden_dim": 32,
            "output_dim": 2 if dataset_name == "METR-LA" else 3,
            "num_layers": 2,
            "dropout": 0.1,
        }
    )
    config["train"].update(
        {
            "epochs": 10,
            "batch_size": 16,
            "lr": 0.001,
            "weight_decay": 0.0,
            "num_workers": 0,
        }
    )
    config["trust"].update(
        {
            "trust_min_std": 0.08,
            "beta_variance": 0.02,
            "beta_rank": 0.05,
            "rank_margin": 0.1,
            "extra_feature_dim": 6,
        }
    )
    config["conflict"]["start_epoch"] = 5
    config["method"] = {
        "two_stage": True,
        "pretrain_epochs": 30,
        "trust_finetune_start_epoch": 31,
        "data_aux_weight": 0.5,
        "correction_l1_weight": 0.01,
        "observed_aux_weight": 0.1,
        "physics_form_weight": 0.0,
        "physics_form_temperature": 0.08,
    }
    config["dataset_residual"] = DATASETS[dataset_name]["residual"]
    return config


def _instantiate(config: dict, model_name: str) -> torch.nn.Module:
    model_cfg = config["model"]
    kwargs = {
        "input_dim": int(model_cfg["input_dim"]),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "output_dim": int(model_cfg["output_dim"]),
        "num_layers": int(model_cfg["num_layers"]),
        "dropout": float(model_cfg["dropout"]),
    }
    if model_name in {"BaseTCN", "FixedPhysics"}:
        return BaseTCNGraph(**kwargs)
    if model_name == "LiteTrustGRIN":
        return LiteTrustGRIN(
            input_dim=int(model_cfg["output_dim"]),
            hidden_dim=48,
            output_dim=int(model_cfg["output_dim"]),
            dropout=float(model_cfg["dropout"]),
            w_min=float(config["trust"].get("w_min", 0.0)),
            use_uncertainty=True,
            extra_feature_dim=int(config["trust"].get("extra_feature_dim", 1)),
        )
    if model_name == "LiteTrustGRINCorrection":
        return LiteTrustGRINCorrection(
            input_dim=int(model_cfg["output_dim"]),
            hidden_dim=48,
            output_dim=int(model_cfg["output_dim"]),
            dropout=float(model_cfg["dropout"]),
            w_min=float(config["trust"].get("w_min", 0.0)),
            use_uncertainty=True,
            extra_feature_dim=int(config["trust"].get("extra_feature_dim", 1)),
        )
    return LiteTrustPINN(
        **kwargs,
        w_min=float(config["trust"].get("w_min", 0.0)),
        use_uncertainty=True,
        extra_feature_dim=int(config["trust"].get("extra_feature_dim", 1)),
    )


def _residual(pred: torch.Tensor, adj: torch.Tensor, scaler, residual_kind: str) -> torch.Tensor:
    if residual_kind == "graph_speed":
        speed_idx = 0 if pred.shape[-1] <= 2 else channel_indices("flow_occupancy_speed")["speed"]
        speed_residual = graph_speed_residual(pred[..., speed_idx : speed_idx + 1], adj)
        pad = torch.zeros_like(speed_residual[:, :1])
        return torch.cat([pad, speed_residual], dim=1)
    return fundamental_residual_from_prediction(pred, normalizer=scaler)


def _forward(model, model_name: str, x_obs, obs_mask, adj, scaler, residual_kind: str):
    if model_name in {"BaseTCN", "FixedPhysics"}:
        pred = model(x_obs, obs_mask, adj)
        return pred, _residual(pred, adj, scaler, residual_kind), None, None, None, None
    if model_name == "LiteTrustGRINCorrection":
        base_output = model(x_obs, obs_mask, adj)
        mu_data = base_output["mu_data"]
        residual_data = _residual(mu_data, adj, scaler, residual_kind)
        log_var = base_output["log_var"]
        anomaly = _trust_extra_features(x_obs, obs_mask, adj, residual_data)
        gate_log_var = log_var.mean(dim=-1, keepdim=True)
        output = model(
            x_obs,
            obs_mask,
            adj,
            residual_abs=residual_data.detach().abs(),
            residual_signed=residual_data.detach(),
            log_var=gate_log_var,
            extra_feature=anomaly,
        )
        pred = output["mu"]
        residual = _residual(pred, adj, scaler, residual_kind)
        return pred, residual, output["trust"], log_var, anomaly, output
    output = model(x_obs, obs_mask, adj)
    pred = output["mu"]
    log_var = output["log_var"]
    residual = _residual(pred, adj, scaler, residual_kind)
    gate_log_var = log_var.mean(dim=-1, keepdim=True)
    anomaly = _trust_extra_features(x_obs, obs_mask, adj, residual)
    trust = model.trust_from_residual(
        output["h"],
        residual.detach().abs(),
        obs_mask,
        log_var=gate_log_var,
        extra_feature=anomaly,
    )
    return pred, residual, trust, log_var, anomaly, output


def _loss_terms(model_name, pred, target, target_mask, obs_mask, residual, trust, log_var, anomaly, epoch, lambda_phys, config, output=None):
    data_mae = masked_mae_loss(pred, target, target_mask)
    hetero_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if log_var is not None:
        hetero = torch.exp(-log_var) * torch.abs(target - pred) + log_var
        hetero_loss = _masked_mean(hetero, target_mask)
    pretrain_only = (
        model_name in {"LiteTrustGRIN", "LiteTrustGRINCorrection"}
        and bool(config.get("method", {}).get("two_stage", False))
        and epoch <= int(config.get("method", {}).get("pretrain_epochs", 0))
    )
    if pretrain_only and model_name == "LiteTrustGRINCorrection" and output is not None and "mu_data" in output:
        data_mae = masked_mae_loss(output["mu_data"], target, target_mask)
    data_loss = data_mae
    if model_name in {"LiteTrustPINN_full", "LiteTrustGRIN", "LiteTrustGRINCorrection"} and not pretrain_only:
        data_loss = data_mae + float(config["uncertainty"]["alpha_unc"]) * hetero_loss
    if model_name == "LiteTrustGRINCorrection" and output is not None and "mu_data" in output:
        observed_weight = float(config.get("method", {}).get("observed_aux_weight", 0.0))
        if observed_weight > 0.0:
            data_loss = data_loss + observed_weight * masked_mae_loss(output["mu_data"], target, obs_mask)
    if model_name == "LiteTrustGRINCorrection" and output is not None and "mu_data" in output:
        aux_weight = float(config.get("method", {}).get("data_aux_weight", 0.5))
        data_loss = data_loss + aux_weight * masked_mae_loss(output["mu_data"], target, target_mask)
    if model_name == "LiteTrustGRINCorrection" and output is not None and "physics_validity" in output and "x_phys" in output:
        validity_weight = float(config.get("method", {}).get("physics_validity_weight", 0.0))
        if validity_weight > 0.0:
            obs_point = obs_mask.detach().mean(dim=-1, keepdim=True)
            data_error = torch.abs(output["mu_data"].detach() - target).mean(dim=-1, keepdim=True)
            phys_error = torch.abs(output["x_phys"].detach() - target).mean(dim=-1, keepdim=True)
            target_validity = (data_error - phys_error).sigmoid().detach()
            validity_loss = _masked_mean(
                F.binary_cross_entropy(output["physics_validity"].clamp(1e-4, 1.0 - 1e-4), target_validity, reduction="none"),
                obs_point,
            )
            data_loss = data_loss + validity_weight * validity_loss

    physics_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    floor_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    smooth_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    conflict_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    variance_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    rank_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    correction_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    form_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)

    if model_name == "FixedPhysics":
        physics_loss = F.smooth_l1_loss(residual, torch.zeros_like(residual))
    elif model_name in {"LiteTrustPINN_full", "LiteTrustGRIN", "LiteTrustGRINCorrection"} and not pretrain_only:
        physics_weight = output.get("phys_weight", trust) if output is not None else trust
        regularize_trust = output.get("correction_trust", trust) if output is not None else trust
        physics_loss = torch.mean(physics_weight * F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none"))
        floor_loss = float(config["trust"]["beta_floor"]) * torch.relu(
            torch.as_tensor(float(config["trust"]["trust_floor"]), device=regularize_trust.device) - regularize_trust.mean()
        ) ** 2
        smooth_loss = float(config["trust"]["beta_smooth"]) * torch.mean(torch.abs(regularize_trust[:, 1:] - regularize_trust[:, :-1]))
        variance_loss = _trust_variance_loss(regularize_trust, config)
        if epoch >= int(config["conflict"]["start_epoch"]):
            data_conf = torch.exp(-log_var.detach().mean(dim=-1, keepdim=True))
            obs_conf = obs_mask.detach().mean(dim=-1, keepdim=True)
            residual_score = _batch_rank(residual.detach().abs())
            anomaly_tail = torch.sigmoid((_anomaly_scalar(anomaly).detach() - 0.6) / 0.15)
            conflict_score = torch.clamp(obs_conf * data_conf * residual_score * anomaly_tail, max=float(config["conflict"]["clip"]))
            conflict_loss = float(config["conflict"]["beta_conflict"]) * torch.mean(conflict_score * physics_weight)
            rank_loss = _trust_ranking_loss(physics_weight, conflict_score, config)
        if model_name == "LiteTrustGRINCorrection" and output is not None and "delta_phys" in output:
            correction_l1 = torch.mean(torch.abs(output["delta_phys"]))
            correction_loss = float(config.get("method", {}).get("correction_l1_weight", 0.01)) * correction_l1
            if "phys_residual" in output and "data_residual" in output:
                projection = torch.relu(output["phys_residual"].abs() - output["data_residual"].detach().abs()).mean()
                correction_loss = correction_loss + float(config.get("method", {}).get("physics_projection_weight", 0.0)) * projection
        if model_name == "LiteTrustGRINCorrection" and output is not None and "physics_form_weights" in output and "physics_form_candidates" in output:
            beta_form = float(config.get("method", {}).get("physics_form_weight", 0.0))
            if beta_form > 0.0:
                temperature = max(float(config.get("method", {}).get("physics_form_temperature", 0.08)), 1e-4)
                candidate_preds = output["mu_data"].unsqueeze(-1) + output["physics_form_candidates"]
                candidate_mae = torch.abs(candidate_preds - target.unsqueeze(-1)).mean(dim=-2)
                utility_target = torch.softmax(-candidate_mae.detach() / temperature, dim=-1)
                form_log_probs = torch.log(output["physics_form_weights"].clamp(1e-4, 1.0))
                form_loss = _masked_mean(
                    F.kl_div(form_log_probs, utility_target, reduction="none").sum(dim=-1, keepdim=True),
                    target_mask.mean(dim=-1, keepdim=True),
                )

    total = data_loss + float(lambda_phys) * physics_loss + floor_loss + smooth_loss + variance_loss + conflict_loss + rank_loss + correction_loss + form_loss
    return total, {
        "data_mae": data_mae,
        "physics_loss": physics_loss,
        "hetero_loss": hetero_loss,
        "conflict_loss": conflict_loss,
        "variance_loss": variance_loss,
        "rank_loss": rank_loss,
        "correction_loss": correction_loss,
        "form_loss": form_loss,
        "residual_abs": residual.detach().abs().mean(),
    }


def _run_epoch(model, model_name, loader, adj, scaler, optimizer, device, epoch, lambda_phys, config):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals = []
    data_maes = []
    physics_losses = []
    conflict_losses = []
    form_losses = []
    residuals = []
    preds = []
    targets = []
    masks = []
    trust_all = []

    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        x_full = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        with torch.set_grad_enabled(train_mode):
            pred, residual, trust, log_var, anomaly, output = _forward(
                model,
                model_name,
                x_obs,
                obs_mask,
                adj,
                scaler,
                config["dataset_residual"],
            )
            total, terms = _loss_terms(
                model_name,
                pred,
                x_full,
                target_mask,
                obs_mask,
                residual,
                trust,
                log_var,
                anomaly,
                epoch,
                lambda_phys,
                config,
                output,
            )
            if train_mode:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        totals.append(float(total.detach().cpu()))
        data_maes.append(float(terms["data_mae"].detach().cpu()))
        physics_losses.append(float(terms["physics_loss"].detach().cpu()))
        conflict_losses.append(float(terms["conflict_loss"].detach().cpu()))
        form_losses.append(float(terms.get("form_loss", torch.zeros((), device=pred.device)).detach().cpu()))
        residuals.append(float(terms["residual_abs"].detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
        if trust is not None:
            trust_all.append(trust.detach().cpu().numpy())

    metrics = compute_metrics(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), np.concatenate(masks, axis=0))
    trust_joined = np.concatenate([v.reshape(-1) for v in trust_all], axis=0) if trust_all else None
    return {
        "total_loss": float(np.mean(totals)),
        "data_mae": float(np.mean(data_maes)),
        "physics_loss": float(np.mean(physics_losses)),
        "conflict_loss": float(np.mean(conflict_losses)),
        "form_loss": float(np.mean(form_losses)),
        "physics_residual": float(np.mean(residuals)),
        "trust_mean": None if trust_joined is None else float(trust_joined.mean()),
        "trust_std": None if trust_joined is None else float(trust_joined.std()),
        **metrics,
    }


def _train_one(dataset_name: str, scenario: str, model_name: str) -> tuple[dict, list[dict]]:
    config = _stage10_config(dataset_name)
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, adj, scaler = _build_scenario_loaders(config, scenario)
    model = _instantiate(config, model_name).to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    logs = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        lambda_phys = 0.0 if model_name == "BaseTCN" else _lambda_schedule(epoch, config["physics"])
        train_stats = _run_epoch(model, model_name, train_loader, adj, scaler, optimizer, device, epoch, lambda_phys, config)
        val_stats = _run_epoch(model, model_name, val_loader, adj, scaler, None, device, epoch, lambda_phys, config)
        logs.append(
            {
                "epoch": epoch,
                "lambda_phys": lambda_phys,
                "train_data_mae": train_stats["data_mae"],
                "val_masked_mae": val_stats["masked_mae"],
                "val_physics_residual": val_stats["physics_residual"],
                "val_trust_mean": val_stats["trust_mean"],
            }
        )
    test_stats = _run_epoch(model, model_name, test_loader, adj, scaler, None, device, int(config["train"]["epochs"]), 0.0, config)
    fallback_used = not (dataset_name == "METR-LA" and str(config["dataset"].get("source", "")).lower() == "hf")
    return {
        "dataset": dataset_name,
        "scenario": scenario,
        "model": model_name,
        "residual": config["dataset_residual"],
        "fallback_used": fallback_used,
        "formal_result": False,
        "MAE": test_stats["mae"],
        "RMSE": test_stats["rmse"],
        "MAPE": test_stats["mape"],
        "masked_MAE": test_stats["masked_mae"],
        "physics_residual": test_stats["physics_residual"],
        "trust_mean": test_stats["trust_mean"],
        "trust_std": test_stats["trust_std"],
    }, logs


def _write_outputs(rows: list[dict], logs_by_key: dict[str, list[dict]]) -> None:
    output_dir = ROOT / "results" / "stage2_three_dataset_quick"
    log_dir = output_dir / "per_model_logs"
    best_by_group = {}
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            group = [r for r in rows if r["dataset"] == dataset and r["scenario"] == scenario]
            best_by_group[(dataset, scenario)] = min(group, key=lambda r: float(r["masked_MAE"]))
    summary_lines = [
        "# Stage 2 Three-Dataset Quick Summary",
        "",
        "All rows used synthetic toy fallback because real PEMS08, PEMS04, and METR-LA loaders/data are not available in this project yet.",
        "These numbers are engineering smoke/trend data only, not formal benchmark results.",
        "",
        "## Best Masked MAE",
        "",
        "| Dataset | Scenario | Best model | Masked MAE |",
        "|---|---|---:|---:|",
    ]
    for (dataset, scenario), row in best_by_group.items():
        summary_lines.append(f"| {dataset} | {scenario} | {row['model']} | {row['masked_MAE']:.6f} |")
    summary_lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- `fallback_used=true` for every row.",
            "- `formal_result=false` for every row.",
            "- METR-LA uses a speed-graph residual on the synthetic multivariate fallback tensor.",
            "- No external baselines beyond BaseTCN, FixedPhysics, LiteTrustPINN_full, and LiteTrustGRIN were run.",
        ]
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        for key, logs in logs_by_key.items():
            with open(log_dir / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
                writer.writeheader()
                writer.writerows(logs)
        with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "datasets": DATASETS,
                    "scenarios": SCENARIOS,
                    "models": MODELS,
                    "epochs": 10,
                    "train_samples": 16,
                    "val_samples": 8,
                    "test_samples": 8,
                    "device": "cpu",
                    "note": "ultra-quick debug subset because 30-epoch CPU/GPU-contended run exceeded the local timeout",
                },
                f,
                sort_keys=False,
            )
    except OSError:
        return


def main() -> None:
    rows = []
    logs_by_key = {}
    for dataset_name in DATASETS:
        for scenario in SCENARIOS:
            for model_name in MODELS:
                print(f"running {dataset_name} {scenario} {model_name}", file=sys.stderr, flush=True)
                row, logs = _train_one(dataset_name, scenario, model_name)
                rows.append(row)
                safe_dataset = dataset_name.replace("-", "_")
                logs_by_key[f"{safe_dataset}__{scenario}__{model_name}"] = logs
    _write_outputs(rows, logs_by_key)
    print(json.dumps({"rows": rows}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
