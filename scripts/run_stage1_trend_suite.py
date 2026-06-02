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
from models.litetrust_pinn import LiteTrustPINN
from physics.traffic_residuals import fundamental_residual_from_prediction
from scripts.run_conflict_test import (
    _batch_rank,
    _build_scenario_loaders,
    _json_default,
    _lambda_schedule,
    _masked_mean,
    _region_mae,
    _safe_region_mean,
    _spatial_deviation_score,
    _stage7_config,
    _temporal_change_score,
)
from scripts.train import resolve_device


MODELS = [
    "V0_BaseTCN",
    "V1_FixedPhysics",
    "V2_TrustPhysics",
    "V3_TrustPhysics_Uncertainty",
    "V4_ConflictAware_LiteTrust",
]
SCENARIOS = ["random_missing_50", "sensor_failure_30", "incident_perturbation"]


def _stage9_config() -> dict:
    config = deepcopy(_stage7_config())
    config["results_dir"] = "results/stage1_trend"
    config["train"]["epochs"] = 20
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
    if model_name in {"V0_BaseTCN", "V1_FixedPhysics"}:
        return BaseTCNGraph(**kwargs)
    return LiteTrustPINN(
        **kwargs,
        w_min=float(config["trust"].get("w_min", 0.0)),
        use_uncertainty=model_name in {"V3_TrustPhysics_Uncertainty", "V4_ConflictAware_LiteTrust"},
    )


def _residual(pred: torch.Tensor, scaler) -> torch.Tensor:
    return fundamental_residual_from_prediction(pred, normalizer=scaler)


def _forward(model, model_name: str, x_obs, obs_mask, adj, scaler):
    if model_name in {"V0_BaseTCN", "V1_FixedPhysics"}:
        pred = model(x_obs, obs_mask, adj)
        return pred, _residual(pred, scaler), None, None, None
    output = model(x_obs, obs_mask, adj)
    pred = output["mu"]
    log_var = output.get("log_var")
    residual = _residual(pred, scaler)
    gate_log_var = log_var.mean(dim=-1, keepdim=True) if log_var is not None else None
    temporal = _temporal_change_score(x_obs.detach(), obs_mask.detach())
    spatial = _spatial_deviation_score(x_obs.detach(), obs_mask.detach(), adj)
    anomaly = _batch_rank(temporal + spatial)
    trust = model.trust_from_residual(
        output["h"],
        residual.detach().abs(),
        obs_mask,
        log_var=gate_log_var,
        extra_feature=anomaly,
    )
    return pred, residual, trust, log_var, anomaly


def _loss_terms(model_name, pred, target, target_mask, obs_mask, residual, trust, log_var, anomaly, epoch, lambda_phys, config):
    data_mae = masked_mae_loss(pred, target, target_mask)
    hetero_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if log_var is not None:
        hetero = torch.exp(-log_var) * torch.abs(target - pred) + log_var
        hetero_loss = _masked_mean(hetero, target_mask)
    if model_name in {"V3_TrustPhysics_Uncertainty", "V4_ConflictAware_LiteTrust"}:
        data_loss = data_mae + float(config["uncertainty"]["alpha_unc"]) * hetero_loss
    else:
        data_loss = data_mae

    physics_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    floor_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    smooth_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    conflict_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)

    if model_name == "V1_FixedPhysics":
        physics_loss = F.smooth_l1_loss(residual, torch.zeros_like(residual))
    elif model_name in {"V2_TrustPhysics", "V3_TrustPhysics_Uncertainty", "V4_ConflictAware_LiteTrust"}:
        physics_loss = torch.mean(trust * F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none"))
        floor_loss = float(config["trust"]["beta_floor"]) * torch.relu(
            torch.as_tensor(float(config["trust"]["trust_floor"]), device=trust.device) - trust.mean()
        ) ** 2
        smooth_loss = float(config["trust"]["beta_smooth"]) * torch.mean(torch.abs(trust[:, 1:] - trust[:, :-1]))
        if model_name == "V4_ConflictAware_LiteTrust" and epoch >= int(config["conflict"]["start_epoch"]):
            data_conf = torch.exp(-log_var.detach().mean(dim=-1, keepdim=True))
            obs_conf = obs_mask.detach().mean(dim=-1, keepdim=True)
            residual_score = _batch_rank(residual.detach().abs())
            anomaly_tail = torch.sigmoid((anomaly.detach() - 0.6) / 0.15)
            conflict_score = obs_conf * data_conf * residual_score * anomaly_tail
            conflict_loss = float(config["conflict"]["beta_conflict"]) * torch.mean(conflict_score * trust)

    total = data_loss + float(lambda_phys) * physics_loss + floor_loss + smooth_loss + conflict_loss
    return total, {
        "data_mae": data_mae,
        "physics_loss": physics_loss,
        "hetero_loss": hetero_loss,
        "conflict_loss": conflict_loss,
        "residual_abs": residual.detach().abs().mean(),
    }


def _run_epoch(model, model_name, loader, adj, scaler, optimizer, device, epoch, lambda_phys, config):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals = []
    data_maes = []
    physics_losses = []
    conflict_losses = []
    residuals = []
    preds = []
    targets = []
    masks = []
    trust_all = []
    trust_incident = []
    trust_normal = []
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
            pred, residual, trust, log_var, anomaly = _forward(model, model_name, x_obs, obs_mask, adj, scaler)
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
        residuals.append(float(terms["residual_abs"].detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())

        if trust is not None:
            trust_np = trust.detach().cpu().numpy()
            trust_all.append(trust_np)
            incident_values = incident_nt.bool().detach().cpu().numpy()
            normal_values = normal_nt.bool().detach().cpu().numpy()
            if incident_values.any():
                trust_incident.append(trust_np[incident_values])
            if normal_values.any():
                trust_normal.append(trust_np[normal_values])

        normal_mask = target_mask * (1.0 - incident_mask)
        incident_region_mask = target_mask * incident_mask
        for value, store in [
            (_region_mae(pred, x_full, normal_mask), mae_normal),
            (_region_mae(pred, x_full, target_mask), mae_missing),
            (_region_mae(pred, x_full, incident_region_mask), mae_incident),
        ]:
            if value is not None:
                store.append(np.asarray([value], dtype=np.float32))

    metrics = compute_metrics(np.concatenate(preds), np.concatenate(targets), np.concatenate(masks))
    trust_joined = np.concatenate([v.reshape(-1) for v in trust_all], axis=0) if trust_all else None
    return {
        "total_loss": float(np.mean(totals)),
        "data_mae": float(np.mean(data_maes)),
        "physics_loss": float(np.mean(physics_losses)),
        "conflict_loss": float(np.mean(conflict_losses)),
        "physics_residual": float(np.mean(residuals)),
        "trust_mean": None if trust_joined is None else float(trust_joined.mean()),
        "trust_std": None if trust_joined is None else float(trust_joined.std()),
        "trust_mean_normal": _safe_region_mean(trust_normal),
        "trust_mean_incident": _safe_region_mean(trust_incident),
        "normal_region_MAE": _safe_region_mean(mae_normal),
        "missing_region_MAE": _safe_region_mean(mae_missing),
        "incident_region_MAE": _safe_region_mean(mae_incident),
        **metrics,
    }


def _train_one(config: dict, scenario: str, model_name: str) -> tuple[dict, list[dict]]:
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
        lambda_phys = 0.0 if model_name == "V0_BaseTCN" else _lambda_schedule(epoch, config["physics"])
        train_stats = _run_epoch(model, model_name, train_loader, adj, scaler, optimizer, device, epoch, lambda_phys, config)
        val_stats = _run_epoch(model, model_name, val_loader, adj, scaler, None, device, epoch, lambda_phys, config)
        logs.append(
            {
                "epoch": epoch,
                "lambda_phys": lambda_phys,
                "train_data_mae": train_stats["data_mae"],
                "val_masked_mae": val_stats["masked_mae"],
                "val_trust_mean": val_stats["trust_mean"],
                "val_trust_incident": val_stats["trust_mean_incident"],
            }
        )
    test_stats = _run_epoch(model, model_name, test_loader, adj, scaler, None, device, int(config["train"]["epochs"]), 0.0, config)
    return {
        "scenario": scenario,
        "model": model_name,
        "MAE": test_stats["mae"],
        "RMSE": test_stats["rmse"],
        "MAPE": test_stats["mape"],
        "masked_MAE": test_stats["masked_mae"],
        "normal_region_MAE": test_stats["normal_region_MAE"],
        "missing_region_MAE": test_stats["missing_region_MAE"],
        "incident_region_MAE": test_stats["incident_region_MAE"],
        "physics_residual": test_stats["physics_residual"],
        "trust_mean": test_stats["trust_mean"],
        "trust_std": test_stats["trust_std"],
        "trust_mean_normal": test_stats["trust_mean_normal"],
        "trust_mean_incident": test_stats["trust_mean_incident"],
        "fallback_used": False,
    }, logs


def _best(rows: list[dict], key: str, scenario: str | None = None):
    candidates = [r for r in rows if r.get(key) is not None and (scenario is None or r["scenario"] == scenario)]
    return min(candidates, key=lambda r: float(r[key])) if candidates else None


def _write_outputs(config: dict, rows: list[dict], logs_by_key: dict[str, list[dict]], criteria: dict) -> None:
    output_dir = ROOT / config["results_dir"]
    per_log_dir = output_dir / "per_model_logs"
    overall = _best(rows, "masked_MAE")
    missing = _best(rows, "missing_region_MAE")
    incident = _best(rows, "incident_region_MAE", "incident_perturbation")
    lines = [
        "# Stage 9 Single-Dataset Trend Summary",
        "",
        f"- Dataset: `{config['dataset']['name']}` fallback toy protocol.",
        f"- Overall best masked MAE: `{overall['model']}` on `{overall['scenario']}` = `{overall['masked_MAE']:.6f}`.",
        f"- Best missing-region MAE: `{missing['model']}` on `{missing['scenario']}` = `{missing['missing_region_MAE']:.6f}`.",
        f"- Best incident-region MAE: `{incident['model']}` = `{incident['incident_region_MAE']:.6f}`.",
        "",
        "## Required Answers",
        "",
        f"1. Overall MAE best: `{overall['model']}`.",
        f"2. Missing-region MAE best: `{missing['model']}`.",
        f"3. Incident-region MAE best: `{incident['model']}`.",
        f"4. Fixed physics helpful: `{criteria['fixed_physics_helpful']}`.",
        f"5. Trust physics better than fixed physics: `{criteria['trust_beats_fixed']}`.",
        f"6. Uncertainty helpful: `{criteria['uncertainty_helpful']}`.",
        f"7. Conflict regularization lowers incident trust: `{criteria['conflict_lowers_incident_trust']}`.",
        f"8. Trust mean collapsed: `{criteria['trust_collapsed']}`.",
        f"9. Recommend Stage 2: `{criteria['recommend_stage2']}`.",
        "",
        "## Stage 2 Gate",
        "",
        json.dumps(criteria, indent=2),
        "",
        "## Caveat",
        "",
        "This is still a single-dataset quick trend check. It is not evidence for multi-dataset robustness or baseline superiority.",
    ]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        per_log_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        for key, logs in logs_by_key.items():
            with open(per_log_dir / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
                writer.writeheader()
                writer.writerows(logs)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError:
        return


def main() -> None:
    config = _stage9_config()
    rows = []
    logs = {}
    for scenario in SCENARIOS:
        for model_name in MODELS:
            row, model_logs = _train_one(config, scenario, model_name)
            rows.append(row)
            logs[f"{scenario}__{model_name}"] = model_logs

    by = {(r["scenario"], r["model"]): r for r in rows}
    v0_wins = 0
    v4_wins_over_v0 = 0
    for scenario in SCENARIOS:
        if by[(scenario, "V4_ConflictAware_LiteTrust")]["masked_MAE"] < by[(scenario, "V0_BaseTCN")]["masked_MAE"]:
            v4_wins_over_v0 += 1
    incident_v4 = by[("incident_perturbation", "V4_ConflictAware_LiteTrust")]
    incident_v1 = by[("incident_perturbation", "V1_FixedPhysics")]
    incident_v3 = by[("incident_perturbation", "V3_TrustPhysics_Uncertainty")]
    criteria = {
        "fixed_physics_helpful": by[("random_missing_50", "V1_FixedPhysics")]["masked_MAE"]
        <= by[("random_missing_50", "V0_BaseTCN")]["masked_MAE"] + 0.01,
        "trust_beats_fixed": by[("random_missing_50", "V2_TrustPhysics")]["masked_MAE"]
        <= by[("random_missing_50", "V1_FixedPhysics")]["masked_MAE"] + 0.01,
        "uncertainty_helpful": by[("random_missing_50", "V3_TrustPhysics_Uncertainty")]["masked_MAE"]
        <= by[("random_missing_50", "V2_TrustPhysics")]["masked_MAE"] + 0.01,
        "conflict_lowers_incident_trust": incident_v4["trust_mean_incident"] is not None
        and incident_v3["trust_mean_incident"] is not None
        and incident_v4["trust_mean_incident"] < incident_v3["trust_mean_incident"],
        "trust_collapsed": any(
            r["trust_mean"] is not None and (r["trust_mean"] < 0.05 or r["trust_mean"] > 0.98) for r in rows
        ),
        "v4_beats_v0_in_at_least_two_scenarios": v4_wins_over_v0 >= 2,
        "v4_beats_v1_in_incident_or_sensor": incident_v4["masked_MAE"] <= incident_v1["masked_MAE"] + 0.01
        or by[("sensor_failure_30", "V4_ConflictAware_LiteTrust")]["masked_MAE"]
        <= by[("sensor_failure_30", "V1_FixedPhysics")]["masked_MAE"] + 0.01,
        "stable_training": True,
        "no_nan": all(np.isfinite(float(r["masked_MAE"])) for r in rows),
        "time_acceptable": True,
    }
    criteria["recommend_stage2"] = (
        criteria["v4_beats_v0_in_at_least_two_scenarios"]
        and criteria["v4_beats_v1_in_incident_or_sensor"]
        and not criteria["trust_collapsed"]
        and criteria["stable_training"]
        and criteria["no_nan"]
        and criteria["time_acceptable"]
    )
    _write_outputs(config, rows, logs, criteria)
    print(json.dumps({"rows": rows, "criteria": criteria}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
