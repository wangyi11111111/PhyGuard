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

from data.datasets import build_dataloaders
from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.litetrust_pinn import LiteTrustPINN
from physics.traffic_residuals import fundamental_residual_from_prediction
from scripts.train import load_config, resolve_device


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _stage6_config() -> dict:
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
        "beta_floor": 0.01,
        "beta_smooth": 0.001,
        "w_min": 0.0,
    }
    config["uncertainty"] = {
        "enabled": True,
        "use_uncertainty_loss": True,
        "alpha_unc": 0.1,
        "logvar_min": -6.0,
        "logvar_max": 3.0,
    }
    config["results_dir"] = "results/v3_uncertainty"
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


def _residual(pred: torch.Tensor, scaler) -> torch.Tensor:
    return fundamental_residual_from_prediction(pred, normalizer=scaler)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _trust_stats(values: list[np.ndarray]) -> dict:
    if not values:
        return {"trust_mean": None, "trust_std": None, "trust_min": None, "trust_max": None}
    trust = np.concatenate([v.reshape(-1) for v in values], axis=0)
    return {
        "trust_mean": float(np.mean(trust)),
        "trust_std": float(np.std(trust)),
        "trust_min": float(np.min(trust)),
        "trust_max": float(np.max(trust)),
    }


def _logvar_stats(values: list[np.ndarray]) -> dict:
    if not values:
        return {"log_var_mean": None, "log_var_std": None}
    log_var = np.concatenate([v.reshape(-1) for v in values], axis=0)
    return {
        "log_var_mean": float(np.mean(log_var)),
        "log_var_std": float(np.std(log_var)),
    }


def _corr(abs_errors: list[np.ndarray], variances: list[np.ndarray]) -> float | None:
    if not abs_errors:
        return None
    err = np.concatenate([v.reshape(-1) for v in abs_errors], axis=0)
    var = np.concatenate([v.reshape(-1) for v in variances], axis=0)
    if err.size < 2 or float(np.std(err)) < 1e-8 or float(np.std(var)) < 1e-8:
        return None
    return float(np.corrcoef(err, var)[0, 1])


def _instantiate(config: dict, use_uncertainty: bool) -> LiteTrustPINN:
    model_cfg = config["model"]
    return LiteTrustPINN(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        w_min=float(config["trust"].get("w_min", 0.0)),
        use_uncertainty=use_uncertainty,
    )


def _forward(model: LiteTrustPINN, x_obs: torch.Tensor, obs_mask: torch.Tensor, adj: torch.Tensor, scaler):
    output = model(x_obs, obs_mask, adj)
    pred = output["mu"]
    log_var = output.get("log_var")
    residual = _residual(pred, scaler)
    gate_log_var = log_var.mean(dim=-1, keepdim=True) if log_var is not None else None
    trust = model.trust_from_residual(output["h"], residual.detach().abs(), obs_mask, log_var=gate_log_var)
    return pred, residual, trust, log_var


def _loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    residual: torch.Tensor,
    trust: torch.Tensor,
    log_var: torch.Tensor | None,
    lambda_phys: float,
    config: dict,
):
    data_mae = masked_mae_loss(pred, target, target_mask)
    hetero_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if log_var is not None and bool(config["uncertainty"].get("use_uncertainty_loss", True)):
        abs_error = torch.abs(target - pred)
        hetero = torch.exp(-log_var) * abs_error + log_var
        hetero_loss = _masked_mean(hetero, target_mask)
    data_loss = data_mae + float(config["uncertainty"]["alpha_unc"]) * hetero_loss

    physics_loss = torch.mean(trust * F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none"))
    trust_floor = float(config["trust"]["trust_floor"])
    beta_floor = float(config["trust"]["beta_floor"])
    beta_smooth = float(config["trust"]["beta_smooth"])
    floor_loss = beta_floor * torch.relu(torch.as_tensor(trust_floor, device=trust.device) - trust.mean()) ** 2
    smooth_loss = beta_smooth * torch.mean(torch.abs(trust[:, 1:] - trust[:, :-1]))
    total_loss = data_loss + float(lambda_phys) * physics_loss + floor_loss + smooth_loss
    return total_loss, {
        "data_loss": data_loss,
        "data_mae": data_mae,
        "hetero_loss": hetero_loss,
        "physics_loss": physics_loss,
        "floor_loss": floor_loss,
        "smooth_loss": smooth_loss,
        "residual_abs": residual.detach().abs().mean(),
    }


def _run_epoch(model, loader, adj, scaler, optimizer, device, lambda_phys, config):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals = []
    data_losses = []
    data_maes = []
    hetero_losses = []
    physics_losses = []
    residual_means = []
    preds = []
    targets = []
    masks = []
    trust_values = []
    logvar_values = []
    abs_error_values = []
    variance_values = []

    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        x_full = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        with torch.set_grad_enabled(train_mode):
            pred, residual, trust, log_var = _forward(model, x_obs, obs_mask, adj, scaler)
            total_loss, terms = _loss_terms(pred, x_full, target_mask, residual, trust, log_var, lambda_phys, config)
            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        totals.append(float(total_loss.detach().cpu()))
        data_losses.append(float(terms["data_loss"].detach().cpu()))
        data_maes.append(float(terms["data_mae"].detach().cpu()))
        hetero_losses.append(float(terms["hetero_loss"].detach().cpu()))
        physics_losses.append(float(terms["physics_loss"].detach().cpu()))
        residual_means.append(float(terms["residual_abs"].detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
        trust_values.append(trust.detach().cpu().numpy())
        if log_var is not None:
            log_np = log_var.detach().cpu().numpy()
            mask_np = target_mask.detach().cpu().numpy().astype(bool)
            err_np = torch.abs(x_full - pred).detach().cpu().numpy()
            var_np = np.exp(log_np)
            logvar_values.append(log_np)
            abs_error_values.append(err_np[mask_np])
            variance_values.append(var_np[mask_np])

    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(masks, axis=0)
    metrics = compute_metrics(pred_np, target_np, mask_np)
    return {
        "total_loss": float(np.mean(totals)),
        "data_loss": float(np.mean(data_losses)),
        "data_mae": float(np.mean(data_maes)),
        "hetero_loss": float(np.mean(hetero_losses)),
        "physics_loss": float(np.mean(physics_losses)),
        "physics_residual": float(np.mean(residual_means)),
        **_trust_stats(trust_values),
        **_logvar_stats(logvar_values),
        "uncertainty_error_correlation": _corr(abs_error_values, variance_values),
        **metrics,
    }


def _train_variant(config: dict, use_uncertainty: bool) -> dict:
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config.get("seed", 1)))
    np.random.seed(int(config.get("seed", 1)))
    train_loader, val_loader, test_loader, adj, scaler, metadata = build_dataloaders(config)
    model = _instantiate(config, use_uncertainty=use_uncertainty).to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    log_rows = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        lambda_phys = _lambda_schedule(epoch, config["physics"])
        train_stats = _run_epoch(model, train_loader, adj, scaler, optimizer, device, lambda_phys, config)
        val_stats = _run_epoch(model, val_loader, adj, scaler, None, device, lambda_phys, config)
        log_rows.append(
            {
                "epoch": epoch,
                "lambda_phys": lambda_phys,
                "train_total_loss": train_stats["total_loss"],
                "train_data_loss": train_stats["data_loss"],
                "train_data_mae": train_stats["data_mae"],
                "train_hetero_loss": train_stats["hetero_loss"],
                "train_physics_loss": train_stats["physics_loss"],
                "val_masked_mae": val_stats["masked_mae"],
                "val_physics_loss": val_stats["physics_loss"],
                "val_physics_residual": val_stats["physics_residual"],
                "trust_mean": val_stats["trust_mean"],
                "trust_std": val_stats["trust_std"],
                "trust_min": val_stats["trust_min"],
                "trust_max": val_stats["trust_max"],
                "log_var_mean": val_stats["log_var_mean"],
                "log_var_std": val_stats["log_var_std"],
                "uncertainty_error_correlation": val_stats["uncertainty_error_correlation"],
            }
        )
    test_stats = _run_epoch(model, test_loader, adj, scaler, None, device, 0.0, config)
    return {
        "test": {
            **test_stats,
            "device": str(device),
            "fallback_used": bool(metadata.get("fallback_used", False)),
        },
        "final_val_masked_mae": float(log_rows[-1]["val_masked_mae"]),
        "min_val_masked_mae": float(min(row["val_masked_mae"] for row in log_rows)),
        "final_trust_mean": float(log_rows[-1]["trust_mean"]),
        "final_trust_std": float(log_rows[-1]["trust_std"]),
        "final_trust_min": float(log_rows[-1]["trust_min"]),
        "final_trust_max": float(log_rows[-1]["trust_max"]),
        "final_log_var_mean": log_rows[-1]["log_var_mean"],
        "final_log_var_std": log_rows[-1]["log_var_std"],
        "final_uncertainty_error_correlation": log_rows[-1]["uncertainty_error_correlation"],
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
        for variant, variant_rows in summary["logs"].items():
            for row in variant_rows:
                rows.append({"variant": variant, **row})
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
    config = _stage6_config()
    v2 = _train_variant(config, use_uncertainty=False)
    v3 = _train_variant(config, use_uncertainty=True)

    corr = v3["final_uncertainty_error_correlation"]
    criteria = {
        "v3_not_worse_than_v2": v3["final_val_masked_mae"] <= v2["final_val_masked_mae"] + 0.01,
        "log_var_has_variation": v3["final_log_var_std"] is not None and float(v3["final_log_var_std"]) > 1e-4,
        "trust_mean_not_collapsed": 0.1 <= v3["final_trust_mean"] <= 0.95,
        "trust_std_positive": v3["final_trust_std"] > 1e-4,
        "uncertainty_correlation_finite": corr is not None and np.isfinite(float(corr)),
        "v3_train_mae_declined": v3["last_train_data_mae"] < v3["first_train_data_mae"],
        "no_nan": bool(
            np.isfinite(v2["final_val_masked_mae"])
            and np.isfinite(v3["final_val_masked_mae"])
            and np.isfinite(v3["final_trust_mean"])
            and np.isfinite(v3["final_trust_std"])
            and np.isfinite(float(v3["final_log_var_mean"]))
            and np.isfinite(float(v3["final_log_var_std"]))
        ),
    }
    criteria["recommend_enter_v4"] = bool(
        criteria["v3_not_worse_than_v2"]
        and criteria["log_var_has_variation"]
        and criteria["trust_mean_not_collapsed"]
        and criteria["trust_std_positive"]
        and criteria["no_nan"]
    )

    summary = {
        "stage": "stage_6_v3_uncertainty",
        "config": config,
        "v2_trust_without_uncertainty": {k: v for k, v in v2.items() if k != "log_rows"},
        "v3_trust_with_uncertainty": {k: v for k, v in v3.items() if k != "log_rows"},
        "comparison": {
            "v3_minus_v2_val_mae": float(v3["final_val_masked_mae"] - v2["final_val_masked_mae"]),
            "v3_minus_v2_test_masked_mae": float(v3["test"]["masked_mae"] - v2["test"]["masked_mae"]),
        },
        "criteria": criteria,
        "logs": {
            "v2_trust_without_uncertainty": v2["log_rows"],
            "v3_trust_with_uncertainty": v3["log_rows"],
        },
    }
    _write_artifacts(summary, config)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
