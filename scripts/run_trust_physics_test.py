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
from models.base_model import BaseTCNGraph
from models.litetrust_pinn import LiteTrustPINN
from physics.traffic_residuals import fundamental_residual_from_prediction
from scripts.train import load_config, resolve_device


def _stage5_config() -> dict:
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
    config["results_dir"] = "results/v2_trust_physics"
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


def _trust_stats(trust_values: list[np.ndarray]) -> dict:
    if not trust_values:
        return {
            "trust_mean": None,
            "trust_std": None,
            "trust_min": None,
            "trust_max": None,
        }
    trust = np.concatenate([v.reshape(-1) for v in trust_values], axis=0)
    return {
        "trust_mean": float(np.mean(trust)),
        "trust_std": float(np.std(trust)),
        "trust_min": float(np.min(trust)),
        "trust_max": float(np.max(trust)),
    }


def _instantiate_model(config: dict, variant: str) -> torch.nn.Module:
    model_cfg = config["model"]
    kwargs = {
        "input_dim": int(model_cfg["input_dim"]),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "output_dim": int(model_cfg["output_dim"]),
        "num_layers": int(model_cfg["num_layers"]),
        "dropout": float(model_cfg["dropout"]),
    }
    if variant == "trust":
        return LiteTrustPINN(**kwargs, w_min=float(config["trust"].get("w_min", 0.0)))
    return BaseTCNGraph(**kwargs)


def _forward(model, variant: str, x_obs: torch.Tensor, obs_mask: torch.Tensor, adj: torch.Tensor, scaler):
    if variant == "trust":
        first = model(x_obs, obs_mask, adj)
        pred = first["mu"]
        residual = _residual(pred, scaler)
        trust = model.trust_from_residual(first["h"], residual.detach().abs(), obs_mask)
        return pred, residual, trust
    pred = model(x_obs, obs_mask, adj)
    residual = _residual(pred, scaler)
    return pred, residual, None


def _loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    residual: torch.Tensor,
    trust: torch.Tensor | None,
    lambda_phys: float,
    config: dict,
) -> tuple[torch.Tensor, dict]:
    data_loss = masked_mae_loss(pred, target, target_mask)
    if trust is None:
        physics_loss = F.smooth_l1_loss(residual, torch.zeros_like(residual))
        floor_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        smooth_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    else:
        physics_loss = torch.mean(trust * F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none"))
        trust_floor = float(config["trust"]["trust_floor"])
        beta_floor = float(config["trust"]["beta_floor"])
        beta_smooth = float(config["trust"]["beta_smooth"])
        floor_loss = beta_floor * torch.relu(torch.as_tensor(trust_floor, device=trust.device) - trust.mean()) ** 2
        smooth_loss = beta_smooth * torch.mean(torch.abs(trust[:, 1:] - trust[:, :-1]))
    total_loss = data_loss + float(lambda_phys) * physics_loss + floor_loss + smooth_loss
    return total_loss, {
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "floor_loss": floor_loss,
        "smooth_loss": smooth_loss,
        "residual_abs": residual.detach().abs().mean(),
    }


def _run_epoch(model, variant, loader, adj, scaler, optimizer, device, lambda_phys, config):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_losses = []
    data_losses = []
    physics_losses = []
    floor_losses = []
    smooth_losses = []
    residual_means = []
    preds = []
    targets = []
    masks = []
    trust_values = []

    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        x_full = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        with torch.set_grad_enabled(train_mode):
            pred, residual, trust = _forward(model, variant, x_obs, obs_mask, adj, scaler)
            total_loss, terms = _loss_terms(pred, x_full, target_mask, residual, trust, lambda_phys, config)
            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_losses.append(float(total_loss.detach().cpu()))
        data_losses.append(float(terms["data_loss"].detach().cpu()))
        physics_losses.append(float(terms["physics_loss"].detach().cpu()))
        floor_losses.append(float(terms["floor_loss"].detach().cpu()))
        smooth_losses.append(float(terms["smooth_loss"].detach().cpu()))
        residual_means.append(float(terms["residual_abs"].detach().cpu()))
        if trust is not None:
            trust_values.append(trust.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(masks, axis=0),
    )
    stats = {
        "total_loss": float(np.mean(total_losses)),
        "data_loss": float(np.mean(data_losses)),
        "physics_loss": float(np.mean(physics_losses)),
        "floor_loss": float(np.mean(floor_losses)),
        "smooth_loss": float(np.mean(smooth_losses)),
        "physics_residual": float(np.mean(residual_means)),
        **_trust_stats(trust_values),
        **metrics,
    }
    return stats


def _train_variant(config: dict, variant: str) -> dict:
    device = resolve_device(config.get("device", "auto"))
    torch.manual_seed(int(config.get("seed", 1)))
    np.random.seed(int(config.get("seed", 1)))
    train_loader, val_loader, test_loader, adj, scaler, metadata = build_dataloaders(config)
    model = _instantiate_model(config, variant).to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    log_rows = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        lambda_phys = _lambda_schedule(epoch, config["physics"]) if variant != "base" else 0.0
        train_stats = _run_epoch(model, variant, train_loader, adj, scaler, optimizer, device, lambda_phys, config)
        val_stats = _run_epoch(model, variant, val_loader, adj, scaler, None, device, lambda_phys, config)
        log_rows.append(
            {
                "epoch": epoch,
                "lambda_phys": lambda_phys,
                "train_data_loss": train_stats["data_loss"],
                "train_physics_loss": train_stats["physics_loss"],
                "train_total_loss": train_stats["total_loss"],
                "train_physics_residual": train_stats["physics_residual"],
                "val_masked_mae": val_stats["masked_mae"],
                "val_physics_loss": val_stats["physics_loss"],
                "val_physics_residual": val_stats["physics_residual"],
                "trust_mean": val_stats["trust_mean"],
                "trust_std": val_stats["trust_std"],
                "trust_min": val_stats["trust_min"],
                "trust_max": val_stats["trust_max"],
            }
        )
    test_stats = _run_epoch(model, variant, test_loader, adj, scaler, None, device, 0.0, config)
    return {
        "test": {
            **test_stats,
            "device": str(device),
            "fallback_used": bool(metadata.get("fallback_used", False)),
        },
        "final_val_masked_mae": float(log_rows[-1]["val_masked_mae"]),
        "min_val_masked_mae": float(min(row["val_masked_mae"] for row in log_rows)),
        "final_val_physics_residual": float(log_rows[-1]["val_physics_residual"]),
        "final_trust_mean": log_rows[-1]["trust_mean"],
        "final_trust_std": log_rows[-1]["trust_std"],
        "final_trust_min": log_rows[-1]["trust_min"],
        "final_trust_max": log_rows[-1]["trust_max"],
        "first_train_data_loss": float(log_rows[0]["train_data_loss"]),
        "last_train_data_loss": float(log_rows[-1]["train_data_loss"]),
        "log_rows": log_rows,
    }


def _write_artifacts(summary: dict, config: dict) -> None:
    output_dir = ROOT / config["results_dir"]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in summary.items() if k != "logs"}, f, indent=2)
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
    config = _stage5_config()
    v0 = _train_variant(config, "base")
    v1 = _train_variant(config, "fixed")
    v2 = _train_variant(config, "trust")

    trust_mean = float(v2["final_trust_mean"])
    trust_std = float(v2["final_trust_std"])
    criteria = {
        "v2_not_worse_than_v1": v2["final_val_masked_mae"] <= v1["final_val_masked_mae"] + 0.01,
        "trust_mean_not_collapsed": 0.1 <= trust_mean <= 0.95,
        "trust_std_positive": trust_std > 1e-4,
        "v2_train_loss_declined": v2["last_train_data_loss"] < v2["first_train_data_loss"],
        "no_nan": bool(
            np.isfinite(v0["final_val_masked_mae"])
            and np.isfinite(v1["final_val_masked_mae"])
            and np.isfinite(v2["final_val_masked_mae"])
            and np.isfinite(trust_mean)
            and np.isfinite(trust_std)
        ),
    }
    criteria["recommend_enter_v3"] = all(criteria.values())

    summary = {
        "stage": "stage_5_v2_trust_physics",
        "config": config,
        "v0_base": {k: v for k, v in v0.items() if k != "log_rows"},
        "v1_fixed_physics": {k: v for k, v in v1.items() if k != "log_rows"},
        "v2_trust_physics": {k: v for k, v in v2.items() if k != "log_rows"},
        "comparison": {
            "v1_minus_v0_val_mae": float(v1["final_val_masked_mae"] - v0["final_val_masked_mae"]),
            "v2_minus_v1_val_mae": float(v2["final_val_masked_mae"] - v1["final_val_masked_mae"]),
            "v2_minus_v0_val_mae": float(v2["final_val_masked_mae"] - v0["final_val_masked_mae"]),
        },
        "criteria": criteria,
        "logs": {
            "v0_base": v0["log_rows"],
            "v1_fixed_physics": v1["log_rows"],
            "v2_trust_physics": v2["log_rows"],
        },
    }
    _write_artifacts(summary, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
