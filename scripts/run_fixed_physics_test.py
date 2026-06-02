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
from physics.traffic_residuals import fundamental_residual_from_prediction
from scripts.train import instantiate_model, load_config, resolve_device


def _stage4_config() -> dict:
    config = deepcopy(load_config(str(ROOT / "configs" / "default.yaml")))
    config["seed"] = 1
    config["device"] = "auto"
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
        "enabled": True,
        "lambda_max": 0.001,
        "warmup_start_epoch": 5,
        "warmup_end_epoch": 15,
        "loss": "smooth_l1",
        "residual": "fundamental_diagram",
    }
    config["results_dir"] = "results/v1_fixed_physics"
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


def _physics_loss_and_residual(pred: torch.Tensor, scaler) -> tuple[torch.Tensor, torch.Tensor]:
    residual = fundamental_residual_from_prediction(pred, normalizer=scaler)
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual))
    return loss, residual.detach().abs().mean()


def _run_epoch(model, loader, adj, scaler, optimizer, device, lambda_phys: float) -> tuple[float, float, float, dict]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_losses = []
    data_losses = []
    physics_losses = []
    residual_means = []
    preds = []
    targets = []
    masks = []

    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        x_full = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)

        with torch.set_grad_enabled(train_mode):
            pred = model(x_obs, obs_mask, adj)
            data_loss = masked_mae_loss(pred, x_full, target_mask)
            physics_loss, residual_mean = _physics_loss_and_residual(pred, scaler)
            total_loss = data_loss + float(lambda_phys) * physics_loss
            if train_mode:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_losses.append(float(total_loss.detach().cpu()))
        data_losses.append(float(data_loss.detach().cpu()))
        physics_losses.append(float(physics_loss.detach().cpu()))
        residual_means.append(float(residual_mean.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(masks, axis=0),
    )
    return (
        float(np.mean(total_losses)),
        float(np.mean(data_losses)),
        float(np.mean(physics_losses)),
        float(np.mean(residual_means)),
        metrics,
    )


def _train_variant(config: dict, use_physics: bool) -> dict:
    device = resolve_device(config.get("device", "auto"))
    torch.manual_seed(int(config.get("seed", 1)))
    np.random.seed(int(config.get("seed", 1)))

    train_loader, val_loader, test_loader, adj, scaler, metadata = build_dataloaders(config)
    model = instantiate_model(config).to(device)
    adj = adj.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    log_rows = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        lambda_phys = _lambda_schedule(epoch, config["physics"]) if use_physics else 0.0
        train_total, train_data, train_phys, train_resid, train_metrics = _run_epoch(
            model, train_loader, adj, scaler, optimizer, device, lambda_phys
        )
        val_total, val_data, val_phys, val_resid, val_metrics = _run_epoch(
            model, val_loader, adj, scaler, optimizer=None, device=device, lambda_phys=lambda_phys
        )
        log_rows.append(
            {
                "epoch": epoch,
                "lambda_phys": lambda_phys,
                "train_total_loss": train_total,
                "train_data_loss": train_data,
                "train_physics_loss": train_phys,
                "train_residual_abs": train_resid,
                "val_data_loss": val_data,
                "val_physics_loss": val_phys,
                "val_residual_abs": val_resid,
                "train_masked_mae": train_metrics["masked_mae"],
                "val_masked_mae": val_metrics["masked_mae"],
            }
        )

    test_total, test_data, test_phys, test_resid, test_metrics = _run_epoch(
        model, test_loader, adj, scaler, optimizer=None, device=device, lambda_phys=0.0
    )
    return {
        "test": {
            **test_metrics,
            "test_total_loss": test_total,
            "test_data_loss": test_data,
            "test_physics_loss": test_phys,
            "physics_residual_abs": test_resid,
            "device": str(device),
            "fallback_used": bool(metadata.get("fallback_used", False)),
        },
        "final_val_masked_mae": float(log_rows[-1]["val_masked_mae"]),
        "min_val_masked_mae": float(min(row["val_masked_mae"] for row in log_rows)),
        "first_train_data_loss": float(log_rows[0]["train_data_loss"]),
        "last_train_data_loss": float(log_rows[-1]["train_data_loss"]),
        "final_val_physics_loss": float(log_rows[-1]["val_physics_loss"]),
        "final_val_residual_abs": float(log_rows[-1]["val_residual_abs"]),
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
    config = _stage4_config()
    v0 = _train_variant(config, use_physics=False)
    v1 = _train_variant(config, use_physics=True)

    mae_delta = float(v1["final_val_masked_mae"] - v0["final_val_masked_mae"])
    residual_delta = float(v1["final_val_residual_abs"] - v0["final_val_residual_abs"])
    criteria = {
        "v0_train_loss_declined": v0["last_train_data_loss"] < v0["first_train_data_loss"],
        "v1_train_loss_declined": v1["last_train_data_loss"] < v1["first_train_data_loss"],
        "fixed_physics_reduces_val_mae": mae_delta < 0.0,
        "fixed_physics_reduces_val_residual": residual_delta < 0.0,
        "no_nan": bool(
            np.isfinite(v0["final_val_masked_mae"])
            and np.isfinite(v1["final_val_masked_mae"])
            and np.isfinite(v0["final_val_residual_abs"])
            and np.isfinite(v1["final_val_residual_abs"])
        ),
    }
    criteria["recommend_enter_v2"] = bool(
        criteria["v1_train_loss_declined"]
        and criteria["no_nan"]
        and mae_delta <= 0.01
        and criteria["fixed_physics_reduces_val_residual"]
    )

    summary = {
        "stage": "stage_4_v1_fixed_physics",
        "config": config,
        "v0_base": {k: v for k, v in v0.items() if k != "log_rows"},
        "v1_fixed_physics": {k: v for k, v in v1.items() if k != "log_rows"},
        "comparison": {
            "val_masked_mae_delta_v1_minus_v0": mae_delta,
            "val_residual_abs_delta_v1_minus_v0": residual_delta,
        },
        "criteria": criteria,
        "logs": {
            "v0_base": v0["log_rows"],
            "v1_fixed_physics": v1["log_rows"],
        },
    }
    _write_artifacts(summary, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
