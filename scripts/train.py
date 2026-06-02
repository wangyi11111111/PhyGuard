from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from data.datasets import build_dataloaders
from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.base_model import BaseTCNGraph


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, output_dir: Path) -> None:
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def instantiate_model(config: dict) -> BaseTCNGraph:
    model_cfg = config["model"]
    return BaseTCNGraph(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )


def run_epoch(model, loader, adj, optimizer, device) -> tuple[float, dict]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
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
            loss = masked_mae_loss(pred, x_full, target_mask)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(masks, axis=0),
    )
    return float(np.mean(losses)), metrics


def fit_pipeline(config: dict) -> dict:
    device = resolve_device(config.get("device", "auto"))
    torch.manual_seed(int(config.get("seed", 1)))
    np.random.seed(int(config.get("seed", 1)))

    train_loader, val_loader, test_loader, adj, scaler, metadata = build_dataloaders(config)
    model = instantiate_model(config).to(device)
    adj = adj.to(device)

    train_cfg = config["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    log_rows = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, adj, optimizer, device)
        val_loss, val_metrics = run_epoch(model, val_loader, adj, optimizer=None, device=device)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_masked_mae": train_metrics["masked_mae"],
                "val_masked_mae": val_metrics["masked_mae"],
            }
        )

    test_loss, test_metrics = run_epoch(model, test_loader, adj, optimizer=None, device=device)
    metrics = {
        **test_metrics,
        "test_loss": test_loss,
        "device": str(device),
        "fallback_used": bool(metadata.get("fallback_used", False)),
    }

    return {"metrics": metrics, "log_rows": log_rows, "config": config}


def save_artifacts(artifacts: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = artifacts["metrics"]
    log_rows = artifacts["log_rows"]
    config = artifacts["config"]

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    save_config(config, output_dir)
    return metrics


def train_pipeline(config: dict, output_dir: str | Path) -> dict:
    artifacts = fit_pipeline(config)
    return save_artifacts(artifacts, output_dir)
