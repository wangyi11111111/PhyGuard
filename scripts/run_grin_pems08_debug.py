from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.grin_baseline import GRINLite
from scripts.run_conflict_test import _json_default
from scripts.run_stage10a_pems08_real_debug import SCENARIOS, _config, _scenario_loaders
from scripts.train import resolve_device


def _run_epoch(model, loader, adj, optimizer, device) -> dict:
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
            missing_loss = masked_mae_loss(pred, x_full, target_mask)
            observed_loss = masked_mae_loss(pred, x_full, obs_mask)
            loss = missing_loss + 0.1 * observed_loss
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
    metrics = compute_metrics(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), np.concatenate(masks, axis=0))
    return {"loss": float(np.mean(losses)), **metrics}


def _train_one(config: dict, scenario: str) -> tuple[dict, list[dict]]:
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, adj, _scaler, metadata = _scenario_loaders(config, scenario)
    model = GRINLite(input_dim=3, hidden_dim=48, output_dim=3, dropout=0.1).to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    logs = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_stats = _run_epoch(model, train_loader, adj, optimizer, device)
        val_stats = _run_epoch(model, val_loader, adj, None, device)
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_masked_mae": train_stats["masked_mae"],
                "val_loss": val_stats["loss"],
                "val_masked_mae": val_stats["masked_mae"],
            }
        )
    test_stats = _run_epoch(model, test_loader, adj, None, device)
    row = {
        "dataset": "PEMS08",
        "scenario": scenario,
        "model": "GRINLite",
        "real_data_used": bool(metadata.get("real_data_used", False)),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        "zip_path": metadata.get("zip_path"),
        "MAE": test_stats["mae"],
        "RMSE": test_stats["rmse"],
        "MAPE": test_stats["mape"],
        "masked_MAE": test_stats["masked_mae"],
    }
    return row, logs


def _write_outputs(config: dict, rows: list[dict], logs_by_key: dict[str, list[dict]]) -> None:
    output_dir = ROOT / "results" / "grin_pems08_debug"
    log_dir = output_dir / "logs"
    lines = [
        "# GRINLite PEMS08 Debug Baseline",
        "",
        "This is a compact GRIN-style graph recurrent imputation baseline, not the official GRIN implementation.",
        "",
        "| Scenario | GRINLite masked MAE |",
        "|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['masked_MAE']:.6f} |")
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
            f.write("\n".join(lines))
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "source": "GRIN-style compact baseline",
                    "dataset": "PEMS08 real debug from ASTGNN zip",
                    "epochs": config["train"]["epochs"],
                    "hidden_dim": 48,
                    "scenarios": SCENARIOS,
                },
                f,
                sort_keys=False,
            )
    except OSError:
        return


def main() -> None:
    config = _config()
    config["train"]["epochs"] = 10
    config["device"] = "cpu"
    rows = []
    logs_by_key = {}
    for scenario in SCENARIOS:
        print(f"running GRINLite PEMS08 {scenario}", file=sys.stderr, flush=True)
        row, logs = _train_one(config, scenario)
        rows.append(row)
        logs_by_key[scenario] = logs
    _write_outputs(config, rows, logs_by_key)
    print(json.dumps({"rows": rows}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
