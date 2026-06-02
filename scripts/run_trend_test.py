from __future__ import annotations

import json
import csv
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import build_dataloaders
from losses.metrics import compute_metrics
from scripts.train import fit_pipeline, load_config


def _stage3_config() -> dict:
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
            "epochs": 10,
            "batch_size": 16,
            "lr": 0.001,
            "weight_decay": 0.0,
            "num_workers": 0,
        }
    )
    config["results_dir"] = "results/v0_base_trend"
    return config


def _observed_train_mean(train_loader) -> np.ndarray:
    channel_sum = None
    channel_count = None
    for batch in train_loader:
        x_full = batch["x_full"].numpy()
        obs_mask = batch["mask"].numpy()
        if channel_sum is None:
            channel_sum = np.zeros((x_full.shape[-1],), dtype=np.float64)
            channel_count = np.zeros((x_full.shape[-1],), dtype=np.float64)
        channel_sum += (x_full * obs_mask).sum(axis=(0, 1, 2))
        channel_count += obs_mask.sum(axis=(0, 1, 2))
    return channel_sum / np.clip(channel_count, 1.0, None)


def _mean_fill_metrics(loader, channel_mean: np.ndarray) -> dict:
    preds = []
    targets = []
    masks = []
    mean = channel_mean.reshape(1, 1, 1, -1).astype(np.float32)
    for batch in loader:
        x_obs = batch["x_obs"].numpy()
        x_full = batch["x_full"].numpy()
        obs_mask = batch["mask"].numpy()
        target_mask = batch["target_mask"].numpy()
        pred = x_obs + (1.0 - obs_mask) * mean
        preds.append(pred)
        targets.append(x_full)
        masks.append(target_mask)
    return compute_metrics(
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(masks, axis=0),
    )


def _naive_baseline(config: dict) -> dict:
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, _adj, _scaler, metadata = build_dataloaders(config)
    channel_mean = _observed_train_mean(train_loader)
    return {
        "channel_mean": channel_mean.tolist(),
        "val": _mean_fill_metrics(val_loader, channel_mean),
        "test": _mean_fill_metrics(test_loader, channel_mean),
        "fallback_used": bool(metadata.get("fallback_used", False)),
    }


def main() -> None:
    config = _stage3_config()
    naive = _naive_baseline(config)
    base_artifacts = fit_pipeline(config)
    log_rows = base_artifacts["log_rows"]
    first_train_loss = float(log_rows[0]["train_loss"])
    last_train_loss = float(log_rows[-1]["train_loss"])
    base_final_val_mae = float(log_rows[-1]["val_masked_mae"])
    base_min_val_mae = float(min(row["val_masked_mae"] for row in log_rows))
    naive_val_mae = float(naive["val"]["masked_mae"])

    criteria = {
        "train_loss_declined": last_train_loss < first_train_loss,
        "base_val_beats_naive": base_final_val_mae < naive_val_mae,
        "no_nan": bool(
            np.isfinite(first_train_loss)
            and np.isfinite(last_train_loss)
            and np.isfinite(base_final_val_mae)
            and np.isfinite(float(base_artifacts["metrics"]["masked_mae"]))
        ),
    }
    criteria["passed"] = all(criteria.values())

    summary = {
        "stage": "stage_3_v0_base_trend",
        "config": config,
        "naive_mean_fill": naive,
        "base_tcn_graph": {
            "test": base_artifacts["metrics"],
            "final_val_masked_mae": base_final_val_mae,
            "min_val_masked_mae": base_min_val_mae,
            "first_train_loss": first_train_loss,
            "last_train_loss": last_train_loss,
        },
        "train_log": log_rows,
        "criteria": criteria,
    }
    output_dir = ROOT / config["results_dir"]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in summary.items() if k != "train_log"}, f, indent=2)
        with open(output_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except OSError as exc:
        expected_files = ["metrics.json", "train_log.csv", "config.yaml"]
        artifacts_exist = all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in expected_files)
        if not artifacts_exist:
            summary["artifact_save_error"] = str(exc)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
