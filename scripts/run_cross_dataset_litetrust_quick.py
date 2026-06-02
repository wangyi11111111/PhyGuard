from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage10a_pems08_real_debug import (
    _config as pems08_config,
    _instantiate,
    _lambda_schedule,
    _run_epoch,
    _scenario_loaders as pems08_loaders,
)
from scripts.run_stage2_three_dataset_quick import _build_scenario_loaders as metrla_loaders


SCENARIOS = ["random_missing_50", "sensor_failure_30"]
MODELS = ["LiteTrustGRINCorrection"]


def _metrla_config(output_dir: Path) -> dict:
    config = deepcopy(pems08_config())
    config["results_dir"] = str(output_dir / "metrla")
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


def _run_dataset(dataset_name: str, config: dict, loader_fn) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        train_loader, val_loader, test_loader, adj, scaler, metadata = _unpack_loader(loader_fn(config, scenario), dataset_name)
        for model_name in MODELS:
            torch.manual_seed(int(config["seed"]))
            np.random.seed(int(config["seed"]))
            device = torch.device("cpu")
            model = _instantiate(config, model_name).to(device)
            adj = adj.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(config["train"]["lr"]),
                weight_decay=float(config["train"].get("weight_decay", 0.0)),
            )
            logs = []
            for epoch in range(1, int(config["train"]["epochs"]) + 1):
                lambda_phys = _lambda_schedule(epoch, config["physics"])
                if model_name in {"LiteTrustGRIN", "LiteTrustGRINCorrection"} and epoch <= int(config.get("method", {}).get("pretrain_epochs", 0)):
                    lambda_phys = 0.0
                train_stats = _run_epoch(model, model_name, train_loader, adj, scaler, optimizer, device, epoch, lambda_phys, config)
                val_stats = _run_epoch(model, model_name, val_loader, adj, scaler, None, device, epoch, lambda_phys, config)
                logs.append((epoch, train_stats, val_stats))
            test_stats = _run_epoch(model, model_name, test_loader, adj, scaler, None, device, int(config["train"]["epochs"]), 0.0, config)
            row = {
                "dataset": dataset_name,
                "scenario": scenario,
                "model": model_name,
                "masked_mae": test_stats["masked_mae"],
                "rmse": test_stats["rmse"],
                "mape": test_stats["mape"],
                "trust_mean": test_stats.get("trust_mean"),
                "trust_std": test_stats.get("trust_std"),
                "form_loss": test_stats.get("form_loss"),
                "first_train_loss": logs[0][1]["total_loss"],
                "last_train_loss": logs[-1][1]["total_loss"],
                "first_val_mae": logs[0][2]["masked_mae"],
                "last_val_mae": logs[-1][2]["masked_mae"],
                "real_data_used": bool(metadata.get("real_data_used", False)),
                "fallback_used": bool(metadata.get("fallback_used", False)),
            }
            rows.append(row)
            print(f"done {dataset_name} {scenario} {model_name}: {row['masked_mae']:.6f}", flush=True)
    return rows


def main() -> None:
    output_dir = Path("results") / "stage10a_pems08_real_debug"
    pems_config = pems08_config()
    pems_config["seed"] = 1
    pems_config["train"]["epochs"] = 10
    pems_config["results_dir"] = str(output_dir / "pems08")
    metrla_config = _metrla_config(output_dir)
    metrla_config["seed"] = 1
    metrla_config["train"]["epochs"] = 10

    rows = _run_dataset("PEMS08", pems_config, pems08_loaders)
    rows += _run_dataset("METR-LA", metrla_config, metrla_loaders)

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
