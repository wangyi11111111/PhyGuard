from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import (
    TrafficWindowDataset,
    _generate_toy_tensor,
    _load_pems08_debug_splits,
    _normalize_splits,
)
from data.corruptions import add_gaussian_noise, incident_perturbation
from data.masks import block_missing_mask, random_missing_mask, sensor_failure_mask, temporal_missing_mask
from scripts.run_stage2_three_dataset_quick import _instantiate, _lambda_schedule, _run_epoch
from scripts.run_conflict_test import _json_default, _stage7_config
from scripts.train import resolve_device


SCENARIOS = ["random_missing_50", "sensor_failure_30"]
EXTENDED_SCENARIOS = [
    "random_missing_50",
    "sensor_failure_30",
    "block_missing",
    "temporal_missing",
    "noise_random_missing",
    "incident_perturbation",
]
MODELS = ["BaseTCN", "FixedPhysics", "LiteTrustPINN_full", "LiteTrustGRIN", "LiteTrustGRINCorrection"]


def _config() -> dict:
    config = deepcopy(_stage7_config())
    config["device"] = "cpu"
    config["results_dir"] = "results/stage10a_pems08_real_debug"
    config["dataset"].update(
        {
            "name": "pems08_debug",
            "root": "data/raw/pems08",
            "zip_path": "E:/ASTGNN-9c2e19b98c4cedf1f35214d8789685b6381b3aad.zip",
            "fallback_to_toy": True,
            "nodes": 20,
            "seq_len": 24,
            "channels": 3,
            "train_samples": 64,
            "val_samples": 16,
            "test_samples": 16,
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
    config["dataset_residual"] = "fundamental"
    config["method"] = {
        "two_stage": True,
        "pretrain_epochs": 0,
        "trust_finetune_start_epoch": 1,
        "data_aux_weight": 0.5,
        "correction_l1_weight": 0.01,
        "physics_projection_weight": 0.02,
        "physics_validity_weight": 0.05,
        "observed_aux_weight": 0.1,
        "physics_form_weight": 0.0,
        "physics_form_temperature": 0.08,
    }
    config["trust"].update(
        {
            "extra_feature_dim": 6,
            "trust_min_std": 0.08,
            "beta_variance": 0.02,
            "beta_rank": 0.05,
            "rank_margin": 0.1,
        }
    )
    return config


def _load_splits(config: dict):
    seed = int(config["seed"])
    dataset_cfg = config["dataset"]
    real = _load_pems08_debug_splits(dataset_cfg, seed)
    if real is not None:
        train_x, val_x, test_x, adj, metadata = real
    else:
        train_x, adj = _generate_toy_tensor(
            int(dataset_cfg["train_samples"]),
            int(dataset_cfg["seq_len"]),
            int(dataset_cfg["nodes"]),
            int(dataset_cfg["channels"]),
            seed,
        )
        val_x, _ = _generate_toy_tensor(
            int(dataset_cfg["val_samples"]),
            int(dataset_cfg["seq_len"]),
            int(dataset_cfg["nodes"]),
            int(dataset_cfg["channels"]),
            seed + 1,
        )
        test_x, _ = _generate_toy_tensor(
            int(dataset_cfg["test_samples"]),
            int(dataset_cfg["seq_len"]),
            int(dataset_cfg["nodes"]),
            int(dataset_cfg["channels"]),
            seed + 2,
        )
        metadata = {
            "dataset_name": "pems08_debug",
            "fallback_used": True,
            "real_data_used": False,
            "fallback_reason": "no .npz file found under data/raw/pems08",
        }
    train_x, val_x, test_x, scaler = _normalize_splits(train_x, val_x, test_x)
    metadata["real_data_used"] = not bool(metadata.get("fallback_used", False))
    return train_x, val_x, test_x, adj, scaler, metadata


def _make_loader(full_x: np.ndarray, mask: np.ndarray, batch_size: int, extra_masks: dict[str, np.ndarray] | None = None) -> DataLoader:
    obs_x = full_x * mask
    return DataLoader(TrafficWindowDataset(full_x, obs_x, mask, extra_masks=extra_masks), batch_size=batch_size, shuffle=False, num_workers=0)


def _make_loader_with_obs(
    full_x: np.ndarray,
    obs_x: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    extra_masks: dict[str, np.ndarray] | None = None,
) -> DataLoader:
    return DataLoader(TrafficWindowDataset(full_x, obs_x, mask, extra_masks=extra_masks), batch_size=batch_size, shuffle=False, num_workers=0)


def _scenario_loaders(config: dict, scenario: str):
    train_x, val_x, test_x, adj, scaler, metadata = _load_splits(config)
    seed = int(config["seed"])
    train_obs_x = None
    val_obs_x = None
    test_obs_x = None
    train_extra_masks: dict[str, np.ndarray] = {}
    val_extra_masks: dict[str, np.ndarray] = {}
    test_extra_masks: dict[str, np.ndarray] = {}
    if scenario == "random_missing_50":
        train_mask = random_missing_mask(train_x.shape, 0.5, seed=seed)
        val_mask = random_missing_mask(val_x.shape, 0.5, seed=seed + 11)
        test_mask = random_missing_mask(test_x.shape, 0.5, seed=seed + 29)
    elif scenario == "sensor_failure_30":
        train_mask = sensor_failure_mask(train_x.shape, 0.3, seed=seed)
        val_mask = sensor_failure_mask(val_x.shape, 0.3, seed=seed + 11)
        test_mask = sensor_failure_mask(test_x.shape, 0.3, seed=seed + 29)
    elif scenario == "block_missing":
        train_mask = block_missing_mask(train_x.shape, adj, block_size=max(2, int(round(train_x.shape[2] * 0.2))), seed=seed)
        val_mask = block_missing_mask(val_x.shape, adj, block_size=max(2, int(round(val_x.shape[2] * 0.2))), seed=seed + 11)
        test_mask = block_missing_mask(test_x.shape, adj, block_size=max(2, int(round(test_x.shape[2] * 0.2))), seed=seed + 29)
    elif scenario == "temporal_missing":
        train_mask = temporal_missing_mask(train_x.shape, 0.3, duration=6, seed=seed)
        val_mask = temporal_missing_mask(val_x.shape, 0.3, duration=6, seed=seed + 11)
        test_mask = temporal_missing_mask(test_x.shape, 0.3, duration=6, seed=seed + 29)
    elif scenario == "noise_random_missing":
        train_mask = random_missing_mask(train_x.shape, 0.5, seed=seed)
        val_mask = random_missing_mask(val_x.shape, 0.5, seed=seed + 11)
        test_mask = random_missing_mask(test_x.shape, 0.5, seed=seed + 29)
        train_obs_x = add_gaussian_noise(train_x, noise_std=0.15, seed=seed + 101) * train_mask
        val_obs_x = add_gaussian_noise(val_x, noise_std=0.15, seed=seed + 102) * val_mask
        test_obs_x = add_gaussian_noise(test_x, noise_std=0.15, seed=seed + 103) * test_mask
        train_extra_masks["noise_region_mask"] = train_mask.astype(np.float32)
        val_extra_masks["noise_region_mask"] = val_mask.astype(np.float32)
        test_extra_masks["noise_region_mask"] = test_mask.astype(np.float32)
    elif scenario == "incident_perturbation":
        train_mask = random_missing_mask(train_x.shape, 0.5, seed=seed)
        val_mask = random_missing_mask(val_x.shape, 0.5, seed=seed + 11)
        test_mask = random_missing_mask(test_x.shape, 0.5, seed=seed + 29)
        train_perturbed, train_incident_mask = incident_perturbation(
            train_x,
            adj,
            drop_ratio=0.5,
            duration=6,
            region_size=max(3, int(round(train_x.shape[2] * 0.2))),
            seed=seed + 201,
            flow_drop_ratio=0.0,
            speed_drop_ratio=0.5,
            return_mask=True,
        )
        val_perturbed, val_incident_mask = incident_perturbation(
            val_x,
            adj,
            drop_ratio=0.5,
            duration=6,
            region_size=max(3, int(round(val_x.shape[2] * 0.2))),
            seed=seed + 202,
            flow_drop_ratio=0.0,
            speed_drop_ratio=0.5,
            return_mask=True,
        )
        test_perturbed, test_incident_mask = incident_perturbation(
            test_x,
            adj,
            drop_ratio=0.5,
            duration=6,
            region_size=max(3, int(round(test_x.shape[2] * 0.2))),
            seed=seed + 203,
            flow_drop_ratio=0.0,
            speed_drop_ratio=0.5,
            return_mask=True,
        )
        train_obs_x = train_perturbed * train_mask
        val_obs_x = val_perturbed * val_mask
        test_obs_x = test_perturbed * test_mask
        train_extra_masks["incident_region_mask"] = train_incident_mask.astype(np.float32)
        val_extra_masks["incident_region_mask"] = val_incident_mask.astype(np.float32)
        test_extra_masks["incident_region_mask"] = test_incident_mask.astype(np.float32)
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    batch_size = int(config["train"]["batch_size"])
    if train_obs_x is not None and val_obs_x is not None and test_obs_x is not None:
        return (
            _make_loader_with_obs(train_x, train_obs_x, train_mask, batch_size, train_extra_masks),
            _make_loader_with_obs(val_x, val_obs_x, val_mask, batch_size, val_extra_masks),
            _make_loader_with_obs(test_x, test_obs_x, test_mask, batch_size, test_extra_masks),
            torch.tensor(adj, dtype=torch.float32),
            scaler,
            metadata,
        )
    return (
        _make_loader(train_x, train_mask, batch_size, train_extra_masks),
        _make_loader(val_x, val_mask, batch_size, val_extra_masks),
        _make_loader(test_x, test_mask, batch_size, test_extra_masks),
        torch.tensor(adj, dtype=torch.float32),
        scaler,
        metadata,
    )


def _train_one(config: dict, scenario: str, model_name: str) -> tuple[dict, list[dict]]:
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, adj, scaler, metadata = _scenario_loaders(config, scenario)
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
        if model_name in {"LiteTrustGRIN", "LiteTrustGRINCorrection"} and epoch <= int(config.get("method", {}).get("pretrain_epochs", 0)):
            lambda_phys = 0.0
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
    return {
        "dataset": "PEMS08",
        "scenario": scenario,
        "model": model_name,
        "real_data_used": bool(metadata.get("real_data_used", False)),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        "real_data_path": metadata.get("real_data_path"),
        "zip_path": metadata.get("zip_path"),
        "zip_data_entry": metadata.get("zip_data_entry"),
        "adjacency_fallback_ring": metadata.get("adjacency_fallback_ring"),
        "fallback_reason": metadata.get("fallback_reason"),
        "MAE": test_stats["mae"],
        "RMSE": test_stats["rmse"],
        "MAPE": test_stats["mape"],
        "masked_MAE": test_stats["masked_mae"],
        "physics_residual": test_stats["physics_residual"],
        "trust_mean": test_stats["trust_mean"],
        "trust_std": test_stats["trust_std"],
    }, logs


def _write_outputs(config: dict, rows: list[dict], logs_by_key: dict[str, list[dict]]) -> None:
    output_dir = ROOT / config["results_dir"]
    log_dir = output_dir / "per_model_logs"
    best_by_scenario = {
        scenario: min([r for r in rows if r["scenario"] == scenario], key=lambda r: float(r["masked_MAE"]))
        for scenario in SCENARIOS
    }
    lines = [
        "# Stage 10A PEMS08 Real Debug",
        "",
        f"- Real data used: `{any(r['real_data_used'] for r in rows)}`",
        f"- Fallback used: `{any(r['fallback_used'] for r in rows)}`",
        "- Models: BaseTCN, FixedPhysics, LiteTrustPINN_full",
        "- Scenarios: random_missing_50, sensor_failure_30",
        f"- Epochs: `{config['train']['epochs']}`",
        "",
        "## Best Masked MAE",
        "",
        "| Scenario | Best model | Masked MAE |",
        "|---|---|---:|",
    ]
    for scenario, row in best_by_scenario.items():
        lines.append(f"| {scenario} | {row['model']} | {row['masked_MAE']:.6f} |")
    lines.extend(
        [
            "",
            "## Full Rows",
            "",
            "| Scenario | Model | Masked MAE | Physics residual | Trust mean | Real data | Fallback |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        trust = "" if row["trust_mean"] is None else f"{row['trust_mean']:.6f}"
        lines.append(
            f"| {row['scenario']} | {row['model']} | {row['masked_MAE']:.6f} | "
            f"{row['physics_residual']:.6f} | {trust} | {row['real_data_used']} | {row['fallback_used']} |"
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
            f.write("\n".join(lines))
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except OSError:
        return


def main() -> None:
    config = _config()
    rows = []
    logs_by_key = {}
    for scenario in SCENARIOS:
        for model_name in MODELS:
            print(f"running PEMS08 {scenario} {model_name}", file=sys.stderr, flush=True)
            row, logs = _train_one(config, scenario, model_name)
            rows.append(row)
            logs_by_key[f"{scenario}__{model_name}"] = logs
    _write_outputs(config, rows, logs_by_key)
    print(json.dumps({"rows": rows}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
