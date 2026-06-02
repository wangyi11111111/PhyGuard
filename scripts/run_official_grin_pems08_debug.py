from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.official_grin_wrapper import DEFAULT_OFFICIAL_GRIN_ROOT, OfficialGRINWrapper
from scripts.run_stage10a_pems08_real_debug import _config, _scenario_loaders
from scripts.train import resolve_device


DEFAULT_SCENARIOS = ["random_missing_50", "noise_random_missing", "incident_perturbation"]


def _run_epoch(model, loader, optimizer, device):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    preds = []
    targets = []
    masks = []
    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        target = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        with torch.set_grad_enabled(train_mode):
            pred = model(x_obs, obs_mask)
            missing_loss = masked_mae_loss(pred, target, target_mask)
            observed_loss = masked_mae_loss(pred, target, obs_mask)
            loss = missing_loss + 0.1 * observed_loss
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(masks, axis=0)
    metrics = compute_metrics(pred_np, target_np, mask_np)
    metrics["loss"] = float(np.mean(losses))
    return metrics, pred_np, target_np, mask_np


def _save_arrays(output_root: Path, scenario: str, pred: np.ndarray, target: np.ndarray, mask: np.ndarray, metrics: dict) -> None:
    run_dir = output_root / scenario
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "pred.npy", pred)
    np.save(run_dir / "true.npy", target)
    np.save(run_dir / "mask.npy", mask)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _train_scenario(config: dict, scenario: str, official_root: Path, epochs: int, output_root: Path):
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, adj, _scaler, metadata = _scenario_loaders(config, scenario)
    model = OfficialGRINWrapper(
        adj=adj.numpy(),
        input_dim=int(config["dataset"]["channels"]),
        hidden_dim=int(config["model"].get("hidden_dim", 32)),
        ff_dim=int(config["model"].get("hidden_dim", 32)),
        dropout=0.0,
        official_root=official_root,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=0.0)
    logs = []
    for epoch in range(1, epochs + 1):
        train_stats, *_ = _run_epoch(model, train_loader, optimizer, device)
        val_stats, *_ = _run_epoch(model, val_loader, None, device)
        logs.append({"epoch": epoch, "train_loss": train_stats["loss"], "val_masked_mae": val_stats["masked_mae"]})
    test_stats, pred, target, mask = _run_epoch(model, test_loader, None, device)
    _save_arrays(output_root, scenario, pred, target, mask, test_stats)
    return {
        "scenario": scenario,
        "model": "OfficialGRINet",
        "epochs": epochs,
        "real_data_used": bool(metadata.get("real_data_used", False)),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        **test_stats,
    }, logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_GRIN_ROOT))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-root", default="C:/Users/21329/litetrust_official_grin_outputs/official_grin_pems08_debug")
    args = parser.parse_args()

    config = _config()
    config["train"]["epochs"] = int(args.epochs)
    config["device"] = "cpu"
    official_root = Path(args.official_root)
    output_root = Path(args.output_root)
    rows = []
    logs_by_scenario = {}
    for scenario in args.scenarios:
        print(f"running official GRIN {scenario}", file=sys.stderr, flush=True)
        row, logs = _train_scenario(config, scenario, official_root, int(args.epochs), output_root)
        rows.append(row)
        logs_by_scenario[scenario] = logs

    try:
        with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(output_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "logs": logs_by_scenario}, f, indent=2)
    except OSError as exc:
        print(f"warning: failed to write summary: {exc}", file=sys.stderr, flush=True)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
