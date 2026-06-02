from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pypots
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from scripts.run_five_baselines_flow_quick import _run_pypots_model, _scenario_data
from scripts.run_maginet_physics_guard_quick import _load_dataset_splits
from scripts.train import resolve_device

try:
    from pypots.imputation import ImputeFormer
except Exception as exc:  # pragma: no cover
    ImputeFormer = None
    IMPUTEFORMER_IMPORT_ERROR = exc
else:
    IMPUTEFORMER_IMPORT_ERROR = None


def _make_imputeformer(n_steps: int, n_features: int, epochs: int, batch_size: int, device: torch.device):
    if ImputeFormer is None:
        raise RuntimeError(f"PyPOTS ImputeFormer import failed: {IMPUTEFORMER_IMPORT_ERROR}")
    return ImputeFormer(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        d_input_embed=16,
        d_learnable_embed=16,
        d_proj=16,
        d_ffn=64,
        n_temporal_heads=2,
        dropout=0.1,
        input_dim=1,
        output_dim=1,
        batch_size=batch_size,
        epochs=epochs,
        patience=None,
        device=device,
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS08", choices=["PEMS03", "PEMS04", "PEMS08", "PEMS08_debug", "METR-LA", "PEMS-BAY"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50"])
    parser.add_argument("--output-dir", default="results/imputeformer_pypots_quick")
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, val_x, test_x, adj, metadata = _load_dataset_splits(args.dataset, args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario in args.scenarios:
        train_obs, train_mask = _scenario_data(train_x, adj, scenario, args.seed)
        val_obs, val_mask = _scenario_data(val_x, adj, scenario, args.seed + 11)
        test_obs, test_mask = _scenario_data(test_x, adj, scenario, args.seed + 29)
        train = (train_x, train_obs, train_mask)
        val = (val_x, val_obs, val_mask)
        test = (test_x, test_obs, test_mask)
        model = _make_imputeformer(
            n_steps=int(train_x.shape[1]),
            n_features=int(train_x.shape[2]),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=device,
        )
        metrics = _run_pypots_model("ImputeFormer_PyPOTS", model, train, val, test)
        rows.append({"dataset": args.dataset, "scenario": scenario, **metrics})

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "pypots_version": getattr(pypots, "__version__", "unknown"),
                    "source": metadata.get("source", metadata.get("dataset_name", "unknown")),
                    "model": "ImputeFormer_PyPOTS",
                    "paper": "ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation, KDD 2024",
                    "hyperparameters": {
                        "n_layers": 2,
                        "d_input_embed": 16,
                        "d_learnable_embed": 16,
                        "d_proj": 16,
                        "d_ffn": 64,
                        "n_temporal_heads": 2,
                        "dropout": 0.1,
                    },
                },
                "rows": rows,
            },
            f,
            indent=2,
        )
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# PyPOTS ImputeFormer Quick Baseline\n\n")
        f.write(f"- dataset: `{args.dataset}`\n")
        f.write(f"- seed: `{args.seed}`\n")
        f.write(f"- epochs: `{args.epochs}`\n")
        f.write(f"- batch_size: `{args.batch_size}`\n")
        f.write(f"- PyPOTS version: `{getattr(pypots, '__version__', 'unknown')}`\n")
        f.write("- paper: `ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation`, KDD 2024\n\n")
        f.write("| Scenario | Model | masked MAE | RMSE | MAPE |\n|---|---|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['scenario']} | {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} |\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
