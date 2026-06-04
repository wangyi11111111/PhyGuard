from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pypots
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduce.run_antileakage_protocol import _load_antileakage_splits
from scripts.run_five_baselines_flow_quick import _run_pypots_model, _scenario_data
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict anti-leakage ImputeFormer baseline.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS03", "PEMS04", "PEMS08", "PEMS-BAY", "METR-LA"])
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/imputeformer_strict_seed1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    metadata = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            load_args = argparse.Namespace(**vars(args))
            train_x, val_x, test_x, adj, meta = _load_antileakage_splits(dataset, load_args)
            metadata[f"{dataset}_seed{seed}"] = meta
            for scenario in args.scenarios:
                print(f"running strict ImputeFormer {dataset} {scenario} seed={seed}", flush=True)
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                model = _make_imputeformer(
                    n_steps=int(train_x.shape[1]),
                    n_features=int(train_x.shape[2]),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    device=device,
                )
                metrics = _run_pypots_model(
                    "ImputeFormer_PyPOTS",
                    model,
                    (train_x, train_obs, train_mask),
                    (val_x, val_obs, val_mask),
                    (test_x, test_obs, test_mask),
                )
                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **metrics})
                _write(output_dir, rows, args, metadata)
    _write(output_dir, rows, args, metadata)
    print(json.dumps(rows, indent=2))
    return 0


def _write(output_dir: Path, rows: list[dict], args: argparse.Namespace, metadata: dict) -> None:
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": "strict_raw_split_imputeformer",
                "args": vars(args),
                "metadata": metadata,
                "pypots_version": getattr(pypots, "__version__", "unknown"),
                "paper": "ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation, KDD 2024",
                "rows": rows,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    raise SystemExit(main())
