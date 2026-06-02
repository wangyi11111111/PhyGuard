from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from scripts.run_five_baselines_flow_quick import _scenario_data
from scripts.run_maginet_physics_guard_quick import _load_dataset_splits, _masked_mae_np, _select_physics_candidate_from_bank
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits, _run_saits_all_splits
from scripts.train import resolve_device


def _best_residual_adapter(base_pack, phys_pack, val_full: np.ndarray, val_mask: np.ndarray, max_gamma: float, steps: int):
    base_train, base_val, base_test = base_pack
    phys_train, phys_val, phys_test = phys_pack
    val_region = 1.0 - val_mask
    best_gamma = 0.0
    best_val = _masked_mae_np(base_val, val_full, val_region)
    best_test = base_test
    for gamma in np.linspace(0.0, max_gamma, steps):
        gamma_f = float(gamma)
        pred_val = base_val + gamma_f * (phys_val - base_val)
        val_mae = _masked_mae_np(pred_val, val_full, val_region)
        if val_mae < best_val:
            best_val = val_mae
            best_gamma = gamma_f
            best_test = base_test + gamma_f * (phys_test - base_test)
    return best_test.astype(np.float32), {"adapter_gamma": best_gamma, "adapter_val_mae": best_val}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS08", choices=["PEMS03", "PEMS04", "PEMS08", "PEMS08_debug", "METR-LA", "PEMS-BAY"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "incident_perturbation", "noise_random_missing"])
    parser.add_argument("--output-dir", default="results/backbone_generality_quick")
    parser.add_argument("--max-gamma", type=float, default=2.5)
    parser.add_argument("--gamma-steps", type=int, default=51)
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
        target_mask = 1.0 - test_mask

        base_packs = {
            "MagiNet": _run_maginet_all_splits(scenario, train, val, test, adj, device, args.epochs),
            "SAITS": _run_saits_all_splits(train, val, test, device, args.epochs),
        }
        for base_name, base_pack in base_packs.items():
            base_metrics = compute_metrics(base_pack[2], test_x, target_mask)
            rows.append({"dataset": args.dataset, "scenario": scenario, "backbone": base_name, "variant": "backbone_only", **base_metrics})
            phys_pack, bank_stats, _bank = _select_physics_candidate_from_bank(train, val, test, adj, base_pack)
            adapted_test, adapter_stats = _best_residual_adapter(
                base_pack,
                phys_pack,
                val_x,
                val_mask,
                max_gamma=args.max_gamma,
                steps=args.gamma_steps,
            )
            adapted_metrics = compute_metrics(adapted_test, test_x, target_mask)
            gain = (base_metrics["masked_mae"] - adapted_metrics["masked_mae"]) / base_metrics["masked_mae"] * 100.0
            rows.append(
                {
                    "dataset": args.dataset,
                    "scenario": scenario,
                    "backbone": base_name,
                    "variant": "backbone_plus_litetrust_residual_adapter",
                    **adapted_metrics,
                    **adapter_stats,
                    **bank_stats,
                    "gain_vs_backbone_%": gain,
                }
            )

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"dataset": args.dataset, "seed": args.seed, "source": metadata.get("source", metadata.get("dataset_name", "unknown"))}, "rows": rows}, f, indent=2)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Backbone Generality Quick\n\n")
        f.write(f"- dataset: `{args.dataset}`\n")
        f.write(f"- seed: `{args.seed}`\n")
        f.write(f"- epochs: `{args.epochs}`\n")
        f.write("- adapter: validation-selected scalar residual correction using the same physics residual bank.\n\n")
        f.write("| Scenario | Backbone | Variant | masked MAE | Gain vs backbone |\n|---|---|---|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['scenario']} | {row['backbone']} | {row['variant']} | {row['masked_mae']:.6f} | {row.get('gain_vs_backbone_%', 0.0):+.2f}% |\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
