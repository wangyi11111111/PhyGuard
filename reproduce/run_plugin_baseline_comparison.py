from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from reproduce.run_antileakage_protocol import _load_antileakage_splits
from reproduce.run_phyguard_plugin_strong_backbones import (
    _features,
    _run_pypots_all_splits,
    _train_plugin,
)
from reproduce.run_strong_external_baselines import _make_saits_strong
from scripts.run_five_baselines_flow_quick import _scenario_data
from scripts.run_maginet_physics_guard_quick import (
    _apply_observed,
    _failure_mode_score,
    _graph_residual_np,
    _rank_np,
)
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits
from scripts.train import resolve_device


PLUGIN_FEATURES = {
    # Pure data-driven residual adapter. It does not use physics residuals or
    # explicit failure-mode evidence.
    "GenericAdapter": [0, 1, 5, 6, 7, 8, 9, 10],
    # A calibrated adapter with a learned gate. It uses ordinary temporal and
    # mask evidence, but no physics residual and no explicit failure score.
    "CalibrationGuard": [0, 1, 5, 6, 7, 8, 9, 10],
    # A reliability guard driven by failure/anomaly cues, without physics
    # residuals. This tests whether failure awareness alone explains PhyGuard.
    "FailureAnomalyGuard": [0, 1, 2, 5, 6, 7, 8, 9, 10, 11],
}


DISPLAY_BACKBONE = {
    "SAITSStrong": "SAITS",
    "MagiNetStrong": "MagiNet",
}


class ResidualPlugin(nn.Module):
    def __init__(self, feature_dim: int, *, hidden_dim: int = 64, correction_clip: float = 0.20, use_gate: bool = True):
        super().__init__()
        self.correction_clip = float(correction_clip)
        self.use_gate = bool(use_gate)
        self.delta = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = torch.tanh(self.delta(features)) * self.correction_clip
        gate = torch.sigmoid(self.gate(features)) if self.use_gate else torch.ones_like(delta)
        return base + gate * delta, gate, delta


def _masked_mae_np(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _select_features(features: np.ndarray, plugin: str) -> np.ndarray:
    return features[..., PLUGIN_FEATURES[plugin]].astype(np.float32)


def _train_baseline_plugin(
    plugin: str,
    train,
    val,
    test,
    adj: np.ndarray,
    base_train: np.ndarray,
    base_val: np.ndarray,
    base_test: np.ndarray,
    *,
    epochs: int,
    seed: int,
    correction_clip: float,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed + 9301)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test

    feat_train = _select_features(_features(base_train, train_obs, train_mask, adj), plugin)
    feat_val = _select_features(_features(base_val, val_obs, val_mask, adj), plugin)
    feat_test = _select_features(_features(base_test, test_obs, test_mask, adj), plugin)

    target_region = (1.0 - train_mask)[..., 0] > 0.0
    x = torch.tensor(feat_train[target_region], dtype=torch.float32)
    base = torch.tensor(base_train[target_region], dtype=torch.float32)
    y = torch.tensor(train_full[target_region], dtype=torch.float32)
    failure = torch.tensor(_failure_mode_score(train_mask, adj)[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    residual_rank = torch.tensor(_rank_np(np.abs(_graph_residual_np(base_train, adj)))[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)

    if plugin == "GenericAdapter":
        weight = torch.ones_like(failure)
        use_gate = False
        harm_weight = 0.0
        gate_weight = 0.0
    elif plugin == "CalibrationGuard":
        weight = torch.ones_like(failure)
        use_gate = True
        harm_weight = 0.05
        gate_weight = 0.15
    elif plugin == "FailureAnomalyGuard":
        weight = 1.0 + 1.25 * failure + 0.25 * residual_rank
        use_gate = True
        harm_weight = 0.05 + 0.15 * failure
        gate_weight = 0.15
    else:
        raise ValueError(f"unknown plugin: {plugin}")

    model = ResidualPlugin(x.shape[-1], correction_clip=correction_clip, use_gate=use_gate)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 9302)
    batch_size = 32768
    best_state = None
    best_val = float("inf")
    val_feat_t = torch.tensor(feat_val, dtype=torch.float32)
    val_base_t = torch.tensor(base_val, dtype=torch.float32)
    val_target_t = torch.tensor(val_full, dtype=torch.float32)
    val_region_t = torch.tensor(1.0 - val_mask, dtype=torch.float32)

    for _epoch in range(max(1, epochs)):
        order = torch.randperm(x.shape[0], generator=generator)
        model.train()
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            pred, gate, delta = model(x[idx], base[idx])
            base_err = torch.abs(base[idx] - y[idx])
            pred_err = torch.abs(pred - y[idx])
            improvement_target = (pred_err.detach() + 1e-6 < base_err.detach()).float()
            rec_loss = torch.mean(pred_err * weight[idx])
            harm_penalty = torch.relu(pred_err - base_err)
            if isinstance(harm_weight, torch.Tensor):
                harm_loss = torch.mean(harm_penalty * harm_weight[idx])
            else:
                harm_loss = torch.mean(harm_penalty) * float(harm_weight)
            if use_gate:
                gate_loss = torch.mean(
                    F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), improvement_target, reduction="none") * weight[idx]
                )
            else:
                gate_loss = torch.zeros((), dtype=pred.dtype)
            delta_loss = torch.mean(torch.abs(delta) * (1.0 - improvement_target) * weight[idx])
            loss = rec_loss + harm_loss + gate_weight * gate_loss + 0.03 * delta_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val, _gate_val, _delta_val = model(val_feat_t, val_base_t)
            val_mae = float((torch.abs(pred_val - val_target_t) * val_region_t).sum() / val_region_t.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test, gate_test, delta_test = model(torch.tensor(feat_test, dtype=torch.float32), torch.tensor(base_test, dtype=torch.float32))
    pred_np = pred_test.numpy().astype(np.float32)
    test_region = 1.0 - test_mask
    base_mae = _masked_mae_np(base_test, test_full, test_region)
    plugin_mae = _masked_mae_np(pred_np, test_full, test_region)
    return pred_np, {
        "plugin_val_mae": best_val,
        "base_test_mae": base_mae,
        "plugin_gain_pct": (base_mae - plugin_mae) / max(base_mae, 1e-8) * 100.0,
        "gate_mean": float((gate_test.numpy() * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "delta_abs_mean": float((np.abs(delta_test.numpy()) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "failure_score_mean": float((_failure_mode_score(test_mask, adj) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
    }


def _write_outputs(output_dir: Path, rows: list[dict], protocol: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    if not df.empty:
        grouped = (
            df.groupby(["backbone", "plugin"], as_index=False)
            .agg(
                masked_mae_mean=("masked_mae", "mean"),
                masked_mae_std=("masked_mae", "std"),
                gain_pct_mean=("plugin_gain_pct", "mean"),
                gain_pct_std=("plugin_gain_pct", "std"),
                gate_mean=("gate_mean", "mean"),
                delta_abs_mean=("delta_abs_mean", "mean"),
            )
            .sort_values(["backbone", "masked_mae_mean"])
        )
        grouped.to_csv(output_dir / "plugin_comparison.csv", index=False)

        scenario = (
            df.groupby(["scenario", "backbone", "plugin"], as_index=False)
            .agg(masked_mae_mean=("masked_mae", "mean"), gain_pct_mean=("plugin_gain_pct", "mean"))
            .sort_values(["scenario", "backbone", "masked_mae_mean"])
        )
        scenario.to_csv(output_dir / "plugin_comparison_by_scenario.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "rows": rows}, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PhyGuard with plug-in baselines under the same backbone outputs.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS03", "PEMS04", "PEMS08", "PEMS-BAY", "METR-LA"])
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--backbones", nargs="+", default=["SAITSStrong", "MagiNetStrong"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plugin-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--correction-clip", type=float, default=0.20)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/plugin_baseline_comparison_seed1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    metadata = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            train_x, val_x, test_x, adj, meta = _load_antileakage_splits(dataset, argparse.Namespace(**vars(args)))
            metadata[f"{dataset}_seed{seed}"] = meta
            for scenario in args.scenarios:
                print(f"running plugin baselines {dataset} {scenario} seed={seed}", flush=True)
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                train = (train_x, train_obs, train_mask)
                val = (val_x, val_obs, val_mask)
                test = (test_x, test_obs, test_mask)
                for backbone in args.backbones:
                    print(f"  backbone {backbone}", flush=True)
                    if backbone == "MagiNetStrong":
                        pred_train, pred_val, pred_test = _run_maginet_all_splits(scenario, train, val, test, adj, device, args.epochs)
                    elif backbone == "SAITSStrong":
                        pred_train, pred_val, pred_test = _run_pypots_all_splits(
                            _make_saits_strong,
                            backbone,
                            train,
                            val,
                            test,
                            device,
                            args.epochs,
                            args.batch_size,
                        )
                    else:
                        raise ValueError(f"unsupported backbone for this trend test: {backbone}")

                    base_train = _apply_observed(train_obs, train_mask, pred_train)
                    base_val = _apply_observed(val_obs, val_mask, pred_val)
                    base_test = _apply_observed(test_obs, test_mask, pred_test)
                    base_metrics = compute_metrics(base_test, test_x, 1.0 - test_mask)
                    base_mae = base_metrics["masked_mae"]
                    base_name = DISPLAY_BACKBONE.get(backbone, backbone)
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "scenario": scenario,
                            "backbone": base_name,
                            "plugin": "Backbone",
                            "model": base_name,
                            "base_test_mae": base_mae,
                            "plugin_gain_pct": 0.0,
                            "gate_mean": np.nan,
                            "delta_abs_mean": np.nan,
                            "failure_score_mean": float((_failure_mode_score(test_mask, adj) * (1.0 - test_mask)).sum() / np.clip((1.0 - test_mask).sum(), 1.0, None)),
                            **base_metrics,
                        }
                    )

                    for plugin in ["GenericAdapter", "CalibrationGuard", "FailureAnomalyGuard"]:
                        plugin_pred, stats = _train_baseline_plugin(
                            plugin,
                            train,
                            val,
                            test,
                            adj,
                            base_train,
                            base_val,
                            base_test,
                            epochs=args.plugin_epochs,
                            seed=seed,
                            correction_clip=args.correction_clip,
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "scenario": scenario,
                                "backbone": base_name,
                                "plugin": plugin,
                                "model": f"{base_name}+{plugin}",
                                **compute_metrics(plugin_pred, test_x, 1.0 - test_mask),
                                **stats,
                            }
                        )

                    phyguard_pred, phyguard_stats = _train_plugin(
                        train,
                        val,
                        test,
                        adj,
                        base_train,
                        base_val,
                        base_test,
                        epochs=args.plugin_epochs,
                        seed=seed,
                        correction_clip=args.correction_clip,
                        harm_mode="failure_aware_soft",
                        fixed_harm_coef=1.0,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "scenario": scenario,
                            "backbone": base_name,
                            "plugin": "PhyGuard",
                            "model": f"{base_name}+PhyGuard",
                            **compute_metrics(phyguard_pred, test_x, 1.0 - test_mask),
                            **phyguard_stats,
                        }
                    )
                    _write_outputs(output_dir, rows, {**vars(args), "metadata": metadata})

    _write_outputs(output_dir, rows, {**vars(args), "metadata": metadata})
    print(pd.read_csv(output_dir / "plugin_comparison.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
