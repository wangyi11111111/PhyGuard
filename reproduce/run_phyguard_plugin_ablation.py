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
from reproduce.run_phyguard_plugin_strong_backbones import _features, _run_pypots_all_splits
from reproduce.run_strong_external_baselines import _make_brits_strong, _make_imputeformer_strong, _make_saits_strong
from scripts.run_five_baselines_flow_quick import _scenario_data
from scripts.run_maginet_physics_guard_quick import _apply_observed, _failure_mode_score, _graph_residual_np, _rank_np
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits
from scripts.train import resolve_device


FEATURE_COLUMNS = {
    "base": [0],
    "local_missing": [1],
    "failure": [2],
    "physics_residual": [3, 4, 11],
    "temporal": [5, 6, 7, 9],
    "graph_delta": [8],
    "observed_error_proxy": [10],
}


VARIANT_ZERO_COLUMNS = {
    "full": [],
    "soft_harm_0.05": [],
    "soft_harm_0.10": [],
    "soft_harm_0.20": [],
    "failure_aware_soft_harm": [],
    "no_physics_residual": FEATURE_COLUMNS["physics_residual"],
    "no_failure_score": FEATURE_COLUMNS["failure"],
    "no_temporal_features": FEATURE_COLUMNS["temporal"],
    "no_observed_error_proxy": FEATURE_COLUMNS["observed_error_proxy"],
    "mask_graph_delta_only": FEATURE_COLUMNS["failure"]
    + FEATURE_COLUMNS["physics_residual"]
    + FEATURE_COLUMNS["temporal"]
    + FEATURE_COLUMNS["observed_error_proxy"],
    "no_gate": [],
    "no_harm_loss": [],
}

VARIANT_HARM_COEF = {
    "no_harm_loss": 0.0,
    "soft_harm_0.05": 0.05,
    "soft_harm_0.10": 0.10,
    "soft_harm_0.20": 0.20,
}


class AblationPlugin(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, correction_clip: float = 0.20, use_gate: bool = True):
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
        self.gate = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = torch.tanh(self.delta(features)) * self.correction_clip
        if self.use_gate:
            gate = torch.sigmoid(self.gate(features))
        else:
            gate = torch.ones_like(delta)
        return base + gate * delta, gate, delta


def _masked_mae_np(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _ablate_features(features: np.ndarray, variant: str) -> np.ndarray:
    out = features.copy()
    for col in VARIANT_ZERO_COLUMNS[variant]:
        out[..., col] = 0.0
    return out


def _train_variant(
    train,
    val,
    test,
    adj: np.ndarray,
    base_train: np.ndarray,
    base_val: np.ndarray,
    base_test: np.ndarray,
    *,
    variant: str,
    epochs: int,
    seed: int,
    correction_clip: float,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed + 8101)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test

    feat_train = _ablate_features(_features(base_train, train_obs, train_mask, adj), variant)
    feat_val = _ablate_features(_features(base_val, val_obs, val_mask, adj), variant)
    feat_test = _ablate_features(_features(base_test, test_obs, test_mask, adj), variant)

    target_region = (1.0 - train_mask)[..., 0] > 0.0
    x = torch.tensor(feat_train[target_region], dtype=torch.float32)
    base = torch.tensor(base_train[target_region], dtype=torch.float32)
    y = torch.tensor(train_full[target_region], dtype=torch.float32)
    failure = torch.tensor(_failure_mode_score(train_mask, adj)[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    residual = torch.tensor(_rank_np(np.abs(_graph_residual_np(base_train, adj)))[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    weight = 1.0 + 0.75 * failure + 0.50 * residual

    model = AblationPlugin(
        feature_dim=x.shape[-1],
        hidden_dim=64,
        correction_clip=correction_clip,
        use_gate=(variant != "no_gate"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 8102)
    batch_size = 32768
    best_state = None
    best_val = float("inf")
    val_feat_t = torch.tensor(feat_val, dtype=torch.float32)
    val_base_t = torch.tensor(base_val, dtype=torch.float32)
    val_target_t = torch.tensor(val_full, dtype=torch.float32)
    val_region_t = torch.tensor(1.0 - val_mask, dtype=torch.float32)
    harm_coef = VARIANT_HARM_COEF.get(variant, 1.0)

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
            harm_penalty = torch.relu(pred_err - base_err) * (1.0 + 2.0 * weight[idx])
            if variant == "failure_aware_soft_harm":
                local_harm_coef = 0.05 + 0.15 * failure[idx]
                harm_loss = torch.mean(harm_penalty * local_harm_coef)
                harm_coef_for_loss = 1.0
            else:
                harm_loss = torch.mean(harm_penalty)
                harm_coef_for_loss = harm_coef
            if variant == "no_gate":
                gate_loss = torch.zeros((), dtype=pred.dtype)
            else:
                gate_loss = torch.mean(
                    F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), improvement_target, reduction="none") * weight[idx]
                )
            delta_loss = torch.mean(torch.abs(delta) * (1.0 - improvement_target) * weight[idx])
            loss = rec_loss + harm_coef_for_loss * harm_loss + 0.15 * gate_loss + 0.03 * delta_loss
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
    stats = {
        "plugin_val_mae": best_val,
        "base_test_mae": base_mae,
        "plugin_gain_pct": (base_mae - plugin_mae) / max(base_mae, 1e-8) * 100.0,
        "gate_mean": float((gate_test.numpy() * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "delta_abs_mean": float((np.abs(delta_test.numpy()) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "failure_score_mean": float((_failure_mode_score(test_mask, adj) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
    }
    return pred_np, stats


def _run_backbone(backbone: str, scenario: str, train, val, test, adj: np.ndarray, device: torch.device, epochs: int, batch_size: int):
    factories = {
        "SAITSStrong": _make_saits_strong,
        "BRITSStrong": _make_brits_strong,
        "ImputeFormerStrong": _make_imputeformer_strong,
    }
    if backbone == "MagiNetStrong":
        return _run_maginet_all_splits(scenario, train, val, test, adj, device, epochs)
    if backbone in factories:
        return _run_pypots_all_splits(factories[backbone], backbone, train, val, test, device, epochs, batch_size)
    raise ValueError(f"unknown backbone: {backbone}")


def _write(output_dir: Path, rows: list[dict], protocol: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "rows": rows}, f, indent=2)


def _aggregate(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "summary.csv")
    variant = df[df["variant"].notna()].copy()
    agg = (
        variant.groupby(["backbone", "scenario", "variant"], as_index=False)
        .agg(
            masked_mae_mean=("masked_mae", "mean"),
            masked_mae_std=("masked_mae", "std"),
            gain_pct_mean=("plugin_gain_pct", "mean"),
            gain_pct_std=("plugin_gain_pct", "std"),
            gate_mean=("gate_mean", "mean"),
            delta_abs_mean=("delta_abs_mean", "mean"),
            wins=("plugin_gain_pct", lambda x: int((x > 0).sum())),
            runs=("plugin_gain_pct", "size"),
        )
        .sort_values(["scenario", "masked_mae_mean"])
    )
    full = agg[agg["variant"] == "full"][["backbone", "scenario", "masked_mae_mean"]].rename(
        columns={"masked_mae_mean": "full_masked_mae_mean"}
    )
    agg = agg.merge(full, on=["backbone", "scenario"], how="left")
    agg["delta_vs_full_pct"] = (agg["full_masked_mae_mean"] - agg["masked_mae_mean"]) / agg["full_masked_mae_mean"] * 100.0
    agg.to_csv(output_dir / "ablation_table.csv", index=False)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# PhyGuard Component Ablation\n\n")
        f.write("Representative ablation with fixed backbone predictions. Lower masked MAE is better.\n\n")
        f.write(agg.to_string(index=False))
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="PhyGuard plugin component ablation.")
    parser.add_argument("--dataset", default="PEMS08")
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--backbone", default="SAITSStrong")
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_ZERO_COLUMNS.keys()))
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
    parser.add_argument("--output-dir", default="results/phyguard_plugin_ablation_pems08_saits_3seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    metadata = {}
    for seed in args.seeds:
        train_x, val_x, test_x, adj, meta = _load_antileakage_splits(args.dataset, argparse.Namespace(**vars(args)))
        metadata[f"{args.dataset}_seed{seed}"] = meta
        for scenario in args.scenarios:
            print(f"running ablation {args.dataset} {scenario} seed={seed} backbone={args.backbone}", flush=True)
            train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
            val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
            test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
            train = (train_x, train_obs, train_mask)
            val = (val_x, val_obs, val_mask)
            test = (test_x, test_obs, test_mask)
            pred_train, pred_val, pred_test = _run_backbone(args.backbone, scenario, train, val, test, adj, device, args.epochs, args.batch_size)
            backbone_metrics = compute_metrics(pred_test, test_x, 1.0 - test_mask)
            base_row = {"dataset": args.dataset, "scenario": scenario, "seed": seed, "backbone": args.backbone}
            rows.append({**base_row, "model": args.backbone, "variant": "backbone", **backbone_metrics})
            base_train = _apply_observed(train_obs, train_mask, pred_train)
            base_val = _apply_observed(val_obs, val_mask, pred_val)
            base_test = _apply_observed(test_obs, test_mask, pred_test)
            for variant in args.variants:
                print(f"  variant {variant}", flush=True)
                pred, stats = _train_variant(
                    train,
                    val,
                    test,
                    adj,
                    base_train,
                    base_val,
                    base_test,
                    variant=variant,
                    epochs=args.plugin_epochs,
                    seed=seed,
                    correction_clip=args.correction_clip,
                )
                rows.append(
                    {
                        **base_row,
                        "model": f"{args.backbone}+PhyGuardPlugin",
                        "variant": variant,
                        **compute_metrics(pred, test_x, 1.0 - test_mask),
                        **stats,
                    }
                )
                _write(output_dir, rows, {**vars(args), "metadata": metadata})
    _write(output_dir, rows, {**vars(args), "metadata": metadata})
    _aggregate(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
