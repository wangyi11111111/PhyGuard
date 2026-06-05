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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from reproduce.run_antileakage_protocol import _load_antileakage_splits
from reproduce.run_plugin_baseline_comparison import (
    GENERIC_FEATURES,
    RELIABILITY_FEATURES,
    ReliabilityConditionedPlugin,
    _masked_mae_np,
    _train_baseline_plugin,
)
from reproduce.run_phyguard_plugin_strong_backbones import _features, _run_pypots_all_splits
from reproduce.run_strong_external_baselines import _make_brits_strong, _make_imputeformer_strong, _make_saits_strong
from scripts.run_five_baselines_flow_quick import _scenario_data
from scripts.run_maginet_physics_guard_quick import _apply_observed, _failure_mode_score, _graph_residual_np, _rank_np
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits
from scripts.train import resolve_device


DISPLAY_BACKBONE = {
    "SAITSStrong": "SAITS",
    "MagiNetStrong": "MagiNet",
    "BRITSStrong": "BRITS",
    "ImputeFormerStrong": "ImputeFormer",
}


ABLATION_VARIANTS = [
    "full",
    "generic_only",
    "no_physics_promotion",
    "no_residual_bank",
    "no_conflict_suppression",
    "no_failure_evidence",
    "no_reliability_gate",
    "no_harm_regularization",
]


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


def _variant_features(features: np.ndarray, variant: str) -> np.ndarray:
    out = features.copy()
    if variant == "no_residual_bank":
        # residual rank, clipped residual, graph delta, temporal consistency, spatial gap
        out[..., [3, 4, 8, 9, 11]] = 0.0
    elif variant == "no_failure_evidence":
        out[..., 2] = 0.0
    return out


def _variant_aligned_delta(features: np.ndarray, variant: str) -> np.ndarray:
    if variant in {"no_physics_promotion", "no_residual_bank"}:
        return np.zeros(features.shape[:-1] + (1,), dtype=np.float32)
    return (0.6 * features[..., 8:9] + 0.4 * features[..., 9:10]).astype(np.float32)


def _train_phypro_variant(
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
    gate_floor: float,
    conflict_coef: float,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed + 12801)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test

    feat_train_full = _variant_features(_features(base_train, train_obs, train_mask, adj), variant)
    feat_val_full = _variant_features(_features(base_val, val_obs, val_mask, adj), variant)
    feat_test_full = _variant_features(_features(base_test, test_obs, test_mask, adj), variant)

    feat_train_g = feat_train_full[..., GENERIC_FEATURES].astype(np.float32)
    feat_val_g = feat_val_full[..., GENERIC_FEATURES].astype(np.float32)
    feat_test_g = feat_test_full[..., GENERIC_FEATURES].astype(np.float32)
    feat_train_r = feat_train_full[..., RELIABILITY_FEATURES].astype(np.float32)
    feat_val_r = feat_val_full[..., RELIABILITY_FEATURES].astype(np.float32)
    feat_test_r = feat_test_full[..., RELIABILITY_FEATURES].astype(np.float32)
    aligned_train = _variant_aligned_delta(feat_train_full, variant)
    aligned_val = _variant_aligned_delta(feat_val_full, variant)
    aligned_test = _variant_aligned_delta(feat_test_full, variant)

    target_region = (1.0 - train_mask)[..., 0] > 0.0
    xg = torch.tensor(feat_train_g[target_region], dtype=torch.float32)
    xr = torch.tensor(feat_train_r[target_region], dtype=torch.float32)
    xa = torch.tensor(aligned_train[target_region], dtype=torch.float32)
    base = torch.tensor(base_train[target_region], dtype=torch.float32)
    y = torch.tensor(train_full[target_region], dtype=torch.float32)
    failure = torch.tensor(_failure_mode_score(train_mask, adj)[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    residual_rank = torch.tensor(_rank_np(np.abs(_graph_residual_np(base_train, adj)))[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    reliability_weight = 1.0 + 0.75 * failure + 0.50 * residual_rank

    local_gate_floor = 1.0 if variant == "no_reliability_gate" else gate_floor
    local_conflict_coef = 0.0 if variant == "no_conflict_suppression" else conflict_coef
    model = ReliabilityConditionedPlugin(
        xg.shape[-1],
        xr.shape[-1],
        correction_clip=correction_clip,
        gate_floor=local_gate_floor,
        conflict_coef=local_conflict_coef,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 12802)
    batch_size = 32768
    best_state = None
    best_val = float("inf")

    val_g_t = torch.tensor(feat_val_g, dtype=torch.float32)
    val_r_t = torch.tensor(feat_val_r, dtype=torch.float32)
    val_a_t = torch.tensor(aligned_val, dtype=torch.float32)
    val_base_t = torch.tensor(base_val, dtype=torch.float32)
    val_target_t = torch.tensor(val_full, dtype=torch.float32)
    val_region_t = torch.tensor(1.0 - val_mask, dtype=torch.float32)

    for _epoch in range(max(1, epochs)):
        order = torch.randperm(xg.shape[0], generator=generator)
        model.train()
        for start in range(0, xg.shape[0], batch_size):
            idx = order[start : start + batch_size]
            pred, gate, delta, beta = model(xg[idx], xr[idx], xa[idx], base[idx])
            generic_pred = base[idx] + delta
            promoted_probe = generic_pred + torch.tanh(xa[idx]) * correction_clip
            base_err = torch.abs(base[idx] - y[idx])
            generic_err = torch.abs(generic_pred.detach() - y[idx])
            promoted_err = torch.abs(promoted_probe.detach() - y[idx])
            pred_err = torch.abs(pred - y[idx])
            utility_target = (generic_err + 1e-6 < base_err.detach()).float()
            promo_target = (promoted_err + 1e-6 < generic_err).float()
            rec_loss = torch.mean(pred_err)
            gate_loss = torch.mean(
                F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), utility_target, reduction="none") * reliability_weight[idx]
            )
            harm = torch.relu(pred_err - torch.minimum(base_err.detach(), generic_err.detach()))
            if variant == "no_harm_regularization":
                harm_loss = torch.zeros((), dtype=pred.dtype)
                promo_harm = torch.zeros((), dtype=pred.dtype)
            else:
                harm_loss = torch.mean(harm * (0.01 + 0.05 * failure[idx] + 0.02 * residual_rank[idx]))
                promo_harm = torch.mean(torch.relu(pred_err - generic_err.detach()) * beta * (0.02 + 0.05 * residual_rank[idx]))
            delta_shrink = torch.mean(torch.abs(delta) * (1.0 - utility_target) * (0.01 + 0.02 * reliability_weight[idx]))
            promo_loss = torch.mean(
                F.binary_cross_entropy(beta.clamp(1e-4, 1.0 - 1e-4), promo_target, reduction="none") * reliability_weight[idx]
            )
            loss = rec_loss + 0.05 * gate_loss + 0.05 * promo_loss + harm_loss + promo_harm + delta_shrink
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val, _gate_val, _delta_val, _beta_val = model(val_g_t, val_r_t, val_a_t, val_base_t)
            val_mae = float((torch.abs(pred_val - val_target_t) * val_region_t).sum() / val_region_t.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test, gate_test, delta_test, beta_test = model(
            torch.tensor(feat_test_g, dtype=torch.float32),
            torch.tensor(feat_test_r, dtype=torch.float32),
            torch.tensor(aligned_test, dtype=torch.float32),
            torch.tensor(base_test, dtype=torch.float32),
        )
    pred_np = pred_test.numpy().astype(np.float32)
    test_region = 1.0 - test_mask
    base_mae = _masked_mae_np(base_test, test_full, test_region)
    plugin_mae = _masked_mae_np(pred_np, test_full, test_region)
    return pred_np, {
        "plugin_val_mae": best_val,
        "base_test_mae": base_mae,
        "plugin_gain_pct": (base_mae - plugin_mae) / max(base_mae, 1e-8) * 100.0,
        "gate_mean": float((gate_test.numpy() * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "promotion_mean": float((beta_test.numpy() * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "delta_abs_mean": float((np.abs(delta_test.numpy()) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "aligned_delta_abs_mean": float((np.abs(aligned_test) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "failure_score_mean": float((_failure_mode_score(test_mask, adj) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
    }


def _write(output_dir: Path, rows: list[dict], protocol: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "rows": rows}, f, indent=2)
    if rows:
        df = pd.DataFrame(rows)
        ablation = (
            df.groupby(["dataset", "scenario", "backbone", "variant"], as_index=False)
            .agg(
                masked_mae_mean=("masked_mae", "mean"),
                masked_mae_std=("masked_mae", "std"),
                gain_pct_mean=("plugin_gain_pct", "mean"),
                gate_mean=("gate_mean", "mean"),
                promotion_mean=("promotion_mean", "mean"),
            )
            .sort_values(["dataset", "scenario", "backbone", "masked_mae_mean"])
        )
        ablation.to_csv(output_dir / "ablation_table.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="PhyPro component ablation.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS08"])
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--backbones", nargs="+", default=["SAITSStrong", "MagiNetStrong"])
    parser.add_argument("--variants", nargs="+", default=ABLATION_VARIANTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plugin-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--correction-clip", type=float, default=0.20)
    parser.add_argument("--phypro-gate-floor", type=float, default=0.95)
    parser.add_argument("--phypro-conflict-coef", type=float, default=0.75)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/phypro_ablation_pems08_saits_maginet_3seed")
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
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                train = (train_x, train_obs, train_mask)
                val = (val_x, val_obs, val_mask)
                test = (test_x, test_obs, test_mask)
                for backbone in args.backbones:
                    print(f"running PhyPro ablation {dataset} {scenario} seed={seed} backbone={backbone}", flush=True)
                    pred_train, pred_val, pred_test = _run_backbone(backbone, scenario, train, val, test, adj, device, args.epochs, args.batch_size)
                    base_train = _apply_observed(train_obs, train_mask, pred_train)
                    base_val = _apply_observed(val_obs, val_mask, pred_val)
                    base_test = _apply_observed(test_obs, test_mask, pred_test)
                    base_name = DISPLAY_BACKBONE.get(backbone, backbone)
                    base_metrics = compute_metrics(base_test, test_x, 1.0 - test_mask)
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "scenario": scenario,
                            "backbone": base_name,
                            "variant": "backbone",
                            "model": base_name,
                            "base_test_mae": base_metrics["masked_mae"],
                            "plugin_gain_pct": 0.0,
                            **base_metrics,
                        }
                    )
                    for variant in args.variants:
                        print(f"  variant {variant}", flush=True)
                        if variant == "generic_only":
                            pred, stats = _train_baseline_plugin(
                                "GenericAdapter",
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
                        else:
                            pred, stats = _train_phypro_variant(
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
                                gate_floor=args.phypro_gate_floor,
                                conflict_coef=args.phypro_conflict_coef,
                            )
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "scenario": scenario,
                                "backbone": base_name,
                                "variant": variant,
                                "model": f"{base_name}+{variant}",
                                "phypro_gate_floor": args.phypro_gate_floor,
                                "phypro_conflict_coef": args.phypro_conflict_coef,
                                **compute_metrics(pred, test_x, 1.0 - test_mask),
                                **stats,
                            }
                        )
                        _write(output_dir, rows, {**vars(args), "metadata": metadata})
    _write(output_dir, rows, {**vars(args), "metadata": metadata})
    print(pd.read_csv(output_dir / "ablation_table.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
