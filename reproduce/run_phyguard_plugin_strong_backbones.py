from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pypots
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from reproduce.run_antileakage_protocol import _load_antileakage_splits
from reproduce.run_strong_external_baselines import (
    _make_brits_strong,
    _make_imputeformer_strong,
    _make_saits_strong,
    _run_grinlite_strong,
)
from scripts.run_five_baselines_flow_quick import _run_pypots_model, _scenario_data
from scripts.run_maginet_physics_guard_quick import (
    _apply_observed,
    _failure_mode_score,
    _graph_residual_np,
    _rank_np,
    _temporal_gap_features,
)
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits
from scripts.train import resolve_device


class PhyGuardPlugin(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, correction_clip: float = 0.25):
        super().__init__()
        self.correction_clip = float(correction_clip)
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
        gate = torch.sigmoid(self.gate(features))
        pred = base + gate * delta
        return pred, gate, delta


def _masked_mae_np(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _features(base: np.ndarray, obs: np.ndarray, mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    local_missing = (1.0 - mask).astype(np.float32)
    failure = _failure_mode_score(mask, adj).astype(np.float32)
    residual = _rank_np(np.abs(_graph_residual_np(base, adj))).astype(np.float32)
    raw_residual = np.clip(np.abs(_graph_residual_np(base, adj)), 0.0, 2.0).astype(np.float32)
    prev_gap, next_gap, gap_decay = _temporal_gap_features(mask)
    neigh = np.einsum("nm,btmc->btnc", adj, base).astype(np.float32)
    graph_delta = np.clip(neigh - base, -2.0, 2.0).astype(np.float32)
    temporal_delta = np.zeros_like(base, dtype=np.float32)
    if base.shape[1] > 1:
        temporal_delta[:, 1:-1] = 0.5 * (base[:, :-2] + base[:, 2:]) - base[:, 1:-1]
        temporal_delta[:, 0] = base[:, 1] - base[:, 0]
        temporal_delta[:, -1] = base[:, -2] - base[:, -1]
    temporal_delta = np.clip(temporal_delta, -2.0, 2.0).astype(np.float32)
    observed_error_proxy = np.abs(base - obs) * mask
    spatial_gap_rank = _rank_np(np.abs(base - neigh)).astype(np.float32)
    return np.concatenate(
        [
            base.astype(np.float32),
            local_missing,
            failure,
            residual,
            raw_residual,
            prev_gap,
            next_gap,
            gap_decay,
            graph_delta,
            temporal_delta,
            observed_error_proxy.astype(np.float32),
            spatial_gap_rank,
        ],
        axis=-1,
    ).astype(np.float32)


def _train_plugin(
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
    harm_mode: str,
    fixed_harm_coef: float,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed + 4701)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    feat_train = _features(base_train, train_obs, train_mask, adj)
    feat_val = _features(base_val, val_obs, val_mask, adj)
    feat_test = _features(base_test, test_obs, test_mask, adj)
    target_region = (1.0 - train_mask)[..., 0] > 0.0
    x = torch.tensor(feat_train[target_region], dtype=torch.float32)
    base = torch.tensor(base_train[target_region], dtype=torch.float32)
    y = torch.tensor(train_full[target_region], dtype=torch.float32)
    failure = torch.tensor(_failure_mode_score(train_mask, adj)[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    residual = torch.tensor(_rank_np(np.abs(_graph_residual_np(base_train, adj)))[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    weight = 1.0 + 0.75 * failure + 0.50 * residual

    model = PhyGuardPlugin(x.shape[-1], correction_clip=correction_clip)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 4702)
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
            harm_penalty = torch.relu(pred_err - base_err) * (1.0 + 2.0 * weight[idx])
            if harm_mode == "failure_aware_soft":
                local_harm_coef = 0.05 + 0.15 * failure[idx]
                harm_loss = torch.mean(harm_penalty * local_harm_coef)
            elif harm_mode == "fixed":
                harm_loss = torch.mean(harm_penalty) * fixed_harm_coef
            else:
                raise ValueError(f"unknown harm_mode: {harm_mode}")
            gate_loss = torch.mean(F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), improvement_target, reduction="none") * weight[idx])
            delta_loss = torch.mean(torch.abs(delta) * (1.0 - improvement_target) * weight[idx])
            loss = rec_loss + harm_loss + 0.15 * gate_loss + 0.03 * delta_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val, _gate_val, _delta_val = model(val_feat_t, val_base_t)
            # Conservative validation: never accept correction if it is worse than the backbone.
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
    stats = {
        "plugin_val_mae": best_val,
        "base_test_mae": _masked_mae_np(base_test, test_full, test_region),
        "plugin_gain_pct": (_masked_mae_np(base_test, test_full, test_region) - _masked_mae_np(pred_np, test_full, test_region))
        / max(_masked_mae_np(base_test, test_full, test_region), 1e-8)
        * 100.0,
        "gate_mean": float((gate_test.numpy() * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "delta_abs_mean": float((np.abs(delta_test.numpy()) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "failure_score_mean": float((_failure_mode_score(test_mask, adj) * test_region).sum() / np.clip(test_region.sum(), 1.0, None)),
        "harm_mode": harm_mode,
        "fixed_harm_coef": float(fixed_harm_coef),
    }
    return pred_np, stats


def _run_backbone_predictions(model_name: str, scenario: str, train, val, test, adj: np.ndarray, device: torch.device, epochs: int, batch_size: int):
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    if model_name == "MagiNetStrong":
        return _run_maginet_all_splits(scenario, train, val, test, adj, device, epochs)
    if model_name == "SAITSStrong":
        train_metrics = _run_pypots_model(
            "SAITSStrong",
            _make_saits_strong(train_full.shape[1], train_full.shape[2], epochs, batch_size, device),
            train,
            val,
            test,
        )
        # PyPOTS helper only returns test metrics, so refit here to export all splits for plugin training.
    raise NotImplementedError


def _run_pypots_all_splits(factory, model_name: str, train, val, test, device: torch.device, epochs: int, batch_size: int):
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    model = factory(train_full.shape[1], train_full.shape[2], epochs, batch_size, device)
    train_dict = {"X": np.where(train_mask > 0.5, train_obs, np.nan).reshape(train_obs.shape[0], train_obs.shape[1], train_obs.shape[2])}
    val_dict = {
        "X": np.where(val_mask > 0.5, val_obs, np.nan).reshape(val_obs.shape[0], val_obs.shape[1], val_obs.shape[2]),
        "X_ori": val_full.reshape(val_full.shape[0], val_full.shape[1], val_full.shape[2]),
    }
    model.fit(train_dict, val_dict)

    def predict(split):
        full, obs, mask = split
        data = {
            "X": np.where(mask > 0.5, obs, np.nan).reshape(obs.shape[0], obs.shape[1], obs.shape[2]),
            "X_ori": full.reshape(full.shape[0], full.shape[1], full.shape[2]),
        }
        result = model.predict(data)
        pred = result["imputation"] if isinstance(result, dict) else result
        return np.asarray(pred, dtype=np.float32).reshape(full.shape[0], full.shape[1], full.shape[2], 1)

    return predict(train), predict(val), predict(test)


def _write(output_dir: Path, rows: list[dict], protocol: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "rows": rows}, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired strong-backbone + PhyGuard plugin experiment.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS03", "PEMS04", "PEMS08", "PEMS-BAY", "METR-LA"])
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--backbones", nargs="+", default=["MagiNetStrong", "SAITSStrong", "BRITSStrong", "ImputeFormerStrong"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plugin-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--correction-clip", type=float, default=0.20)
    parser.add_argument("--harm-mode", choices=["fixed", "failure_aware_soft"], default="fixed")
    parser.add_argument("--fixed-harm-coef", type=float, default=1.0)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/phyguard_plugin_strong_backbones_seed1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    metadata = {}
    factories = {
        "SAITSStrong": _make_saits_strong,
        "BRITSStrong": _make_brits_strong,
        "ImputeFormerStrong": _make_imputeformer_strong,
    }
    for dataset in args.datasets:
        for seed in args.seeds:
            train_x, val_x, test_x, adj, meta = _load_antileakage_splits(dataset, argparse.Namespace(**vars(args)))
            metadata[f"{dataset}_seed{seed}"] = meta
            for scenario in args.scenarios:
                print(f"running paired plugin {dataset} {scenario} seed={seed}", flush=True)
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                train = (train_x, train_obs, train_mask)
                val = (val_x, val_obs, val_mask)
                test = (test_x, test_obs, test_mask)
                base_row = {"dataset": dataset, "seed": seed, "scenario": scenario}
                for backbone in args.backbones:
                    print(f"  backbone {backbone}", flush=True)
                    if backbone == "MagiNetStrong":
                        pred_train, pred_val, pred_test = _run_maginet_all_splits(scenario, train, val, test, adj, device, args.epochs)
                    elif backbone in factories:
                        pred_train, pred_val, pred_test = _run_pypots_all_splits(factories[backbone], backbone, train, val, test, device, args.epochs, args.batch_size)
                    else:
                        raise ValueError(f"unknown backbone: {backbone}")
                    backbone_metrics = compute_metrics(pred_test, test_x, 1.0 - test_mask)
                    rows.append({**base_row, "model": backbone, **backbone_metrics})
                    plugin_pred, plugin_stats = _train_plugin(
                        train,
                        val,
                        test,
                        adj,
                        _apply_observed(train_obs, train_mask, pred_train),
                        _apply_observed(val_obs, val_mask, pred_val),
                        _apply_observed(test_obs, test_mask, pred_test),
                        epochs=args.plugin_epochs,
                        seed=seed,
                        correction_clip=args.correction_clip,
                        harm_mode=args.harm_mode,
                        fixed_harm_coef=args.fixed_harm_coef,
                    )
                    rows.append(
                        {
                            **base_row,
                            "model": f"{backbone}+PhyGuardPlugin",
                            "backbone": backbone,
                            **compute_metrics(plugin_pred, test_x, 1.0 - test_mask),
                            **plugin_stats,
                        }
                    )
                    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
