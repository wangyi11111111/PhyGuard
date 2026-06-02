from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from models.litetrust_pinn import _node_failure_signal
from scripts.run_five_baselines_flow_quick import (
    BRITS,
    PYPOTS_IMPORT_ERROR,
    SAITS,
    _load_flow_splits,
    _run_grinlite,
    _run_knn,
    _run_maginet,
    _run_pypots_model,
    _scenario_data,
)
from scripts.run_strong_candidate_fusion_flow_quick import (
    _physics_candidate,
    _run_maginet_all_splits,
    _run_saits_all_splits,
)
from scripts.train import resolve_device
from data.datasets import _load_metrla_hf_splits_cached, _normalize_splits


CANDIDATE_NAMES = [
    "MagiNet",
    "SAITS",
    "PhysicsFromMagi",
    "absolute_node_missing_regime",
    "contrast_sensor_regime",
    "residual_verified_regime",
]


class ResidualUtilityRouter(nn.Module):
    def __init__(self, input_dim: int, num_candidates: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, num_candidates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _rank_np(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    order = np.argsort(flat, axis=1)
    ranks = np.zeros_like(flat, dtype=np.float32)
    values = np.linspace(0.0, 1.0, flat.shape[1], dtype=np.float32)[None, :]
    np.put_along_axis(ranks, order, np.repeat(values, flat.shape[0], axis=0), axis=1)
    return ranks.reshape(x.shape)


def _missing_features(mask: np.ndarray, adj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_missing = 1.0 - mask
    node_missing = 1.0 - mask.mean(axis=(1, 3), keepdims=True)
    node_missing = np.repeat(node_missing, mask.shape[1], axis=1)
    neighbor_obs = np.einsum("nm,btmc->btnc", adj, mask).mean(axis=-1, keepdims=True)
    neighbor_missing = 1.0 - neighbor_obs
    node_failure = 1.0 / (1.0 + np.exp(-((node_missing - local_missing) / 0.15)))
    return local_missing.astype(np.float32), node_missing.astype(np.float32), neighbor_missing.astype(np.float32), node_failure.astype(np.float32)


def _candidate_predictions(x_magi: np.ndarray, x_saits: np.ndarray, x_phys: np.ndarray, obs: np.ndarray, mask: np.ndarray, adj: np.ndarray):
    local_missing, node_missing, neighbor_missing, node_failure = _missing_features(mask, adj)
    node_contrast = node_missing - neighbor_missing
    absolute_sensor = 1.0 / (1.0 + np.exp(-((node_missing - 0.7) / 0.12))) * np.clip(local_missing, 0.0, 1.0)
    absolute_regime = np.maximum(absolute_sensor, node_failure * absolute_sensor)
    contrast_sensor = 1.0 / (1.0 + np.exp(-((node_contrast - 0.25) / 0.10))) * np.clip(local_missing, 0.0, 1.0)

    residual_magi = np.zeros_like(x_magi)
    residual_magi[:, 1:] = x_magi[:, 1:] - x_magi[:, :-1] + (x_magi[:, :-1] - np.einsum("nm,btmc->btnc", adj, x_magi[:, :-1]))
    residual_phys = np.zeros_like(x_phys)
    residual_phys[:, 1:] = x_phys[:, 1:] - x_phys[:, :-1] + (x_phys[:, :-1] - np.einsum("nm,btmc->btnc", adj, x_phys[:, :-1]))
    residual_magi_rank = _rank_np(np.abs(residual_magi))
    residual_phys_rank = _rank_np(np.abs(residual_phys))
    residual_prefers_phys = 1.0 / (1.0 + np.exp(-((residual_magi_rank - residual_phys_rank - 0.05) / 0.12)))
    pred_gap_rank = _rank_np(np.abs(x_magi - x_phys))
    candidate_agreement = 1.0 / (1.0 + np.exp(-((0.70 - pred_gap_rank) / 0.12)))
    intermittent_region = 1.0 / (1.0 + np.exp(-((0.20 - np.abs(node_contrast)) / 0.10))) * np.clip(local_missing, 0.0, 1.0)
    residual_verified = np.maximum(contrast_sensor, 0.25 * intermittent_region * residual_prefers_phys * candidate_agreement)

    preds = {
        "MagiNet": x_magi,
        "SAITS": x_saits,
        "PhysicsFromMagi": x_phys,
        "absolute_node_missing_regime": (1.0 - absolute_regime) * x_magi + absolute_regime * x_phys,
        "contrast_sensor_regime": (1.0 - contrast_sensor) * x_magi + contrast_sensor * x_phys,
        "residual_verified_regime": (1.0 - residual_verified) * x_magi + residual_verified * x_phys,
    }
    regimes = {
        "absolute_node_missing_regime": absolute_regime.astype(np.float32),
        "contrast_sensor_regime": contrast_sensor.astype(np.float32),
        "residual_verified_regime": residual_verified.astype(np.float32),
    }
    features = np.concatenate(
        [
            local_missing,
            node_missing,
            neighbor_missing,
            node_failure,
            node_contrast.astype(np.float32),
            residual_magi_rank.astype(np.float32),
            residual_phys_rank.astype(np.float32),
            (residual_magi_rank - residual_phys_rank).astype(np.float32),
            pred_gap_rank.astype(np.float32),
            regimes["absolute_node_missing_regime"],
            regimes["contrast_sensor_regime"],
            regimes["residual_verified_regime"],
            np.abs(x_magi - x_saits).astype(np.float32),
            np.abs(x_magi - x_phys).astype(np.float32),
            np.abs(x_saits - x_phys).astype(np.float32),
        ],
        axis=-1,
    )
    return preds, regimes, features.astype(np.float32)


def _train_router(train_pack, seed: int):
    full, obs, mask, adj, x_magi, x_saits, x_phys = train_pack
    preds, _regimes, features = _candidate_predictions(x_magi, x_saits, x_phys, obs, mask, adj)
    candidate = np.stack([preds[name] for name in CANDIDATE_NAMES], axis=-1)
    target_mask = 1.0 - mask
    errors = np.abs(candidate - full[..., None]).mean(axis=-2)
    labels = np.argmin(errors, axis=-1).astype(np.int64)
    valid = target_mask[..., 0] > 0.0

    x = torch.tensor(features[valid], dtype=torch.float32)
    y = torch.tensor(labels[valid], dtype=torch.long)
    local_missing = torch.tensor((1.0 - mask)[..., 0][valid], dtype=torch.float32)
    node_missing = torch.tensor(features[..., 1][valid], dtype=torch.float32)
    neighbor_missing = torch.tensor(features[..., 2][valid], dtype=torch.float32)
    node_failure = _node_failure_signal(node_missing[:, None], local_missing[:, None]).squeeze(-1)
    weights = 1.0 + 1.5 * local_missing + 3.0 * node_failure + torch.clamp(node_missing - neighbor_missing, min=0.0)

    max_samples = 120_000
    if x.shape[0] > max_samples:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=generator)[:max_samples]
        x, y, weights = x[idx], y[idx], weights[idx]

    router = ResidualUtilityRouter(x.shape[-1], len(CANDIDATE_NAMES))
    opt = torch.optim.Adam(router.parameters(), lr=0.002, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 101)
    batch_size = 4096
    for _ in range(50):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            logits = router(x[idx])
            loss = torch.mean(F.cross_entropy(logits, y[idx], reduction="none") * weights[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    router.eval()
    with torch.no_grad():
        train_acc = float((torch.argmax(router(x), dim=-1) == y).float().mean())
        counts = torch.bincount(y, minlength=len(CANDIDATE_NAMES)).float()
        counts = counts / counts.sum().clamp_min(1.0)
    return router, {f"label_share_{name}": float(counts[i]) for i, name in enumerate(CANDIDATE_NAMES)} | {"router_train_acc": train_acc}


def _region_mae(pred: np.ndarray, target: np.ndarray, region_mask: np.ndarray) -> float:
    denom = float(np.clip(region_mask.sum(), 1.0, None))
    return float((np.abs(pred - target) * region_mask).sum() / denom)


def _diagnostic_metrics(
    pred_map: dict[str, np.ndarray],
    target: np.ndarray,
    target_mask: np.ndarray,
    sensor_like: np.ndarray,
) -> dict:
    sensor_region = target_mask * (sensor_like >= 0.15).astype(np.float32)
    nonsensor_region = target_mask * (sensor_like < 0.15).astype(np.float32)
    diag = {
        "sensor_region_ratio": float(sensor_region.sum() / np.clip(target_mask.sum(), 1.0, None)),
    }
    for name, pred in pred_map.items():
        key = name.lower().replace("-", "_")
        diag[f"{key}_target_mae"] = _region_mae(pred, target, target_mask)
        diag[f"{key}_sensor_mae"] = _region_mae(pred, target, sensor_region)
        diag[f"{key}_nonsensor_mae"] = _region_mae(pred, target, nonsensor_region)
    return diag


def _predict_confidence_guarded(router, split_pack, val_selected_name: str):
    full, obs, mask, adj, x_magi, x_saits, x_phys = split_pack
    preds, regimes, features = _candidate_predictions(x_magi, x_saits, x_phys, obs, mask, adj)
    with torch.no_grad():
        logits = router(torch.tensor(features, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).numpy().astype(np.float32)
    candidate = np.stack([preds[name] for name in CANDIDATE_NAMES], axis=-1)
    soft = np.sum(candidate * probs[..., None, :], axis=-1)
    hard_idx = np.argmax(probs, axis=-1)
    hard = np.take_along_axis(candidate, hard_idx[..., None, None], axis=-1)[..., 0]
    confidence = probs.max(axis=-1, keepdims=True)
    sensor_like = regimes["contrast_sensor_regime"]
    fallback = preds[val_selected_name]
    pred = np.where(
        confidence >= 0.50,
        np.where(sensor_like >= 0.15, hard, soft),
        fallback,
    ).astype(np.float32)
    if val_selected_name == "SAITS" and float(sensor_like.mean()) >= 0.15:
        preserved_pred = preds["SAITS"].astype(np.float32)
        preserve_reason = "scenario_level_saits_preservation"
    else:
        preserved_pred = pred
        preserve_reason = "confidence_guarded"
    stats = {
        "router_confidence_mean": float(confidence.mean()),
        "router_sensor_like_mean": float(sensor_like.mean()),
        "fallback_candidate": val_selected_name,
        "preserve_reason": preserve_reason,
    }
    for i, name in enumerate(CANDIDATE_NAMES):
        stats[f"utility_weight_{name}"] = float(probs[..., i].mean())
    pred_map = {
        "MagiNet": preds["MagiNet"],
        "SAITS": preds["SAITS"],
        "PhysicsFromMagi": preds["PhysicsFromMagi"],
        "ConfidenceGuarded": pred,
        "SAITSPreserved": preserved_pred,
    }
    stats.update(_diagnostic_metrics(pred_map, full, 1.0 - mask, sensor_like))
    return pred, preserved_pred, stats


def _load_dataset_splits(dataset: str, seed: int):
    dataset_key = dataset.lower()
    if dataset_key in {"pems08", "pems08_debug"}:
        train_x, val_x, test_x, adj, scaler, metadata = _load_flow_splits(seed)
        return train_x, val_x, test_x, np.asarray(adj, dtype=np.float32), metadata
    if dataset_key in {"metr-la", "metrla"}:
        train_x, val_x, test_x, adj, metadata = _load_metrla_hf_splits_cached(64, 16, 16)
        train_x, val_x, test_x, _scaler = _normalize_splits(train_x, val_x, test_x)
        return train_x[:, :12, ..., :1], val_x[:, :12, ..., :1], test_x[:, :12, ..., :1], np.asarray(adj, dtype=np.float32), metadata
    raise ValueError(f"unsupported dataset: {dataset}")


def _run_confidence_guarded(train, val, test, adj: np.ndarray, device: torch.device, epochs: int, seed: int, scenario: str):
    magi_train, magi_val, magi_test = _run_maginet_all_splits(scenario, train, val, test, adj, device, epochs)
    saits_train, saits_val, saits_test = _run_saits_all_splits(train, val, test, device, epochs)
    phys_train = _physics_candidate(magi_train, train[1], train[2], adj)
    phys_val = _physics_candidate(magi_val, val[1], val[2], adj)
    phys_test = _physics_candidate(magi_test, test[1], test[2], adj)

    router, router_stats = _train_router((train[0], train[1], train[2], adj, magi_train, saits_train, phys_train), seed)
    val_preds, _val_regimes, _val_features = _candidate_predictions(magi_val, saits_val, phys_val, val[1], val[2], adj)
    val_selected = min(
        CANDIDATE_NAMES,
        key=lambda name: compute_metrics(val_preds[name], val[0], 1.0 - val[2])["masked_mae"],
    )
    pred, preserved_pred, pred_stats = _predict_confidence_guarded(router, (test[0], test[1], test[2], adj, magi_test, saits_test, phys_test), val_selected)
    guarded = {
        **compute_metrics(pred, test[0], 1.0 - test[2]),
        "model": "ConfidenceGuardedUtilityRouter",
        **router_stats,
        **pred_stats,
    }
    preserved = {
        **compute_metrics(preserved_pred, test[0], 1.0 - test[2]),
        "model": "SAITSPreservedConfidenceGuarded",
        **router_stats,
        **pred_stats,
    }
    return [
        {"model": "MagiNet", **compute_metrics(magi_test, test[0], 1.0 - test[2])},
        {"model": "SAITS", **compute_metrics(saits_test, test[0], 1.0 - test[2])},
        {"model": "PhysicsFromMagi", **compute_metrics(phys_test, test[0], 1.0 - test[2])},
        guarded,
        preserved,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS08", choices=["PEMS08", "METR-LA"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--output-dir", default="results/stage2_three_dataset_quick")
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, val_x, test_x, adj_np, _metadata = _load_dataset_splits(args.dataset, args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in args.scenarios:
        print(f"running confidence guarded five-baseline quick {scenario}", flush=True)
        train_obs, train_mask = _scenario_data(train_x, adj_np, scenario, args.seed)
        val_obs, val_mask = _scenario_data(val_x, adj_np, scenario, args.seed + 11)
        test_obs, test_mask = _scenario_data(test_x, adj_np, scenario, args.seed + 29)
        train = (train_x, train_obs, train_mask)
        val = (val_x, val_obs, val_mask)
        test = (test_x, test_obs, test_mask)

        rows.append({"scenario": scenario, **_run_knn(train, val, test)})
        if BRITS is None or SAITS is None:
            raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
        rows.append(
            {
                "scenario": scenario,
                **_run_pypots_model(
                    "BRITS",
                    BRITS(
                        n_steps=train_x.shape[1],
                        n_features=train_x.shape[2],
                        rnn_hidden_size=32,
                        batch_size=16,
                        epochs=args.epochs,
                        patience=None,
                        device=device,
                        verbose=False,
                    ),
                    train,
                    val,
                    test,
                ),
            }
        )
        rows.append({"scenario": scenario, **_run_grinlite(train, val, test, adj_np, device, args.epochs)})
        for method_row in _run_confidence_guarded(train, val, test, adj_np, device, args.epochs, args.seed, scenario):
            rows.append({"scenario": scenario, **method_row})

    print(json.dumps(rows, indent=2))

    csv_path = output_dir / "confidence_guarded_five_baseline_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "confidence_guarded_five_baseline_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ConfidenceGuarded Five-Baseline Quick\n\n")
        f.write(f"- dataset: {args.dataset}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- epochs: {args.epochs}\n")
        for scenario in args.scenarios:
            subset = [row for row in rows if row["scenario"] == scenario]
            best_external = min(
                (
                    row
                    for row in subset
                    if row["model"] not in {"ConfidenceGuardedUtilityRouter", "SAITSPreservedConfidenceGuarded", "PhysicsFromMagi"}
                ),
                key=lambda row: row["masked_mae"],
            )
            ours = next(row for row in subset if row["model"] == "SAITSPreservedConfidenceGuarded")
            gain = (best_external["masked_mae"] - ours["masked_mae"]) / best_external["masked_mae"] * 100.0
            f.write(f"\n## {scenario}\n\n")
            f.write(f"- best external: `{best_external['model']}` `{best_external['masked_mae']:.6f}`\n")
            f.write(f"- SAITSPreservedConfidenceGuarded: `{ours['masked_mae']:.6f}`\n")
            f.write(f"- gain vs best external: `{gain:+.2f}%`\n\n")
            f.write("| Model | masked MAE | RMSE | MAPE |\n|---|---:|---:|---:|\n")
            for row in subset:
                f.write(f"| {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} |\n")

if __name__ == "__main__":
    main()
