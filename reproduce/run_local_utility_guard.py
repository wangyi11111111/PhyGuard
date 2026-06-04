from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.metrics import compute_metrics
from reproduce.run_antileakage_protocol import _load_antileakage_splits
from scripts.run_five_baselines_flow_quick import _run_grinlite, _run_knn, _scenario_data
from scripts.run_maginet_physics_guard_quick import (
    _apply_observed,
    _bidirectional_temporal_target,
    _failure_mode_score,
    _graph_residual_np,
    _physics_candidate_bank,
    _rank_np,
    _selector_features,
)
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits, _run_saits_all_splits
from scripts.train import resolve_device


class LocalUtilityGuard(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _masked_mae_np(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _candidate_pack(train, val, test, adj: np.ndarray, device: torch.device, epochs: int, scenario: str) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    magi_train, magi_val, magi_test = _run_maginet_all_splits(scenario, train, val, test, adj, device, epochs)
    saits_train, saits_val, saits_test = _run_saits_all_splits(train, val, test, device, epochs)

    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test

    train_bidir, _ = _bidirectional_temporal_target(magi_train, train_obs, train_mask)
    val_bidir, _ = _bidirectional_temporal_target(magi_val, val_obs, val_mask)
    test_bidir, _ = _bidirectional_temporal_target(magi_test, test_obs, test_mask)

    train_candidates = {
        "MagiNet": _apply_observed(train_obs, train_mask, magi_train),
        "TemporalSAITS": _apply_observed(train_obs, train_mask, saits_train),
        "TemporalBidirObs": _apply_observed(train_obs, train_mask, train_bidir),
    }
    val_candidates = {
        "MagiNet": _apply_observed(val_obs, val_mask, magi_val),
        "TemporalSAITS": _apply_observed(val_obs, val_mask, saits_val),
        "TemporalBidirObs": _apply_observed(val_obs, val_mask, val_bidir),
    }
    test_candidates = {
        "MagiNet": _apply_observed(test_obs, test_mask, magi_test),
        "TemporalSAITS": _apply_observed(test_obs, test_mask, saits_test),
        "TemporalBidirObs": _apply_observed(test_obs, test_mask, test_bidir),
    }

    train_candidates.update(_physics_candidate_bank(magi_train, train_obs, train_mask, adj))
    val_candidates.update(_physics_candidate_bank(magi_val, val_obs, val_mask, adj))
    test_candidates.update(_physics_candidate_bank(magi_test, test_obs, test_mask, adj))
    names = sorted(train_candidates)
    return names, train_candidates, val_candidates, test_candidates


def _build_guard_features(base: np.ndarray, candidates: dict[str, np.ndarray], names: list[str], obs: np.ndarray, mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    per_candidate = []
    failure = _failure_mode_score(mask, adj)
    for index, name in enumerate(names):
        pred = candidates[name]
        selector = _selector_features(base, pred, obs, mask, adj)
        residual = _rank_np(np.abs(_graph_residual_np(pred, adj)))
        cand_gap = _rank_np(np.abs(pred - base))
        temporal = np.zeros_like(pred, dtype=np.float32)
        temporal[:, 1:] = np.abs(pred[:, 1:] - pred[:, :-1])
        temporal = _rank_np(temporal)
        is_temporal = np.full_like(pred, 1.0 if name.startswith("Temporal") else 0.0, dtype=np.float32)
        is_physics = np.full_like(pred, 1.0 if name.startswith("Physics") else 0.0, dtype=np.float32)
        is_base = np.full_like(pred, 1.0 if name == "MagiNet" else 0.0, dtype=np.float32)
        identity = np.zeros((*pred.shape[:-1], len(names)), dtype=np.float32)
        identity[..., index] = 1.0
        per_candidate.append(
            np.concatenate(
                [
                    selector,
                    residual,
                    cand_gap,
                    temporal,
                    failure,
                    is_temporal,
                    is_physics,
                    is_base,
                    identity,
                ],
                axis=-1,
            ).astype(np.float32)
        )
    return np.stack(per_candidate, axis=-2).astype(np.float32)


def _stack_candidates(candidates: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.stack([candidates[name] for name in names], axis=-2).astype(np.float32)


def _train_local_utility_guard(
    train,
    val,
    test,
    adj: np.ndarray,
    names: list[str],
    train_candidates: dict[str, np.ndarray],
    val_candidates: dict[str, np.ndarray],
    test_candidates: dict[str, np.ndarray],
    *,
    epochs: int,
    seed: int,
    tau: float,
    router_temperature: float,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed + 3203)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    base_train = train_candidates["MagiNet"]
    base_val = val_candidates["MagiNet"]
    base_test = test_candidates["MagiNet"]

    feat_train = _build_guard_features(base_train, train_candidates, names, train_obs, train_mask, adj)
    feat_val = _build_guard_features(base_val, val_candidates, names, val_obs, val_mask, adj)
    feat_test = _build_guard_features(base_test, test_candidates, names, test_obs, test_mask, adj)
    cand_train = _stack_candidates(train_candidates, names)
    cand_val = _stack_candidates(val_candidates, names)
    cand_test = _stack_candidates(test_candidates, names)

    valid = (1.0 - train_mask)[..., 0] > 0.0
    x = torch.tensor(feat_train[valid], dtype=torch.float32)
    cand = torch.tensor(cand_train[valid], dtype=torch.float32)
    y = torch.tensor(train_full[valid], dtype=torch.float32)
    base = torch.tensor(base_train[valid], dtype=torch.float32)
    cand_err = torch.abs(cand.detach() - y.unsqueeze(-2))
    utility_target = torch.softmax(-cand_err.mean(dim=-1) / float(tau), dim=-1).detach()
    oracle_idx = cand_err.mean(dim=-1).argmin(dim=-1).detach()
    best_err = cand_err.mean(dim=-1).min(dim=-1).values.detach()
    failure = _failure_mode_score(train_mask, adj)[..., 0][valid]
    missing_weight = 1.0 + torch.tensor(failure, dtype=torch.float32)

    model = LocalUtilityGuard(x.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 3204)
    batch_size = 32768
    best_state = None
    best_val = float("inf")

    val_region_t = torch.tensor(1.0 - val_mask, dtype=torch.float32)
    val_target_t = torch.tensor(val_full, dtype=torch.float32)
    val_feat_t = torch.tensor(feat_val, dtype=torch.float32)
    val_cand_t = torch.tensor(cand_val, dtype=torch.float32)

    for _epoch in range(max(1, epochs)):
        order = torch.randperm(x.shape[0], generator=generator)
        model.train()
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            logits = model(x[idx])
            pi = torch.softmax(logits, dim=-1)
            pred = torch.sum(pi.unsqueeze(-1) * cand[idx], dim=-2)
            pred_err = torch.abs(pred - y[idx]).mean(dim=-1)
            base_err = torch.abs(base[idx] - y[idx]).mean(dim=-1)
            rec_loss = torch.mean(pred_err * missing_weight[idx])
            utility_loss = torch.mean(
                F.kl_div(torch.log(pi.clamp_min(1e-6)), utility_target[idx], reduction="none").sum(dim=-1) * missing_weight[idx]
            )
            oracle_loss = torch.mean(F.cross_entropy(logits, oracle_idx[idx], reduction="none") * missing_weight[idx])
            harm_loss = torch.mean(torch.relu(pred_err - torch.minimum(base_err, best_err[idx])) * missing_weight[idx])
            entropy = torch.mean(torch.sum(pi * torch.log(pi.clamp_min(1e-6)), dim=-1).neg())
            loss = rec_loss + 1.50 * oracle_loss + 0.40 * utility_loss + 0.80 * harm_loss + 0.01 * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(val_feat_t) / max(float(router_temperature), 1e-3)
            pi_val = torch.softmax(logits_val, dim=-1)
            pred_val = torch.sum(pi_val.unsqueeze(-1) * val_cand_t, dim=-2)
            val_mae = float((torch.abs(pred_val - val_target_t) * val_region_t).sum() / val_region_t.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_feat_t = torch.tensor(feat_test, dtype=torch.float32)
        test_cand_t = torch.tensor(cand_test, dtype=torch.float32)
        pi_test = torch.softmax(model(test_feat_t) / max(float(router_temperature), 1e-3), dim=-1)
        pred_test = torch.sum(pi_test.unsqueeze(-1) * test_cand_t, dim=-2).numpy().astype(np.float32)
        pi_np = pi_test.numpy().astype(np.float32)

    target_region = 1.0 - test_mask
    stats: dict[str, float] = {
        "guard_val_mae": best_val,
        "guard_entropy": float((-np.sum(pi_np * np.log(np.clip(pi_np, 1e-6, None)), axis=-1) * target_region[..., 0]).sum() / np.clip(target_region[..., 0].sum(), 1.0, None)),
        "failure_score_mean": float((_failure_mode_score(test_mask, adj) * target_region).sum() / np.clip(target_region.sum(), 1.0, None)),
    }
    for i, name in enumerate(names):
        weight = pi_np[..., i : i + 1]
        stats[f"weight_{name}"] = float((weight * target_region).sum() / np.clip(target_region.sum(), 1.0, None))
        stats[f"candidate_mae_{name}"] = _masked_mae_np(test_candidates[name], test_full, target_region)
    return pred_test, stats


def _write_outputs(output_dir: Path, rows: list[dict], protocol: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-only local utility guard for PhyGuard-Hard.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS08"])
    parser.add_argument("--scenarios", nargs="+", default=["sensor_failure_30"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--guard-epochs", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.08)
    parser.add_argument("--router-temperature", type=float, default=0.35)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/local_utility_guard")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    metadata: dict[str, dict] = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            load_args = argparse.Namespace(**vars(args))
            train_x, val_x, test_x, adj, meta = _load_antileakage_splits(dataset, load_args)
            metadata[f"{dataset}_seed{seed}"] = meta
            for scenario in args.scenarios:
                print(f"running local utility guard {dataset} {scenario} seed={seed}", flush=True)
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                train = (train_x, train_obs, train_mask)
                val = (val_x, val_obs, val_mask)
                test = (test_x, test_obs, test_mask)
                target_region = 1.0 - test_mask
                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **_run_knn(train, val, test)})
                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **_run_grinlite(train, val, test, adj, device, args.epochs)})
                names, train_cand, val_cand, test_cand = _candidate_pack(train, val, test, adj, device, args.epochs, scenario)
                for model_name in ["MagiNet", "TemporalSAITS", "TemporalBidirObs"]:
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "scenario": scenario,
                            "model": model_name if model_name != "TemporalSAITS" else "SAITS",
                            **compute_metrics(test_cand[model_name], test_x, target_region),
                        }
                    )
                pred, stats = _train_local_utility_guard(
                    train,
                    val,
                    test,
                    adj,
                    names,
                    train_cand,
                    val_cand,
                    test_cand,
                    epochs=args.guard_epochs,
                    seed=seed,
                    tau=args.tau,
                    router_temperature=args.router_temperature,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "scenario": scenario,
                        "model": "PhyGuardHardLocalUtility",
                        **compute_metrics(pred, test_x, target_region),
                        **stats,
                    }
                )
                _write_outputs(output_dir, rows, {**vars(args), "metadata": metadata, "protocol": "train_only_local_utility_guard"})
    _write_outputs(output_dir, rows, {**vars(args), "metadata": metadata, "protocol": "train_only_local_utility_guard"})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
