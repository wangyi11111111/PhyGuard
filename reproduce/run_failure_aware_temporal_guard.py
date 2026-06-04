from __future__ import annotations

import argparse
import csv
import json
import sys
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
from scripts.run_five_baselines_flow_quick import (
    BRITS,
    PYPOTS_IMPORT_ERROR,
    _run_grinlite,
    _run_knn,
    _run_pypots_model,
    _scenario_data,
)
from scripts.run_maginet_physics_guard_quick import METHOD_NAME, _failure_mode_score, _run_one_scenario
import scripts.run_maginet_physics_guard_quick as guard_module
from scripts.run_temporal_anchor_litetrust_quick import _train_temporal_anchor_litetrust
from scripts.run_strong_candidate_fusion_flow_quick import _run_saits_all_splits
from scripts.train import resolve_device


def _region_mae(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _failure_aware_fuse(
    phyguard: np.ndarray,
    temporal: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    adj: np.ndarray,
    *,
    power: float,
    max_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    target_region = 1.0 - mask
    failure = _failure_mode_score(mask, adj)
    weight = np.clip(failure, 0.0, float(max_weight)) ** float(power)
    fused = ((1.0 - weight) * phyguard + weight * temporal).astype(np.float32)
    sensor_region = target_region * (failure >= 0.5).astype(np.float32)
    nonsensor_region = target_region * (failure < 0.5).astype(np.float32)
    stats = {
        "failure_weight_mean": float((weight * target_region).sum() / np.clip(target_region.sum(), 1.0, None)),
        "failure_weight_sensor_mean": float((weight * sensor_region).sum() / np.clip(sensor_region.sum(), 1.0, None)),
        "failure_weight_nonsensor_mean": float((weight * nonsensor_region).sum() / np.clip(nonsensor_region.sum(), 1.0, None)),
        "sensor_region_mae": _region_mae(fused, target, sensor_region),
        "nonsensor_region_mae": _region_mae(fused, target, nonsensor_region),
    }
    return fused, stats


class MaskedTemporalAttentionAnchor(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_proj = nn.Linear(2 * self.num_nodes + 2, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, self.num_nodes)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = obs[..., 0] * mask[..., 0]
        m = mask[..., 0]
        b, t, _n = x.shape
        pos = torch.linspace(0.0, 1.0, t, dtype=x.dtype, device=x.device)[None, :, None].expand(b, -1, -1)
        pos_feat = torch.cat([torch.sin(2.0 * torch.pi * pos), torch.cos(2.0 * torch.pi * pos)], dim=-1)
        h = self.input_proj(torch.cat([x, m, pos_feat], dim=-1))
        h = self.encoder(h)
        pred = self.output(h)[..., None]
        return mask * obs + (1.0 - mask) * pred


def _train_masked_temporal_attention_anchor(
    train,
    val,
    test,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed + 911)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    model = MaskedTemporalAttentionAnchor(num_nodes=train_full.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_t = torch.tensor(train_full, dtype=torch.float32, device=device)
    train_o = torch.tensor(train_obs, dtype=torch.float32, device=device)
    train_m = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_full, dtype=torch.float32, device=device)
    val_o = torch.tensor(val_obs, dtype=torch.float32, device=device)
    val_m = torch.tensor(val_mask, dtype=torch.float32, device=device)
    batch_size = 16
    best_state = None
    best_val = float("inf")
    generator = torch.Generator(device=device).manual_seed(seed + 912)
    for _epoch in range(max(1, epochs)):
        model.train()
        order = torch.randperm(train_t.shape[0], generator=generator, device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred = model(train_o[idx], train_m[idx])
            target = train_t[idx]
            target_region = 1.0 - train_m[idx]
            node_missing = 1.0 - train_m[idx].mean(dim=(1, 3), keepdim=True)
            node_missing = node_missing.expand(-1, train_m.shape[1], -1, -1)
            weight = target_region * (1.0 + 1.5 * node_missing)
            missing_loss = torch.sum(torch.abs(pred - target) * weight) / weight.sum().clamp_min(1.0)
            observed_loss = torch.sum(torch.abs(pred - target) * train_m[idx]) / train_m[idx].sum().clamp_min(1.0)
            smooth_loss = torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1])) if pred.shape[1] > 1 else torch.zeros((), device=device)
            loss = missing_loss + 0.03 * observed_loss + 0.002 * smooth_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_o, val_m)
            val_region = 1.0 - val_m
            val_mae = float((torch.abs(val_pred - val_t) * val_region).sum() / val_region.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(
            torch.tensor(test_obs, dtype=torch.float32, device=device),
            torch.tensor(test_mask, dtype=torch.float32, device=device),
        ).cpu().numpy().astype(np.float32)
    return test_pred, {"attention_anchor_val_mae": best_val}


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
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Failure-Aware Temporal PhyGuard\n\n")
        for key, value in protocol.items():
            f.write(f"- {key}: `{value}`\n")
        for dataset in sorted({row["dataset"] for row in rows}):
            f.write(f"\n## {dataset}\n\n")
            for scenario in sorted({row["scenario"] for row in rows if row["dataset"] == dataset}):
                subset = [row for row in rows if row["dataset"] == dataset and row["scenario"] == scenario]
                external = [row for row in subset if row["model"] in {"KNN", "BRITS", "GRINLite", "MagiNet", "SAITS"}]
                best_external = min(external, key=lambda row: row["masked_mae"]) if external else None
                ours = next((row for row in subset if row["model"] == "FailureAwareTemporalPhyGuard"), None)
                f.write(f"### {scenario}\n\n")
                if best_external and ours:
                    gain = (best_external["masked_mae"] - ours["masked_mae"]) / best_external["masked_mae"] * 100.0
                    f.write(f"- best external: `{best_external['model']}` `{best_external['masked_mae']:.6f}`\n")
                    f.write(f"- FailureAwareTemporalPhyGuard: `{ours['masked_mae']:.6f}`\n")
                    f.write(f"- gain vs best external: `{gain:+.2f}%`\n")
                    f.write(f"- failure weight mean: `{ours.get('failure_weight_mean', float('nan')):.6f}`\n\n")
                f.write("| Model | masked MAE | RMSE | MAPE |\n|---|---:|---:|---:|\n")
                for row in sorted(subset, key=lambda item: item.get("masked_mae", float("inf"))):
                    f.write(f"| {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict failure-aware temporal PhyGuard experiment.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS03", "PEMS04", "PEMS08"])
    parser.add_argument("--scenarios", nargs="+", default=["sensor_failure_30"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--guard-epochs", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--fixed-correction-key", default="RegionAmplitudeScaled@1.50")
    parser.add_argument("--guard-ablation", default="no_temporal_evidence_bank")
    parser.add_argument("--anchor-type", choices=["attention", "graph"], default="attention")
    parser.add_argument("--temporal-alpha", type=float, default=1.0)
    parser.add_argument("--failure-power", type=float, default=1.0)
    parser.add_argument("--max-failure-weight", type=float, default=1.0)
    parser.add_argument("--skip-pypots-baselines", action="store_true")
    parser.add_argument("--output-dir", default="results/failure_aware_temporal_guard")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    protocol = vars(args).copy()
    protocol["protocol"] = "strict_raw_split_failure_aware_temporal_phyguard"

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    guard_module.FIXED_CORRECTION_KEY = args.fixed_correction_key or None
    rows: list[dict] = []
    metadata: dict[str, dict] = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            load_args = argparse.Namespace(**vars(args))
            train_x, val_x, test_x, adj, meta = _load_antileakage_splits(dataset, load_args)
            metadata[f"{dataset}_seed{seed}"] = meta
            for scenario in args.scenarios:
                print(f"running failure-aware temporal PhyGuard {dataset} {scenario} seed={seed}", flush=True)
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                train = (train_x, train_obs, train_mask)
                val = (val_x, val_obs, val_mask)
                test = (test_x, test_obs, test_mask)
                target_region = 1.0 - test_mask

                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **_run_knn(train, val, test)})
                if not args.skip_pypots_baselines:
                    if BRITS is None:
                        raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
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
                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **_run_grinlite(train, val, test, adj, device, args.epochs)})

                phyguard_rows, phyguard_predictions = _run_one_scenario(
                    train,
                    val,
                    test,
                    adj,
                    device,
                    args.epochs,
                    args.guard_epochs,
                    seed,
                    scenario,
                    args.guard_ablation,
                    dataset=dataset,
                    return_predictions=True,
                )
                for row in phyguard_rows:
                    if row["model"] in {METHOD_NAME, "MagiNet", "PhysicsFromMagi"}:
                        rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **row})

                saits_train, saits_val, saits_test = _run_saits_all_splits(train, val, test, device, args.epochs)
                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, "model": "SAITS", **compute_metrics(saits_test, test_x, target_region)})

                if args.anchor_type == "attention":
                    temporal_test, temporal_stats = _train_masked_temporal_attention_anchor(
                        train,
                        val,
                        test,
                        device,
                        args.guard_epochs,
                        seed,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "scenario": scenario,
                            "model": "MaskedTemporalAttentionAnchor",
                            **compute_metrics(temporal_test, test_x, target_region),
                            **temporal_stats,
                        }
                    )
                else:
                    temporal_rows, temporal_predictions = _train_temporal_anchor_litetrust(
                        train,
                        val,
                        test,
                        adj,
                        device,
                        args.epochs,
                        seed,
                        fixed_alpha=args.temporal_alpha,
                    )
                    temporal_test = temporal_predictions["test_calibrated"]
                    for row in temporal_rows:
                        if row["model"] in {"TemporalAnchor", "TemporalAnchorPhysicsGuarded", "TemporalAnchorPhysicsCalibrated"}:
                            rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **row})

                fused, stats = _failure_aware_fuse(
                    phyguard_predictions["phyguard"],
                    temporal_test,
                    test_x,
                    test_mask,
                    adj,
                    power=args.failure_power,
                    max_weight=args.max_failure_weight,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "scenario": scenario,
                        "model": "FailureAwareTemporalPhyGuard",
                        **compute_metrics(fused, test_x, target_region),
                        **stats,
                    }
                )
                _write_outputs(output_dir, rows, {**protocol, "metadata": metadata})
    _write_outputs(output_dir, rows, {**protocol, "metadata": metadata})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
