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
from models.grin_baseline import GRINLite
from reproduce.run_antileakage_protocol import _load_antileakage_splits
from scripts.run_five_baselines_flow_quick import (
    BRITS,
    PYPOTS_IMPORT_ERROR,
    SAITS,
    _run_knn,
    _run_pypots_model,
    _scenario_data,
)
from scripts.run_imputeformer_pypots_quick import _make_imputeformer
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits
from scripts.train import resolve_device


def _run_grinlite_strong(train, val, test, adj: np.ndarray, device: torch.device, epochs: int) -> dict:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    model = GRINLite(input_dim=1, hidden_dim=64, output_dim=1, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    train_t = torch.tensor(train_full, dtype=torch.float32, device=device)
    train_o = torch.tensor(train_obs, dtype=torch.float32, device=device)
    train_m = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_full, dtype=torch.float32, device=device)
    val_o = torch.tensor(val_obs, dtype=torch.float32, device=device)
    val_m = torch.tensor(val_mask, dtype=torch.float32, device=device)
    best_state = None
    best_val = float("inf")
    for _epoch in range(max(1, epochs)):
        model.train()
        pred = model(train_o, train_m, adj_t)
        target_region = 1.0 - train_m
        loss = torch.sum(torch.abs(pred - train_t) * target_region) / target_region.sum().clamp_min(1.0)
        observed_loss = torch.sum(torch.abs(pred - train_t) * train_m) / train_m.sum().clamp_min(1.0)
        loss = loss + 0.05 * observed_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_o, val_m, adj_t)
            val_region = 1.0 - val_m
            val_mae = float((torch.abs(val_pred - val_t) * val_region).sum() / val_region.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(test_obs, dtype=torch.float32, device=device),
            torch.tensor(test_mask, dtype=torch.float32, device=device),
            adj_t,
        )
    return {**compute_metrics(pred.cpu().numpy(), test_full, 1.0 - test_mask), "model": "GRINLiteStrong", "best_val_mae": best_val}


def _make_saits_strong(n_steps: int, n_features: int, epochs: int, batch_size: int, device: torch.device):
    if SAITS is None:
        raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
    return SAITS(
        n_steps=n_steps,
        n_features=n_features,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_k=32,
        d_v=32,
        d_ffn=256,
        dropout=0.1,
        attn_dropout=0.1,
        batch_size=batch_size,
        epochs=epochs,
        patience=None,
        device=device,
        verbose=False,
    )


def _make_brits_strong(n_steps: int, n_features: int, epochs: int, batch_size: int, device: torch.device):
    if BRITS is None:
        raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
    return BRITS(
        n_steps=n_steps,
        n_features=n_features,
        rnn_hidden_size=128,
        batch_size=batch_size,
        epochs=epochs,
        patience=None,
        device=device,
        verbose=False,
    )


def _make_imputeformer_strong(n_steps: int, n_features: int, epochs: int, batch_size: int, device: torch.device):
    model = _make_imputeformer(n_steps, n_features, epochs, batch_size, device)
    return model


def _write(output_dir: Path, rows: list[dict], protocol: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "rows": rows}, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strong local external baseline protocol.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS03", "PEMS04", "PEMS08", "PEMS-BAY", "METR-LA"])
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--models", nargs="+", default=["KNN", "GRINLiteStrong", "MagiNet", "BRITSStrong", "SAITSStrong", "ImputeFormerStrong"])
    parser.add_argument("--output-dir", default="results/strong_external_baselines_5x3_seed1")
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
                print(f"running strong baselines {dataset} {scenario} seed={seed}", flush=True)
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                train = (train_x, train_obs, train_mask)
                val = (val_x, val_obs, val_mask)
                test = (test_x, test_obs, test_mask)
                base = {"dataset": dataset, "seed": seed, "scenario": scenario}
                if "KNN" in args.models:
                    rows.append({**base, **_run_knn(train, val, test)})
                    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
                if "GRINLiteStrong" in args.models:
                    rows.append({**base, **_run_grinlite_strong(train, val, test, adj, device, args.epochs)})
                    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
                if "MagiNet" in args.models:
                    try:
                        _m_train, _m_val, magi_test = _run_maginet_all_splits(scenario, train, val, test, adj, device, args.epochs)
                        rows.append({**base, "model": "MagiNetStrong", **compute_metrics(magi_test, test_x, 1.0 - test_mask)})
                    except Exception as exc:
                        rows.append({**base, "model": "MagiNetStrong", "error": str(exc)})
                    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
                if "BRITSStrong" in args.models:
                    rows.append({**base, **_run_pypots_model("BRITSStrong", _make_brits_strong(train_x.shape[1], train_x.shape[2], args.epochs, args.batch_size, device), train, val, test)})
                    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
                if "SAITSStrong" in args.models:
                    rows.append({**base, **_run_pypots_model("SAITSStrong", _make_saits_strong(train_x.shape[1], train_x.shape[2], args.epochs, args.batch_size, device), train, val, test)})
                    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
                if "ImputeFormerStrong" in args.models:
                    rows.append({**base, **_run_pypots_model("ImputeFormerStrong", _make_imputeformer_strong(train_x.shape[1], train_x.shape[2], args.epochs, args.batch_size, device), train, val, test)})
                    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
    _write(output_dir, rows, {**vars(args), "metadata": metadata, "pypots_version": getattr(pypots, "__version__", "unknown")})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
