from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.impute import KNNImputer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.corruptions import add_gaussian_noise, incident_perturbation
from data.masks import random_missing_mask, sensor_failure_mask
from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.grin_baseline import GRINLite
from models.litetrust_pinn import LiteTrustFusion
from scripts.run_conflict_test import _trust_extra_features
from scripts.run_stage10a_pems08_real_debug import _config as _stage10a_config
from scripts.run_stage10a_pems08_real_debug import _load_splits
from scripts.train import resolve_device

try:
    from pypots.imputation import BRITS, SAITS
except Exception as exc:  # pragma: no cover
    BRITS = SAITS = None
    PYPOTS_IMPORT_ERROR = exc
else:
    PYPOTS_IMPORT_ERROR = None


MAGI_ROOT = Path(os.environ.get("LITETRUST_MAGI_ROOT", "C:/tmp/MagiNet/MagiNet-main"))
BENCH_ROOT = Path("C:/tmp/five_baseline_flow_quick")


def _load_flow_splits(seed: int):
    config = deepcopy(_stage10a_config())
    config["seed"] = int(seed)
    train_x, val_x, test_x, adj, scaler, metadata = _load_splits(config)
    return train_x[:, :12, ..., :1], val_x[:, :12, ..., :1], test_x[:, :12, ..., :1], adj, scaler, metadata


def _scenario_data(full: np.ndarray, adj: np.ndarray, scenario: str, seed: int):
    if scenario.startswith("random_missing_"):
        try:
            missing_pct = float(scenario.rsplit("_", 1)[-1]) / 100.0
        except ValueError as exc:
            raise ValueError(f"invalid random-missing scenario: {scenario}") from exc
        if not 0.0 < missing_pct < 1.0:
            raise ValueError(f"random-missing ratio must be in (0, 1), got {missing_pct} from {scenario}")
        mask = random_missing_mask(full.shape, missing_pct, seed=seed)
        obs = full * mask
    elif scenario == "sensor_failure_30":
        mask = sensor_failure_mask(full.shape, 0.3, seed=seed)
        obs = full * mask
    elif scenario == "incident_perturbation":
        mask = random_missing_mask(full.shape, 0.5, seed=seed)
        perturbed = incident_perturbation(
            full,
            adj,
            drop_ratio=0.5,
            duration=6,
            region_size=max(3, int(round(full.shape[2] * 0.2))),
            seed=seed + 7,
            flow_drop_ratio=0.5,
            speed_drop_ratio=0.0,
        )
        obs = perturbed * mask
    elif scenario == "noise_random_missing":
        mask = random_missing_mask(full.shape, 0.5, seed=seed)
        obs = add_gaussian_noise(full, noise_std=0.15, seed=seed + 7) * mask
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    return obs.astype(np.float32), mask.astype(np.float32)


def _masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return compute_metrics(pred, target, 1.0 - mask)["masked_mae"]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_bn1t(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected 4D tensor, got {x.shape}")
    if x.shape[-1] == 1:
        return x
    if x.shape[2] == 1:
        return np.transpose(x, (0, 1, 3, 2))
    return x


def _flow_to_time_feature(full: np.ndarray) -> np.ndarray:
    b, t, n, _ = full.shape
    pos = np.linspace(0.0, 1.0, t, dtype=np.float32)[None, None, :, None]
    pos = np.repeat(pos, b, axis=0)
    pos = np.repeat(pos, n, axis=1)
    return pos


def _prepare_magi_files(root: Path, scenario: str, train: tuple[np.ndarray, np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray, np.ndarray], adj: np.ndarray, epochs: int) -> Path:
    scenario_root = root / "datasets" / "PEMS08_debug"
    proc_root = scenario_root / "processed" / scenario
    proc_root.mkdir(parents=True, exist_ok=True)
    scenario_root.mkdir(parents=True, exist_ok=True)
    seq_len = int(train[0].shape[1])
    with open(scenario_root / "adj_mx.pkl", "wb") as f:
        pickle.dump(adj.astype(np.float32), f)
    with open(scenario_root / "data_pos.pkl", "wb") as f:
        # Positional channel only needs the time feature in channel 1.
        total_steps = (train[0].shape[0] + val[0].shape[0] + test[0].shape[0]) * seq_len
        pos = np.zeros((total_steps, train[0].shape[2], 2), dtype=np.float32)
        for i in range(total_steps):
            pos[i, :, 1] = (i % seq_len) / max(1, seq_len - 1)
        pickle.dump(pos, f)
    index = {}
    cursor = 0
    for split_name, split_full in (("train", train[0]), ("valid", val[0]), ("test", test[0])):
        indices = []
        for _ in range(split_full.shape[0]):
            indices.append((cursor, cursor + seq_len))
            cursor += seq_len
        index[split_name] = indices
    with open(scenario_root / "processed" / f"index_{seq_len}.pkl", "wb") as f:
        pickle.dump(index, f)
    with open(scenario_root / "processed" / "normalization.pkl", "wb") as f:
        pickle.dump({"args": {"mean": 0.0, "std": 1.0}}, f)
    for split_name, (full, obs, mask) in (("train", train), ("valid", val), ("test", test)):
        # MagiNet expects [B, N, L]
        target = np.transpose(full[..., 0], (0, 2, 1))
        miss = np.transpose(obs[..., 0], (0, 2, 1))
        obs_mask = np.transpose(mask[..., 0], (0, 2, 1))
        payload = {
            "data": np.where(obs_mask > 0.5, miss, np.nan).astype(np.float32),
            "mask": obs_mask.astype(np.float32),
            "target": target.astype(np.float32),
        }
        with open(proc_root / f"{split_name}_{scenario}_ms0.5_seqlen_{seq_len}.pkl", "wb") as f:
            pickle.dump(payload, f)
    config = {
        "data": {
            "dataset": "PEMS08_debug",
            "miss_mechanism": scenario,
            "miss_ratio": 0.5,
            "batch_size": 16,
            "test_batch_size": 16,
            "val_batch_size": 16,
        },
        "model": {
            "num_nodes": int(train[0].shape[2]),
            "hidden_size": 16,
            "in_channel": 1,
            "seqlen": seq_len,
            "st_block": 2,
            "K": 3,
            "d_model": 64,
            "n_heads": 2,
        },
        "train": {
            "cuda": 0,
            "lr": 1e-3,
            "epochs": int(epochs),
            "save_model_path": "save_model/",
            "result_path": str(root / "results").replace("\\", "/") + "/",
        },
    }
    config_path = root / f"maginet_{scenario}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return config_path


def _run_maginet(scenario: str, train, val, test, adj: np.ndarray, bench_root: Path, epochs: int) -> dict:
    if not MAGI_ROOT.exists():
        raise FileNotFoundError(f"MagiNet repo not found: {MAGI_ROOT}")
    config_path = _prepare_magi_files(MAGI_ROOT, scenario, train, val, test, adj, epochs)
    cmd = [sys.executable, "main.py", "--config_path", str(config_path), "--seed", "1"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(MAGI_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MagiNet failed for {scenario}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    result_path = MAGI_ROOT / "results" / "PEMS08_debug" / f"result_{scenario}_ms0.5_seed1.pkl"
    if not result_path.exists():
        raise FileNotFoundError(f"MagiNet result missing: {result_path}")
    with open(result_path, "rb") as f:
        payload = pickle.load(f)
    pred = np.asarray(payload["imputed_data"], dtype=np.float32)
    target = np.asarray(payload["groundtruth"], dtype=np.float32)
    mask = np.asarray(payload["missed_data"], dtype=np.float32)
    pred = np.transpose(pred, (0, 3, 1, 2))
    target = np.transpose(target, (0, 3, 1, 2))
    mask = np.transpose(np.where(np.isnan(mask), 0.0, 1.0), (0, 3, 1, 2))
    metrics = compute_metrics(pred, target, 1.0 - mask)
    metrics.update({"train_time_sec": time.time() - t0, "model": "MagiNet"})
    return metrics


def _run_knn(train, val, test) -> dict:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    imp = KNNImputer(n_neighbors=3, weights="distance")
    train_flat = train_obs.reshape(train_obs.shape[0], -1)
    val_flat = val_obs.reshape(val_obs.shape[0], -1)
    test_flat = test_obs.reshape(test_obs.shape[0], -1)
    imp.fit(train_flat)
    pred = imp.transform(test_flat).reshape(test_obs.shape)
    return {**compute_metrics(pred, test_full, 1.0 - test_mask), "model": "KNN"}


def _run_grinlite(train, val, test, adj: np.ndarray, device: torch.device, epochs: int) -> dict:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    model = GRINLite(input_dim=1, hidden_dim=32, output_dim=1, dropout=0.1).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_val = float("inf")
    train_t = torch.tensor(train_full, dtype=torch.float32, device=device)
    train_o = torch.tensor(train_obs, dtype=torch.float32, device=device)
    train_m = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_full, dtype=torch.float32, device=device)
    val_o = torch.tensor(val_obs, dtype=torch.float32, device=device)
    val_m = torch.tensor(val_mask, dtype=torch.float32, device=device)
    batch_size = 16
    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_t.shape[0], device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred = model(train_o[idx], train_m[idx], adj_t)
            if pred.shape != train_t[idx].shape:
                pred = pred.transpose(1, 2)
            loss = torch.mean(torch.abs(pred - train_t[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(val_o, val_m, adj_t)
            if pred.shape != val_t.shape:
                pred = pred.transpose(1, 2)
            val_mae = float(torch.mean(torch.abs(pred - val_t)).cpu())
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(test_obs, dtype=torch.float32, device=device), torch.tensor(test_mask, dtype=torch.float32, device=device), adj_t)
        if pred.shape != torch.tensor(test_full).shape:
            pred = pred.transpose(1, 2)
    pred_np = pred.cpu().numpy()
    return {**compute_metrics(pred_np, test_full, 1.0 - test_mask), "model": "GRINLite"}


def _run_litetrust_fusion(train, val, test, adj: np.ndarray, device: torch.device, epochs: int, scenario: str) -> dict:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    model = LiteTrustFusion(
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        dropout=0.1,
        extra_feature_dim=6,
        sensor_layers=2,
        sensor_heads=4,
        router_temperature=0.55,
    ).to(device)
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None
    best_val = float("inf")
    train_t = torch.tensor(train_full, dtype=torch.float32, device=device)
    train_o = torch.tensor(train_obs, dtype=torch.float32, device=device)
    train_m = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_full, dtype=torch.float32, device=device)
    val_o = torch.tensor(val_obs, dtype=torch.float32, device=device)
    val_m = torch.tensor(val_mask, dtype=torch.float32, device=device)
    batch_size = 16
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_t.shape[0], device=device)
        pretrain = epoch <= max(1, epochs // 4)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            output = model(train_o[idx], train_m[idx], adj_t)
            pred = output["mu"]
            global_pred = output["mu_data"]
            sensor_pred = output["x_sensor"]
            phys_pred = output["x_phys"]
            target = train_t[idx]
            target_mask = train_m[idx] * 0.0 + (1.0 - train_m[idx])
            data_loss = masked_mae_loss(pred, target, target_mask)
            global_loss = masked_mae_loss(global_pred, target, target_mask)
            sensor_loss = masked_mae_loss(sensor_pred, target, target_mask)
            phys_loss = masked_mae_loss(phys_pred, target, target_mask)
            if scenario == "sensor_failure_30":
                aux_loss = 0.05 * global_loss + 0.25 * sensor_loss + 0.05 * phys_loss
            elif scenario == "incident_perturbation":
                aux_loss = 0.2 * global_loss + 0.08 * sensor_loss + 0.1 * phys_loss
            else:
                aux_loss = 0.2 * global_loss + 0.08 * sensor_loss + 0.08 * phys_loss

            utility_target = None
            utility_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
            harm_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
            if not pretrain and "expert_weights" in output:
                err_global = torch.abs(global_pred.detach() - target).mean(dim=-1, keepdim=True)
                err_sensor = torch.abs(sensor_pred.detach() - target).mean(dim=-1, keepdim=True)
                err_phys = torch.abs(phys_pred.detach() - target).mean(dim=-1, keepdim=True)
                err_stack = torch.cat([err_global, err_sensor, err_phys], dim=-1)
                utility_target = torch.softmax(-err_stack / 0.08, dim=-1).detach()
                gate_mask = target_mask.mean(dim=-1, keepdim=True)
                utility_loss = torch.sum(
                    torch.nn.functional.binary_cross_entropy(
                        output["expert_weights"].clamp(1e-4, 1.0 - 1e-4),
                        utility_target,
                        reduction="none",
                    )
                    * gate_mask
                ) / torch.clamp(gate_mask.sum(), min=1.0)
                final_err = torch.abs(pred - target).mean(dim=-1, keepdim=True)
                best_err = torch.minimum(err_global, torch.minimum(err_sensor, err_phys))
                harm_loss = torch.sum(torch.relu(final_err - best_err) * gate_mask) / torch.clamp(gate_mask.sum(), min=1.0)

            physics_residual = output["x_phys"][:, 1:] - output["x_phys"][:, :-1]
            physics_residual = torch.cat([torch.zeros_like(output["x_phys"][:, :1]), physics_residual], dim=1)
            physics_reg = torch.mean(output["physics_trust"] * torch.nn.functional.smooth_l1_loss(
                physics_residual, torch.zeros_like(physics_residual), reduction="none"
            ))
            trust_floor = torch.relu(torch.tensor(0.2, device=device) - output["physics_trust"].mean()) ** 2

            if scenario == "sensor_failure_30":
                utility_weight = 0.12
                harm_weight = 0.12
                physics_weight = 0.008
            elif scenario == "incident_perturbation":
                utility_weight = 0.1
                harm_weight = 0.18
                physics_weight = 0.01
            else:
                utility_weight = 0.1
                harm_weight = 0.15
                physics_weight = 0.01
            loss = data_loss + aux_loss + utility_weight * utility_loss + harm_weight * harm_loss + physics_weight * physics_reg + 0.01 * trust_floor
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(val_o, val_m, adj_t)["mu"]
            val_mae = float(torch.mean(torch.abs(pred - val_t)).cpu())
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(test_obs, dtype=torch.float32, device=device), torch.tensor(test_mask, dtype=torch.float32, device=device), adj_t)
        pred = output["mu"]
    pred_np = pred.cpu().numpy()
    stats = compute_metrics(pred_np, test_full, 1.0 - test_mask)
    stats.update(
        {
            "model": "LiteTrustFusion",
            "global_weight_mean": float(output["global_weight"].mean().cpu()),
            "sensor_weight_mean": float(output["sensor_weight"].mean().cpu()),
            "phys_weight_mean": float(output["phys_weight"].mean().cpu()),
            "physics_trust_mean": float(output["physics_trust"].mean().cpu()),
        }
    )
    return stats


def _run_pypots_model(model_name: str, model, train, val, test) -> dict:
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    train_dict = {"X": np.where(train_mask > 0.5, train_obs, np.nan).reshape(train_obs.shape[0], train_obs.shape[1], train_obs.shape[2])}
    val_dict = {
        "X": np.where(val_mask > 0.5, val_obs, np.nan).reshape(val_obs.shape[0], val_obs.shape[1], val_obs.shape[2]),
        "X_ori": val_full.reshape(val_full.shape[0], val_full.shape[1], val_full.shape[2]),
    }
    test_dict = {
        "X": np.where(test_mask > 0.5, test_obs, np.nan).reshape(test_obs.shape[0], test_obs.shape[1], test_obs.shape[2]),
        "X_ori": test_full.reshape(test_full.shape[0], test_full.shape[1], test_full.shape[2]),
    }
    model.fit(train_dict, val_dict)
    result = model.predict(test_dict)
    pred = result["imputation"] if isinstance(result, dict) else result
    pred = np.asarray(pred, dtype=np.float32)
    pred = pred.reshape(test_full.shape[0], test_full.shape[1], test_full.shape[2], 1)
    return {**compute_metrics(pred, test_full, 1.0 - test_mask), "model": model_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--output-dir", default="results/five_baselines_flow_quick")
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, val_x, test_x, adj, _scaler, metadata = _load_flow_splits(args.seed)
    adj_np = _to_numpy(adj).astype(np.float32)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in args.scenarios:
        train_obs, train_mask = _scenario_data(train_x, adj_np, scenario, args.seed)
        val_obs, val_mask = _scenario_data(val_x, adj_np, scenario, args.seed + 11)
        test_obs, test_mask = _scenario_data(test_x, adj_np, scenario, args.seed + 29)
        train = (train_x, train_obs, train_mask)
        val = (val_x, val_obs, val_mask)
        test = (test_x, test_obs, test_mask)

        magi_metrics = _run_maginet(scenario, train, val, test, adj_np, BENCH_ROOT, args.epochs)
        rows.append({"scenario": scenario, **magi_metrics})

        knn_metrics = _run_knn(train, val, test)
        rows.append({"scenario": scenario, **knn_metrics})

        if BRITS is None or SAITS is None:
            raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
        brits = BRITS(
            n_steps=train_x.shape[1],
            n_features=train_x.shape[2],
            rnn_hidden_size=32,
            batch_size=16,
            epochs=args.epochs,
            patience=None,
            device=device,
            verbose=False,
        )
        brits_metrics = _run_pypots_model("BRITS", brits, train, val, test)
        rows.append({"scenario": scenario, **brits_metrics})

        saits = SAITS(
            n_steps=train_x.shape[1],
            n_features=train_x.shape[2],
            n_layers=2,
            d_model=64,
            n_heads=4,
            d_k=16,
            d_v=16,
            d_ffn=64,
            dropout=0.1,
            attn_dropout=0.1,
            batch_size=16,
            epochs=args.epochs,
            patience=None,
            device=device,
            verbose=False,
        )
        saits_metrics = _run_pypots_model("SAITS", saits, train, val, test)
        rows.append({"scenario": scenario, **saits_metrics})

        grin_metrics = _run_grinlite(train, val, test, adj_np, device, args.epochs)
        rows.append({"scenario": scenario, **grin_metrics})

        fusion_metrics = _run_litetrust_fusion(train, val, test, adj_np, device, args.epochs, scenario)
        rows.append({"scenario": scenario, **fusion_metrics})

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Five Baseline Quick Benchmark\n\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- task: flow-only reconstruction on PEMS08 debug, 20 nodes, {train_x.shape[1]} steps\n\n")
        for scenario in args.scenarios:
            subset = [r for r in rows if r["scenario"] == scenario]
            f.write(f"## {scenario}\n\n")
            f.write("| Model | masked MAE | RMSE | MAPE |\n|---|---:|---:|---:|\n")
            for row in subset:
                f.write(f"| {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} |\n")
            f.write("\n")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
