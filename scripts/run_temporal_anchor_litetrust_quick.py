from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import _load_metrla_hf_splits_cached, _normalize_splits
from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.litetrust_pinn import MaskAwareGraphRepairV2, TemporalAnchorPhysicsGuarded, graph_flow_residual_full
from scripts.run_five_baselines_flow_quick import (
    BRITS,
    PYPOTS_IMPORT_ERROR,
    SAITS,
    _load_flow_splits,
    _run_grinlite,
    _run_knn,
    _run_pypots_model,
    _scenario_data,
)
from scripts.run_strong_candidate_fusion_flow_quick import (
    _physics_candidate as _numpy_physics_candidate,
    _run_maginet_all_splits,
    _run_saits_all_splits,
)
from scripts.train import resolve_device


def _load_dataset_splits(dataset: str, seed: int):
    key = dataset.lower()
    if key in {"pems08", "pems08_debug"}:
        train_x, val_x, test_x, adj, _scaler, metadata = _load_flow_splits(seed)
        return train_x[:, :12, ..., :1], val_x[:, :12, ..., :1], test_x[:, :12, ..., :1], np.asarray(adj, dtype=np.float32), metadata
    if key in {"metr-la", "metrla"}:
        train_x, val_x, test_x, adj, metadata = _load_metrla_hf_splits_cached(64, 16, 16)
        train_x, val_x, test_x, _scaler = _normalize_splits(train_x, val_x, test_x)
        return train_x[:, :12, ..., :1], val_x[:, :12, ..., :1], test_x[:, :12, ..., :1], np.asarray(adj, dtype=np.float32), metadata
    raise ValueError(f"unsupported dataset: {dataset}")


def _region_mae(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _fixed_physics_candidate(anchor: torch.Tensor, output: dict) -> torch.Tensor:
    return anchor + 0.20 * output["graph_delta"].detach() + 0.10 * output["temporal_delta"].detach()


def _physics_residual(x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 1:
        return graph_flow_residual_full(x[..., 0:1], adj)
    return torch.zeros_like(x[..., :1])


def _train_temporal_anchor_litetrust(
    train,
    val,
    test,
    adj: np.ndarray,
    device: torch.device,
    epochs: int,
    seed: int,
    fixed_alpha: float | None = None,
) -> list[dict]:
    torch.manual_seed(seed)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    model = TemporalAnchorPhysicsGuarded(
        input_dim=train_full.shape[-1],
        hidden_dim=48,
        output_dim=train_full.shape[-1],
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        correction_clip=0.6,
    ).to(device)
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
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_t.shape[0], device=device)
        pretrain_anchor = epoch <= max(2, epochs // 3)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            target = train_t[idx]
            target_mask = 1.0 - train_m[idx]
            out = model(train_o[idx], train_m[idx], adj_t)
            pred = out["mu"]
            anchor = out["x_anchor"]
            fixed_phys = _fixed_physics_candidate(anchor, out)
            extra = out["extra_feature"]
            local_missing = extra[..., 2:3]
            residual_rank = extra[..., 3:4]
            node_missing = extra[..., 4:5]
            neighbor_missing = extra[..., 5:6]
            node_failure = torch.sigmoid((node_missing - 0.70) / 0.12) * local_missing
            incident_like = torch.clamp(extra[..., 0:1] + extra[..., 1:2] + residual_rank, 0.0, 2.0)
            balance = target_mask * (1.0 + 1.5 * node_failure + 0.8 * incident_like + 0.5 * torch.clamp(node_missing - neighbor_missing, min=0.0))
            data_loss = torch.sum(torch.abs(pred - target) * balance) / balance.sum().clamp_min(1.0)
            anchor_loss = masked_mae_loss(anchor, target, target_mask)
            anchor_full_loss = masked_mae_loss(anchor, target, torch.ones_like(target))
            observed_loss = masked_mae_loss(pred, target, train_m[idx]) * 0.05
            anchor_err = torch.abs(anchor.detach() - target).mean(dim=-1, keepdim=True)
            fixed_phys_err = torch.abs(fixed_phys.detach() - target).mean(dim=-1, keepdim=True)
            utility_target = torch.sigmoid((anchor_err - fixed_phys_err) / 0.06)
            gate_loss = torch.sum(
                F.binary_cross_entropy(out["correction_gate"].clamp(1e-4, 1.0 - 1e-4), utility_target, reduction="none")
                * balance
            ) / balance.sum().clamp_min(1.0)
            harm_loss = torch.sum(torch.relu(torch.abs(pred - target) - torch.abs(anchor.detach() - target)) * balance) / balance.sum().clamp_min(1.0)
            residual_after = _physics_residual(out["x_corrected"], adj_t)
            phys_weight = target_mask * torch.clamp(1.0 - 0.9 * node_failure + 0.2 * residual_rank, 0.0, 1.0)
            physics_loss = torch.sum(F.smooth_l1_loss(residual_after, torch.zeros_like(residual_after), reduction="none") * phys_weight) / phys_weight.sum().clamp_min(1.0)
            if pretrain_anchor:
                loss = 0.65 * anchor_loss + 0.35 * anchor_full_loss + observed_loss
            else:
                loss = data_loss + 0.15 * anchor_loss + 0.15 * anchor_full_loss + 0.25 * gate_loss + 0.45 * harm_loss + 0.008 * physics_loss + observed_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_o, val_m, adj_t)
            val_metric = masked_mae_loss(val_out["mu"], val_t, 1.0 - val_m)
            val_mae = float(val_metric.cpu())
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_out = model(train_o, train_m, adj_t)
        train_anchor = train_out["x_anchor"].cpu().numpy().astype(np.float32)
        train_pred = train_out["mu"].cpu().numpy().astype(np.float32)
        val_out = model(val_o, val_m, adj_t)
        val_anchor = val_out["x_anchor"].cpu().numpy().astype(np.float32)
        val_pred = val_out["mu"].cpu().numpy().astype(np.float32)
    if fixed_alpha is None:
        best_alpha = 1.0
        best_alpha_val = float("inf")
        for alpha in np.linspace(0.0, 1.0, 21):
            candidate = val_anchor + float(alpha) * (val_pred - val_anchor)
            mae = compute_metrics(candidate, val_full, 1.0 - val_mask)["masked_mae"]
            if mae < best_alpha_val:
                best_alpha_val = float(mae)
                best_alpha = float(alpha)
    else:
        best_alpha = float(fixed_alpha)
        candidate = val_anchor + best_alpha * (val_pred - val_anchor)
        best_alpha_val = float(compute_metrics(candidate, val_full, 1.0 - val_mask)["masked_mae"])

    with torch.no_grad():
        test_out = model(
            torch.tensor(test_obs, dtype=torch.float32, device=device),
            torch.tensor(test_mask, dtype=torch.float32, device=device),
            adj_t,
        )
        pred = test_out["mu"].cpu().numpy().astype(np.float32)
        anchor = test_out["x_anchor"].cpu().numpy().astype(np.float32)
        calibrated = anchor + best_alpha * (pred - anchor)
        gate = test_out["correction_gate"].cpu().numpy().astype(np.float32)
        delta = test_out["delta"].cpu().numpy().astype(np.float32)
        residual_anchor = test_out["residual_anchor"].abs().cpu().numpy().astype(np.float32)
        residual_after = _physics_residual(test_out["x_corrected"], adj_t).abs().cpu().numpy().astype(np.float32)
        extra = test_out["extra_feature"].cpu().numpy().astype(np.float32)
    target_mask_np = 1.0 - test_mask
    node_failure_np = (1.0 / (1.0 + np.exp(-((extra[..., 4:5] - 0.70) / 0.12))) * extra[..., 2:3]).astype(np.float32)
    sensor_region = target_mask_np * (node_failure_np >= 0.5).astype(np.float32)
    nonsensor_region = target_mask_np * (node_failure_np < 0.5).astype(np.float32)
    rows = [
        {
            "model": "TemporalAnchor",
            **compute_metrics(anchor, test_full, target_mask_np),
            "gate_mean": 0.0,
            "delta_abs_mean": 0.0,
            "sensor_region_mae": _region_mae(anchor, test_full, sensor_region),
            "nonsensor_region_mae": _region_mae(anchor, test_full, nonsensor_region),
            "residual_before": float((residual_anchor * target_mask_np).sum() / np.clip(target_mask_np.sum(), 1.0, None)),
            "residual_after": float((residual_anchor * target_mask_np).sum() / np.clip(target_mask_np.sum(), 1.0, None)),
        },
        {
            "model": "TemporalAnchorPhysicsGuarded",
            **compute_metrics(pred, test_full, target_mask_np),
            "gate_mean": float(gate.mean()),
            "gate_sensor_mean": float((gate * sensor_region).sum() / np.clip(sensor_region.sum(), 1.0, None)),
            "gate_nonsensor_mean": float((gate * nonsensor_region).sum() / np.clip(nonsensor_region.sum(), 1.0, None)),
            "delta_abs_mean": float(np.abs(delta).mean()),
            "sensor_region_mae": _region_mae(pred, test_full, sensor_region),
            "nonsensor_region_mae": _region_mae(pred, test_full, nonsensor_region),
            "residual_before": float((residual_anchor * target_mask_np).sum() / np.clip(target_mask_np.sum(), 1.0, None)),
            "residual_after": float((residual_after * target_mask_np).sum() / np.clip(target_mask_np.sum(), 1.0, None)),
            "best_val_mae": best_val,
            "calibration_alpha": best_alpha,
            "calibration_val_mae": best_alpha_val,
        },
        {
            "model": "TemporalAnchorPhysicsCalibrated",
            **compute_metrics(calibrated, test_full, target_mask_np),
            "gate_mean": float(gate.mean() * best_alpha),
            "gate_sensor_mean": float((gate * sensor_region).sum() / np.clip(sensor_region.sum(), 1.0, None) * best_alpha),
            "gate_nonsensor_mean": float((gate * nonsensor_region).sum() / np.clip(nonsensor_region.sum(), 1.0, None) * best_alpha),
            "delta_abs_mean": float(np.abs(delta).mean() * best_alpha),
            "sensor_region_mae": _region_mae(calibrated, test_full, sensor_region),
            "nonsensor_region_mae": _region_mae(calibrated, test_full, nonsensor_region),
            "residual_before": float((residual_anchor * target_mask_np).sum() / np.clip(target_mask_np.sum(), 1.0, None)),
            "residual_after": float((residual_after * target_mask_np).sum() / np.clip(target_mask_np.sum(), 1.0, None)),
            "best_val_mae": best_val,
            "calibration_alpha": best_alpha,
            "calibration_val_mae": best_alpha_val,
        },
    ]
    predictions = {
        "train_anchor": train_anchor,
        "val_anchor": val_anchor,
        "test_anchor": anchor,
        "train_guarded": train_pred,
        "val_guarded": val_pred,
        "test_guarded": pred,
        "train_calibrated": train_anchor + best_alpha * (train_pred - train_anchor),
        "val_calibrated": val_anchor + best_alpha * (val_pred - val_anchor),
        "test_calibrated": calibrated,
    }
    return rows, predictions


def _torch_rank(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    order = torch.argsort(flat, dim=1)
    ranks = torch.zeros_like(flat)
    values = torch.linspace(0.0, 1.0, flat.shape[1], dtype=x.dtype, device=x.device)[None, :]
    ranks.scatter_(1, order, values.expand(flat.shape[0], -1))
    return ranks.reshape_as(x)


def _torch_graph_residual(x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 1:
        return torch.zeros_like(x[..., :1])
    return graph_flow_residual_full(x[..., 0:1], adj).abs()


def _train_mask_aware_graph_repair(
    train,
    val,
    test,
    adj: np.ndarray,
    device: torch.device,
    epochs: int,
    seed: int,
    teacher_pack: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    distill_weight: float = 0.45,
    residual_aware_distill: bool = False,
):
    torch.manual_seed(seed + 17)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    model = MaskAwareGraphRepairV2(
        train_full.shape[-1],
        hidden_dim=48,
        output_dim=train_full.shape[-1],
        num_blocks=2,
        num_heads=4,
        k_order=3,
        dropout=0.1,
        top_k=min(16, int(adj.shape[0])),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_t = torch.tensor(train_full, dtype=torch.float32, device=device)
    train_o = torch.tensor(train_obs, dtype=torch.float32, device=device)
    train_m = torch.tensor(train_mask, dtype=torch.float32, device=device)
    val_t = torch.tensor(val_full, dtype=torch.float32, device=device)
    val_o = torch.tensor(val_obs, dtype=torch.float32, device=device)
    val_m = torch.tensor(val_mask, dtype=torch.float32, device=device)
    train_teacher = val_teacher = test_teacher = None
    if teacher_pack is not None:
        train_teacher = torch.tensor(teacher_pack[0], dtype=torch.float32, device=device)
        val_teacher = torch.tensor(teacher_pack[1], dtype=torch.float32, device=device)
        test_teacher = torch.tensor(teacher_pack[2], dtype=torch.float32, device=device)
    batch_size = 16
    best_state = None
    best_val = float("inf")
    for _epoch in range(max(1, epochs)):
        model.train()
        order = torch.randperm(train_t.shape[0], device=device)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            out = model(train_o[idx], train_m[idx], adj_t)
            target_mask = 1.0 - train_m[idx]
            obs_mask = train_m[idx]
            node_missing = 1.0 - train_m[idx].mean(dim=(1, 3))[:, None, :, None]
            node_missing = node_missing.expand(-1, train_m.shape[1], -1, -1)
            neighbor_obs = torch.einsum("nm,btmc->btnc", adj_t, train_m[idx]).mean(dim=-1, keepdim=True)
            region_weight = target_mask * (1.0 + 1.0 * node_missing + 0.5 * (1.0 - neighbor_obs))
            missing_loss = torch.sum(torch.abs(out["mu"] - train_t[idx]) * region_weight) / region_weight.sum().clamp_min(1.0)
            full_loss = masked_mae_loss(out["x_graph_repair"], train_t[idx], torch.ones_like(train_t[idx]))
            obs_loss = masked_mae_loss(out["mu"], train_t[idx], obs_mask)
            graph_entropy = -torch.sum(out["learned_graph"].clamp_min(1e-6) * torch.log(out["learned_graph"].clamp_min(1e-6)), dim=-1).mean()
            distill_loss = torch.zeros((), dtype=train_t.dtype, device=device)
            teacher_harm = torch.zeros((), dtype=train_t.dtype, device=device)
            if train_teacher is not None:
                teacher = train_teacher[idx]
                teacher_err = torch.abs(teacher.detach() - train_t[idx])
                student_err = torch.abs(out["mu"] - train_t[idx])
                if residual_aware_distill:
                    observed_fill_err = torch.abs(train_t[idx] - train_o[idx])
                    teacher_advantage = torch.sigmoid((observed_fill_err - teacher_err.detach()) / 0.08)
                    teacher_residual = _torch_graph_residual(teacher.detach(), adj_t)
                    student_residual = _torch_graph_residual(out["mu"].detach(), adj_t)
                    teacher_residual_rank = _torch_rank(teacher_residual)
                    residual_advantage = torch.sigmoid((student_residual - teacher_residual) / 0.08)
                    physics_safe = torch.sigmoid((0.65 - teacher_residual_rank) / 0.12)
                    distill_confidence = torch.clamp(teacher_advantage * (0.35 + 0.65 * residual_advantage) * physics_safe, 0.0, 1.0)
                    distill_region = region_weight * distill_confidence
                else:
                    teacher_better = (teacher_err < torch.abs(train_t[idx] - train_o[idx])).float()
                    distill_region = region_weight * (1.0 + 0.8 * teacher_better)
                distill_loss = torch.sum(torch.abs(out["mu"] - teacher.detach()) * distill_region) / distill_region.sum().clamp_min(1.0)
                teacher_harm = torch.sum(torch.relu(student_err - teacher_err.detach()) * distill_region) / distill_region.sum().clamp_min(1.0)
            loss = (
                missing_loss
                + 0.25 * full_loss
                + 0.03 * obs_loss
                + float(distill_weight) * distill_loss
                + 0.15 * teacher_harm
                - 0.0002 * graph_entropy
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_out = model(val_o, val_m, adj_t)
            val_target_mae = masked_mae_loss(val_out["mu"], val_t, 1.0 - val_m)
            if val_teacher is not None:
                if residual_aware_distill:
                    val_mae = float(val_target_mae.cpu())
                else:
                    val_distill = masked_mae_loss(val_out["mu"], val_teacher, 1.0 - val_m)
                    val_mae = float((val_target_mae + 0.15 * val_distill).cpu())
            else:
                val_mae = float(val_target_mae.cpu())
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(obs: np.ndarray, mask: np.ndarray):
        with torch.no_grad():
            out = model(torch.tensor(obs, dtype=torch.float32, device=device), torch.tensor(mask, dtype=torch.float32, device=device), adj_t)
        return out["mu"].cpu().numpy().astype(np.float32), out

    train_pred, _train_out = predict(train_obs, train_mask)
    val_pred, _val_out = predict(val_obs, val_mask)
    test_pred, test_out = predict(test_obs, test_mask)
    metrics = compute_metrics(test_pred, test_full, 1.0 - test_mask)
    metrics.update(
        {
            "model": "MaskAwareGraphRepairV2",
            "best_val_mae": best_val,
            "graph_confidence_mean": float(test_out["graph_confidence"].mean().cpu()),
            "distilled_from_maginet": teacher_pack is not None,
            "residual_aware_distill": residual_aware_distill,
        }
    )
    return metrics, (train_pred, val_pred, test_pred)


class InternalUtilityRouter(torch.nn.Module):
    def __init__(self, feature_dim: int, num_candidates: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, num_candidates),
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


def _graph_residual_np(x: np.ndarray, adj: np.ndarray) -> np.ndarray:
    residual = np.zeros_like(x, dtype=np.float32)
    residual[:, 1:] = x[:, 1:] - x[:, :-1] + (x[:, :-1] - np.einsum("nm,btmc->btnc", adj, x[:, :-1]))
    return residual.astype(np.float32)


def _router_features(magi: np.ndarray, anchor: np.ndarray, phys: np.ndarray, mask: np.ndarray, adj: np.ndarray) -> np.ndarray:
    local_missing = (1.0 - mask).astype(np.float32)
    node_missing = 1.0 - mask.mean(axis=(1, 3), keepdims=True)
    node_missing = np.repeat(node_missing, mask.shape[1], axis=1).astype(np.float32)
    neighbor_obs = np.einsum("nm,btmc->btnc", adj, mask).mean(axis=-1, keepdims=True)
    neighbor_missing = (1.0 - neighbor_obs).astype(np.float32)
    node_failure = (1.0 / (1.0 + np.exp(-((node_missing - 0.70) / 0.12))) * local_missing).astype(np.float32)
    res_magi = _rank_np(np.abs(_graph_residual_np(magi, adj)))
    res_anchor = _rank_np(np.abs(_graph_residual_np(anchor, adj)))
    res_phys = _rank_np(np.abs(_graph_residual_np(phys, adj)))
    return np.concatenate(
        [
            local_missing,
            node_missing,
            neighbor_missing,
            node_failure,
            res_magi,
            res_anchor,
            res_phys,
            np.abs(magi - anchor).astype(np.float32),
            np.abs(magi - phys).astype(np.float32),
            np.abs(anchor - phys).astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def _train_internal_fusion(train, val, test, adj: np.ndarray, graph_pack, anchor_pack, phys_pack, seed: int) -> dict:
    train_full, _train_obs, train_mask = train
    val_full, _val_obs, val_mask = val
    test_full, _test_obs, test_mask = test
    graph_train, graph_val, graph_test = graph_pack
    anchor_train, anchor_val, anchor_test = anchor_pack
    phys_train, phys_val, phys_test = phys_pack
    names = ["MaskAwareGraphRepairV2", "TemporalAnchorPhysicsCalibrated", "PhysicsFromGraphRepair"]
    train_candidates = np.stack([graph_train, anchor_train, phys_train], axis=-1)
    train_errors = np.abs(train_candidates - train_full[..., None]).mean(axis=-2)
    labels = np.argmin(train_errors, axis=-1).astype(np.int64)
    valid = (1.0 - train_mask)[..., 0] > 0.0
    x = torch.tensor(_router_features(graph_train, anchor_train, phys_train, train_mask, adj)[valid], dtype=torch.float32)
    y = torch.tensor(labels[valid], dtype=torch.long)
    local_missing = x[:, 0]
    node_failure = x[:, 3]
    residual_pressure = torch.maximum(x[:, 4], torch.maximum(x[:, 5], x[:, 6]))
    weights = 1.0 + 1.5 * local_missing + 2.0 * node_failure + 0.5 * residual_pressure
    max_samples = 120_000
    if x.shape[0] > max_samples:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=generator)[:max_samples]
        x, y, weights = x[idx], y[idx], weights[idx]
    router = InternalUtilityRouter(x.shape[-1], len(names))
    opt = torch.optim.AdamW(router.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 301)
    batch_size = 4096
    for _epoch in range(60):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            loss = torch.mean(F.cross_entropy(router(x[idx]), y[idx], reduction="none") * weights[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(split_mask, split_candidates):
        magi, anchor, phys = split_candidates
        features = torch.tensor(_router_features(magi, anchor, phys, split_mask, adj), dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(router(features), dim=-1).numpy().astype(np.float32)
        cand = np.stack([magi, anchor, phys], axis=-1)
        soft = np.sum(cand * probs[..., None, :], axis=-1)
        hard_idx = np.argmax(probs, axis=-1)
        hard = np.take_along_axis(cand, hard_idx[..., None, None], axis=-1)[..., 0]
        confidence = probs.max(axis=-1, keepdims=True)
        pred = np.where(confidence >= 0.55, hard, soft).astype(np.float32)
        return pred, probs

    val_pred, val_probs = predict(val_mask, (graph_val, anchor_val, phys_val))
    test_pred, test_probs = predict(test_mask, (graph_test, anchor_test, phys_test))
    val_candidates = {
        "router": val_pred,
        "MaskAwareGraphRepairV2": graph_val,
        "TemporalAnchorPhysicsCalibrated": anchor_val,
        "PhysicsFromGraphRepair": phys_val,
    }
    test_candidates = {
        "router": test_pred,
        "MaskAwareGraphRepairV2": graph_test,
        "TemporalAnchorPhysicsCalibrated": anchor_test,
        "PhysicsFromGraphRepair": phys_test,
    }
    selected = min(val_candidates, key=lambda name: compute_metrics(val_candidates[name], val_full, 1.0 - val_mask)["masked_mae"])
    final_pred = test_candidates[selected]
    metrics = compute_metrics(final_pred, test_full, 1.0 - test_mask)
    metrics.update(
        {
            "model": "TemporalAnchorLiteTrustFusion",
            "selected_output": selected,
            "utility_weight_graph_repair": float(test_probs[..., 0].mean()),
            "utility_weight_temporal_anchor": float(test_probs[..., 1].mean()),
            "utility_weight_physics": float(test_probs[..., 2].mean()),
        }
    )
    for i, name in enumerate(names):
        metrics[f"label_share_{name}"] = float(np.mean(labels[valid] == i))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS08", choices=["PEMS08", "METR-LA"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "sensor_failure_30", "incident_perturbation"])
    parser.add_argument("--distill-maginet", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=0.45)
    parser.add_argument("--residual-aware-distill", action="store_true")
    parser.add_argument("--output-dir", default="results/temporal_anchor_litetrust_quick")
    args = parser.parse_args()

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, val_x, test_x, adj, metadata = _load_dataset_splits(args.dataset, args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in args.scenarios:
        print(f"running temporal-anchor LiteTrust {args.dataset} {scenario}", flush=True)
        train_obs, train_mask = _scenario_data(train_x, adj, scenario, args.seed)
        val_obs, val_mask = _scenario_data(val_x, adj, scenario, args.seed + 11)
        test_obs, test_mask = _scenario_data(test_x, adj, scenario, args.seed + 29)
        train = (train_x, train_obs, train_mask)
        val = (val_x, val_obs, val_mask)
        test = (test_x, test_obs, test_mask)
        rows.append({"dataset": args.dataset, "scenario": scenario, **_run_knn(train, val, test)})
        if BRITS is None or SAITS is None:
            raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
        rows.append(
            {
                "dataset": args.dataset,
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
        rows.append({"dataset": args.dataset, "scenario": scenario, **_run_grinlite(train, val, test, adj, device, args.epochs)})
        magi_train, magi_val, magi_test = _run_maginet_all_splits(scenario, train, val, test, adj, device, args.epochs)
        saits_train, saits_val, saits_test = _run_saits_all_splits(train, val, test, device, args.epochs)
        rows.append({"dataset": args.dataset, "scenario": scenario, "model": "MagiNet", **compute_metrics(magi_test, test_x, 1.0 - test_mask)})
        rows.append({"dataset": args.dataset, "scenario": scenario, "model": "SAITS", **compute_metrics(saits_test, test_x, 1.0 - test_mask)})
        teacher_pack = (magi_train, magi_val, magi_test) if args.distill_maginet else None
        graph_metrics, graph_pack = _train_mask_aware_graph_repair(
            train,
            val,
            test,
            adj,
            device,
            args.epochs,
            args.seed,
            teacher_pack=teacher_pack,
            distill_weight=args.distill_weight,
            residual_aware_distill=args.residual_aware_distill,
        )
        rows.append({"dataset": args.dataset, "scenario": scenario, **graph_metrics})
        graph_train, graph_val, graph_test = graph_pack
        phys_train = _numpy_physics_candidate(graph_train, train_obs, train_mask, adj)
        phys_val = _numpy_physics_candidate(graph_val, val_obs, val_mask, adj)
        phys_test = _numpy_physics_candidate(graph_test, test_obs, test_mask, adj)
        rows.append({"dataset": args.dataset, "scenario": scenario, "model": "PhysicsFromGraphRepair", **compute_metrics(phys_test, test_x, 1.0 - test_mask)})
        temporal_rows, temporal_predictions = _train_temporal_anchor_litetrust(train, val, test, adj, device, args.epochs, args.seed)
        for row in temporal_rows:
            rows.append({"dataset": args.dataset, "scenario": scenario, **row})
        fusion = _train_internal_fusion(
            train,
            val,
            test,
            adj,
            (graph_train, graph_val, graph_test),
            (
                temporal_predictions["train_calibrated"],
                temporal_predictions["val_calibrated"],
                temporal_predictions["test_calibrated"],
            ),
            (phys_train, phys_val, phys_test),
            args.seed,
        )
        rows.append({"dataset": args.dataset, "scenario": scenario, **fusion})

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Temporal Anchor LiteTrust Quick\n\n")
        f.write(f"- dataset: {args.dataset}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- distill_maginet: {args.distill_maginet}\n")
        f.write(f"- distill_weight: {args.distill_weight}\n")
        f.write(f"- residual_aware_distill: {args.residual_aware_distill}\n")
        f.write(f"- source: {metadata.get('source', metadata.get('dataset_name', 'unknown'))}\n\n")
        f.write("This run treats temporal self-attention as an internal anchor, not as an external preserved SAITS output.\n\n")
        for scenario in args.scenarios:
            subset = [row for row in rows if row["scenario"] == scenario]
            external = [row for row in subset if row["model"] in {"KNN", "BRITS", "GRINLite", "MagiNet", "SAITS"}]
            best_external = min(external, key=lambda row: row["masked_mae"])
            ours = next(row for row in subset if row["model"] == "TemporalAnchorLiteTrustFusion")
            gain = (best_external["masked_mae"] - ours["masked_mae"]) / best_external["masked_mae"] * 100.0
            f.write(f"## {scenario}\n\n")
            f.write(f"- best external: `{best_external['model']}` `{best_external['masked_mae']:.6f}`\n")
            f.write(f"- TemporalAnchorLiteTrustFusion: `{ours['masked_mae']:.6f}`\n")
            f.write(f"- gain vs best external: `{gain:+.2f}%`\n")
            f.write(f"- selected output: `{ours.get('selected_output', '')}`\n")
            f.write(f"- calibrated physics alpha: `{ours.get('calibration_alpha', float('nan')):.2f}`\n")
            f.write(f"- gate mean: `{ours.get('gate_mean', float('nan')):.6f}`; sensor gate: `{ours.get('gate_sensor_mean', float('nan')):.6f}`; nonsensor gate: `{ours.get('gate_nonsensor_mean', float('nan')):.6f}`\n")
            f.write(f"- residual before/after: `{ours.get('residual_before', float('nan')):.6f}` / `{ours.get('residual_after', float('nan')):.6f}`\n\n")
            f.write("| Model | masked MAE | RMSE | MAPE | sensor MAE | nonsensor MAE | gate | delta | residual before | residual after |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in subset:
                f.write(
                    f"| {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} | "
                    f"{row.get('sensor_region_mae', float('nan')):.6f} | {row.get('nonsensor_region_mae', float('nan')):.6f} | "
                    f"{row.get('gate_mean', float('nan')):.6f} | {row.get('delta_abs_mean', float('nan')):.6f} | "
                    f"{row.get('residual_before', float('nan')):.6f} | {row.get('residual_after', float('nan')):.6f} |\n"
                )
            f.write("\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
