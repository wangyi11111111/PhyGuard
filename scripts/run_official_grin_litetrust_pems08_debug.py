from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.official_grin_wrapper import DEFAULT_OFFICIAL_GRIN_ROOT, OfficialGRINLiteTrustCorrection
from scripts.run_conflict_test import _trust_extra_features
from scripts.run_stage10a_pems08_real_debug import _config, _scenario_loaders
from scripts.run_stage2_three_dataset_quick import _residual
from scripts.train import resolve_device


DEFAULT_SCENARIOS = ["random_missing_50", "noise_random_missing", "incident_perturbation"]


def _masked_mean_array(values: np.ndarray, mask: np.ndarray) -> float | None:
    if mask.shape[-1] != values.shape[-1]:
        mask = mask.mean(axis=-1, keepdims=True)
    denom = float(mask.sum())
    if denom <= 0.0:
        return None
    return float((values * mask).sum() / denom)


def _forward(model, x_obs, obs_mask, adj, scaler, config):
    with torch.no_grad():
        mu_data = model.grin(x_obs, obs_mask)
        residual_data = _residual(mu_data, adj, scaler, config["dataset_residual"])
        extra = _trust_extra_features(x_obs, obs_mask, adj, residual_data)
        if bool(config.get("method", {}).get("disable_physics_features", False)):
            residual_data = torch.zeros_like(residual_data)
            extra = extra.clone()
            if extra.shape[-1] >= 4:
                extra[..., 3:4] = 0.0
    output = model(
        x_obs,
        obs_mask,
        adj,
        residual_signed=residual_data.detach(),
        extra_feature=extra,
    )
    output["data_residual"] = residual_data
    output["physics_residual"] = _residual(output["mu"], adj, scaler, config["dataset_residual"])
    return output


def _run_epoch(model, loader, adj, scaler, optimizer, device, config):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    preds = []
    data_preds = []
    targets = []
    masks = []
    phys_weights = []
    generic_weights = []
    prior_weights = []
    region_gates = []
    deltas = []
    explicit_phys_deltas = []
    projection_strengths = []
    fd_strengths = []
    graph_strengths = []
    temporal_strengths = []
    residuals = []
    region_masks: dict[str, list[np.ndarray]] = {
        "missing": [],
        "noise": [],
        "incident": [],
        "incident_missing": [],
    }
    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        target = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        with torch.set_grad_enabled(train_mode):
            output = _forward(model, x_obs, obs_mask, adj, scaler, config)
            pred = output["mu"]
            data_loss = masked_mae_loss(pred, target, target_mask)
            data_aux = masked_mae_loss(output["mu_data"], target, target_mask)
            observed_aux = masked_mae_loss(output["mu_data"], target, obs_mask)
            correction_l1 = torch.mean(torch.abs(output["final_delta"]))
            gate_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
            harm = torch.zeros((), dtype=pred.dtype, device=pred.device)
            oracle_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
            utility_gate_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
            utility_harm = torch.zeros((), dtype=pred.dtype, device=pred.device)
            if bool(config.get("method", {}).get("improvement_gate_loss", False)):
                final_error = torch.abs(pred - target).mean(dim=-1, keepdim=True)
                data_error = torch.abs(output["mu_data"].detach() - target).mean(dim=-1, keepdim=True)
                improvement = data_error - final_error.detach()
                gate_target = torch.sigmoid(improvement / 0.1).detach()
                gate_mask = target_mask.mean(dim=-1, keepdim=True)
                gate_loss = torch.sum(
                    F.binary_cross_entropy(output["phys_weight"].clamp(1e-4, 1.0 - 1e-4), gate_target, reduction="none") * gate_mask
                ) / torch.clamp(gate_mask.sum(), min=1.0)
                harm = torch.sum(output["phys_weight"] * torch.relu(final_error - data_error) * gate_mask) / torch.clamp(gate_mask.sum(), min=1.0)
            if bool(config.get("method", {}).get("oracle_correction_loss", False)):
                oracle_delta = target - output["mu_data"].detach()
                flow_alignment = (
                    torch.sign(oracle_delta[..., 0:1]) == torch.sign(-output["data_residual"].detach())
                ).to(pred.dtype)
                error_scale = torch.clamp(torch.abs(oracle_delta).mean(dim=-1, keepdim=True) / 0.5, min=0.0, max=1.0)
                missing_gate = target_mask.mean(dim=-1, keepdim=True)
                physics_explainable = missing_gate * (0.25 + 0.75 * flow_alignment) * error_scale
                delta_target = torch.clamp(oracle_delta, min=-0.5, max=0.5)
                delta_error = F.smooth_l1_loss(output["delta_phys"], delta_target, reduction="none").mean(dim=-1, keepdim=True)
                oracle_loss = torch.sum(delta_error * physics_explainable) / torch.clamp(physics_explainable.sum(), min=1.0)
            if bool(config.get("method", {}).get("contrastive_utility_gate", False)):
                gate_mask = target_mask.mean(dim=-1, keepdim=True)
                err_generic = torch.abs(output["x_generic"].detach() - target).mean(dim=-1, keepdim=True)
                err_phys = torch.abs(output["x_phys"].detach() - target).mean(dim=-1, keepdim=True)
                utility_target = torch.sigmoid((err_generic - err_phys) / 0.05).detach()
                utility_gate_loss = torch.sum(
                    F.binary_cross_entropy(output["phys_weight"].clamp(1e-4, 1.0 - 1e-4), utility_target, reduction="none") * gate_mask
                ) / torch.clamp(gate_mask.sum(), min=1.0)
                gated_err = torch.abs(pred - target).mean(dim=-1, keepdim=True)
                best_expert_err = torch.minimum(err_generic, err_phys)
                utility_harm = torch.sum(torch.relu(gated_err - best_expert_err) * gate_mask) / torch.clamp(gate_mask.sum(), min=1.0)
            physics_reg = torch.mean(output["phys_weight"] * torch.nn.functional.smooth_l1_loss(
                output["physics_residual"], torch.zeros_like(output["physics_residual"]), reduction="none"
            ))
            loss = (
                data_loss
                + 0.2 * data_aux
                + 0.05 * observed_aux
                + 0.005 * correction_l1
                + 0.001 * physics_reg
                + float(config.get("method", {}).get("gate_loss_weight", 0.0)) * gate_loss
                + float(config.get("method", {}).get("harm_loss_weight", 0.0)) * harm
                + float(config.get("method", {}).get("oracle_correction_weight", 0.0)) * oracle_loss
                + float(config.get("method", {}).get("utility_gate_weight", 0.0)) * utility_gate_loss
                + float(config.get("method", {}).get("utility_harm_weight", 0.0)) * utility_harm
            )
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        data_preds.append(output["mu_data"].detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
        phys_weights.append(output["phys_weight"].detach().cpu().numpy())
        generic_weights.append(output["generic_weight"].detach().cpu().numpy())
        prior_weights.append(output["prior_phys_weight"].detach().cpu().numpy())
        region_gates.append(output.get("region_gate", torch.zeros_like(output["phys_weight"])).detach().cpu().numpy())
        deltas.append(output["delta_phys"].detach().abs().cpu().numpy())
        if "explicit_phys_delta" in output:
            explicit_phys_deltas.append(output["explicit_phys_delta"].detach().abs().cpu().numpy())
        if "physics_projection_strength" in output:
            projection_strengths.append(output["physics_projection_strength"].detach().cpu().numpy())
        if "fd_projection_strength" in output:
            fd_strengths.append(output["fd_projection_strength"].detach().cpu().numpy())
        if "graph_projection_strength" in output:
            graph_strengths.append(output["graph_projection_strength"].detach().cpu().numpy())
        if "temporal_projection_strength" in output:
            temporal_strengths.append(output["temporal_projection_strength"].detach().cpu().numpy())
        residuals.append(float(output["physics_residual"].detach().abs().mean().cpu()))
        missing_region = target_mask.detach().cpu().numpy()
        region_masks["missing"].append(missing_region)
        if "noise_region_mask" in batch:
            region_masks["noise"].append(batch["noise_region_mask"].detach().cpu().numpy())
        if "incident_region_mask" in batch:
            incident_region = batch["incident_region_mask"].detach().cpu().numpy()
            region_masks["incident"].append(incident_region)
            region_masks["incident_missing"].append(incident_region * missing_region)
    pred_np = np.concatenate(preds, axis=0)
    data_np = np.concatenate(data_preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(masks, axis=0)
    metrics = compute_metrics(pred_np, target_np, mask_np)
    data_metrics = compute_metrics(data_np, target_np, mask_np)
    metrics["loss"] = float(np.mean(losses))
    metrics["data_masked_mae"] = data_metrics["masked_mae"]
    phys_weight_np = np.concatenate(phys_weights, axis=0)
    generic_weight_np = np.concatenate(generic_weights, axis=0)
    metrics["phys_weight_mean"] = float(phys_weight_np.mean())
    metrics["generic_weight_mean"] = float(generic_weight_np.mean())
    metrics["prior_phys_weight_mean"] = float(np.concatenate(prior_weights, axis=0).mean())
    metrics["region_gate_mean"] = float(np.concatenate(region_gates, axis=0).mean())
    metrics["delta_phys_mean"] = float(np.concatenate(deltas, axis=0).mean())
    if explicit_phys_deltas:
        metrics["explicit_phys_delta_mean"] = float(np.concatenate(explicit_phys_deltas, axis=0).mean())
    if projection_strengths:
        metrics["physics_projection_strength_mean"] = float(np.concatenate(projection_strengths, axis=0).mean())
    if fd_strengths:
        metrics["fd_projection_strength_mean"] = float(np.concatenate(fd_strengths, axis=0).mean())
    if graph_strengths:
        metrics["graph_projection_strength_mean"] = float(np.concatenate(graph_strengths, axis=0).mean())
    if temporal_strengths:
        metrics["temporal_projection_strength_mean"] = float(np.concatenate(temporal_strengths, axis=0).mean())
    metrics["physics_residual"] = float(np.mean(residuals))
    for region_name, mask_parts in region_masks.items():
        if not mask_parts:
            continue
        region_mask = np.concatenate(mask_parts, axis=0)
        phys_value = _masked_mean_array(phys_weight_np, region_mask)
        generic_value = _masked_mean_array(generic_weight_np, region_mask)
        if phys_value is not None:
            metrics[f"phys_weight_{region_name}_mean"] = phys_value
            metrics[f"generic_weight_{region_name}_mean"] = generic_value
    return metrics, pred_np, target_np, mask_np


def _run_grin_epoch(model, loader, optimizer, device):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    preds = []
    targets = []
    masks = []
    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        target = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        with torch.set_grad_enabled(train_mode):
            pred = model.grin(x_obs, obs_mask)
            loss = masked_mae_loss(pred, target, target_mask) + 0.1 * masked_mae_loss(pred, target, obs_mask)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.grin.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(masks, axis=0)
    metrics = compute_metrics(pred_np, target_np, mask_np)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def _save_arrays(output_root: Path, scenario: str, pred: np.ndarray, target: np.ndarray, mask: np.ndarray, metrics: dict) -> None:
    run_dir = output_root / scenario
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "pred.npy", pred)
    np.save(run_dir / "true.npy", target)
    np.save(run_dir / "mask.npy", mask)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _train_scenario(config: dict, scenario: str, official_root: Path, epochs: int, output_root: Path, pretrain_epochs: int):
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, adj, scaler, metadata = _scenario_loaders(config, scenario)
    adj = adj.to(device)
    model = OfficialGRINLiteTrustCorrection(
        adj=adj.detach().cpu().numpy(),
        input_dim=int(config["dataset"]["channels"]),
        hidden_dim=int(config["model"].get("hidden_dim", 32)),
        ff_dim=int(config["model"].get("hidden_dim", 32)),
        dropout=0.0,
        correction_clip=0.5,
        flow_only_correction=bool(config.get("method", {}).get("flow_only_correction", False)),
        correction_mode=str(config.get("method", {}).get("correction_mode", "mixed")),
        official_root=official_root,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=0.0)
    logs = []
    for epoch in range(1, pretrain_epochs + 1):
        train_stats = _run_grin_epoch(model, train_loader, optimizer, device)
        val_stats = _run_grin_epoch(model, val_loader, None, device)
        logs.append({"epoch": epoch, "phase": "grin_pretrain", "train_loss": train_stats["loss"], "val_masked_mae": val_stats["masked_mae"]})
    if pretrain_epochs > 0:
        for param in model.grin.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(config["train"]["lr"]),
            weight_decay=0.0,
        )
    for epoch in range(1, epochs + 1):
        train_stats, *_ = _run_epoch(model, train_loader, adj, scaler, optimizer, device, config)
        val_stats, *_ = _run_epoch(model, val_loader, adj, scaler, None, device, config)
        logs.append({"epoch": pretrain_epochs + epoch, "phase": "correction", "train_loss": train_stats["loss"], "val_masked_mae": val_stats["masked_mae"]})
    test_stats, pred, target, mask = _run_epoch(model, test_loader, adj, scaler, None, device, config)
    _save_arrays(output_root, scenario, pred, target, mask, test_stats)
    return {
        "scenario": scenario,
        "model": "OfficialGRIN_LiteTrustCorrection",
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "real_data_used": bool(metadata.get("real_data_used", False)),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        **test_stats,
    }, logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_GRIN_ROOT))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--improvement-gate-loss", action="store_true")
    parser.add_argument("--oracle-correction-loss", action="store_true")
    parser.add_argument("--flow-only-correction", action="store_true")
    parser.add_argument("--disable-physics-features", action="store_true")
    parser.add_argument("--correction-mode", choices=["mixed", "generic", "physics", "gated"], default="mixed")
    parser.add_argument("--contrastive-utility-gate", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--output-root", default="C:/Users/21329/litetrust_official_grin_outputs/official_grin_litetrust_pems08_debug")
    args = parser.parse_args()

    config = _config()
    config["train"]["epochs"] = int(args.epochs)
    config["device"] = "cpu"
    config.setdefault("method", {})
    config["method"]["improvement_gate_loss"] = bool(args.improvement_gate_loss)
    config["method"]["gate_loss_weight"] = 0.05 if args.improvement_gate_loss else 0.0
    config["method"]["harm_loss_weight"] = 0.2 if args.improvement_gate_loss else 0.0
    config["method"]["oracle_correction_loss"] = bool(args.oracle_correction_loss)
    config["method"]["oracle_correction_weight"] = 0.1 if args.oracle_correction_loss else 0.0
    config["method"]["flow_only_correction"] = bool(args.flow_only_correction)
    config["method"]["disable_physics_features"] = bool(args.disable_physics_features)
    config["method"]["correction_mode"] = args.correction_mode
    config["method"]["contrastive_utility_gate"] = bool(args.contrastive_utility_gate)
    config["method"]["utility_gate_weight"] = 0.1 if args.contrastive_utility_gate else 0.0
    config["method"]["utility_harm_weight"] = 0.2 if args.contrastive_utility_gate else 0.0
    output_root = Path(args.output_root)
    rows = []
    logs_by_scenario = {}
    for scenario in args.scenarios:
        print(f"running official GRIN + LiteTrust {scenario}", file=sys.stderr, flush=True)
        row, logs = _train_scenario(
            config,
            scenario,
            Path(args.official_root),
            int(args.epochs),
            output_root,
            int(args.pretrain_epochs),
        )
        rows.append(row)
        logs_by_scenario[scenario] = logs
    try:
        with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(output_root / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "logs": logs_by_scenario}, f, indent=2)
    except OSError as exc:
        print(f"warning: failed to write summary: {exc}", file=sys.stderr, flush=True)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
