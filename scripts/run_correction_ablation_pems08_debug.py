from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.losses import masked_mae_loss
from losses.metrics import compute_metrics
from models.grin_baseline import GRINLite
from models.litetrust_pinn import LiteTrustGRINCorrection, LiteTrustGRINReliabilityRouter, _node_failure_signal
from scripts.run_conflict_test import _json_default, _trust_extra_features
from scripts.run_stage10a_pems08_real_debug import SCENARIOS, _config, _scenario_loaders
from scripts.run_stage2_three_dataset_quick import _lambda_schedule, _residual, _loss_terms
from scripts.train import resolve_device


VARIANTS = [
    "GRINLite",
    "GRINLite_graph_delta",
    "LiteTrust_delta_only",
    "CorrectionV2_soft_router",
    "CorrectionV2_no_physics",
    "CorrectionV2_hard_router",
    "ReliabilityRouter",
    "ReliabilityRouter_validity",
    "ReliabilityRouter_spatial_physics",
    "ReliabilityRouter_directional_physics",
    "ReliabilityRouter_no_physics",
]


def _masked_mean_np(values: np.ndarray, mask: np.ndarray) -> float | None:
    denom = float(mask.sum())
    if denom <= 0.0:
        return None
    return float((values * mask).sum() / denom)


def _instantiate(variant: str, config: dict) -> torch.nn.Module:
    if variant in {"GRINLite", "GRINLite_graph_delta"}:
        return GRINLite(input_dim=3, hidden_dim=48, output_dim=3, dropout=0.1)
    if variant in {
        "ReliabilityRouter",
        "ReliabilityRouter_validity",
        "ReliabilityRouter_no_validity",
        "ReliabilityRouter_spatial_physics",
        "ReliabilityRouter_directional_physics",
        "ReliabilityRouter_no_physics",
    }:
        return LiteTrustGRINReliabilityRouter(
            input_dim=3,
            hidden_dim=48,
            output_dim=3,
            dropout=0.1,
            w_min=float(config["trust"].get("w_min", 0.0)),
            use_uncertainty=True,
            extra_feature_dim=int(config["trust"].get("extra_feature_dim", 6)),
            use_phys_expert=variant != "ReliabilityRouter_no_physics",
            correction_clip=1.0,
            router_temperature=0.7,
            use_validity_gate=variant == "ReliabilityRouter_validity",
            use_spatial_physics=variant == "ReliabilityRouter_spatial_physics",
            use_directional_physics=variant == "ReliabilityRouter_directional_physics",
        )
    use_graph_delta = variant in {"CorrectionV2_soft_router", "CorrectionV2_no_physics", "CorrectionV2_hard_router"}
    failure_routing = "hard" if variant == "CorrectionV2_hard_router" else "soft_prior"
    return LiteTrustGRINCorrection(
        input_dim=3,
        hidden_dim=48,
        output_dim=3,
        dropout=0.1,
        w_min=float(config["trust"].get("w_min", 0.0)),
        use_uncertainty=True,
        extra_feature_dim=int(config["trust"].get("extra_feature_dim", 6)),
        use_graph_delta=use_graph_delta,
        use_phys_expert=variant != "CorrectionV2_no_physics",
        failure_routing=failure_routing,
    )


def _graph_delta(mu_data: torch.Tensor, x_obs: torch.Tensor, obs_mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    node_missing = 1.0 - obs_mask.detach().mean(dim=(1, 3))[:, None, :, None]
    failed_sensor = _node_failure_signal(node_missing, local_missing, temperature=0.2)
    graph_context = obs_mask * x_obs + (1.0 - obs_mask) * mu_data
    neigh_context = torch.einsum("nm,btmc->btnc", adj, graph_context)
    return failed_sensor * (neigh_context - mu_data)


def _forward_variant(model, variant: str, x_obs, obs_mask, adj, scaler, config, epoch: int):
    if variant in {"GRINLite", "GRINLite_graph_delta"}:
        mu_data = model(x_obs, obs_mask, adj)
        graph_delta = _graph_delta(mu_data, x_obs, obs_mask, adj) if variant == "GRINLite_graph_delta" else torch.zeros_like(mu_data)
        mu = mu_data + graph_delta
        return {
            "mu": mu,
            "mu_data": mu_data,
            "graph_delta": graph_delta,
            "delta_phys": torch.zeros_like(mu_data),
            "trust": None,
            "physics_residual": _residual(mu, adj, scaler, config["dataset_residual"]),
        }

    base = model(x_obs, obs_mask, adj)
    mu_data = base["mu_data"]
    residual_data = _residual(mu_data, adj, scaler, config["dataset_residual"])
    extra = _trust_extra_features(x_obs, obs_mask, adj, residual_data)
    gate_log_var = base["log_var"].mean(dim=-1, keepdim=True)
    output = model(
        x_obs,
        obs_mask,
        adj,
        residual_abs=residual_data.detach().abs(),
        residual_signed=residual_data.detach(),
        log_var=gate_log_var,
        extra_feature=extra,
    )
    output["physics_residual"] = _residual(output["mu"], adj, scaler, config["dataset_residual"])
    output["data_residual"] = residual_data
    if "x_phys" in output:
        output["phys_residual"] = _residual(output["x_phys"], adj, scaler, config["dataset_residual"])
    return output


def _train_loss(variant: str, output: dict, target: torch.Tensor, obs_mask: torch.Tensor, target_mask: torch.Tensor, epoch: int, config: dict):
    if variant in {"GRINLite", "GRINLite_graph_delta"}:
        missing = masked_mae_loss(output["mu_data"], target, target_mask)
        observed = masked_mae_loss(output["mu_data"], target, obs_mask)
        return missing + 0.1 * observed
    raise ValueError(f"_train_loss is only used for GRINLite variants, got {variant}")


def _run_epoch(model, variant: str, loader, adj, scaler, optimizer, device, epoch: int, config: dict):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    pred_final = []
    pred_data = []
    targets = []
    target_masks = []
    obs_masks = []
    trusts = []
    data_weights = []
    graph_weights = []
    phys_weights = []
    correction_trusts = []
    projection_gammas = []
    physics_validities = []
    spatial_phys_gates = []
    spatial_phys_deltas = []
    directional_phys_gates = []
    directional_phys_deltas = []
    directional_shifts = []
    directional_conservation_residuals = []
    graph_deltas = []
    phys_deltas = []
    residuals = []

    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        target = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        with torch.set_grad_enabled(train_mode):
            output = _forward_variant(model, variant, x_obs, obs_mask, adj, scaler, config, epoch)
            if variant in {"GRINLite", "GRINLite_graph_delta"}:
                loss = _train_loss(variant, output, target, obs_mask, target_mask, epoch, config)
            else:
                lambda_phys = _lambda_schedule(epoch, config["physics"])
                if epoch <= int(config.get("method", {}).get("pretrain_epochs", 0)):
                    lambda_phys = 0.0
                extra = _trust_extra_features(x_obs, obs_mask, adj, _residual(output["mu_data"], adj, scaler, config["dataset_residual"]))
                loss, _ = _loss_terms(
                    "LiteTrustGRINCorrection",
                    output["mu"],
                    target,
                    target_mask,
                    obs_mask,
                    output["physics_residual"],
                    output["trust"],
                    output["log_var"],
                    extra,
                    epoch,
                    lambda_phys,
                    config,
                    output,
                )
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        pred_final.append(output["mu"].detach().cpu().numpy())
        pred_data.append(output["mu_data"].detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        target_masks.append(target_mask.detach().cpu().numpy())
        obs_masks.append(obs_mask.detach().cpu().numpy())
        graph_deltas.append(output["graph_delta"].detach().abs().cpu().numpy())
        phys_deltas.append(output["delta_phys"].detach().abs().cpu().numpy())
        residuals.append(float(output["physics_residual"].detach().abs().mean().cpu()))
        if output["trust"] is not None:
            trusts.append(output["trust"].detach().cpu().numpy())
        if "data_weight" in output:
            data_weights.append(output["data_weight"].detach().cpu().numpy())
            graph_weights.append(output["graph_weight"].detach().cpu().numpy())
            phys_weights.append(output["phys_weight"].detach().cpu().numpy())
            correction_trusts.append(output["correction_trust"].detach().cpu().numpy())
        if "projection_gamma" in output:
            projection_gammas.append(output["projection_gamma"].detach().cpu().numpy())
        if "physics_validity" in output:
            physics_validities.append(output["physics_validity"].detach().cpu().numpy())
        if "spatial_phys_gate" in output:
            spatial_phys_gates.append(output["spatial_phys_gate"].detach().cpu().numpy())
        if "spatial_phys_delta" in output:
            spatial_phys_deltas.append(output["spatial_phys_delta"].detach().abs().cpu().numpy())
        if "directional_phys_gate" in output:
            directional_phys_gates.append(output["directional_phys_gate"].detach().cpu().numpy())
        if "directional_phys_delta" in output:
            directional_phys_deltas.append(output["directional_phys_delta"].detach().abs().cpu().numpy())
        if "directional_shift" in output:
            directional_shifts.append(output["directional_shift"].detach().cpu().numpy())
        if "directional_conservation_residual" in output:
            directional_conservation_residuals.append(output["directional_conservation_residual"].detach().cpu().numpy())

    pred_final_np = np.concatenate(pred_final, axis=0)
    pred_data_np = np.concatenate(pred_data, axis=0)
    target_np = np.concatenate(targets, axis=0)
    target_mask_np = np.concatenate(target_masks, axis=0)
    obs_mask_np = np.concatenate(obs_masks, axis=0)
    graph_delta_np = np.concatenate(graph_deltas, axis=0)
    phys_delta_np = np.concatenate(phys_deltas, axis=0)
    final_metrics = compute_metrics(pred_final_np, target_np, target_mask_np)
    data_metrics = compute_metrics(pred_data_np, target_np, target_mask_np)

    trust_np = np.concatenate(trusts, axis=0) if trusts else None
    data_weight_np = np.concatenate(data_weights, axis=0) if data_weights else None
    graph_weight_np = np.concatenate(graph_weights, axis=0) if graph_weights else None
    phys_weight_np = np.concatenate(phys_weights, axis=0) if phys_weights else None
    correction_trust_np = np.concatenate(correction_trusts, axis=0) if correction_trusts else None
    projection_gamma_np = np.concatenate(projection_gammas, axis=0) if projection_gammas else None
    physics_validity_np = np.concatenate(physics_validities, axis=0) if physics_validities else None
    spatial_phys_gate_np = np.concatenate(spatial_phys_gates, axis=0) if spatial_phys_gates else None
    spatial_phys_delta_np = np.concatenate(spatial_phys_deltas, axis=0) if spatial_phys_deltas else None
    directional_phys_gate_np = np.concatenate(directional_phys_gates, axis=0) if directional_phys_gates else None
    directional_phys_delta_np = np.concatenate(directional_phys_deltas, axis=0) if directional_phys_deltas else None
    directional_shift_np = np.concatenate(directional_shifts, axis=0) if directional_shifts else None
    directional_conservation_residual_np = (
        np.concatenate(directional_conservation_residuals, axis=0) if directional_conservation_residuals else None
    )
    failed_node_mask = (1.0 - obs_mask_np.mean(axis=(1, 3), keepdims=False))[:, None, :, None] > 0.9
    failed_node_mask = np.repeat(failed_node_mask, obs_mask_np.shape[1], axis=1)
    failed_node_mask = np.repeat(failed_node_mask, obs_mask_np.shape[-1], axis=-1)
    normal_node_mask = ~failed_node_mask
    return {
        "loss": float(np.mean(losses)),
        "masked_mae_final": final_metrics["masked_mae"],
        "masked_mae_data": data_metrics["masked_mae"],
        "mae_delta_final_minus_data": final_metrics["masked_mae"] - data_metrics["masked_mae"],
        "physics_residual": float(np.mean(residuals)),
        "trust_mean": None if trust_np is None else float(trust_np.mean()),
        "trust_std": None if trust_np is None else float(trust_np.std()),
        "trust_observed_mean": None if trust_np is None else _masked_mean_np(np.repeat(trust_np, obs_mask_np.shape[-1], axis=-1), obs_mask_np),
        "trust_missing_mean": None if trust_np is None else _masked_mean_np(np.repeat(trust_np, target_mask_np.shape[-1], axis=-1), target_mask_np),
        "trust_failed_node_mean": None if trust_np is None else _masked_mean_np(np.repeat(trust_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
        "trust_normal_node_mean": None if trust_np is None else _masked_mean_np(np.repeat(trust_np, target_mask_np.shape[-1], axis=-1), normal_node_mask.astype(np.float32)),
        "data_weight_mean": None if data_weight_np is None else float(data_weight_np.mean()),
        "graph_weight_mean": None if graph_weight_np is None else float(graph_weight_np.mean()),
        "phys_weight_mean": None if phys_weight_np is None else float(phys_weight_np.mean()),
        "correction_trust_mean": None if correction_trust_np is None else float(correction_trust_np.mean()),
        "graph_weight_failed_node_mean": None if graph_weight_np is None else _masked_mean_np(np.repeat(graph_weight_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
        "phys_weight_failed_node_mean": None if phys_weight_np is None else _masked_mean_np(np.repeat(phys_weight_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
        "projection_gamma_mean": None if projection_gamma_np is None else float(projection_gamma_np.mean()),
        "projection_gamma_missing_mean": None if projection_gamma_np is None else _masked_mean_np(np.repeat(projection_gamma_np, target_mask_np.shape[-1], axis=-1), target_mask_np),
        "projection_gamma_failed_node_mean": None if projection_gamma_np is None else _masked_mean_np(np.repeat(projection_gamma_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
        "physics_validity_mean": None if physics_validity_np is None else float(physics_validity_np.mean()),
        "physics_validity_missing_mean": None if physics_validity_np is None else _masked_mean_np(np.repeat(physics_validity_np, target_mask_np.shape[-1], axis=-1), target_mask_np),
        "physics_validity_failed_node_mean": None if physics_validity_np is None else _masked_mean_np(np.repeat(physics_validity_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
        "spatial_phys_gate_mean": None if spatial_phys_gate_np is None else float(spatial_phys_gate_np.mean()),
        "spatial_phys_gate_missing_mean": None if spatial_phys_gate_np is None else _masked_mean_np(np.repeat(spatial_phys_gate_np, target_mask_np.shape[-1], axis=-1), target_mask_np),
        "spatial_phys_gate_failed_node_mean": None if spatial_phys_gate_np is None else _masked_mean_np(np.repeat(spatial_phys_gate_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
        "spatial_phys_delta_mean": None if spatial_phys_delta_np is None else float(spatial_phys_delta_np.mean()),
        "spatial_phys_delta_missing_mean": None if spatial_phys_delta_np is None else _masked_mean_np(spatial_phys_delta_np, target_mask_np),
        "spatial_phys_delta_failed_node_mean": None if spatial_phys_delta_np is None else _masked_mean_np(spatial_phys_delta_np, failed_node_mask.astype(np.float32)),
        "directional_phys_gate_mean": None if directional_phys_gate_np is None else float(directional_phys_gate_np.mean()),
        "directional_phys_gate_missing_mean": None if directional_phys_gate_np is None else _masked_mean_np(np.repeat(directional_phys_gate_np, target_mask_np.shape[-1], axis=-1), target_mask_np),
        "directional_phys_gate_failed_node_mean": None if directional_phys_gate_np is None else _masked_mean_np(np.repeat(directional_phys_gate_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
        "directional_phys_delta_mean": None if directional_phys_delta_np is None else float(directional_phys_delta_np.mean()),
        "directional_phys_delta_missing_mean": None if directional_phys_delta_np is None else _masked_mean_np(directional_phys_delta_np, target_mask_np),
        "directional_phys_delta_failed_node_mean": None if directional_phys_delta_np is None else _masked_mean_np(directional_phys_delta_np, failed_node_mask.astype(np.float32)),
        "directional_shift_mean": None if directional_shift_np is None else float(directional_shift_np.mean()),
        "directional_shift_failed_node_mean": None if directional_shift_np is None else _masked_mean_np(
            np.repeat(directional_shift_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)
        ),
        "directional_conservation_residual_mean": None if directional_conservation_residual_np is None else float(directional_conservation_residual_np.mean()),
        "directional_conservation_residual_failed_node_mean": None if directional_conservation_residual_np is None else _masked_mean_np(
            np.repeat(directional_conservation_residual_np, target_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)
        ),
        "graph_delta_failed_mean": _masked_mean_np(graph_delta_np, failed_node_mask.astype(np.float32)),
        "graph_delta_missing_mean": _masked_mean_np(graph_delta_np, target_mask_np),
        "delta_phys_mean": float(phys_delta_np.mean()),
        "delta_phys_missing_mean": _masked_mean_np(phys_delta_np, target_mask_np),
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }


def _train_variant(config: dict, scenario: str, variant: str):
    device = resolve_device(config.get("device", "cpu"))
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    train_loader, val_loader, test_loader, adj, scaler, metadata = _scenario_loaders(config, scenario)
    model = _instantiate(variant, config).to(device)
    adj = adj.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    logs = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_stats = _run_epoch(model, variant, train_loader, adj, scaler, optimizer, device, epoch, config)
        val_stats = _run_epoch(model, variant, val_loader, adj, scaler, None, device, epoch, config)
        logs.append({"epoch": epoch, "train_loss": train_stats["loss"], "val_masked_mae": val_stats["masked_mae_final"]})
    test_stats = _run_epoch(model, variant, test_loader, adj, scaler, None, device, int(config["train"]["epochs"]), config)
    row = {
        "dataset": "PEMS08",
        "scenario": scenario,
        "variant": variant,
        "real_data_used": bool(metadata.get("real_data_used", False)),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        "epochs": int(config["train"]["epochs"]),
        **{k: v for k, v in test_stats.items() if k != "loss"},
    }
    return row, logs


def _write_outputs(rows: list[dict], logs: dict[str, list[dict]]) -> None:
    output_dir = ROOT / "results" / "correction_soft_router_pems08_debug"
    log_dir = output_dir / "logs"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    lines = [
        "# Correction V2 Soft Router Ablation",
        "",
        "| Scenario | Variant | Final masked MAE | Data-branch masked MAE | Phys weight | Graph weight | Correction trust |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        phys_weight = "" if row["phys_weight_mean"] is None else f"{row['phys_weight_mean']:.6f}"
        graph_weight = "" if row["graph_weight_mean"] is None else f"{row['graph_weight_mean']:.6f}"
        correction_trust = "" if row["correction_trust_mean"] is None else f"{row['correction_trust_mean']:.6f}"
        lines.append(
            f"| {row['scenario']} | {row['variant']} | {row['masked_mae_final']:.6f} | "
            f"{row['masked_mae_data']:.6f} | {phys_weight} | {graph_weight} | {correction_trust} |"
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        for key, values in logs.items():
            with open(log_dir / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(values[0].keys()))
                writer.writeheader()
                writer.writerows(values)
        with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"variants": VARIANTS, "scenarios": SCENARIOS, "epochs": 50}, f, sort_keys=False)
    except OSError:
        return


def main() -> None:
    config = _config()
    config["train"]["epochs"] = 50
    config["method"]["pretrain_epochs"] = 0
    rows = []
    logs = {}
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            print(f"running ablation {scenario} {variant}", file=sys.stderr, flush=True)
            row, variant_logs = _train_variant(config, scenario, variant)
            rows.append(row)
            logs[f"{scenario}__{variant}"] = variant_logs
    _write_outputs(rows, logs)
    print(json.dumps({"rows": rows}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
