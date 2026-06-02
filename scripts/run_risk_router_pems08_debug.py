from __future__ import annotations

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
from models.grin_baseline import GRINLite
from models.litetrust_pinn import LiteTrustGRINCorrection, LiteTrustGRINRiskRouter
from scripts.run_conflict_test import _json_default, _trust_extra_features
from scripts.run_stage10a_pems08_real_debug import SCENARIOS, _config, _scenario_loaders
from scripts.run_stage2_three_dataset_quick import _residual
from scripts.train import resolve_device


VARIANTS = [
    "GRINLite",
    "GRINLite_graph_delta",
    "SoftRouter_bounded",
    "RiskRouter",
    "RiskRouter_no_physics",
]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _masked_mean_np(values: np.ndarray, mask: np.ndarray) -> float | None:
    denom = float(mask.sum())
    if denom <= 0.0:
        return None
    return float((values * mask).sum() / denom)


def _instantiate(variant: str, config: dict) -> torch.nn.Module:
    if variant in {"GRINLite", "GRINLite_graph_delta"}:
        return GRINLite(input_dim=3, hidden_dim=48, output_dim=3, dropout=0.1)
    if variant == "SoftRouter_bounded":
        return LiteTrustGRINCorrection(
            input_dim=3,
            hidden_dim=48,
            output_dim=3,
            dropout=0.1,
            use_uncertainty=True,
            extra_feature_dim=int(config["trust"].get("extra_feature_dim", 6)),
            use_graph_delta=True,
            use_phys_expert=True,
            failure_routing="soft_prior",
            correction_clip=1.0,
        )
    return LiteTrustGRINRiskRouter(
        input_dim=3,
        hidden_dim=48,
        output_dim=3,
        dropout=0.1,
        use_uncertainty=True,
        extra_feature_dim=int(config["trust"].get("extra_feature_dim", 6)),
        use_phys_expert=variant != "RiskRouter_no_physics",
        correction_clip=1.0,
        risk_temperature=0.5,
    )


def _graph_delta(mu_data: torch.Tensor, x_obs: torch.Tensor, obs_mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    local_missing = 1.0 - obs_mask.detach().mean(dim=-1, keepdim=True)
    graph_context = obs_mask * x_obs + (1.0 - obs_mask) * mu_data
    neigh_context = torch.einsum("nm,btmc->btnc", adj, graph_context)
    return local_missing * (neigh_context - mu_data)


def _forward(model, variant: str, x_obs, obs_mask, adj, scaler, config):
    if variant in {"GRINLite", "GRINLite_graph_delta"}:
        mu_data = model(x_obs, obs_mask, adj)
        graph_delta = _graph_delta(mu_data, x_obs, obs_mask, adj) if variant == "GRINLite_graph_delta" else torch.zeros_like(mu_data)
        mu = mu_data + graph_delta
        return {
            "mu": mu,
            "mu_data": mu_data,
            "x_graph": mu_data + graph_delta,
            "x_phys": mu_data,
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
    if "x_graph" not in output:
        output["x_graph"] = output["mu_data"] + output["graph_delta"]
    if "x_phys" not in output:
        output["x_phys"] = output["mu_data"] + output["delta_phys"]
    output["physics_residual"] = _residual(output["mu"], adj, scaler, config["dataset_residual"])
    output["data_residual"] = residual_data
    output["phys_residual"] = _residual(output["x_phys"], adj, scaler, config["dataset_residual"])
    return output


def _counterfactual_mask(obs_mask: torch.Tensor, rate: float) -> torch.Tensor:
    observed_point = (obs_mask.mean(dim=-1, keepdim=True) > 0.99).float()
    sampled = (torch.rand_like(observed_point) < rate).float() * observed_point
    return sampled.repeat_interleave(obs_mask.shape[-1], dim=-1)


def _risk_losses(output: dict, target: torch.Tensor, cf_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    cf_point = (cf_mask.mean(dim=-1, keepdim=True) > 0.0).float()
    if cf_point.sum() <= 0:
        zero = target.new_zeros(())
        return {"risk_loss": zero, "oracle_loss": zero, "oracle_acc": zero}
    experts = torch.stack([output["mu_data"], output["x_graph"], output["x_phys"]], dim=-2)
    expert_errors = torch.mean(torch.abs(experts - target.unsqueeze(-2)), dim=-1).detach()
    risk_pred = output["risk_pred"]
    risk_loss = _masked_mean(F.smooth_l1_loss(risk_pred, expert_errors, reduction="none"), cf_point.repeat_interleave(3, dim=-1))
    oracle = torch.argmin(expert_errors, dim=-1)
    log_weights = torch.log(output["expert_weights"].clamp_min(1e-6))
    oracle_loss = _masked_mean(F.nll_loss(log_weights.permute(0, 3, 1, 2), oracle, reduction="none").unsqueeze(-1), cf_point)
    chosen = torch.argmax(output["expert_weights"], dim=-1)
    oracle_acc = _masked_mean((chosen == oracle).float().unsqueeze(-1), cf_point)
    return {"risk_loss": risk_loss, "oracle_loss": oracle_loss, "oracle_acc": oracle_acc}


def _physics_projection_loss(output: dict) -> torch.Tensor:
    if "phys_residual" not in output or "data_residual" not in output:
        return output["mu"].new_zeros(())
    return torch.relu(output["phys_residual"].abs() - output["data_residual"].detach().abs()).mean()


def _train_loss(variant: str, output: dict, target: torch.Tensor, obs_mask: torch.Tensor, target_mask: torch.Tensor, cf_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    train_mask = torch.clamp(target_mask + cf_mask, max=1.0)
    if variant in {"GRINLite", "GRINLite_graph_delta"}:
        loss = masked_mae_loss(output["mu_data"], target, train_mask) + 0.1 * masked_mae_loss(output["mu_data"], target, obs_mask)
        return loss, {"risk_loss": 0.0, "oracle_loss": 0.0, "oracle_acc": 0.0, "proj_loss": 0.0}
    recon = masked_mae_loss(output["mu"], target, train_mask)
    data_aux = masked_mae_loss(output["mu_data"], target, train_mask)
    correction_l1 = torch.mean(torch.abs(output["delta_phys"]))
    proj_loss = _physics_projection_loss(output)
    loss = recon + 0.3 * data_aux + 0.01 * correction_l1 + 0.05 * proj_loss
    terms = {"risk_loss": 0.0, "oracle_loss": 0.0, "oracle_acc": 0.0, "proj_loss": float(proj_loss.detach().cpu())}
    if variant.startswith("RiskRouter"):
        risk_terms = _risk_losses(output, target, cf_mask)
        loss = loss + 0.2 * risk_terms["risk_loss"] + 0.1 * risk_terms["oracle_loss"]
        terms.update({k: float(v.detach().cpu()) for k, v in risk_terms.items()})
    return loss, terms


def _run_epoch(model, variant: str, loader, adj, scaler, optimizer, device, config: dict):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    risk_losses = []
    oracle_losses = []
    oracle_accs = []
    proj_losses = []
    pred_final = []
    targets = []
    target_masks = []
    obs_masks = []
    weights = []
    risk_preds = []
    graph_deltas = []
    phys_deltas = []
    residuals = []

    for batch in loader:
        x_full = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        cf_mask = _counterfactual_mask(obs_mask, 0.2) if train_mode else torch.zeros_like(obs_mask)
        train_mask = obs_mask * (1.0 - cf_mask)
        x_obs = x_full * train_mask
        with torch.set_grad_enabled(train_mode):
            output = _forward(model, variant, x_obs, train_mask, adj, scaler, config)
            loss, terms = _train_loss(variant, output, x_full, obs_mask, target_mask, cf_mask)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        risk_losses.append(float(terms["risk_loss"]))
        oracle_losses.append(float(terms["oracle_loss"]))
        oracle_accs.append(float(terms["oracle_acc"]))
        proj_losses.append(float(terms["proj_loss"]))
        pred_final.append(output["mu"].detach().cpu().numpy())
        targets.append(x_full.detach().cpu().numpy())
        target_masks.append(target_mask.detach().cpu().numpy())
        obs_masks.append(obs_mask.detach().cpu().numpy())
        graph_deltas.append(output["graph_delta"].detach().abs().cpu().numpy())
        phys_deltas.append(output["delta_phys"].detach().abs().cpu().numpy())
        residuals.append(float(output["physics_residual"].detach().abs().mean().cpu()))
        if "expert_weights" in output:
            weights.append(output["expert_weights"].detach().cpu().numpy())
        if "risk_pred" in output:
            risk_preds.append(output["risk_pred"].detach().cpu().numpy())

    pred_np = np.concatenate(pred_final, axis=0)
    target_np = np.concatenate(targets, axis=0)
    target_mask_np = np.concatenate(target_masks, axis=0)
    obs_mask_np = np.concatenate(obs_masks, axis=0)
    graph_delta_np = np.concatenate(graph_deltas, axis=0)
    phys_delta_np = np.concatenate(phys_deltas, axis=0)
    metrics = compute_metrics(pred_np, target_np, target_mask_np)
    failed_node_mask = (1.0 - obs_mask_np.mean(axis=(1, 3), keepdims=False))[:, None, :, None] > 0.9
    failed_node_mask = np.repeat(failed_node_mask, obs_mask_np.shape[1], axis=1)
    failed_node_mask = np.repeat(failed_node_mask, obs_mask_np.shape[-1], axis=-1)
    weight_np = np.concatenate(weights, axis=0) if weights else None
    risk_np = np.concatenate(risk_preds, axis=0) if risk_preds else None
    row = {
        "loss": float(np.mean(losses)),
        "masked_mae": metrics["masked_mae"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "physics_residual": float(np.mean(residuals)),
        "risk_loss": float(np.mean(risk_losses)),
        "oracle_loss": float(np.mean(oracle_losses)),
        "oracle_acc": float(np.mean(oracle_accs)),
        "proj_loss": float(np.mean(proj_losses)),
        "graph_delta_failed_mean": _masked_mean_np(graph_delta_np, failed_node_mask.astype(np.float32)),
        "delta_phys_missing_mean": _masked_mean_np(phys_delta_np, target_mask_np),
    }
    if weight_np is not None:
        expanded = np.repeat(weight_np, obs_mask_np.shape[-1], axis=-1)
        row.update(
            {
                "data_weight_mean": float(weight_np[..., 0].mean()),
                "graph_weight_mean": float(weight_np[..., 1].mean()),
                "phys_weight_mean": float(weight_np[..., 2].mean()),
                "graph_weight_failed_mean": _masked_mean_np(np.repeat(weight_np[..., 1:2], obs_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
                "phys_weight_failed_mean": _masked_mean_np(np.repeat(weight_np[..., 2:3], obs_mask_np.shape[-1], axis=-1), failed_node_mask.astype(np.float32)),
                "phys_weight_missing_mean": _masked_mean_np(np.repeat(weight_np[..., 2:3], target_mask_np.shape[-1], axis=-1), target_mask_np),
            }
        )
    else:
        row.update({"data_weight_mean": None, "graph_weight_mean": None, "phys_weight_mean": None, "graph_weight_failed_mean": None, "phys_weight_failed_mean": None, "phys_weight_missing_mean": None})
    if risk_np is not None:
        row.update(
            {
                "risk_data_mean": float(risk_np[..., 0].mean()),
                "risk_graph_mean": float(risk_np[..., 1].mean()),
                "risk_phys_mean": float(risk_np[..., 2].mean()),
            }
        )
    else:
        row.update({"risk_data_mean": None, "risk_graph_mean": None, "risk_phys_mean": None})
    return row


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
        train_stats = _run_epoch(model, variant, train_loader, adj, scaler, optimizer, device, config)
        val_stats = _run_epoch(model, variant, val_loader, adj, scaler, None, device, config)
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_risk_loss": train_stats["risk_loss"],
                "train_oracle_acc": train_stats["oracle_acc"],
                "val_masked_mae": val_stats["masked_mae"],
                "val_phys_weight": val_stats["phys_weight_mean"],
            }
        )
    test_stats = _run_epoch(model, variant, test_loader, adj, scaler, None, device, config)
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
    output_dir = ROOT / "results" / "risk_router_pems08_debug"
    log_dir = output_dir / "logs"
    fieldnames = sorted({key for row in rows for key in row.keys()})
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
    except OSError:
        return


def main() -> None:
    config = _config()
    config["train"]["epochs"] = 30
    config["method"]["pretrain_epochs"] = 0
    rows = []
    logs = {}
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            print(f"running risk-router {scenario} {variant}", file=sys.stderr, flush=True)
            row, variant_logs = _train_variant(config, scenario, variant)
            rows.append(row)
            logs[f"{scenario}__{variant}"] = variant_logs
    _write_outputs(rows, logs)
    print(json.dumps({"rows": rows}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
