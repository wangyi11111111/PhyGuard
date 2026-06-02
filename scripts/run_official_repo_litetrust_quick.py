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
from models.official_grin_wrapper import DEFAULT_OFFICIAL_GRIN_ROOT
from scripts.run_stage10a_pems08_real_debug import _config, _scenario_loaders
from scripts.train import resolve_device


def _load_official_models(official_root: Path):
    root_str = str(official_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from lib.nn.models import GRINet, LiteTrustGRINet

    return GRINet, LiteTrustGRINet


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * mask) / torch.clamp(mask.sum(), min=1.0)


def _balanced_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred.clamp(1e-4, 1.0 - 1e-4), target, reduction="none")
    pos_mask = mask * target
    neg_mask = mask * (1.0 - target)
    pos_loss = _masked_mean(bce, pos_mask)
    neg_loss = _masked_mean(bce, neg_mask)
    return 0.5 * (pos_loss + neg_loss)


def _balanced_class_weights(labels: torch.Tensor, mask: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    valid = labels[mask > 0.5]
    if valid.numel() < num_classes:
        return torch.ones(num_classes, dtype=torch.float32, device=labels.device)
    counts = torch.bincount(valid.reshape(-1).long(), minlength=num_classes).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / torch.clamp(weights.mean(), min=1e-6)
    return weights.clamp(0.5, 3.0)


def _promotion_margin_for_scenario(scenario: str | None, base_margin: float) -> float:
    if scenario == "random_missing_50":
        return 0.75 * base_margin
    if scenario == "noise_random_missing":
        return 1.25 * base_margin
    if scenario == "incident_perturbation":
        return 1.50 * base_margin
    return base_margin


def _set_litetrust_mode(
    model,
    *,
    generic_only: bool = False,
    contrastive_verifier: bool = False,
    physics_verified: bool = False,
    physics_candidate: bool = False,
    physics_promoted: bool = False,
    discrete_physics_promotion: bool = False,
    physics_harm: bool = False,
    harm_suppressed: bool = False,
    region_adaptive: bool = False,
) -> None:
    if not hasattr(model, "grin"):
        return
    model.generic_only_correction = bool(generic_only)
    model.contrastive_utility_verifier = bool(contrastive_verifier)
    model.physics_verified_correction = bool(physics_verified)
    model.physics_candidate_correction = bool(physics_candidate)
    model.physics_promoted_correction = bool(physics_promoted)
    model.discrete_physics_promotion = bool(discrete_physics_promotion)
    model.physics_harm_verifier = bool(physics_harm)
    model.harm_suppressed_correction = bool(harm_suppressed)
    model.region_adaptive_correction = bool(region_adaptive)
    model.utility_router_correction = False
    model.physics_vetted_correction = False
    model.selective_correction = False


def _set_trainable(model, trainable_prefixes: tuple[str, ...]) -> list[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        param.requires_grad_(any(name.startswith(prefix) for prefix in trainable_prefixes))
    return [param for param in model.parameters() if param.requires_grad]


SCENARIO_TOKENS = {
    "random_missing_50": [1.0, 0.0, 0.0],
    "noise_random_missing": [0.0, 1.0, 0.0],
    "incident_perturbation": [0.0, 0.0, 1.0],
}


def _scenario_tensor(scenario: str | None, device) -> torch.Tensor | None:
    if scenario is None:
        return None
    values = SCENARIO_TOKENS.get(scenario)
    if values is None:
        values = [0.0, 0.0, 0.0]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _model_forward(model, x_obs, obs_mask, grin_only: bool = False, return_details: bool = False, scenario_token=None):
    if grin_only and hasattr(model, "grin"):
        return model.grin(x_obs, obs_mask)
    if return_details and hasattr(model, "grin"):
        return model(x_obs, obs_mask, return_details=True, scenario=scenario_token)
    if hasattr(model, "grin"):
        return model(x_obs, obs_mask, scenario=scenario_token)
    return model(x_obs, obs_mask)


def _train_epoch(
    model,
    loader,
    optimizer,
    device,
    grin_only: bool = False,
    utility_loss: bool = False,
    selective_loss: bool = False,
    physics_verified_loss: bool = False,
    contrastive_verifier_loss: bool = False,
    balanced_verifier_loss: bool = False,
    hard_negative_verifier_loss: bool = False,
    physics_harm_verifier_loss: bool = False,
    harm_regularized_loss: bool = False,
    harm_suppressed_loss: bool = False,
    promotion_loss: bool = False,
    discrete_promotion_loss: bool = False,
    target_only_loss: bool = False,
    hard_negative_margin: float = 0.02,
    harm_hard_weight: float = 1.0,
    harm_safe_weight: float = 0.35,
    harm_utility_target: bool = False,
    harm_utility_temperature: float = 0.05,
    diagnostics: bool = False,
    scenario: str | None = None,
):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    losses = []
    preds = []
    targets = []
    masks = []
    detail_preds = {key: [] for key in ["mu_data", "x_generic", "x_generic_v2", "x_generic_v3", "x_generic_v4", "x_vetted", "x_phys", "x_fused", "x_physics_promoted", "x_discrete_physics_promoted", "x_router", "x_verified", "x_harm_verified", "x_harm_suppressed", "oracle_best"]}
    detail_stats = {
        "phys_weight_sum": 0.0,
        "phys_weight_count": 0.0,
        "correction_conf_sum": 0.0,
        "correction_conf_count": 0.0,
        "harm_rate_sum": 0.0,
        "harm_rate_count": 0.0,
        "router_weight_sum": [0.0, 0.0, 0.0, 0.0],
        "router_weight_count": 0.0,
        "verifier_gate_sum": 0.0,
        "verifier_gate_count": 0.0,
        "verifier_gate_pos_sum": 0.0,
        "verifier_gate_pos_count": 0.0,
        "verifier_gate_neg_sum": 0.0,
        "verifier_gate_neg_count": 0.0,
        "generic_better_sum": 0.0,
        "generic_better_count": 0.0,
        "hard_negative_sum": 0.0,
        "hard_negative_count": 0.0,
        "verifier_gate_hard_neg_sum": 0.0,
        "verifier_gate_hard_neg_count": 0.0,
        "verifier_gate_safe_sum": 0.0,
        "verifier_gate_safe_count": 0.0,
        "harm_prob_sum": 0.0,
        "harm_prob_count": 0.0,
        "harm_prob_hard_neg_sum": 0.0,
        "harm_prob_hard_neg_count": 0.0,
        "harm_prob_safe_sum": 0.0,
        "harm_prob_safe_count": 0.0,
        "harm_keep_sum": 0.0,
        "harm_keep_count": 0.0,
        "harm_keep_hard_neg_sum": 0.0,
        "harm_keep_hard_neg_count": 0.0,
        "harm_keep_safe_sum": 0.0,
        "harm_keep_safe_count": 0.0,
        "harm_region_gate_sum": 0.0,
        "harm_region_gate_count": 0.0,
        "harm_general_prob_sum": 0.0,
        "harm_general_prob_count": 0.0,
        "harm_sensor_prob_sum": 0.0,
        "harm_sensor_prob_count": 0.0,
        "correction_allowance_sum": 0.0,
        "correction_allowance_count": 0.0,
        "correction_allowance_hard_neg_sum": 0.0,
        "correction_allowance_hard_neg_count": 0.0,
        "correction_allowance_safe_sum": 0.0,
        "correction_allowance_safe_count": 0.0,
        "physics_promotion_sum": 0.0,
        "physics_promotion_count": 0.0,
        "physics_promotion_phys_better_sum": 0.0,
        "physics_promotion_phys_better_count": 0.0,
        "physics_promotion_fused_better_sum": 0.0,
        "physics_promotion_fused_better_count": 0.0,
        "physics_promotion_mode_mean_sum": [0.0, 0.0, 0.0],
        "physics_promotion_mode_count": 0.0,
        "physics_promotion_mode_target_sum": [0.0, 0.0, 0.0],
        "physics_promotion_mode_target_count": 0.0,
        "physics_promotion_mode_clear_sum": 0.0,
        "physics_promotion_mode_clear_count": 0.0,
        "harm_tp_sum": 0.0,
        "harm_pred_sum": 0.0,
        "harm_actual_sum": 0.0,
        "generic_v2_strength_sum": 0.0,
        "generic_v2_strength_count": 0.0,
        "generic_v2_weight_sum": [0.0, 0.0, 0.0],
        "generic_v2_weight_count": 0.0,
        "generic_v3_gain_sum": 0.0,
        "generic_v3_gain_count": 0.0,
        "generic_v3_refine_abs_sum": 0.0,
        "generic_v3_refine_abs_count": 0.0,
        "generic_v4_scale_sum": 0.0,
        "generic_v4_scale_count": 0.0,
        "residual_before_sum": 0.0,
        "residual_after_data_sum": 0.0,
        "residual_after_verified_sum": 0.0,
        "residual_count": 0.0,
    }
    for batch in loader:
        x_obs = batch["x_obs"].to(device)
        target = batch["x_full"].to(device)
        obs_mask = batch["mask"].to(device) > 0.5
        target_mask = batch["target_mask"].to(device)
        scenario_token = _scenario_tensor(scenario, device)
        with torch.set_grad_enabled(train_mode):
            output = _model_forward(
                model,
                x_obs,
                obs_mask,
                grin_only=grin_only,
                return_details=((utility_loss or selective_loss or physics_verified_loss or contrastive_verifier_loss or balanced_verifier_loss or hard_negative_verifier_loss or physics_harm_verifier_loss or harm_regularized_loss or harm_suppressed_loss or promotion_loss) and train_mode) or diagnostics,
                scenario_token=scenario_token,
            )
            if isinstance(output, dict):
                pred = output["mu"]
            else:
                pred = output[0] if isinstance(output, tuple) else output
            loss = masked_mae_loss(pred, target, target_mask)
            if not target_only_loss:
                loss = loss + 0.1 * masked_mae_loss(pred, target, obs_mask.float())
            if isinstance(output, dict):
                base_err = torch.abs(output["mu_data"].detach() - target)
                final_err = torch.abs(pred - target)
                generic_source = output.get("x_error", output["x_generic"])
                data_candidate = generic_source
                generic_err = torch.abs(generic_source.detach() - target)
                phys_err = torch.abs(output["x_phys"].detach() - target)
                if utility_loss or selective_loss or physics_verified_loss:
                    best_candidate_err = torch.minimum(base_err, torch.minimum(generic_err, phys_err))
                    harm_loss = _masked_mean(torch.relu(final_err - best_candidate_err), target_mask)
                    delta_target = torch.clamp(target - output["mu_data"].detach(), min=-0.5, max=0.5)
                    calibrated_delta = output.get("error_delta", output["final_delta"])
                    if "error_gamma" in output:
                        calibrated_delta = output["error_gamma"] * calibrated_delta
                    delta_loss = _masked_mean(
                        F.smooth_l1_loss(calibrated_delta, delta_target, reduction="none"),
                        target_mask,
                    )
                    utility_target = torch.sigmoid((generic_err - phys_err) / 0.05).detach()
                    gate_loss = _masked_mean(
                        F.binary_cross_entropy(
                            output["phys_weight"].clamp(1e-4, 1.0 - 1e-4),
                            utility_target,
                            reduction="none",
                        ),
                        target_mask,
                    )
                    loss = loss + 0.5 * delta_loss + 0.2 * harm_loss + 0.05 * gate_loss
                if selective_loss and "correction_conf" in output and "x_fused" in output:
                    fused_err = torch.abs(output["x_fused"].detach() - target)
                    select_target = torch.sigmoid((base_err - fused_err) / 0.05).detach()
                    select_loss = _masked_mean(
                        F.binary_cross_entropy(
                            output["correction_conf"].clamp(1e-4, 1.0 - 1e-4),
                            select_target,
                            reduction="none",
                        ),
                        target_mask,
                    )
                    base_harm_loss = _masked_mean(torch.relu(final_err - base_err), target_mask)
                    loss = loss + 0.1 * select_loss + 0.3 * base_harm_loss
                if selective_loss and "router_weights" in output and "x_fused" in output:
                    candidate_errors = torch.stack(
                        [
                            torch.abs(output["mu_data"].detach() - target),
                            torch.abs(output.get("x_error", output["x_generic"]).detach() - target),
                            torch.abs(output["x_phys"].detach() - target),
                            torch.abs(output["x_fused"].detach() - target),
                        ],
                        dim=3,
                    )
                    best_candidate = torch.argmin(candidate_errors, dim=3)
                    router_target = F.one_hot(best_candidate, num_classes=4).permute(0, 1, 2, 4, 3).float()
                    router_ce = -torch.sum(
                        router_target * torch.log(output["router_weights"].clamp_min(1e-6)),
                        dim=3,
                    )
                    router_loss = _masked_mean(router_ce, target_mask)
                    loss = loss + 0.1 * router_loss
                if physics_verified_loss and "verifier_gate" in output:
                    data_candidate = output.get("x_error", output["x_generic"])
                    data_err = torch.abs(data_candidate.detach() - target)
                    verify_target = torch.sigmoid((base_err - data_err) / 0.05).detach()
                    verify_loss = _masked_mean(
                        F.binary_cross_entropy(
                            output["verifier_gate"].clamp(1e-4, 1.0 - 1e-4),
                            verify_target,
                            reduction="none",
                        ),
                        target_mask,
                    )
                    base_harm_loss = _masked_mean(torch.relu(final_err - base_err), target_mask)
                    node_target_mask = target_mask.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
                    phys_verify_loss = _masked_mean(
                        torch.relu(output["residual_after_verified_abs"] - output["residual_before_abs"]),
                        node_target_mask,
                    )
                    loss = loss + 0.2 * verify_loss + 0.3 * base_harm_loss + 0.005 * phys_verify_loss
                if contrastive_verifier_loss and "verifier_gate" in output:
                    data_candidate = output.get("x_error", output["x_generic"])
                    data_err = torch.abs(data_candidate.detach() - target)
                    verify_target = (data_err < base_err).float().detach()
                    verify_loss = _masked_mean(
                        F.binary_cross_entropy(
                            output["verifier_gate"].clamp(1e-4, 1.0 - 1e-4),
                            verify_target,
                            reduction="none",
                        ),
                        target_mask,
                    )
                    base_harm_loss = _masked_mean(torch.relu(final_err - base_err), target_mask)
                    loss = loss + 0.75 * verify_loss + 0.25 * base_harm_loss
                if balanced_verifier_loss and "verifier_gate" in output:
                    data_candidate = output.get("x_error", output["x_generic"])
                    data_err = torch.abs(data_candidate.detach() - target)
                    verify_target = (data_err < base_err).float().detach()
                    train_gate = output.get("verifier_gate_raw", output["verifier_gate"])
                    verify_loss = _balanced_bce(train_gate, verify_target, target_mask)
                    harm = torch.relu(final_err - base_err)
                    hard_negative = (data_err > base_err + 0.02).float().detach()
                    hard_neg_loss = _masked_mean(
                        F.binary_cross_entropy(
                            output.get("verifier_gate_raw", output["verifier_gate"]).clamp(1e-4, 1.0 - 1e-4),
                            verify_target,
                            reduction="none",
                        ),
                        target_mask * hard_negative,
                    )
                    base_harm_loss = _masked_mean(harm, target_mask)
                    loss = loss + 1.0 * verify_loss + 0.5 * hard_neg_loss + 0.25 * base_harm_loss
                if hard_negative_verifier_loss and "verifier_gate" in output:
                    data_candidate = output.get("x_error", output["x_generic"])
                    data_err = torch.abs(data_candidate.detach() - target)
                    hard_negative = (data_err > base_err + float(hard_negative_margin)).float().detach()
                    verify_target = (1.0 - hard_negative).detach()
                    train_gate = output.get("verifier_gate_raw", output["verifier_gate"])
                    bce = F.binary_cross_entropy(
                        train_gate.clamp(1e-4, 1.0 - 1e-4),
                        verify_target,
                        reduction="none",
                    )
                    hard_neg_loss = _masked_mean(bce, target_mask * hard_negative)
                    safe_loss = _masked_mean(bce, target_mask * (1.0 - hard_negative))
                    base_harm_loss = _masked_mean(torch.relu(final_err - base_err), target_mask)
                    loss = loss + 1.0 * hard_neg_loss + 0.25 * safe_loss + 0.15 * base_harm_loss
                if physics_harm_verifier_loss and "harm_prob" in output:
                    data_candidate = output.get("x_error", output["x_generic"])
                    data_err = torch.abs(data_candidate.detach() - target)
                    hard_negative = (data_err > base_err + float(hard_negative_margin)).float().detach()
                    train_harm = output["harm_prob"].clamp(1e-4, 1.0 - 1e-4)
                    bce = F.binary_cross_entropy(train_harm, hard_negative, reduction="none")
                    hard_loss = _masked_mean(bce, target_mask * hard_negative)
                    safe_loss = _masked_mean(bce, target_mask * (1.0 - hard_negative))
                    final_harm = torch.relu(final_err - base_err)
                    if harm_utility_target and "harm_keep" in output:
                        delta = data_candidate.detach() - output["mu_data"].detach()
                        target_delta = target - output["mu_data"].detach()
                        same_direction = (delta * target_delta > 0).float()
                        keep_target = same_direction * torch.clamp(
                            torch.abs(target_delta) / torch.clamp(torch.abs(delta), min=1e-3),
                            min=0.0,
                            max=1.0,
                        )
                        keep_loss = _masked_mean(
                            F.smooth_l1_loss(output["harm_keep"], keep_target.detach(), reduction="none"),
                            target_mask,
                        )
                        loss = loss + 1.0 * keep_loss + 0.2 * _masked_mean(final_harm, target_mask)
                    else:
                        loss = (
                            loss
                            + float(harm_hard_weight) * hard_loss
                            + float(harm_safe_weight) * safe_loss
                            + 0.25 * _masked_mean(final_harm, target_mask)
                        )
                if harm_regularized_loss and "harm_prob" in output:
                    hard_negative = (final_err.detach() > base_err + float(hard_negative_margin)).float()
                    utility_target = torch.sigmoid(
                        (base_err - torch.abs(data_candidate.detach() - target)) / max(float(harm_utility_temperature), 1e-4)
                    ).detach()
                    valid_utility = utility_target[target_mask > 0.5]
                    if valid_utility.numel() >= 4:
                        hard_thr = torch.quantile(valid_utility, 0.25)
                        safe_thr = torch.quantile(valid_utility, 0.75)
                    else:
                        hard_thr = torch.tensor(0.35, dtype=utility_target.dtype, device=utility_target.device)
                        safe_thr = torch.tensor(0.65, dtype=utility_target.dtype, device=utility_target.device)
                    hard_conf = (utility_target <= hard_thr).float()
                    safe_conf = (utility_target >= safe_thr).float()
                    hard_region = (hard_conf + hard_negative).clamp(0.0, 1.0)
                    safe_region = safe_conf
                    ambig_region = (1.0 - hard_region - safe_region).clamp(0.0, 1.0)
                    hard_weight = torch.full_like(target_mask, 0.5) + 0.5 * hard_region
                    safe_weight = torch.full_like(target_mask, 0.5) + 0.5 * safe_region
                    ambig_weight = torch.full_like(target_mask, 0.2) + 0.3 * ambig_region
                    gate_loss = 0.0
                    general_loss = 0.0
                    sensor_loss = 0.0
                    if harm_utility_target:
                        harm_target = (utility_target < 0.5).float()
                        hard_region_scalar = hard_region.mean(dim=-1, keepdim=True)
                        harm_reg_loss = _masked_mean(
                            F.binary_cross_entropy(
                                output["harm_prob"].clamp(1e-4, 1.0 - 1.0e-4),
                                harm_target,
                                reduction="none",
                            ),
                            target_mask * (0.5 * hard_weight + 0.5 * safe_weight + ambig_weight),
                        )
                        gate_loss = _masked_mean(
                            F.binary_cross_entropy(
                                output["harm_region_gate"].clamp(1e-4, 1.0 - 1.0e-4),
                                hard_region_scalar,
                                reduction="none",
                            ),
                            target_mask.mean(dim=-1, keepdim=True),
                        ) if "harm_region_gate" in output else 0.0
                        general_loss = _masked_mean(
                            F.binary_cross_entropy(
                                output["harm_general_prob"].clamp(1e-4, 1.0 - 1.0e-4),
                                harm_target,
                                reduction="none",
                            ),
                            target_mask * safe_weight,
                        ) if "harm_general_prob" in output else 0.0
                        sensor_loss = _masked_mean(
                            F.binary_cross_entropy(
                                output["harm_sensor_prob"].clamp(1e-4, 1.0 - 1.0e-4),
                                harm_target,
                                reduction="none",
                            ),
                            target_mask * hard_weight,
                        ) if "harm_sensor_prob" in output else 0.0
                        if "harm_keep" in output:
                            keep_target = utility_target
                            keep_hard_loss = _masked_mean(
                                F.smooth_l1_loss(output["harm_keep"], keep_target, reduction="none"),
                                target_mask * hard_weight,
                            )
                            keep_safe_loss = _masked_mean(
                                F.smooth_l1_loss(output["harm_keep"], keep_target, reduction="none"),
                                target_mask * safe_weight,
                            )
                            keep_ambig_loss = _masked_mean(
                                F.smooth_l1_loss(output["harm_keep"], keep_target, reduction="none"),
                                target_mask * ambig_weight,
                            )
                            keep_loss = 0.45 * keep_hard_loss + 0.45 * keep_safe_loss + 0.10 * keep_ambig_loss
                            keep_hard_mean = _masked_mean(output["harm_keep"], target_mask * hard_region)
                            keep_safe_mean = _masked_mean(output["harm_keep"], target_mask * safe_region)
                            keep_margin = torch.relu(keep_hard_mean - keep_safe_mean + 0.05)
                            harm_hard_mean = _masked_mean(output["harm_prob"], target_mask * hard_conf)
                            harm_safe_mean = _masked_mean(output["harm_prob"], target_mask * safe_conf)
                            harm_margin = torch.relu(harm_safe_mean - harm_hard_mean + 0.05)
                        else:
                            keep_loss = 0.0
                            keep_margin = 0.0
                            harm_margin = 0.0
                    else:
                        harm_reg_loss = _masked_mean(
                            F.binary_cross_entropy(
                                output["harm_prob"].clamp(1e-4, 1.0 - 1e-4),
                                hard_negative,
                                reduction="none",
                            ),
                            target_mask * (0.5 * hard_weight + 0.5 * safe_weight + ambig_weight),
                        )
                        keep_loss = 0.0
                        keep_margin = 0.0
                        harm_margin = 0.0
                    base_harm_loss = _masked_mean(
                        torch.relu(final_err - torch.minimum(base_err, generic_err) - float(hard_negative_margin)),
                        target_mask * (0.6 + 0.9 * hard_negative),
                    )
                    correction_mag = torch.abs(pred - output["mu_data"].detach())
                    harm_weighted_mag = _masked_mean(
                        output["harm_prob"].detach() * correction_mag,
                        target_mask * hard_negative,
                    )
                    loss = loss + 0.25 * harm_reg_loss + 0.45 * gate_loss + 0.45 * general_loss + 0.45 * sensor_loss + 0.35 * keep_loss + 0.25 * keep_margin + 0.25 * harm_margin + 0.45 * base_harm_loss + 0.03 * harm_weighted_mag
                if harm_suppressed_loss and "correction_allowance" in output and "harm_prob" in output:
                    data_candidate = output.get("x_error", output["x_generic"])
                    data_err = torch.abs(data_candidate.detach() - target)
                    utility_target = torch.sigmoid(
                        (base_err - data_err) / max(float(harm_utility_temperature), 1e-4)
                    ).detach()
                    hard_negative = (data_err > base_err + float(hard_negative_margin)).float().detach()
                    valid_utility = utility_target[target_mask > 0.5]
                    if valid_utility.numel() >= 4:
                        hard_thr = torch.quantile(valid_utility, 0.25)
                        safe_thr = torch.quantile(valid_utility, 0.75)
                    else:
                        hard_thr = torch.tensor(0.35, dtype=utility_target.dtype, device=utility_target.device)
                        safe_thr = torch.tensor(0.65, dtype=utility_target.dtype, device=utility_target.device)
                    hard_region = ((utility_target <= hard_thr).float() + hard_negative).clamp(0.0, 1.0)
                    safe_region = (utility_target >= safe_thr).float()
                    sample_weight = 0.4 + 0.8 * hard_region + 0.5 * safe_region
                    harm_target = (utility_target < 0.5).float()
                    harm_loss = _masked_mean(
                        F.binary_cross_entropy(
                            output["harm_prob"].clamp(1e-4, 1.0 - 1e-4),
                            harm_target,
                            reduction="none",
                        ),
                        target_mask * sample_weight,
                    )
                    allowance_target = utility_target
                    allowance_for_loss = output.get("correction_allowance_raw", output["correction_allowance"])
                    allowance_loss = _masked_mean(
                        F.smooth_l1_loss(
                            allowance_for_loss,
                            allowance_target,
                            reduction="none",
                        ),
                        target_mask * sample_weight,
                    )
                    allowance_hard_mean = _masked_mean(allowance_for_loss, target_mask * hard_region)
                    allowance_safe_mean = _masked_mean(allowance_for_loss, target_mask * safe_region)
                    allowance_margin = torch.relu(allowance_hard_mean - allowance_safe_mean + 0.05)
                    final_harm_loss = _masked_mean(
                        torch.relu(final_err - base_err - float(hard_negative_margin)),
                        target_mask * (0.5 + hard_region),
                    )
                    suppressed_delta = torch.abs(pred - output["mu_data"].detach())
                    harm_weighted_mag = _masked_mean(
                        output["harm_prob"].detach() * suppressed_delta,
                        target_mask * (0.5 + hard_region),
                    )
                    loss = (
                        loss
                        + 0.35 * harm_loss
                        + 0.55 * allowance_loss
                        + 0.35 * allowance_margin
                        + 0.50 * final_harm_loss
                        + 0.04 * harm_weighted_mag
                    )
                if promotion_loss and "physics_promotion_score" in output:
                    phys_err = torch.abs(output["x_phys"].detach() - target)
                    fused_err = torch.abs(output["x_fused"].detach() - target)
                    promotion_score = output["physics_promotion_score"].clamp(1e-4, 1.0 - 1.0e-4)
                    gap = fused_err - phys_err
                    valid_gap = gap[target_mask > 0.5]
                    if valid_gap.numel() >= 4:
                        hard_thr = torch.quantile(valid_gap, 0.70)
                        safe_thr = torch.quantile(valid_gap, 0.30)
                    else:
                        hard_thr = torch.tensor(float(hard_negative_margin), dtype=gap.dtype, device=gap.device)
                        safe_thr = torch.tensor(-float(hard_negative_margin), dtype=gap.dtype, device=gap.device)
                    hard_region = (gap >= hard_thr).float()
                    safe_region = (gap <= safe_thr).float()
                    ambig_region = (1.0 - hard_region - safe_region).clamp(0.0, 1.0)
                    hard_weight = 0.5 + 0.5 * hard_region
                    safe_weight = 0.5 + 0.5 * safe_region
                    ambig_weight = 0.15 + 0.25 * ambig_region
                    hard_loss = _masked_mean(
                        F.binary_cross_entropy(
                            promotion_score,
                            torch.ones_like(promotion_score),
                            reduction="none",
                        ),
                        target_mask * hard_weight,
                    )
                    safe_loss = _masked_mean(
                        F.binary_cross_entropy(
                            promotion_score,
                            torch.zeros_like(promotion_score),
                            reduction="none",
                        ),
                        target_mask * safe_weight,
                    )
                    ambig_loss = _masked_mean(
                        F.smooth_l1_loss(
                            promotion_score,
                            0.5 * torch.ones_like(promotion_score),
                            reduction="none",
                        ),
                        target_mask * ambig_weight,
                    )
                    promotion_hard_mean = _masked_mean(promotion_score, target_mask * hard_region)
                    promotion_safe_mean = _masked_mean(promotion_score, target_mask * safe_region)
                    promotion_margin = torch.relu(promotion_safe_mean - promotion_hard_mean + 0.1)
                    best_err = torch.minimum(phys_err, fused_err)
                    promotion_harm = _masked_mean(torch.relu(final_err - best_err - float(hard_negative_margin)), target_mask)
                    loss = loss + 0.55 * hard_loss + 0.55 * safe_loss + 0.20 * ambig_loss + 0.35 * promotion_margin + 0.40 * promotion_harm
                if discrete_promotion_loss and "physics_promotion_mode_logits" in output:
                    fused_err = torch.abs(output["x_fused"].detach() - target)
                    phys_err = torch.abs(output["x_phys"].detach() - target)
                    generic_err = torch.abs(output["x_generic"].detach() - target)
                    candidate_errors = torch.stack([fused_err, phys_err, generic_err], dim=3).permute(0, 1, 2, 4, 3)
                    fused_mode_err = candidate_errors[..., 0]
                    phys_mode_err = candidate_errors[..., 1]
                    generic_mode_err = candidate_errors[..., 2]
                    phys_margin = _promotion_margin_for_scenario(scenario, float(hard_negative_margin))
                    generic_margin = 0.75 * phys_margin
                    phys_gain = fused_mode_err - phys_mode_err
                    generic_gain = fused_mode_err - generic_mode_err
                    phys_clear = phys_gain > phys_margin
                    generic_clear = generic_gain > generic_margin
                    phys_target = (phys_clear & (phys_gain >= generic_gain)).float()
                    generic_target = (generic_clear & (generic_gain > phys_gain)).float()
                    fused_target = (1.0 - phys_target - generic_target).clamp(0.0, 1.0)
                    mode_target = torch.zeros_like(candidate_errors)
                    mode_target[..., 0] = fused_target
                    mode_target[..., 1] = phys_target
                    mode_target[..., 2] = generic_target
                    best_idx = torch.argmax(mode_target, dim=-1)
                    top2 = torch.topk(candidate_errors, 2, dim=-1, largest=False).values
                    gap = top2[..., 1] - top2[..., 0]
                    valid_gap = gap[target_mask > 0.5]
                    if valid_gap.numel() >= 4:
                        hard_thr = torch.quantile(valid_gap, 0.70)
                        safe_thr = torch.quantile(valid_gap, 0.30)
                    else:
                        hard_thr = torch.tensor(float(hard_negative_margin), dtype=gap.dtype, device=gap.device)
                        safe_thr = torch.tensor(-float(hard_negative_margin), dtype=gap.dtype, device=gap.device)
                    hard_region = (gap >= hard_thr).float()
                    safe_region = (gap <= safe_thr).float()
                    ambig_region = (1.0 - hard_region - safe_region).clamp(0.0, 1.0)
                    mode_logits = output["physics_promotion_mode_logits"].permute(0, 1, 2, 4, 3)
                    mode_log_probs = F.log_softmax(mode_logits, dim=-1)
                    mode_ce = -(mode_target * mode_log_probs).sum(dim=-1)
                    class_weights = _balanced_class_weights(best_idx, target_mask, num_classes=3)
                    class_weight_map = torch.sum(mode_target * class_weights.view(1, 1, 1, 1, 3), dim=-1)
                    specialist_mass = mode_target[..., 1] + mode_target[..., 2]
                    failure_weight = 1.0 + 1.0 * specialist_mass + 0.35 * hard_region + 0.15 * ambig_region
                    mode_weight = class_weight_map * failure_weight
                    mode_loss = _masked_mean(mode_ce, target_mask * mode_weight)
                    mode_probs = output["physics_promotion_mode_probs"]
                    target_alignment = torch.sum(mode_probs.permute(0, 1, 2, 4, 3) * mode_target, dim=-1)
                    selected_mean = _masked_mean(target_alignment, target_mask)
                    margin_loss = _masked_mean(torch.relu(0.7 - target_alignment), target_mask * (hard_region + specialist_mass).clamp(0.0, 1.0))
                    best_err = torch.minimum(fused_err, torch.minimum(phys_err, generic_err))
                    promotion_harm = _masked_mean(torch.relu(final_err - best_err - float(hard_negative_margin)), target_mask)
                    loss = loss + 0.9 * mode_loss + 0.15 * margin_loss + 0.35 * promotion_harm - 0.05 * selected_mean
            if isinstance(output, tuple):
                for aux in output[1:]:
                    if aux.ndim == pred.ndim + 1:
                        aux = aux.mean(dim=0)
                    loss = loss + 0.1 * masked_mae_loss(aux, target, target_mask)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        masks.append(target_mask.detach().cpu().numpy())
        if diagnostics and isinstance(output, dict):
            candidates = []
            for key in ["mu_data", "x_generic", "x_generic_v2", "x_generic_v3", "x_generic_v4", "x_vetted", "x_phys", "x_fused", "x_physics_promoted", "x_discrete_physics_promoted", "x_router", "x_verified", "x_harm_verified", "x_harm_suppressed"]:
                value = output.get(key, pred)
                candidates.append(value.detach())
                detail_preds[key].append(value.detach().cpu().numpy())
            stacked = torch.stack(candidates, dim=0)
            errors = torch.abs(stacked - target.unsqueeze(0))
            best_index = torch.argmin(errors, dim=0, keepdim=True)
            oracle = torch.gather(stacked, 0, best_index).squeeze(0)
            detail_preds["oracle_best"].append(oracle.detach().cpu().numpy())
            base_err = torch.abs(output["mu_data"].detach() - target)
            final_err = torch.abs(pred.detach() - target)
            harm = (final_err > base_err).float()
            detail_stats["harm_rate_sum"] += float(torch.sum(harm * target_mask).detach().cpu())
            detail_stats["harm_rate_count"] += float(torch.clamp(target_mask.sum(), min=1.0).detach().cpu())
            if "phys_weight" in output:
                detail_stats["phys_weight_sum"] += float(torch.sum(output["phys_weight"].detach() * target_mask).cpu())
                detail_stats["phys_weight_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
            if "correction_conf" in output:
                detail_stats["correction_conf_sum"] += float(torch.sum(output["correction_conf"].detach() * target_mask).cpu())
                detail_stats["correction_conf_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
            if "router_weights" in output:
                for idx in range(4):
                    detail_stats["router_weight_sum"][idx] += float(
                        torch.sum(output["router_weights"][:, :, :, idx, :].detach() * target_mask).cpu()
                    )
                detail_stats["router_weight_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
            if "verifier_gate" in output:
                detail_stats["verifier_gate_sum"] += float(torch.sum(output["verifier_gate"].detach() * target_mask).cpu())
                detail_stats["verifier_gate_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                generic_source = output.get("x_error", output["x_generic"])
                generic_err_for_stats = torch.abs(generic_source.detach() - target)
                generic_better = (generic_err_for_stats < base_err).float()
                hard_negative = (generic_err_for_stats > base_err + 0.02).float()
                pos_mask = target_mask * generic_better
                neg_mask = target_mask * (1.0 - generic_better)
                hard_neg_mask = target_mask * hard_negative
                safe_mask = target_mask * (1.0 - hard_negative)
                detail_stats["verifier_gate_pos_sum"] += float(torch.sum(output["verifier_gate"].detach() * pos_mask).cpu())
                detail_stats["verifier_gate_pos_count"] += float(torch.clamp(pos_mask.sum(), min=1.0).cpu())
                detail_stats["verifier_gate_neg_sum"] += float(torch.sum(output["verifier_gate"].detach() * neg_mask).cpu())
                detail_stats["verifier_gate_neg_count"] += float(torch.clamp(neg_mask.sum(), min=1.0).cpu())
                detail_stats["generic_better_sum"] += float(torch.sum(generic_better * target_mask).cpu())
                detail_stats["generic_better_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                detail_stats["hard_negative_sum"] += float(torch.sum(hard_negative * target_mask).cpu())
                detail_stats["hard_negative_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                detail_stats["verifier_gate_hard_neg_sum"] += float(torch.sum(output["verifier_gate"].detach() * hard_neg_mask).cpu())
                detail_stats["verifier_gate_hard_neg_count"] += float(torch.clamp(hard_neg_mask.sum(), min=1.0).cpu())
                detail_stats["verifier_gate_safe_sum"] += float(torch.sum(output["verifier_gate"].detach() * safe_mask).cpu())
                detail_stats["verifier_gate_safe_count"] += float(torch.clamp(safe_mask.sum(), min=1.0).cpu())
                if "harm_prob" in output:
                    harm_prob = output["harm_prob"].detach()
                    harm_keep = output.get("harm_keep", 1.0 - output["harm_prob"]).detach()
                    pred_harm = (harm_prob > 0.5).float()
                    detail_stats["harm_prob_sum"] += float(torch.sum(harm_prob * target_mask).cpu())
                    detail_stats["harm_prob_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                    detail_stats["harm_prob_hard_neg_sum"] += float(torch.sum(harm_prob * hard_neg_mask).cpu())
                    detail_stats["harm_prob_hard_neg_count"] += float(torch.clamp(hard_neg_mask.sum(), min=1.0).cpu())
                    detail_stats["harm_prob_safe_sum"] += float(torch.sum(harm_prob * safe_mask).cpu())
                    detail_stats["harm_prob_safe_count"] += float(torch.clamp(safe_mask.sum(), min=1.0).cpu())
                    detail_stats["harm_keep_sum"] += float(torch.sum(harm_keep * target_mask).cpu())
                    detail_stats["harm_keep_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                    detail_stats["harm_keep_hard_neg_sum"] += float(torch.sum(harm_keep * hard_neg_mask).cpu())
                    detail_stats["harm_keep_hard_neg_count"] += float(torch.clamp(hard_neg_mask.sum(), min=1.0).cpu())
                    detail_stats["harm_keep_safe_sum"] += float(torch.sum(harm_keep * safe_mask).cpu())
                    detail_stats["harm_keep_safe_count"] += float(torch.clamp(safe_mask.sum(), min=1.0).cpu())
                    if "harm_region_gate" in output:
                        detail_stats["harm_region_gate_sum"] += float(torch.sum(output["harm_region_gate"].detach() * target_mask).cpu())
                        detail_stats["harm_region_gate_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                    if "harm_general_prob" in output:
                        detail_stats["harm_general_prob_sum"] += float(torch.sum(output["harm_general_prob"].detach() * target_mask).cpu())
                        detail_stats["harm_general_prob_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                    if "harm_sensor_prob" in output:
                        detail_stats["harm_sensor_prob_sum"] += float(torch.sum(output["harm_sensor_prob"].detach() * target_mask).cpu())
                        detail_stats["harm_sensor_prob_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                    if "correction_allowance" in output:
                        allowance = output["correction_allowance"].detach()
                        detail_stats["correction_allowance_sum"] += float(torch.sum(allowance * target_mask).cpu())
                        detail_stats["correction_allowance_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                        detail_stats["correction_allowance_hard_neg_sum"] += float(torch.sum(allowance * hard_neg_mask).cpu())
                        detail_stats["correction_allowance_hard_neg_count"] += float(torch.clamp(hard_neg_mask.sum(), min=1.0).cpu())
                        detail_stats["correction_allowance_safe_sum"] += float(torch.sum(allowance * safe_mask).cpu())
                        detail_stats["correction_allowance_safe_count"] += float(torch.clamp(safe_mask.sum(), min=1.0).cpu())
                    if "physics_promotion_score" in output and "x_phys" in output and "x_fused" in output:
                        promotion = output["physics_promotion_score"].detach()
                        phys_err = torch.abs(output["x_phys"].detach() - target)
                        fused_err = torch.abs(output["x_fused"].detach() - target)
                        phys_better = (phys_err < fused_err).float()
                        fused_better = 1.0 - phys_better
                        detail_stats["physics_promotion_sum"] += float(torch.sum(promotion * target_mask).cpu())
                        detail_stats["physics_promotion_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                        detail_stats["physics_promotion_phys_better_sum"] += float(torch.sum(promotion * target_mask * phys_better).cpu())
                        detail_stats["physics_promotion_phys_better_count"] += float(torch.clamp((target_mask * phys_better).sum(), min=1.0).cpu())
                        detail_stats["physics_promotion_fused_better_sum"] += float(torch.sum(promotion * target_mask * fused_better).cpu())
                        detail_stats["physics_promotion_fused_better_count"] += float(torch.clamp((target_mask * fused_better).sum(), min=1.0).cpu())
                    if "physics_promotion_mode_probs" in output:
                        mode_probs = output["physics_promotion_mode_probs"].detach()
                        detail_stats["physics_promotion_mode_mean_sum"][0] += float(torch.sum(mode_probs[:, :, :, 0, :] * target_mask).cpu())
                        detail_stats["physics_promotion_mode_mean_sum"][1] += float(torch.sum(mode_probs[:, :, :, 1, :] * target_mask).cpu())
                        detail_stats["physics_promotion_mode_mean_sum"][2] += float(torch.sum(mode_probs[:, :, :, 2, :] * target_mask).cpu())
                        detail_stats["physics_promotion_mode_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                        if "x_fused" in output and "x_phys" in output and "x_generic" in output:
                            fused_err = torch.abs(output["x_fused"].detach() - target)
                            phys_err = torch.abs(output["x_phys"].detach() - target)
                            generic_err = torch.abs(output["x_generic"].detach() - target)
                            candidate_errors = torch.stack([fused_err, phys_err, generic_err], dim=3).permute(0, 1, 2, 4, 3)
                            fused_mode_err = candidate_errors[..., 0]
                            phys_mode_err = candidate_errors[..., 1]
                            generic_mode_err = candidate_errors[..., 2]
                            phys_margin = _promotion_margin_for_scenario(scenario, float(hard_negative_margin))
                            generic_margin = 0.75 * phys_margin
                            phys_gain = fused_mode_err - phys_mode_err
                            generic_gain = fused_mode_err - generic_mode_err
                            phys_clear = phys_gain > phys_margin
                            generic_clear = generic_gain > generic_margin
                            phys_target = (phys_clear & (phys_gain >= generic_gain)).float()
                            generic_target = (generic_clear & (generic_gain > phys_gain)).float()
                            fused_target = (1.0 - phys_target - generic_target).clamp(0.0, 1.0)
                            mode_target = torch.zeros_like(candidate_errors)
                            mode_target[..., 0] = fused_target
                            mode_target[..., 1] = phys_target
                            mode_target[..., 2] = generic_target
                            detail_stats["physics_promotion_mode_target_sum"][0] += float(
                                torch.sum(mode_target[..., 0] * target_mask).cpu()
                            )
                            detail_stats["physics_promotion_mode_target_sum"][1] += float(
                                torch.sum(mode_target[..., 1] * target_mask).cpu()
                            )
                            detail_stats["physics_promotion_mode_target_sum"][2] += float(
                                torch.sum(mode_target[..., 2] * target_mask).cpu()
                            )
                            detail_stats["physics_promotion_mode_target_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                            detail_stats["physics_promotion_mode_clear_sum"] += float(torch.sum(phys_target * target_mask).cpu())
                            detail_stats["physics_promotion_mode_clear_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
                    detail_stats["harm_tp_sum"] += float(torch.sum(pred_harm * hard_negative * target_mask).cpu())
                    detail_stats["harm_pred_sum"] += float(torch.sum(pred_harm * target_mask).cpu())
                    detail_stats["harm_actual_sum"] += float(torch.sum(hard_negative * target_mask).cpu())
            if "residual_before_abs" in output:
                node_target_mask = target_mask.mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
                detail_stats["residual_before_sum"] += float(torch.sum(output["residual_before_abs"].detach() * node_target_mask).cpu())
                detail_stats["residual_after_data_sum"] += float(torch.sum(output["residual_after_data_abs"].detach() * node_target_mask).cpu())
                detail_stats["residual_after_verified_sum"] += float(torch.sum(output["residual_after_verified_abs"].detach() * node_target_mask).cpu())
                detail_stats["residual_count"] += float(torch.clamp(node_target_mask.sum(), min=1.0).cpu())
            if "generic_v2_strength" in output:
                detail_stats["generic_v2_strength_sum"] += float(torch.sum(output["generic_v2_strength"].detach() * target_mask).cpu())
                detail_stats["generic_v2_strength_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
            if "generic_v2_weights" in output:
                for idx in range(3):
                    detail_stats["generic_v2_weight_sum"][idx] += float(
                        torch.sum(output["generic_v2_weights"][:, :, :, idx, :].detach() * target_mask).cpu()
                    )
                detail_stats["generic_v2_weight_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
            if "generic_v3_gain" in output:
                detail_stats["generic_v3_gain_sum"] += float(torch.sum(output["generic_v3_gain"].detach() * target_mask).cpu())
                detail_stats["generic_v3_gain_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
            if "generic_v3_refine" in output:
                detail_stats["generic_v3_refine_abs_sum"] += float(torch.sum(output["generic_v3_refine"].detach().abs() * target_mask).cpu())
                detail_stats["generic_v3_refine_abs_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
            if "generic_v4_scale" in output:
                detail_stats["generic_v4_scale_sum"] += float(torch.sum(output["generic_v4_scale"].detach() * target_mask).cpu())
                detail_stats["generic_v4_scale_count"] += float(torch.clamp(target_mask.sum(), min=1.0).cpu())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    mask_np = np.concatenate(masks, axis=0)
    metrics = compute_metrics(pred_np, target_np, mask_np)
    metrics["loss"] = float(np.mean(losses))
    if diagnostics and detail_preds["mu_data"]:
        for key, values in detail_preds.items():
            candidate_np = np.concatenate(values, axis=0)
            candidate_metrics = compute_metrics(candidate_np, target_np, mask_np)
            metrics[f"{key}_masked_mae"] = candidate_metrics["masked_mae"]
        if detail_stats["phys_weight_count"] > 0:
            metrics["phys_weight_mean"] = detail_stats["phys_weight_sum"] / detail_stats["phys_weight_count"]
        if detail_stats["correction_conf_count"] > 0:
            metrics["correction_conf_mean"] = detail_stats["correction_conf_sum"] / detail_stats["correction_conf_count"]
        if detail_stats["harm_rate_count"] > 0:
            metrics["harm_rate_vs_grin"] = detail_stats["harm_rate_sum"] / detail_stats["harm_rate_count"]
        if detail_stats["router_weight_count"] > 0:
            for name, value in zip(["grin", "generic", "physics", "fused"], detail_stats["router_weight_sum"]):
                metrics[f"router_weight_{name}_mean"] = value / detail_stats["router_weight_count"]
        if detail_stats["verifier_gate_count"] > 0:
            metrics["verifier_gate_mean"] = detail_stats["verifier_gate_sum"] / detail_stats["verifier_gate_count"]
            metrics["verifier_gate_pos_mean"] = detail_stats["verifier_gate_pos_sum"] / detail_stats["verifier_gate_pos_count"]
            metrics["verifier_gate_neg_mean"] = detail_stats["verifier_gate_neg_sum"] / detail_stats["verifier_gate_neg_count"]
            metrics["generic_better_ratio"] = detail_stats["generic_better_sum"] / detail_stats["generic_better_count"]
            metrics["hard_negative_ratio"] = detail_stats["hard_negative_sum"] / detail_stats["hard_negative_count"]
            metrics["verifier_gate_hard_negative_mean"] = detail_stats["verifier_gate_hard_neg_sum"] / detail_stats["verifier_gate_hard_neg_count"]
            metrics["verifier_gate_safe_mean"] = detail_stats["verifier_gate_safe_sum"] / detail_stats["verifier_gate_safe_count"]
        if detail_stats["harm_prob_count"] > 0:
            metrics["harm_prob_mean"] = detail_stats["harm_prob_sum"] / detail_stats["harm_prob_count"]
            metrics["harm_prob_hard_negative_mean"] = detail_stats["harm_prob_hard_neg_sum"] / detail_stats["harm_prob_hard_neg_count"]
            metrics["harm_prob_safe_mean"] = detail_stats["harm_prob_safe_sum"] / detail_stats["harm_prob_safe_count"]
            metrics["harm_keep_mean"] = detail_stats["harm_keep_sum"] / detail_stats["harm_keep_count"]
            metrics["harm_keep_hard_negative_mean"] = detail_stats["harm_keep_hard_neg_sum"] / detail_stats["harm_keep_hard_neg_count"]
            metrics["harm_keep_safe_mean"] = detail_stats["harm_keep_safe_sum"] / detail_stats["harm_keep_safe_count"]
            metrics["harm_precision_at_05"] = detail_stats["harm_tp_sum"] / max(detail_stats["harm_pred_sum"], 1.0)
            metrics["harm_recall_at_05"] = detail_stats["harm_tp_sum"] / max(detail_stats["harm_actual_sum"], 1.0)
        if detail_stats["harm_region_gate_count"] > 0:
            metrics["harm_region_gate_mean"] = detail_stats["harm_region_gate_sum"] / detail_stats["harm_region_gate_count"]
        if detail_stats["harm_general_prob_count"] > 0:
            metrics["harm_general_prob_mean"] = detail_stats["harm_general_prob_sum"] / detail_stats["harm_general_prob_count"]
        if detail_stats["harm_sensor_prob_count"] > 0:
            metrics["harm_sensor_prob_mean"] = detail_stats["harm_sensor_prob_sum"] / detail_stats["harm_sensor_prob_count"]
        if detail_stats["correction_allowance_count"] > 0:
            metrics["correction_allowance_mean"] = detail_stats["correction_allowance_sum"] / detail_stats["correction_allowance_count"]
            metrics["correction_allowance_hard_negative_mean"] = detail_stats["correction_allowance_hard_neg_sum"] / detail_stats["correction_allowance_hard_neg_count"]
            metrics["correction_allowance_safe_mean"] = detail_stats["correction_allowance_safe_sum"] / detail_stats["correction_allowance_safe_count"]
        if detail_stats["physics_promotion_count"] > 0:
            metrics["physics_promotion_mean"] = detail_stats["physics_promotion_sum"] / detail_stats["physics_promotion_count"]
            metrics["physics_promotion_phys_better_mean"] = detail_stats["physics_promotion_phys_better_sum"] / detail_stats["physics_promotion_phys_better_count"]
            metrics["physics_promotion_fused_better_mean"] = detail_stats["physics_promotion_fused_better_sum"] / detail_stats["physics_promotion_fused_better_count"]
        if detail_stats["physics_promotion_mode_count"] > 0:
            metrics["physics_promotion_mode_fused_mean"] = detail_stats["physics_promotion_mode_mean_sum"][0] / detail_stats["physics_promotion_mode_count"]
            metrics["physics_promotion_mode_phys_mean"] = detail_stats["physics_promotion_mode_mean_sum"][1] / detail_stats["physics_promotion_mode_count"]
            metrics["physics_promotion_mode_generic_mean"] = detail_stats["physics_promotion_mode_mean_sum"][2] / detail_stats["physics_promotion_mode_count"]
        if detail_stats["physics_promotion_mode_target_count"] > 0:
            metrics["physics_promotion_mode_target_fused_mean"] = detail_stats["physics_promotion_mode_target_sum"][0] / detail_stats["physics_promotion_mode_target_count"]
            metrics["physics_promotion_mode_target_phys_mean"] = detail_stats["physics_promotion_mode_target_sum"][1] / detail_stats["physics_promotion_mode_target_count"]
            metrics["physics_promotion_mode_target_generic_mean"] = detail_stats["physics_promotion_mode_target_sum"][2] / detail_stats["physics_promotion_mode_target_count"]
        if detail_stats["physics_promotion_mode_clear_count"] > 0:
            metrics["physics_promotion_mode_physics_clear_fraction"] = detail_stats["physics_promotion_mode_clear_sum"] / detail_stats["physics_promotion_mode_clear_count"]
        if detail_stats["residual_count"] > 0:
            metrics["residual_before_mean"] = detail_stats["residual_before_sum"] / detail_stats["residual_count"]
            metrics["residual_after_data_mean"] = detail_stats["residual_after_data_sum"] / detail_stats["residual_count"]
            metrics["residual_after_verified_mean"] = detail_stats["residual_after_verified_sum"] / detail_stats["residual_count"]
        if detail_stats["generic_v2_strength_count"] > 0:
            metrics["generic_v2_strength_mean"] = detail_stats["generic_v2_strength_sum"] / detail_stats["generic_v2_strength_count"]
        if detail_stats["generic_v2_weight_count"] > 0:
            for name, value in zip(["local", "graph", "temporal"], detail_stats["generic_v2_weight_sum"]):
                metrics[f"generic_v2_weight_{name}_mean"] = value / detail_stats["generic_v2_weight_count"]
        if detail_stats["generic_v3_gain_count"] > 0:
            metrics["generic_v3_gain_mean"] = detail_stats["generic_v3_gain_sum"] / detail_stats["generic_v3_gain_count"]
        if detail_stats["generic_v3_refine_abs_count"] > 0:
            metrics["generic_v3_refine_abs_mean"] = detail_stats["generic_v3_refine_abs_sum"] / detail_stats["generic_v3_refine_abs_count"]
        if detail_stats["generic_v4_scale_count"] > 0:
            metrics["generic_v4_scale_mean"] = detail_stats["generic_v4_scale_sum"] / detail_stats["generic_v4_scale_count"]
    return metrics


BLEND_CANDIDATES = [
    "mu_data",
    "x_generic",
    "x_generic_v3",
    "x_generic_v4",
    "x_verified",
    "x_harm_suppressed",
    "x_router",
    "x_fused",
]


def _collect_candidate_arrays(model, loader, device, scenario: str | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    model.eval()
    preds: dict[str, list[np.ndarray]] = {key: [] for key in BLEND_CANDIDATES}
    targets = []
    masks = []
    scenario_token = _scenario_tensor(scenario, device)
    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"].to(device)
            target = batch["x_full"].to(device)
            obs_mask = batch["mask"].to(device) > 0.5
            target_mask = batch["target_mask"].to(device)
            output = _model_forward(model, x_obs, obs_mask, return_details=True, scenario_token=scenario_token)
            if not isinstance(output, dict):
                continue
            for key in BLEND_CANDIDATES:
                value = output.get(key)
                if value is not None:
                    preds[key].append(value.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            masks.append(target_mask.detach().cpu().numpy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in preds.items() if values}
    return arrays, np.concatenate(targets, axis=0), np.concatenate(masks, axis=0)


def _masked_mae_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    denom = max(float(mask.sum()), 1.0)
    return float(np.sum(np.abs(pred - target) * mask) / denom)


def _fit_validation_blend(
    val_arrays: dict[str, np.ndarray],
    val_target: np.ndarray,
    val_mask: np.ndarray,
    test_arrays: dict[str, np.ndarray],
    test_target: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, float | str]:
    keys = [key for key in BLEND_CANDIDATES if key in val_arrays and key in test_arrays]
    if not keys:
        return {}
    val_stack = np.stack([val_arrays[key] for key in keys], axis=0)
    test_stack = np.stack([test_arrays[key] for key in keys], axis=0)
    rng = np.random.default_rng(1)
    candidates = []
    for idx in range(len(keys)):
        weight = np.zeros(len(keys), dtype=np.float32)
        weight[idx] = 1.0
        candidates.append(weight)
    if "mu_data" in keys and "x_generic_v3" in keys:
        weight = np.zeros(len(keys), dtype=np.float32)
        weight[keys.index("mu_data")] = 0.15
        weight[keys.index("x_generic_v3")] = 0.85
        candidates.append(weight)
    if "x_generic" in keys and "x_generic_v3" in keys:
        for alpha in np.linspace(0.1, 0.9, 9):
            weight = np.zeros(len(keys), dtype=np.float32)
            weight[keys.index("x_generic")] = float(1.0 - alpha)
            weight[keys.index("x_generic_v3")] = float(alpha)
            candidates.append(weight)
    for _ in range(2048):
        candidates.append(rng.dirichlet(np.ones(len(keys))).astype(np.float32))
    best_weight = None
    best_val = float("inf")
    for weight in candidates:
        pred = np.tensordot(weight, val_stack, axes=(0, 0))
        value = _masked_mae_np(pred, val_target, val_mask)
        if value < best_val:
            best_val = value
            best_weight = weight
    assert best_weight is not None
    test_pred = np.tensordot(best_weight, test_stack, axes=(0, 0))
    test_metrics = compute_metrics(test_pred, test_target, test_mask)
    result: dict[str, float | str] = {
        "val_blend_masked_mae": best_val,
        "val_blend_test_masked_mae": float(test_metrics["masked_mae"]),
        "val_blend_test_mae": float(test_metrics["mae"]),
        "val_blend_test_rmse": float(test_metrics["rmse"]),
        "val_blend_test_mape": float(test_metrics["mape"]),
        "val_blend_candidates": ",".join(keys),
    }
    for key, weight in zip(keys, best_weight):
        result[f"val_blend_weight_{key}"] = float(weight)
    return result


def _fit_validation_selection(
    val_arrays: dict[str, np.ndarray],
    val_target: np.ndarray,
    val_mask: np.ndarray,
    test_arrays: dict[str, np.ndarray],
    test_target: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, float | str]:
    keys = [key for key in BLEND_CANDIDATES if key in val_arrays and key in test_arrays]
    if not keys:
        return {}
    val_scores = {key: _masked_mae_np(val_arrays[key], val_target, val_mask) for key in keys}
    best_key = min(val_scores, key=val_scores.get)
    test_metrics = compute_metrics(test_arrays[best_key], test_target, test_mask)
    result: dict[str, float | str] = {
        "val_selected_candidate": best_key,
        "val_selected_candidate_val_masked_mae": float(val_scores[best_key]),
        "val_selected_candidate_test_masked_mae": float(test_metrics["masked_mae"]),
        "val_selected_candidate_test_mae": float(test_metrics["mae"]),
        "val_selected_candidate_test_rmse": float(test_metrics["rmse"]),
        "val_selected_candidate_test_mape": float(test_metrics["mape"]),
    }
    for key, value in val_scores.items():
        result[f"val_candidate_{key}_masked_mae"] = float(value)
    return result


def _run_one(
    model_name: str,
    model_cls,
    train_loader,
    val_loader,
    test_loader,
    adj,
    config,
    device,
    epochs: int,
    pretrain_epochs: int = 0,
    utility_loss: bool = False,
    use_error_calibrator: bool = False,
    selective_correction: bool = False,
    physics_vetted_correction: bool = False,
    generic_only_correction: bool = False,
    utility_router_correction: bool = False,
    physics_verified_correction: bool = False,
    contrastive_utility_verifier: bool = False,
    two_stage_verifier: bool = False,
    verifier_epochs: int = 5,
    verifier_min_gate: float = 0.0,
    hard_negative_verifier: bool = False,
    hard_negative_margin: float = 0.02,
    generic_v2_correction: bool = False,
    generic_v3_correction: bool = False,
    generic_v4_correction: bool = False,
    physics_candidate_correction: bool = False,
    physics_promoted_correction: bool = False,
    learned_physics_promotion: bool = False,
    discrete_physics_promotion: bool = False,
    physics_harm_verifier: bool = False,
    two_stage_harm_verifier: bool = False,
    harm_regularized_correction: bool = False,
    harm_suppressed_correction: bool = False,
    region_adaptive_correction: bool = False,
    harm_keep_min: float = 0.0,
    sparse_harm_verifier: bool = False,
    harm_threshold: float = 0.5,
    harm_temperature: float = 0.05,
    harm_hard_weight: float = 1.0,
    harm_safe_weight: float = 0.35,
    harm_utility_target: bool = False,
    harm_utility_temperature: float = 0.05,
    scenario: str | None = None,
    scenario_aware: bool = False,
    diagnostics: bool = False,
    val_blend: bool = False,
    val_select_candidate: bool = False,
    target_only_loss: bool = False,
    correction_clip: float | None = None,
    save_grin_path: Path | None = None,
    load_grin_path: Path | None = None,
    eval_grin_only: bool = False,
):
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    kwargs = {
        "adj": adj.detach().cpu().numpy(),
        "d_in": int(config["dataset"]["channels"]),
        "d_hidden": int(config["model"].get("hidden_dim", 32)),
        "d_ff": int(config["model"].get("hidden_dim", 32)),
        "ff_dropout": 0.0,
        "n_layers": 1,
        "kernel_size": 2,
        "decoder_order": 1,
        "d_u": 0,
        "d_emb": 8,
        "layer_norm": False,
        "merge": "mlp",
        "impute_only_holes": True,
    }
    if model_name == "official_repo_litetrust_grin":
        if correction_clip is not None:
            kwargs["correction_clip"] = float(correction_clip)
        kwargs["use_error_calibrator"] = bool(use_error_calibrator or two_stage_verifier or two_stage_harm_verifier or harm_regularized_correction or harm_suppressed_correction)
        kwargs["selective_correction"] = bool(selective_correction)
        kwargs["physics_vetted_correction"] = bool(physics_vetted_correction)
        kwargs["generic_only_correction"] = bool(generic_only_correction or harm_regularized_correction)
        kwargs["utility_router_correction"] = bool(utility_router_correction)
        kwargs["physics_verified_correction"] = bool(physics_verified_correction)
        kwargs["contrastive_utility_verifier"] = bool(contrastive_utility_verifier)
        kwargs["verifier_min_gate"] = float(verifier_min_gate)
        kwargs["generic_v2_correction"] = bool(generic_v2_correction)
        kwargs["generic_v3_correction"] = bool(generic_v3_correction)
        kwargs["generic_v4_correction"] = bool(generic_v4_correction)
        kwargs["physics_candidate_correction"] = bool(physics_candidate_correction)
        kwargs["physics_promoted_correction"] = bool(physics_promoted_correction)
        kwargs["learned_physics_promotion"] = bool(learned_physics_promotion)
        kwargs["discrete_physics_promotion"] = bool(discrete_physics_promotion)
        kwargs["physics_harm_verifier"] = bool(physics_harm_verifier or two_stage_harm_verifier)
        kwargs["harm_suppressed_correction"] = bool(harm_suppressed_correction)
        kwargs["region_adaptive_correction"] = bool(region_adaptive_correction)
        kwargs["harm_keep_min"] = float(harm_keep_min)
        kwargs["sparse_harm_verifier"] = bool(sparse_harm_verifier)
        kwargs["harm_threshold"] = float(harm_threshold)
        kwargs["harm_temperature"] = float(harm_temperature)
        kwargs["scenario_dim"] = len(next(iter(SCENARIO_TOKENS.values()))) if scenario_aware else 0
    model = model_cls(**kwargs).to(device)
    logs = []
    if load_grin_path is not None and load_grin_path.exists():
        payload = torch.load(load_grin_path, map_location=device)
        state_dict = payload.get("state_dict", payload)
        if hasattr(model, "grin"):
            model.grin.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    if pretrain_epochs > 0 and hasattr(model, "grin") and load_grin_path is None:
        optimizer = torch.optim.Adam(model.grin.parameters(), lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, pretrain_epochs + 1):
            train_stats = _train_epoch(model, train_loader, optimizer, device, grin_only=True, scenario=scenario)
            val_stats = _train_epoch(model, val_loader, None, device, grin_only=True, scenario=scenario)
            logs.append(
                {
                    "epoch": epoch,
                    "model": model_name,
                    "phase": "grin_pretrain",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
    if save_grin_path is not None:
            save_grin_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.grin.state_dict(),
                    "scenario": scenario,
                    "pretrain_epochs": pretrain_epochs,
                },
                save_grin_path,
            )
    if discrete_physics_promotion and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, discrete_physics_promotion=True)
        trainable = _set_trainable(model, ("physics_promotion_mode_head",))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                promotion_loss=True,
                discrete_promotion_loss=True,
                hard_negative_margin=float(hard_negative_margin),
                scenario=scenario,
            )
            val_stats = _train_epoch(
                model,
                val_loader,
                None,
                device,
                hard_negative_margin=float(hard_negative_margin),
                scenario=scenario,
            )
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "discrete_physics_promotion",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            hard_negative_margin=float(hard_negative_margin),
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            **test_stats,
        }, logs
    if physics_promoted_correction and learned_physics_promotion and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, physics_promoted=True)
        trainable = _set_trainable(model, ("physics_promotion_head",))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                promotion_loss=True,
                hard_negative_margin=float(hard_negative_margin),
                harm_utility_temperature=float(harm_utility_temperature),
                scenario=scenario,
            )
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "learned_physics_promotion",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            **test_stats,
        }, logs
    if (physics_candidate_correction or physics_promoted_correction) and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, physics_candidate=bool(physics_candidate_correction), physics_promoted=bool(physics_promoted_correction))
        trainable = _set_trainable(model, ("physics_head", "physics_projection_head", "fd_gain", "phys_bias"))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(model, train_loader, optimizer, device, scenario=scenario)
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "physics_promoted" if physics_promoted_correction else "physics_candidate",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            **test_stats,
        }, logs
    if harm_suppressed_correction and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, generic_only=True)
        trainable = _set_trainable(model, ("generic_head", "error_calibrator", "error_shrinkage_head"))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(model, train_loader, optimizer, device, scenario=scenario)
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "generic_correction",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )

        for module_name in ("physics_harm_head", "physics_harm_sensor_head", "physics_harm_gate_head"):
            for layer in getattr(model, module_name):
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
        if hasattr(model, "correction_allowance_head"):
            for layer in model.correction_allowance_head:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
            torch.nn.init.constant_(model.correction_allowance_head[-1].bias, 2.0)
        _set_litetrust_mode(model, harm_suppressed=True)
        trainable = _set_trainable(
            model,
            ("physics_harm_head", "physics_harm_sensor_head", "physics_harm_gate_head", "correction_allowance_head"),
        )
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, int(verifier_epochs) + 1):
            train_stats = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                harm_suppressed_loss=True,
                hard_negative_margin=float(hard_negative_margin),
                harm_hard_weight=float(harm_hard_weight),
                harm_safe_weight=float(harm_safe_weight),
                harm_utility_target=bool(harm_utility_target),
                harm_utility_temperature=float(harm_utility_temperature),
                scenario=scenario,
            )
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epochs + epoch,
                    "model": model_name,
                    "phase": "harm_suppression",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            "verifier_epochs": int(verifier_epochs),
            **test_stats,
        }, logs
    if region_adaptive_correction and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, generic_only=True, region_adaptive=True)
        trainable = _set_trainable(
            model,
            (
                "generic_head",
                "physics_head",
                "physics_projection_head",
                "correction_allowance_head",
                "physics_harm_head",
                "physics_harm_sensor_head",
                "physics_harm_gate_head",
                "error_calibrator",
                "error_shrinkage_head",
            ),
        )
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                utility_loss=True,
                harm_suppressed_loss=True,
                hard_negative_margin=float(hard_negative_margin),
                harm_hard_weight=float(harm_hard_weight),
                harm_safe_weight=float(harm_safe_weight),
                harm_utility_target=bool(harm_utility_target),
                harm_utility_temperature=float(harm_utility_temperature),
                target_only_loss=bool(target_only_loss),
                scenario=scenario,
            )
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "region_adaptive_correction",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            **test_stats,
        }, logs
    if harm_regularized_correction and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, generic_only=True)
        trainable = _set_trainable(model, ("generic_head", "error_calibrator", "error_shrinkage_head"))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(model, train_loader, optimizer, device, scenario=scenario)
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "generic_correction",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )

        for layer in model.physics_harm_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        _set_litetrust_mode(model, generic_only=True)
        trainable = _set_trainable(
            model,
            ("generic_head", "error_calibrator", "error_shrinkage_head", "physics_harm_head", "physics_harm_sensor_head", "physics_harm_gate_head"),
        )
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, int(verifier_epochs) + 1):
            train_stats = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                physics_harm_verifier_loss=False,
                harm_regularized_loss=True,
                hard_negative_margin=float(hard_negative_margin),
                harm_hard_weight=float(harm_hard_weight),
                harm_safe_weight=float(harm_safe_weight),
                harm_utility_target=bool(harm_utility_target),
                harm_utility_temperature=float(harm_utility_temperature),
                scenario=scenario,
            )
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epochs + epoch,
                    "model": model_name,
                    "phase": "harm_regularizer",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            "verifier_epochs": int(verifier_epochs),
            **test_stats,
        }, logs
    if two_stage_harm_verifier and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, generic_only=True)
        trainable = _set_trainable(model, ("generic_head", "error_calibrator", "error_shrinkage_head"))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(model, train_loader, optimizer, device, scenario=scenario)
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "generic_correction",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )

        for layer in model.physics_harm_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        _set_litetrust_mode(model, generic_only=True)
        trainable = _set_trainable(
            model,
            ("physics_harm_head", "physics_harm_sensor_head", "physics_harm_gate_head"),
        )
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, int(verifier_epochs) + 1):
            train_stats = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                harm_regularized_loss=True,
                hard_negative_margin=float(hard_negative_margin),
                harm_hard_weight=float(harm_hard_weight),
                harm_safe_weight=float(harm_safe_weight),
                harm_utility_target=bool(harm_utility_target),
                harm_utility_temperature=float(harm_utility_temperature),
                scenario=scenario,
            )
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epochs + epoch,
                    "model": model_name,
                    "phase": "harm_verifier",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            "verifier_epochs": int(verifier_epochs),
            **test_stats,
        }, logs
    if two_stage_verifier and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        _set_litetrust_mode(model, generic_only=True)
        trainable = _set_trainable(model, ("generic_head", "error_calibrator", "error_shrinkage_head"))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, epochs + 1):
            train_stats = _train_epoch(model, train_loader, optimizer, device, scenario=scenario)
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epoch,
                    "model": model_name,
                    "phase": "generic_correction",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )

        for layer in model.physics_verifier_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        _set_litetrust_mode(model, contrastive_verifier=True)
        trainable = _set_trainable(model, ("physics_verifier_head",))
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
        for epoch in range(1, int(verifier_epochs) + 1):
            train_stats = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                balanced_verifier_loss=not bool(hard_negative_verifier),
                hard_negative_verifier_loss=bool(hard_negative_verifier),
                hard_negative_margin=float(hard_negative_margin),
                scenario=scenario,
            )
            val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
            logs.append(
                {
                    "epoch": pretrain_epochs + epochs + epoch,
                    "model": model_name,
                    "phase": "verifier",
                    "train_loss": train_stats["loss"],
                    "val_masked_mae": val_stats["masked_mae"],
                }
            )
        test_stats = _train_epoch(
            model,
            test_loader,
            None,
            device,
            diagnostics=diagnostics and not eval_grin_only,
            scenario=scenario,
        )
        return {
            "model": model_name,
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            "verifier_epochs": int(verifier_epochs),
            **test_stats,
        }, logs
    if pretrain_epochs > 0 and hasattr(model, "grin"):
        for param in model.grin.parameters():
            param.requires_grad_(False)
        trainable = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=float(config["train"]["lr"]), weight_decay=0.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]), weight_decay=0.0)
    for epoch in range(1, epochs + 1):
        train_stats = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            utility_loss=utility_loss and pretrain_epochs > 0,
            selective_loss=(selective_correction or utility_router_correction) and pretrain_epochs > 0,
            physics_verified_loss=physics_verified_correction and pretrain_epochs > 0,
            contrastive_verifier_loss=contrastive_utility_verifier and pretrain_epochs > 0,
            physics_harm_verifier_loss=physics_harm_verifier and pretrain_epochs > 0,
            harm_regularized_loss=harm_regularized_correction and pretrain_epochs > 0,
            harm_hard_weight=float(harm_hard_weight),
            harm_safe_weight=float(harm_safe_weight),
            harm_utility_target=bool(harm_utility_target),
            harm_utility_temperature=float(harm_utility_temperature),
            target_only_loss=bool(target_only_loss),
            scenario=scenario,
        )
        val_stats = _train_epoch(model, val_loader, None, device, scenario=scenario)
        logs.append(
            {
                "epoch": pretrain_epochs + epoch,
                "model": model_name,
                "phase": "main" if pretrain_epochs <= 0 else "correction",
                "train_loss": train_stats["loss"],
                "val_masked_mae": val_stats["masked_mae"],
            }
        )
    test_stats = _train_epoch(
        model,
        test_loader,
        None,
        device,
        grin_only=eval_grin_only,
        diagnostics=diagnostics and not eval_grin_only,
        scenario=scenario,
    )
    if (val_blend or val_select_candidate) and hasattr(model, "grin") and not eval_grin_only:
        val_arrays, val_target, val_mask = _collect_candidate_arrays(model, val_loader, device, scenario=scenario)
        test_arrays, test_target, test_mask = _collect_candidate_arrays(model, test_loader, device, scenario=scenario)
        posthoc_stats = (
            _fit_validation_selection(val_arrays, val_target, val_mask, test_arrays, test_target, test_mask)
            if val_select_candidate
            else _fit_validation_blend(val_arrays, val_target, val_mask, test_arrays, test_target, test_mask)
        )
        if posthoc_stats:
            test_stats["raw_masked_mae"] = test_stats["masked_mae"]
            test_stats["raw_mae"] = test_stats["mae"]
            test_stats["raw_rmse"] = test_stats["rmse"]
            test_stats["raw_mape"] = test_stats["mape"]
            test_stats.update(posthoc_stats)
            if val_select_candidate:
                test_stats["masked_mae"] = float(posthoc_stats["val_selected_candidate_test_masked_mae"])
                test_stats["mae"] = float(posthoc_stats["val_selected_candidate_test_mae"])
                test_stats["rmse"] = float(posthoc_stats["val_selected_candidate_test_rmse"])
                test_stats["mape"] = float(posthoc_stats["val_selected_candidate_test_mape"])
            else:
                test_stats["masked_mae"] = float(posthoc_stats["val_blend_test_masked_mae"])
                test_stats["mae"] = float(posthoc_stats["val_blend_test_mae"])
                test_stats["rmse"] = float(posthoc_stats["val_blend_test_rmse"])
                test_stats["mape"] = float(posthoc_stats["val_blend_test_mape"])
    return {
        "model": model_name,
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        **test_stats,
    }, logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_GRIN_ROOT))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--utility-loss", action="store_true")
    parser.add_argument("--use-error-calibrator", action="store_true")
    parser.add_argument("--selective-correction", action="store_true")
    parser.add_argument("--physics-vetted-correction", action="store_true")
    parser.add_argument("--generic-only-correction", action="store_true")
    parser.add_argument("--utility-router-correction", action="store_true")
    parser.add_argument("--physics-verified-correction", action="store_true")
    parser.add_argument("--contrastive-utility-verifier", action="store_true")
    parser.add_argument("--two-stage-verifier", action="store_true")
    parser.add_argument("--physics-harm-verifier", action="store_true")
    parser.add_argument("--two-stage-harm-verifier", action="store_true")
    parser.add_argument("--harm-regularized-correction", action="store_true")
    parser.add_argument("--harm-suppressed-correction", action="store_true")
    parser.add_argument("--verifier-epochs", type=int, default=5)
    parser.add_argument("--verifier-min-gate", type=float, default=0.0)
    parser.add_argument("--harm-keep-min", type=float, default=0.0)
    parser.add_argument("--sparse-harm-verifier", action="store_true")
    parser.add_argument("--harm-threshold", type=float, default=0.5)
    parser.add_argument("--harm-temperature", type=float, default=0.05)
    parser.add_argument("--harm-hard-weight", type=float, default=1.0)
    parser.add_argument("--harm-safe-weight", type=float, default=0.35)
    parser.add_argument("--harm-utility-target", action="store_true")
    parser.add_argument("--hard-negative-verifier", action="store_true")
    parser.add_argument("--hard-negative-margin", type=float, default=0.02)
    parser.add_argument("--generic-v2-correction", action="store_true")
    parser.add_argument("--generic-v3-correction", action="store_true")
    parser.add_argument("--generic-v4-correction", action="store_true")
    parser.add_argument("--physics-candidate-correction", action="store_true")
    parser.add_argument("--physics-promoted-correction", action="store_true")
    parser.add_argument("--learned-physics-promotion", action="store_true")
    parser.add_argument("--discrete-physics-promotion", action="store_true")
    parser.add_argument("--region-adaptive-correction", action="store_true")
    parser.add_argument("--scenario-aware", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--val-blend", action="store_true")
    parser.add_argument("--val-select-candidate", action="store_true")
    parser.add_argument("--target-only-loss", action="store_true")
    parser.add_argument("--correction-clip", type=float, default=None)
    parser.add_argument("--cache-dir", default="C:/Users/21329/litetrust_official_grin_outputs/grin_cache")
    parser.add_argument("--save-grin-cache", action="store_true")
    parser.add_argument("--load-grin-cache", action="store_true")
    parser.add_argument("--litetrust-only", action="store_true")
    parser.add_argument("--eval-grin-only", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50"])
    parser.add_argument("--output-dir", default="results/official_repo_litetrust_quick")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    official_root = Path(args.official_root)
    GRINet, LiteTrustGRINet = _load_official_models(official_root)
    config = _config()
    if args.seed is not None:
        config["seed"] = int(args.seed)
    config["device"] = "cpu"
    config["train"]["epochs"] = int(args.epochs)
    device = resolve_device(config.get("device", "cpu"))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    logs = []
    for scenario in args.scenarios:
        train_loader, val_loader, test_loader, adj, _scaler, metadata = _scenario_loaders(config, scenario)
        adj = adj.to(device)
        model_pairs = [("official_repo_grin", GRINet), ("official_repo_litetrust_grin", LiteTrustGRINet)]
        if bool(args.litetrust_only):
            model_pairs = [("official_repo_litetrust_grin", LiteTrustGRINet)]
        for model_name, model_cls in model_pairs:
            print(f"running {scenario} {model_name}", file=sys.stderr, flush=True)
            pretrain_epochs = int(args.pretrain_epochs) if model_name == "official_repo_litetrust_grin" else 0
            train_epochs = int(args.pretrain_epochs) if model_name == "official_repo_grin" and int(args.pretrain_epochs) > 0 else int(args.epochs)
            cache_path = Path(args.cache_dir) / f"{scenario}__grin_{int(args.pretrain_epochs)}e.pt"
            save_grin_path = cache_path if bool(args.save_grin_cache) and model_name == "official_repo_litetrust_grin" else None
            load_grin_path = cache_path if bool(args.load_grin_cache) and model_name == "official_repo_litetrust_grin" else None
            row, model_logs = _run_one(
                model_name,
                model_cls,
                train_loader,
                val_loader,
                test_loader,
                adj,
                config,
                device,
                train_epochs,
                pretrain_epochs=pretrain_epochs,
                utility_loss=bool(args.utility_loss),
                use_error_calibrator=bool(args.use_error_calibrator),
                selective_correction=bool(args.selective_correction),
                physics_vetted_correction=bool(args.physics_vetted_correction),
                generic_only_correction=bool(args.generic_only_correction),
                utility_router_correction=bool(args.utility_router_correction),
                physics_verified_correction=bool(args.physics_verified_correction),
                contrastive_utility_verifier=bool(args.contrastive_utility_verifier),
                two_stage_verifier=bool(args.two_stage_verifier),
                verifier_epochs=int(args.verifier_epochs),
                verifier_min_gate=float(args.verifier_min_gate),
                hard_negative_verifier=bool(args.hard_negative_verifier),
                hard_negative_margin=float(args.hard_negative_margin),
                generic_v2_correction=bool(args.generic_v2_correction),
                generic_v3_correction=bool(args.generic_v3_correction),
                generic_v4_correction=bool(args.generic_v4_correction),
                physics_candidate_correction=bool(args.physics_candidate_correction),
                physics_promoted_correction=bool(args.physics_promoted_correction),
                learned_physics_promotion=bool(args.learned_physics_promotion),
                discrete_physics_promotion=bool(args.discrete_physics_promotion),
                region_adaptive_correction=bool(args.region_adaptive_correction),
                physics_harm_verifier=bool(args.physics_harm_verifier),
                two_stage_harm_verifier=bool(args.two_stage_harm_verifier),
                harm_regularized_correction=bool(args.harm_regularized_correction),
                harm_suppressed_correction=bool(args.harm_suppressed_correction),
                harm_keep_min=float(args.harm_keep_min),
                sparse_harm_verifier=bool(args.sparse_harm_verifier),
                harm_threshold=float(args.harm_threshold),
                harm_temperature=float(args.harm_temperature),
                harm_hard_weight=float(args.harm_hard_weight),
                harm_safe_weight=float(args.harm_safe_weight),
                harm_utility_target=bool(args.harm_utility_target),
                harm_utility_temperature=float(args.harm_temperature),
                scenario=scenario,
                scenario_aware=bool(args.scenario_aware),
                diagnostics=bool(args.diagnostics),
                val_blend=bool(args.val_blend),
                val_select_candidate=bool(args.val_select_candidate),
                target_only_loss=bool(args.target_only_loss),
                correction_clip=args.correction_clip,
                save_grin_path=save_grin_path,
                load_grin_path=load_grin_path,
                eval_grin_only=bool(args.eval_grin_only),
            )
            row.update(
                {
                    "scenario": scenario,
                    "real_data_used": bool(metadata.get("real_data_used", False)),
                    "fallback_used": bool(metadata.get("fallback_used", False)),
                }
            )
            rows.append(row)
            logs.extend({"scenario": scenario, **item} for item in model_logs)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    if logs:
        with open(output_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            writer.writeheader()
            writer.writerows(logs)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "logs": logs}, f, indent=2)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
