from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduce.run_antileakage_protocol import _load_antileakage_splits
from reproduce.run_plugin_baseline_comparison import (
    GENERIC_FEATURES,
    RELIABILITY_FEATURES,
    ReliabilityConditionedPlugin,
)
from reproduce.run_phyguard_plugin_strong_backbones import _features
from scripts.run_five_baselines_flow_quick import _scenario_data
from scripts.run_maginet_physics_guard_quick import (
    _apply_observed,
    _failure_mode_score,
    _graph_residual_np,
    _rank_np,
)
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits
from scripts.train import resolve_device


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 320,
            "figure.dpi": 140,
        }
    )


def _masked_mae(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _train_phypro_with_aux(
    train,
    val,
    test,
    adj: np.ndarray,
    base_train: np.ndarray,
    base_val: np.ndarray,
    base_test: np.ndarray,
    *,
    seed: int,
    epochs: int,
    correction_clip: float,
    gate_floor: float,
    conflict_coef: float,
):
    torch.manual_seed(seed + 10401)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test

    feat_train_full = _features(base_train, train_obs, train_mask, adj)
    feat_val_full = _features(base_val, val_obs, val_mask, adj)
    feat_test_full = _features(base_test, test_obs, test_mask, adj)

    feat_train_g = feat_train_full[..., GENERIC_FEATURES].astype(np.float32)
    feat_val_g = feat_val_full[..., GENERIC_FEATURES].astype(np.float32)
    feat_test_g = feat_test_full[..., GENERIC_FEATURES].astype(np.float32)
    feat_train_r = feat_train_full[..., RELIABILITY_FEATURES].astype(np.float32)
    feat_val_r = feat_val_full[..., RELIABILITY_FEATURES].astype(np.float32)
    feat_test_r = feat_test_full[..., RELIABILITY_FEATURES].astype(np.float32)

    aligned_train = (0.6 * feat_train_full[..., 8:9] + 0.4 * feat_train_full[..., 9:10]).astype(np.float32)
    aligned_val = (0.6 * feat_val_full[..., 8:9] + 0.4 * feat_val_full[..., 9:10]).astype(np.float32)
    aligned_test = (0.6 * feat_test_full[..., 8:9] + 0.4 * feat_test_full[..., 9:10]).astype(np.float32)

    target_region = (1.0 - train_mask)[..., 0] > 0.0
    xg = torch.tensor(feat_train_g[target_region], dtype=torch.float32)
    xr = torch.tensor(feat_train_r[target_region], dtype=torch.float32)
    xa = torch.tensor(aligned_train[target_region], dtype=torch.float32)
    base = torch.tensor(base_train[target_region], dtype=torch.float32)
    y = torch.tensor(train_full[target_region], dtype=torch.float32)
    failure = torch.tensor(_failure_mode_score(train_mask, adj)[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    residual_rank = torch.tensor(
        _rank_np(np.abs(_graph_residual_np(base_train, adj)))[..., 0][target_region],
        dtype=torch.float32,
    ).unsqueeze(-1)
    reliability_weight = 1.0 + 0.75 * failure + 0.50 * residual_rank

    model = ReliabilityConditionedPlugin(
        xg.shape[-1],
        xr.shape[-1],
        correction_clip=correction_clip,
        gate_floor=gate_floor,
        conflict_coef=conflict_coef,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 10402)
    batch_size = 32768
    best_state = None
    best_val = float("inf")

    val_g_t = torch.tensor(feat_val_g, dtype=torch.float32)
    val_r_t = torch.tensor(feat_val_r, dtype=torch.float32)
    val_a_t = torch.tensor(aligned_val, dtype=torch.float32)
    val_base_t = torch.tensor(base_val, dtype=torch.float32)
    val_target_t = torch.tensor(val_full, dtype=torch.float32)
    val_region_t = torch.tensor(1.0 - val_mask, dtype=torch.float32)

    for _epoch in range(max(1, epochs)):
        order = torch.randperm(xg.shape[0], generator=generator)
        model.train()
        for start in range(0, xg.shape[0], batch_size):
            idx = order[start : start + batch_size]
            pred, gate, delta, beta = model(xg[idx], xr[idx], xa[idx], base[idx])
            generic_pred = base[idx] + delta
            promoted_probe = generic_pred + torch.tanh(xa[idx]) * correction_clip
            base_err = torch.abs(base[idx] - y[idx])
            generic_err = torch.abs(generic_pred.detach() - y[idx])
            promoted_err = torch.abs(promoted_probe.detach() - y[idx])
            pred_err = torch.abs(pred - y[idx])
            utility_target = (generic_err + 1e-6 < base_err.detach()).float()
            promo_target = (promoted_err + 1e-6 < generic_err).float()
            rec_loss = torch.mean(pred_err)
            gate_loss = torch.mean(
                F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), utility_target, reduction="none")
                * reliability_weight[idx]
            )
            harm = torch.relu(pred_err - torch.minimum(base_err.detach(), generic_err.detach()))
            harm_loss = torch.mean(harm * (0.01 + 0.05 * failure[idx] + 0.02 * residual_rank[idx]))
            delta_shrink = torch.mean(torch.abs(delta) * (1.0 - utility_target) * (0.01 + 0.02 * reliability_weight[idx]))
            promo_loss = torch.mean(
                F.binary_cross_entropy(beta.clamp(1e-4, 1.0 - 1e-4), promo_target, reduction="none")
                * reliability_weight[idx]
            )
            promo_harm = torch.mean(torch.relu(pred_err - generic_err.detach()) * beta * (0.02 + 0.05 * residual_rank[idx]))
            loss = rec_loss + 0.05 * gate_loss + 0.05 * promo_loss + harm_loss + promo_harm + delta_shrink
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val, _, _, _ = model(val_g_t, val_r_t, val_a_t, val_base_t)
            val_mae = float((torch.abs(pred_val - val_target_t) * val_region_t).sum() / val_region_t.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test, gate_test, delta_test, beta_test = model(
            torch.tensor(feat_test_g, dtype=torch.float32),
            torch.tensor(feat_test_r, dtype=torch.float32),
            torch.tensor(aligned_test, dtype=torch.float32),
            torch.tensor(base_test, dtype=torch.float32),
        )
    return (
        pred_test.numpy().astype(np.float32),
        gate_test.numpy().astype(np.float32),
        delta_test.numpy().astype(np.float32),
        beta_test.numpy().astype(np.float32),
        aligned_test.astype(np.float32),
        best_val,
    )


def _select_case(true: np.ndarray, base: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[int, np.ndarray]:
    region = 1.0 - mask
    base_err = np.abs(base - true) * region
    pred_err = np.abs(pred - true) * region
    gain = (base_err - pred_err).sum(axis=(1, 2, 3))
    sample = int(np.argmax(gain))
    node_gain = (base_err[sample] - pred_err[sample]).sum(axis=(0, 2))
    missing_count = region[sample].sum(axis=(0, 2))
    score = node_gain + 0.02 * missing_count
    nodes = np.argsort(-score)[:64]
    return sample, np.sort(nodes)


def _plot_case(
    output_dir: Path,
    true,
    obs,
    mask,
    base,
    pred,
    gate,
    delta,
    beta,
    aligned,
    failure,
    sample: int,
    nodes: np.ndarray,
    correction_clip: float,
) -> None:
    sl = (sample, slice(None), nodes, 0)
    true_m = true[sl].T
    obs_m = obs[sl].T
    mask_m = mask[sl].T
    base_m = base[sl].T
    pred_m = pred[sl].T
    gate_m = gate[sl].T
    delta_m = delta[sl].T
    beta_m = beta[sl].T
    aligned_m = aligned[sl].T
    failure_m = failure[sl].T
    reduction = np.abs(base_m - true_m) - np.abs(pred_m - true_m)
    promoted_m = beta_m * np.tanh(aligned_m) * correction_clip
    final_delta_m = pred_m - base_m

    vmax_reduction = max(float(np.nanmax(np.abs(reduction))), 1e-6)
    vmax_delta = max(float(np.nanmax(np.abs(delta_m))), 1e-6)
    vmax_aligned = max(float(np.nanmax(np.abs(aligned_m))), 1e-6)
    vmax_promoted = max(float(np.nanmax(np.abs(promoted_m))), 1e-6)
    vmax_final_delta = max(float(np.nanmax(np.abs(final_delta_m))), 1e-6)
    fig, axes = plt.subplots(3, 4, figsize=(10.8, 6.4), constrained_layout=True)
    panels = [
        ("Ground truth", true_m, "viridis", None),
        ("Observed input", np.where(mask_m > 0.5, obs_m, np.nan), "viridis", None),
        ("Missing region", 1.0 - mask_m, "gray_r", (0, 1)),
        ("Failure evidence", failure_m, "magma", (0, 1)),
        ("Backbone prediction", base_m, "viridis", None),
        ("PhyPro prediction", pred_m, "viridis", None),
        ("Error reduction", reduction, "RdBu", (-vmax_reduction, vmax_reduction)),
        ("Final correction", final_delta_m, "RdBu", (-vmax_final_delta, vmax_final_delta)),
        ("Learned correction", delta_m, "RdBu", (-vmax_delta, vmax_delta)),
        ("Physics direction", aligned_m, "RdBu", (-vmax_aligned, vmax_aligned)),
        ("Promotion weight", beta_m, "YlGn", (0, 1)),
        ("Physics promotion", promoted_m, "RdBu", (-vmax_promoted, vmax_promoted)),
    ]
    for idx, (ax, (label, data, cmap, limits)) in enumerate(zip(axes.flat, panels)):
        extent = (0, true.shape[1], data.shape[0], 0)
        kwargs = {"aspect": "auto", "interpolation": "nearest", "cmap": cmap, "extent": extent}
        if limits is not None:
            kwargs.update({"vmin": limits[0], "vmax": limits[1]})
        im = ax.imshow(data, **kwargs)
        ax.set_title(label, pad=4)
        ax.set_xticks([0, true.shape[1] // 2, true.shape[1]])
        ax.set_yticks([])
        if idx >= 8:
            ax.set_xlabel("Time step")
        if idx % 4 == 0:
            ax.set_ylabel("Selected sensors")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.018)

    # Store small side panels as arrays even though they are not displayed in the
    # main figure, so the paper can later swap a panel without rerunning.
    np.savez_compressed(
        output_dir / "auxiliary_panel_arrays.npz",
        learned_delta=delta_m,
        physics_aligned_delta=aligned_m,
        delta_vmax=np.array(vmax_delta),
        aligned_vmax=np.array(vmax_aligned),
    )
    fig.savefig(output_dir / "figure_phypro_case.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / "figure_phypro_case.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    split_specs = [
        ("a", panels[0:4]),
        ("b", panels[4:8]),
        ("c", panels[8:12]),
    ]
    for suffix, row_panels in split_specs:
        row_fig, row_axes = plt.subplots(1, 4, figsize=(7.4, 2.35), constrained_layout=True)
        for idx, (ax, (label, data, cmap, limits)) in enumerate(zip(row_axes.flat, row_panels)):
            extent = (0, true.shape[1], data.shape[0], 0)
            kwargs = {"aspect": "auto", "interpolation": "nearest", "cmap": cmap, "extent": extent}
            if limits is not None:
                kwargs.update({"vmin": limits[0], "vmax": limits[1]})
            im = ax.imshow(data, **kwargs)
            ax.set_title(label, pad=4)
            ax.set_xticks([0, true.shape[1] // 2, true.shape[1]])
            ax.set_yticks([])
            ax.set_xlabel("Time step")
            if idx == 0:
                ax.set_ylabel("Selected sensors")
            row_fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
        row_fig.savefig(output_dir / f"figure_phypro_case_{suffix}.png", bbox_inches="tight", pad_inches=0.03)
        row_fig.savefig(output_dir / f"figure_phypro_case_{suffix}.pdf", bbox_inches="tight", pad_inches=0.03)
        plt.close(row_fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS-BAY")
    parser.add_argument("--scenario", default="random_missing_50")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plugin-epochs", type=int, default=40)
    parser.add_argument("--correction-clip", type=float, default=0.20)
    parser.add_argument("--phypro-gate-floor", type=float, default=0.95)
    parser.add_argument("--phypro-conflict-coef", type=float, default=0.75)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/phypro_visual_case")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, val_x, test_x, adj, meta = _load_antileakage_splits(args.dataset, argparse.Namespace(**vars(args)))
    train_obs, train_mask = _scenario_data(train_x, adj, args.scenario, args.seed)
    val_obs, val_mask = _scenario_data(val_x, adj, args.scenario, args.seed + 11)
    test_obs, test_mask = _scenario_data(test_x, adj, args.scenario, args.seed + 29)
    train = (train_x, train_obs, train_mask)
    val = (val_x, val_obs, val_mask)
    test = (test_x, test_obs, test_mask)

    pred_train, pred_val, pred_test = _run_maginet_all_splits(args.scenario, train, val, test, adj, device, args.epochs)
    base_train = _apply_observed(train_obs, train_mask, pred_train)
    base_val = _apply_observed(val_obs, val_mask, pred_val)
    base_test = _apply_observed(test_obs, test_mask, pred_test)
    pred, gate, delta, beta, aligned, best_val = _train_phypro_with_aux(
        train,
        val,
        test,
        adj,
        base_train,
        base_val,
        base_test,
        seed=args.seed,
        epochs=args.plugin_epochs,
        correction_clip=args.correction_clip,
        gate_floor=args.phypro_gate_floor,
        conflict_coef=args.phypro_conflict_coef,
    )
    failure = _failure_mode_score(test_mask, adj).astype(np.float32)
    sample, nodes = _select_case(test_x, base_test, pred, test_mask)
    region = 1.0 - test_mask
    base_mae = _masked_mae(base_test, test_x, region)
    phypro_mae = _masked_mae(pred, test_x, region)
    gain = (base_mae - phypro_mae) / max(base_mae, 1e-8) * 100.0

    np.savez_compressed(
        output_dir / "case_arrays.npz",
        true=test_x,
        obs=test_obs,
        mask=test_mask,
        backbone=base_test,
        phypro=pred,
        reliability_gate=gate,
        learned_delta=delta,
        promotion=beta,
        physics_aligned_delta=aligned,
        failure_score=failure,
        sample=np.array(sample),
        nodes=nodes,
        metadata=np.array(str(meta)),
    )
    _plot_case(
        output_dir,
        test_x,
        test_obs,
        test_mask,
        base_test,
        pred,
        gate,
        delta,
        beta,
        aligned,
        failure,
        sample,
        nodes,
        args.correction_clip,
    )
    with open(output_dir / "case_summary.md", "w", encoding="utf-8") as f:
        f.write("# PhyPro visual case\n\n")
        f.write(f"- dataset: {args.dataset}\n")
        f.write(f"- scenario: {args.scenario}\n")
        f.write(f"- backbone: MagiNet\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- sample: {sample}\n")
        f.write(f"- selected sensors: {len(nodes)}\n")
        f.write(f"- backbone masked MAE: {base_mae:.6f}\n")
        f.write(f"- PhyPro masked MAE: {phypro_mae:.6f}\n")
        f.write(f"- gain: {gain:.2f}%\n")
        f.write(f"- plugin validation MAE: {best_val:.6f}\n")
        f.write(f"- reliability gate mean: {float((gate * region).sum() / np.clip(region.sum(), 1.0, None)):.6f}\n")
        f.write(f"- promotion mean: {float((beta * region).sum() / np.clip(region.sum(), 1.0, None)):.6f}\n")
    print(f"wrote PhyPro visual case to {output_dir}")
    print(f"base_mae={base_mae:.6f} phypro_mae={phypro_mae:.6f} gain={gain:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
