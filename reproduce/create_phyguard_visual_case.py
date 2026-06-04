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
from reproduce.run_phyguard_plugin_strong_backbones import PhyGuardPlugin, _features
from scripts.run_five_baselines_flow_quick import _scenario_data
from scripts.run_maginet_physics_guard_quick import _apply_observed, _failure_mode_score, _graph_residual_np, _rank_np
from scripts.run_strong_candidate_fusion_flow_quick import _run_maginet_all_splits
from scripts.train import resolve_device


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def _masked_mae(pred: np.ndarray, target: np.ndarray, region: np.ndarray) -> float:
    return float((np.abs(pred - target) * region).sum() / np.clip(region.sum(), 1.0, None))


def _train_plugin_with_aux(train, val, test, adj, base_train, base_val, base_test, *, seed: int, epochs: int, correction_clip: float):
    torch.manual_seed(seed + 9101)
    train_full, train_obs, train_mask = train
    val_full, val_obs, val_mask = val
    test_full, test_obs, test_mask = test
    feat_train = _features(base_train, train_obs, train_mask, adj)
    feat_val = _features(base_val, val_obs, val_mask, adj)
    feat_test = _features(base_test, test_obs, test_mask, adj)

    target_region = (1.0 - train_mask)[..., 0] > 0.0
    x = torch.tensor(feat_train[target_region], dtype=torch.float32)
    base = torch.tensor(base_train[target_region], dtype=torch.float32)
    y = torch.tensor(train_full[target_region], dtype=torch.float32)
    failure = torch.tensor(_failure_mode_score(train_mask, adj)[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    residual = torch.tensor(_rank_np(np.abs(_graph_residual_np(base_train, adj)))[..., 0][target_region], dtype=torch.float32).unsqueeze(-1)
    weight = 1.0 + 0.75 * failure + 0.50 * residual

    model = PhyGuardPlugin(x.shape[-1], hidden_dim=64, correction_clip=correction_clip)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    generator = torch.Generator().manual_seed(seed + 9102)
    best_state = None
    best_val = float("inf")
    val_feat_t = torch.tensor(feat_val, dtype=torch.float32)
    val_base_t = torch.tensor(base_val, dtype=torch.float32)
    val_target_t = torch.tensor(val_full, dtype=torch.float32)
    val_region_t = torch.tensor(1.0 - val_mask, dtype=torch.float32)

    for _epoch in range(epochs):
        order = torch.randperm(x.shape[0], generator=generator)
        model.train()
        for start in range(0, x.shape[0], 32768):
            idx = order[start : start + 32768]
            pred, gate, delta = model(x[idx], base[idx])
            base_err = torch.abs(base[idx] - y[idx])
            pred_err = torch.abs(pred - y[idx])
            improvement_target = (pred_err.detach() + 1e-6 < base_err.detach()).float()
            rec_loss = torch.mean(pred_err * weight[idx])
            harm_penalty = torch.relu(pred_err - base_err) * (1.0 + 2.0 * weight[idx])
            local_harm_coef = 0.05 + 0.15 * failure[idx]
            harm_loss = torch.mean(harm_penalty * local_harm_coef)
            gate_loss = torch.mean(F.binary_cross_entropy(gate.clamp(1e-4, 1.0 - 1e-4), improvement_target, reduction="none") * weight[idx])
            delta_loss = torch.mean(torch.abs(delta) * (1.0 - improvement_target) * weight[idx])
            loss = rec_loss + harm_loss + 0.15 * gate_loss + 0.03 * delta_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val, _, _ = model(val_feat_t, val_base_t)
            val_mae = float((torch.abs(pred_val - val_target_t) * val_region_t).sum() / val_region_t.sum().clamp_min(1.0))
        if val_mae < best_val:
            best_val = val_mae
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test, gate_test, delta_test = model(torch.tensor(feat_test, dtype=torch.float32), torch.tensor(base_test, dtype=torch.float32))
    return pred_test.numpy().astype(np.float32), gate_test.numpy().astype(np.float32), delta_test.numpy().astype(np.float32), best_val


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
    nodes = np.sort(nodes)
    return sample, nodes


def _plot_case(output_dir: Path, true, obs, mask, base, pred, gate, delta, failure, sample: int, nodes: np.ndarray) -> None:
    t = np.arange(true.shape[1])
    sl = (sample, slice(None), nodes, 0)
    true_m = true[sl].T
    obs_m = obs[sl].T
    mask_m = mask[sl].T
    base_m = base[sl].T
    pred_m = pred[sl].T
    gate_m = gate[sl].T
    delta_m = delta[sl].T
    failure_m = failure[sl].T
    base_err = np.abs(base_m - true_m)
    pred_err = np.abs(pred_m - true_m)
    reduction = base_err - pred_err

    fig, axes = plt.subplots(3, 3, figsize=(8.2, 5.8), constrained_layout=True)
    panels = [
        ("Ground truth", true_m, "viridis", None),
        ("Observed input", np.where(mask_m > 0.5, obs_m, np.nan), "viridis", None),
        ("Missing mask", 1.0 - mask_m, "gray_r", (0, 1)),
        ("Backbone prediction", base_m, "viridis", None),
        ("PhyGuard prediction", pred_m, "viridis", None),
        ("Error reduction", reduction, "RdBu", (-np.nanmax(np.abs(reduction)), np.nanmax(np.abs(reduction)))),
        ("Gate g(i,t)", gate_m, "YlGnBu", (0, 1)),
        ("Correction delta", delta_m, "RdBu", (-np.nanmax(np.abs(delta_m)), np.nanmax(np.abs(delta_m)))),
        ("Failure score", failure_m, "magma", (0, 1)),
    ]
    for idx, (ax, (title, data, cmap, limits)) in enumerate(zip(axes.flat, panels)):
        extent = (0, len(t), data.shape[0], 0)
        if limits is None:
            im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, extent=extent)
        else:
            im = ax.imshow(
                data,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                extent=extent,
            )
        ax.set_title(title)
        if idx >= 6:
            ax.set_xlabel("Time step")
        else:
            ax.set_xlabel("")
        if idx % 3 == 0:
            ax.set_ylabel("Selected node")
        else:
            ax.set_ylabel("")
        tick_idx = [0, len(t) // 2, len(t)]
        ax.set_xticks(tick_idx, [str(i) for i in tick_idx])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(output_dir / "figure_phyguard_case.png")
    fig.savefig(output_dir / "figure_phyguard_case.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS-BAY")
    parser.add_argument("--scenario", default="random_missing_50")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plugin-epochs", type=int, default=40)
    parser.add_argument("--correction-clip", type=float, default=0.20)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/phyguard_visual_case")
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
    pred, gate, delta, best_val = _train_plugin_with_aux(
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
    )
    failure = _failure_mode_score(test_mask, adj).astype(np.float32)
    sample, nodes = _select_case(test_x, base_test, pred, test_mask)
    np.savez_compressed(
        output_dir / "case_arrays.npz",
        true=test_x,
        obs=test_obs,
        mask=test_mask,
        backbone=base_test,
        phyguard=pred,
        gate=gate,
        delta=delta,
        failure_score=failure,
        sample=np.array(sample),
        nodes=nodes,
        metadata=np.array(str(meta)),
    )
    _plot_case(output_dir, test_x, test_obs, test_mask, base_test, pred, gate, delta, failure, sample, nodes)
    region = 1.0 - test_mask
    base_mae = _masked_mae(base_test, test_x, region)
    phy_mae = _masked_mae(pred, test_x, region)
    with open(output_dir / "case_summary.md", "w", encoding="utf-8") as f:
        f.write("# PhyGuard visual case\n\n")
        f.write(f"- dataset: {args.dataset}\n")
        f.write(f"- scenario: {args.scenario}\n")
        f.write(f"- seed: {args.seed}\n")
        f.write(f"- sample: {sample}\n")
        f.write(f"- selected nodes: {len(nodes)}\n")
        f.write(f"- backbone masked MAE: {base_mae:.6f}\n")
        f.write(f"- PhyGuard masked MAE: {phy_mae:.6f}\n")
        f.write(f"- gain: {(base_mae - phy_mae) / base_mae * 100.0:.2f}%\n")
        f.write(f"- plugin validation MAE: {best_val:.6f}\n")
    print(f"wrote visual case to {output_dir}")
    print(f"base_mae={base_mae:.6f} phyguard_mae={phy_mae:.6f} gain={(base_mae - phy_mae) / base_mae * 100.0:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
