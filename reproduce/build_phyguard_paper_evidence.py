from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduce.run_phyguard_plugin_strong_backbones import PhyGuardPlugin


DATASET_NODES = {
    "PEMS03": 358,
    "PEMS04": 307,
    "PEMS08": 170,
    "PEMS-BAY": 325,
    "METR-LA": 207,
}

SCENARIO_LABELS = {
    "random_missing_50": "Random\nmissing",
    "sensor_failure_30": "Sensor\nfailure",
    "incident_perturbation": "Incident",
}

DISPLAY_NAME_MAP = {
    "MagiNetStrong": "MagiNet",
    "SAITSStrong": "SAITS",
    "BRITSStrong": "BRITS",
    "ImputeFormerStrong": "ImputeFormer",
    "MagiNetStrong+PhyGuardPlugin": "MagiNet + PhyGuard",
    "SAITSStrong+PhyGuardPlugin": "SAITS + PhyGuard",
    "BRITSStrong+PhyGuardPlugin": "BRITS + PhyGuard",
    "ImputeFormerStrong+PhyGuardPlugin": "ImputeFormer + PhyGuard",
}


def _display_name(value: object) -> object:
    if not isinstance(value, str):
        return value
    return DISPLAY_NAME_MAP.get(value, value.replace("+PhyGuardPlugin", " + PhyGuard"))

OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
]


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
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        }
    )


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _benchmark_plugin(feature_dim: int, hidden_dim: int, nodes: int, device: torch.device) -> float:
    model = PhyGuardPlugin(feature_dim=feature_dim, hidden_dim=hidden_dim, correction_clip=0.20).to(device)
    model.eval()
    batch, steps, channels = 16, 12, 1
    features = torch.randn(batch, steps, nodes, feature_dim, device=device)
    base = torch.randn(batch, steps, nodes, channels, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(features, base)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if device.type == "cuda":
            start.record()
            for _ in range(100):
                model(features, base)
            end.record()
            torch.cuda.synchronize()
            return float(start.elapsed_time(end) / 100.0)
        import time

        t0 = time.perf_counter()
        for _ in range(100):
            model(features, base)
        return float((time.perf_counter() - t0) * 1000.0 / 100.0)


def _load_pairs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_path = input_dir / "all_rows.csv"
    if not all_path.exists():
        all_path = input_dir / "summary.csv"
    all_rows = pd.read_csv(all_path)
    pairs = pd.read_csv(input_dir / "paired_rows.csv")
    return all_rows, pairs


def _write_complexity(out: Path, hidden_dim: int, feature_dim: int) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhyGuardPlugin(feature_dim=feature_dim, hidden_dim=hidden_dim, correction_clip=0.20)
    rows = []
    for dataset, nodes in DATASET_NODES.items():
        rows.append(
            {
                "dataset": dataset,
                "nodes": nodes,
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "extra_trainable_params": _count_params(model),
                "benchmark_device": str(device),
                "extra_forward_ms_per_batch": _benchmark_plugin(feature_dim, hidden_dim, nodes, device),
                "batch_shape": f"16x12x{nodes}x1",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "complexity_table.csv", index=False)
    return df


def _write_tables(all_rows: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> dict[str, pd.DataFrame]:
    generality = (
        pairs.groupby("backbone", as_index=False)
        .agg(
            base_mae_mean=("base_mae", "mean"),
            phyguard_mae_mean=("phyguard_mae", "mean"),
            gain_pct_mean=("gain_pct", "mean"),
            gain_pct_std=("gain_pct", "std"),
            gate_mean=("gate_mean", "mean"),
            delta_abs_mean=("delta_abs_mean", "mean"),
        )
        .sort_values("gain_pct_mean", ascending=False)
    )
    generality.to_csv(out / "generality_table.csv", index=False)

    scenario = (
        pairs.groupby("scenario", as_index=False)
        .agg(
            base_mae_mean=("base_mae", "mean"),
            phyguard_mae_mean=("phyguard_mae", "mean"),
            gain_pct_mean=("gain_pct", "mean"),
            gain_pct_std=("gain_pct", "std"),
            gate_mean=("gate_mean", "mean"),
            delta_abs_mean=("delta_abs_mean", "mean"),
            failure_score_mean=("failure_score_mean", "mean"),
        )
        .sort_values("scenario")
    )
    scenario.to_csv(out / "explainability_by_scenario.csv", index=False)

    backbone_scenario = (
        pairs.groupby(["backbone", "scenario"], as_index=False)
        .agg(
            gain_pct_mean=("gain_pct", "mean"),
            gain_pct_std=("gain_pct", "std"),
            gate_mean=("gate_mean", "mean"),
            delta_abs_mean=("delta_abs_mean", "mean"),
            failure_score_mean=("failure_score_mean", "mean"),
        )
        .sort_values(["backbone", "scenario"])
    )
    backbone_scenario.to_csv(out / "explainability_by_backbone_scenario.csv", index=False)

    mean_model = all_rows.groupby(["dataset", "scenario", "model"], as_index=False).agg(
        masked_mae_mean=("masked_mae", "mean"),
        masked_mae_std=("masked_mae", "std"),
    )
    best_rows = []
    for (dataset, scenario_name), sub in mean_model.groupby(["dataset", "scenario"]):
        base_sub = sub[~sub["model"].str.contains(r"\+PhyGuardPlugin", regex=True)]
        plugin_sub = sub[sub["model"].str.contains(r"\+PhyGuardPlugin", regex=True)]
        best_base = base_sub.loc[base_sub["masked_mae_mean"].idxmin()]
        best_plugin = plugin_sub.loc[plugin_sub["masked_mae_mean"].idxmin()]
        best_rows.append(
            {
                "dataset": dataset,
                "scenario": scenario_name,
                "best_base_model": best_base["model"],
                "best_base_mae_mean": best_base["masked_mae_mean"],
                "best_base_mae_std": best_base["masked_mae_std"],
                "best_plugin_model": best_plugin["model"],
                "best_plugin_mae_mean": best_plugin["masked_mae_mean"],
                "best_plugin_mae_std": best_plugin["masked_mae_std"],
                "best_gain_pct": (best_base["masked_mae_mean"] - best_plugin["masked_mae_mean"])
                / best_base["masked_mae_mean"]
                * 100.0,
            }
        )
    best = pd.DataFrame(best_rows).sort_values(["dataset", "scenario"])
    best.to_csv(out / "best_model_gain_table.csv", index=False)

    all_rows.groupby(["dataset", "scenario", "model"], as_index=False).agg(
        masked_mae_mean=("masked_mae", "mean"),
        masked_mae_std=("masked_mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
    ).to_csv(out / "full_metric_table.csv", index=False)

    return {
        "generality": generality,
        "scenario": scenario,
        "backbone_scenario": backbone_scenario,
        "best": best,
    }


def _plot_gain_heatmap(best: pd.DataFrame, out: Path) -> None:
    best = best.copy()
    for col in ["best_base_model", "best_plugin_model"]:
        if col in best.columns:
            best[col] = best[col].map(_display_name)
    pivot = best.pivot(index="dataset", columns="scenario", values="best_gain_pct")
    pivot = pivot[[s for s in SCENARIO_LABELS if s in pivot.columns]]
    fig, ax = plt.subplots(figsize=(5.9, 3.25))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [SCENARIO_LABELS.get(c, c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.tick_params(axis="x", pad=7)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Dataset")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}%", ha="center", va="center", color="black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Gain (%)")
    fig.tight_layout(pad=0.7)
    fig.savefig(out / "figure_best_gain_heatmap.png")
    fig.savefig(out / "figure_best_gain_heatmap.pdf")
    plt.close(fig)


def _plot_backbone_scenario(backbone_scenario: pd.DataFrame, out: Path) -> None:
    pivot = backbone_scenario.pivot(index="backbone", columns="scenario", values="gain_pct_mean")
    pivot = pivot[[s for s in SCENARIO_LABELS if s in pivot.columns]]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    x = np.arange(len(pivot.index))
    width = 0.24
    for offset, col in enumerate(pivot.columns):
        ax.bar(x + (offset - 1) * width, pivot[col], width, label=SCENARIO_LABELS.get(col, col))
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(x, pivot.index, rotation=18, ha="right")
    ax.set_ylabel("Gain over backbone (%)")
    ax.set_title("Backbone-agnostic improvement")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(pad=0.8)
    fig.savefig(out / "figure_backbone_scenario_gain.png")
    fig.savefig(out / "figure_backbone_scenario_gain.pdf")
    plt.close(fig)


def _plot_explainability(scenario: pd.DataFrame, out: Path) -> None:
    scenario = scenario.copy()
    scenario["label"] = scenario["scenario"].map(SCENARIO_LABELS).fillna(scenario["scenario"])
    x = np.arange(len(scenario))
    fig, ax1 = plt.subplots(figsize=(5.8, 2.9))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.18, scenario["gate_mean"], width=0.34, label="Gate mean", color=OKABE_ITO[0])
    ax1.bar(x + 0.18, scenario["failure_score_mean"], width=0.34, label="Failure score", color=OKABE_ITO[2])
    ax2.plot(x, scenario["delta_abs_mean"], marker="o", color=OKABE_ITO[1], label="|Delta| mean")
    ax1.set_xticks(x, scenario["label"], rotation=15, ha="right")
    ax1.set_ylabel("Gate / failure score")
    ax2.set_ylabel("Mean absolute correction")
    ax1.set_ylim(0, max(1.0, float(scenario[["gate_mean", "failure_score_mean"]].max().max()) * 1.15))
    ax1.set_title("Local reliability signals by scenario")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(pad=0.8)
    fig.savefig(out / "figure_explainability_signals.png")
    fig.savefig(out / "figure_explainability_signals.pdf")
    plt.close(fig)


def _write_summary(tables: dict[str, pd.DataFrame], complexity: pd.DataFrame, out: Path) -> None:
    with open(out / "paper_evidence_summary.md", "w", encoding="utf-8") as f:
        f.write("# PhyGuard Paper Evidence Tables\n\n")
        f.write("This folder is derived from the finalized 5 datasets x 3 scenarios x 3 seeds paired experiment.\n")
        f.write("Paper-facing tables use the original model names; implementation-strength labels remain internal run identifiers only.\n\n")
        f.write("## Complexity\n\n")
        f.write(complexity.to_string(index=False))
        f.write("\n\n## Generality\n\n")
        f.write(tables["generality"].assign(backbone=tables["generality"]["backbone"].map(_display_name)).to_string(index=False))
        f.write("\n\n## Explainability by scenario\n\n")
        f.write(tables["scenario"].to_string(index=False))
        f.write("\n\n## Best baseline vs best PhyGuard variant\n\n")
        best = tables["best"].copy()
        for col in ["best_base_model", "best_plugin_model"]:
            best[col] = best[col].map(_display_name)
        f.write(best.to_string(index=False))
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/phyguard_plugin_strong_backbones_5x3_3seed_final")
    parser.add_argument("--output-dir", default="results/phyguard_paper_evidence")
    parser.add_argument("--feature-dim", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = Path(__file__).resolve().parents[1] / input_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()

    all_rows, pairs = _load_pairs(input_dir)
    tables = _write_tables(all_rows, pairs, output_dir)
    complexity = _write_complexity(output_dir, args.hidden_dim, args.feature_dim)
    _plot_gain_heatmap(tables["best"], output_dir)
    _write_summary(tables, complexity, output_dir)
    print(f"wrote paper evidence to {output_dir}")
    print(tables["generality"].to_string(index=False))
    print(tables["scenario"].to_string(index=False))
    print(complexity.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
