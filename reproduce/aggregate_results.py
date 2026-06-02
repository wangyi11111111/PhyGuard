from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PHYGUARD_MODEL = "PhyGuard"
EXTERNAL_MODELS = ["KNN", "GRINLite", "MagiNet", "SAITS", "BRITS", "ImputeFormer_PyPOTS"]


def _read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "masked_mae" not in df.columns:
        raise ValueError(f"{path} has no masked_mae column")
    if "seed" not in df.columns and "_seed" in path.parent.name:
        df["seed"] = int(path.parent.name.rsplit("_seed", 1)[1])
    return df


def _mean_std(values: pd.Series) -> str:
    vals = values.dropna().astype(float)
    if len(vals) == 0:
        return ""
    if len(vals) == 1:
        return f"{vals.iloc[0]:.4f}"
    return f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"


def _pivot(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.pivot_table(
        index=["dataset", "scenario", "seed"],
        columns="model",
        values="masked_mae",
        aggfunc="min",
    ).reset_index()
    df.columns.name = None
    for model in EXTERNAL_MODELS + [PHYGUARD_MODEL]:
        if model not in df.columns:
            df[model] = pd.NA
    df["best_external_mae"] = df[EXTERNAL_MODELS].min(axis=1)
    df["best_external"] = df[EXTERNAL_MODELS].idxmin(axis=1)
    df["gain_vs_best_%"] = (df["best_external_mae"] - df[PHYGUARD_MODEL]) / df["best_external_mae"] * 100.0
    df["win_best"] = df[PHYGUARD_MODEL] < df["best_external_mae"]
    return df


def _summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row["runs"] = len(group)
        row["PhyGuard"] = _mean_std(group[PHYGUARD_MODEL])
        row["Best external"] = _mean_std(group["best_external_mae"])
        row["gain_vs_best_%"] = f"{group['gain_vs_best_%'].mean():.2f}"
        row["wins"] = f"{int(group['win_best'].sum())}/{len(group)}"
        row["best_external_mode"] = group["best_external"].mode().iat[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate reproduction results.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [_read_summary(path) for path in input_root.rglob("summary.csv")]
    if not frames:
        raise FileNotFoundError(f"No summary.csv files found under {input_root}")

    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(output_dir / "all_rows.csv", index=False)
    pivot = _pivot(rows)
    pivot.to_csv(output_dir / "per_seed_pivot.csv", index=False)
    _summarize(pivot, ["scenario"]).to_csv(output_dir / "main_by_scenario.csv", index=False)
    _summarize(pivot, ["dataset"]).to_csv(output_dir / "main_by_dataset.csv", index=False)
    _summarize(pivot.assign(scope="overall"), ["scope"]).to_csv(output_dir / "main_overall.csv", index=False)
    print(f"Wrote aggregate tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

