from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_conflict_test import _json_default
from scripts.run_correction_ablation_pems08_debug import _train_variant
from scripts.run_stage10a_pems08_real_debug import EXTENDED_SCENARIOS, _config


VARIANTS = [
    "GRINLite",
    "ReliabilityRouter",
    "ReliabilityRouter_validity",
    "ReliabilityRouter_directional_physics",
    "ReliabilityRouter_no_physics",
]


def _write_outputs(rows: list[dict], epochs: int) -> None:
    output_dir = ROOT / "results" / "scenario_sweep_pems08_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PEMS08 Debug Scenario Sweep",
        "",
        "Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, seed 1. This is a scenario diagnostic, not a paper benchmark.",
        "",
        f"Epochs: `{epochs}`",
        "",
        "| Scenario | Variant | Masked MAE | Data MAE | Phys weight | Graph failed | Directional gate failed | Physics residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['variant']} | {row['masked_mae_final']:.6f} | "
            f"{row['masked_mae_data']:.6f} | {_fmt(row.get('phys_weight_mean'))} | "
            f"{_fmt(row.get('graph_weight_failed_node_mean'))} | "
            f"{_fmt(row.get('directional_phys_gate_failed_node_mean'))} | "
            f"{row['physics_residual']:.6f} |"
        )

    lines.extend(["", "## Best By Scenario", ""])
    for scenario in EXTENDED_SCENARIOS:
        group = [row for row in rows if row["scenario"] == scenario]
        best = min(group, key=lambda item: float(item["masked_mae_final"]))
        lines.append(f"- `{scenario}`: `{best['variant']}` with masked MAE `{best['masked_mae_final']:.6f}`.")

    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": EXTENDED_SCENARIOS, "variants": VARIANTS, "epochs": epochs}, f, sort_keys=False)


def _fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def main() -> None:
    config = _config()
    epochs = 30
    config["train"]["epochs"] = epochs
    config["method"]["pretrain_epochs"] = 0
    rows = []
    for scenario in EXTENDED_SCENARIOS:
        for variant in VARIANTS:
            print(f"running scenario sweep {scenario} {variant}", file=sys.stderr, flush=True)
            row, _logs = _train_variant(config, scenario, variant)
            rows.append(row)
    print(json.dumps({"rows": rows}, indent=2, default=_json_default))
    try:
        _write_outputs(rows, epochs)
    except OSError as exc:
        print(f"warning: failed to write scenario sweep outputs: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
