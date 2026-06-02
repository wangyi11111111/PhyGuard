from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.datasets import (
    _edge_csv_to_adjacency,
    _load_npz_array,
    _metrla_feature_columns,
    _normalize_splits,
    _window_time_series,
)
from losses.metrics import compute_metrics
from scripts.run_five_baselines_flow_quick import (
    BRITS,
    PYPOTS_IMPORT_ERROR,
    _run_grinlite,
    _run_knn,
    _run_pypots_model,
    _scenario_data,
)
from scripts.run_maginet_physics_guard_quick import (
    METHOD_NAME,
    ZENODO_TRAFFIC_URLS,
    _download_file,
    _edge_csv_to_adjacency_with_sensor_ids,
    _run_one_scenario,
)
from scripts.train import resolve_device


def _segment_length(samples: int, seq_len: int, stride: int) -> int:
    return seq_len + max(0, samples - 1) * stride


def _window_strict_raw_splits(
    series: np.ndarray,
    *,
    seq_len: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    stride: int,
    gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    train_len = _segment_length(train_samples, seq_len, stride)
    val_len = _segment_length(val_samples, seq_len, stride)
    test_len = _segment_length(test_samples, seq_len, stride)

    train_start = 0
    train_end = train_start + train_len
    val_start = train_end + gap
    val_end = val_start + val_len
    test_start = val_end + gap
    test_end = test_start + test_len
    if series.shape[0] < test_end:
        raise ValueError(
            "not enough raw timesteps for strict anti-leakage split: "
            f"need {test_end}, got {series.shape[0]}. "
            "Reduce samples or stride."
        )

    train_x = _window_time_series(series[train_start:train_end], seq_len, train_samples, stride=stride)
    val_x = _window_time_series(series[val_start:val_end], seq_len, val_samples, stride=stride)
    test_x = _window_time_series(series[test_start:test_end], seq_len, test_samples, stride=stride)
    metadata = {
        "split_policy": "raw_split_before_windowing",
        "seq_len": seq_len,
        "stride": stride,
        "gap": gap,
        "raw_boundaries": {
            "train": [train_start, train_end],
            "val": [val_start, val_end],
            "test": [test_start, test_end],
        },
        "cross_split_raw_overlap": False,
        "min_cross_split_gap": gap,
    }
    return train_x, val_x, test_x, metadata


def _load_strict_zenodo_pems(
    dataset_name: str,
    *,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    seq_len: int,
    stride: int,
    gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    urls = ZENODO_TRAFFIC_URLS[dataset_name]
    root = Path("C:/tmp/litetrust_data") / dataset_name.lower()
    npz_path = _download_file(urls["npz"], root / f"{dataset_name}.npz")
    csv_path = _download_file(urls["csv"], root / f"{dataset_name}.csv")
    txt_path = _download_file(urls["txt"], root / f"{dataset_name}.txt") if "txt" in urls else None
    series = _load_npz_array(npz_path)
    train_x, val_x, test_x, split_meta = _window_strict_raw_splits(
        series,
        seq_len=seq_len,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        stride=stride,
        gap=gap,
    )
    if txt_path is not None:
        adj = _edge_csv_to_adjacency_with_sensor_ids(csv_path, txt_path, int(series.shape[1]))
    else:
        adj = _edge_csv_to_adjacency(csv_path.read_text(encoding="utf-8"), int(series.shape[1]))
    metadata = {
        "dataset_name": dataset_name,
        "source": "zenodo_7816008",
        "data_path": str(npz_path),
        "adj_path": str(csv_path),
        "sensor_id_path": str(txt_path) if txt_path is not None else None,
        "series_shape": list(series.shape),
        "split_samples": [train_samples, val_samples, test_samples],
        **split_meta,
    }
    return train_x, val_x, test_x, adj, metadata


def _load_metrla_hf_split_strided(split_name: str, max_windows: int, stride: int) -> tuple[np.ndarray, dict]:
    parquet_path = hf_hub_download(repo_id="witgaw/METR-LA", repo_type="dataset", filename=f"{split_name}.parquet")
    feature_cols = _metrla_feature_columns()
    df = pd.read_parquet(parquet_path, columns=["node_id", "t0_timestamp", *feature_cols])
    df = df.sort_values(["t0_timestamp", "node_id"], kind="mergesort")
    timestamps = df["t0_timestamp"].drop_duplicates().tolist()[:: int(stride)][: int(max_windows)]
    samples: list[np.ndarray] = []
    for timestamp in timestamps:
        group = df[df["t0_timestamp"] == timestamp].sort_values("node_id", kind="mergesort")
        node_values = group[feature_cols].to_numpy(dtype=np.float32)
        seq_len = len(feature_cols) // 2
        sample = node_values.reshape(node_values.shape[0], seq_len, 2).transpose(1, 0, 2)
        samples.append(sample)
    if not samples:
        raise ValueError(f"no METR-LA windows were built for split {split_name!r}.")
    return np.stack(samples, axis=0).astype(np.float32), {
        "split": split_name,
        "parquet_path": parquet_path,
        "windows": len(samples),
        "timestamp_stride": stride,
        "first_timestamp": str(timestamps[0]) if timestamps else None,
        "last_timestamp": str(timestamps[-1]) if timestamps else None,
    }


def _load_strict_metrla_hf(
    *,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    train_x, train_meta = _load_metrla_hf_split_strided("train", train_samples, stride)
    val_x, val_meta = _load_metrla_hf_split_strided("val", val_samples, stride)
    test_x, test_meta = _load_metrla_hf_split_strided("test", test_samples, stride)
    adj_path = hf_hub_download(repo_id="witgaw/METR-LA", repo_type="dataset", filename="sensor_graph/adj_mx.npy")
    adj = np.asarray(np.load(adj_path, allow_pickle=True), dtype=np.float32)
    degree = adj.sum(axis=1, keepdims=True)
    adj = adj / np.clip(degree, 1.0, None)
    metadata = {
        "dataset_name": "METR-LA",
        "source": "huggingface:witgaw/METR-LA",
        "split_policy": "hf_predefined_split_with_timestamp_subsampling",
        "cross_split_raw_overlap": "not_reconstructed_from_hf_windows",
        "train_meta": train_meta,
        "val_meta": val_meta,
        "test_meta": test_meta,
        "adjacency_path": adj_path,
        "adjacency_shape": list(adj.shape),
    }
    return train_x, val_x, test_x, adj, metadata


def _load_antileakage_splits(dataset: str, args: argparse.Namespace):
    key = dataset.lower()
    if key in {"pems03", "pems04", "pems08"}:
        train_x, val_x, test_x, adj, metadata = _load_strict_zenodo_pems(
            key.upper(),
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            seq_len=args.seq_len,
            stride=args.stride,
            gap=args.gap,
        )
    elif key in {"metr-la", "metrla"}:
        train_x, val_x, test_x, adj, metadata = _load_strict_metrla_hf(
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            stride=args.stride,
        )
    else:
        raise ValueError(f"unsupported anti-leakage dataset: {dataset}")

    train_x, val_x, test_x, _scaler = _normalize_splits(train_x, val_x, test_x)
    return (
        train_x[:, : args.seq_len, ..., :1],
        val_x[:, : args.seq_len, ..., :1],
        test_x[:, : args.seq_len, ..., :1],
        np.asarray(adj, dtype=np.float32),
        metadata,
    )


def _external_rows(subset: list[dict]) -> list[dict]:
    return [row for row in subset if row.get("model") in {"KNN", "BRITS", "GRINLite", "MagiNet", "SAITS"}]


def _write_outputs(output_dir: Path, rows: list[dict], protocol: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# PhyGuard Anti-Leakage Validation\n\n")
        f.write("This run splits raw time before windowing where raw series are available. ")
        f.write("The goal is to check whether gains survive a stricter, anti-inflation protocol.\n\n")
        for key, value in protocol.items():
            if key != "datasets":
                f.write(f"- {key}: `{value}`\n")
        f.write("\n")
        for dataset in sorted({row["dataset"] for row in rows}):
            f.write(f"## {dataset}\n\n")
            for scenario in sorted({row["scenario"] for row in rows if row["dataset"] == dataset}):
                subset = [row for row in rows if row["dataset"] == dataset and row["scenario"] == scenario]
                ours = next((row for row in subset if row.get("model") == METHOD_NAME), None)
                externals = _external_rows(subset)
                best_external = min(externals, key=lambda row: row["masked_mae"]) if externals else None
                f.write(f"### {scenario}\n\n")
                if ours is not None and best_external is not None:
                    gain = (best_external["masked_mae"] - ours["masked_mae"]) / best_external["masked_mae"] * 100.0
                    f.write(f"- best external: `{best_external['model']}` `{best_external['masked_mae']:.6f}`\n")
                    f.write(f"- {METHOD_NAME}: `{ours['masked_mae']:.6f}`\n")
                    f.write(f"- gain vs best external: `{gain:+.2f}%`\n")
                    f.write(f"- selected: `{ours.get('safe_selected_key', ours.get('safe_selected', ''))}`\n")
                f.write("| Model | masked MAE | RMSE | MAPE |\n|---|---:|---:|---:|\n")
                for row in sorted(subset, key=lambda item: item.get("masked_mae", float("inf"))):
                    f.write(f"| {row['model']} | {row['masked_mae']:.6f} | {row['rmse']:.6f} | {row['mape']:.6f} |\n")
                f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a stricter anti-leakage PhyGuard protocol.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS08"])
    parser.add_argument("--scenarios", nargs="+", default=["random_missing_50", "incident_perturbation", "sensor_failure_30"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--guard-epochs", type=int, default=20)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--gap", type=int, default=12)
    parser.add_argument("--output-dir", default="results/antileakage_validation")
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["full", "no_physics_residual_bank", "no_temporal_evidence_bank"],
        help="Use no_temporal_evidence_bank for a stricter check that excludes the strongest temporal interpolation bank.",
    )
    parser.add_argument("--skip-pypots-baselines", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    protocol = {
        "protocol": "anti_leakage_raw_split_before_windowing",
        "epochs": args.epochs,
        "guard_epochs": args.guard_epochs,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "gap": args.gap,
        "datasets": args.datasets,
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "ablation": args.ablation,
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    metadata_by_run: dict[str, dict] = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"loading strict splits dataset={dataset} seed={seed}", flush=True)
            train_x, val_x, test_x, adj, metadata = _load_antileakage_splits(dataset, args)
            metadata_by_run[f"{dataset}_seed{seed}"] = metadata
            for scenario in args.scenarios:
                print(f"running anti-leakage {dataset} {scenario} seed={seed}", flush=True)
                train_obs, train_mask = _scenario_data(train_x, adj, scenario, seed)
                val_obs, val_mask = _scenario_data(val_x, adj, scenario, seed + 11)
                test_obs, test_mask = _scenario_data(test_x, adj, scenario, seed + 29)
                train = (train_x, train_obs, train_mask)
                val = (val_x, val_obs, val_mask)
                test = (test_x, test_obs, test_mask)

                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **_run_knn(train, val, test)})
                if not args.skip_pypots_baselines:
                    if BRITS is None:
                        raise RuntimeError(f"pypots import failed: {PYPOTS_IMPORT_ERROR}")
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
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
                rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, **_run_grinlite(train, val, test, adj, device, args.epochs)})
                for row in _run_one_scenario(
                    train,
                    val,
                    test,
                    adj,
                    device,
                    args.epochs,
                    args.guard_epochs,
                    seed,
                    scenario,
                    args.ablation,
                    dataset=dataset,
                ):
                    rows.append({"dataset": dataset, "seed": seed, "scenario": scenario, "ablation": args.ablation, **row})
                _write_outputs(output_dir, rows, {**protocol, "metadata": metadata_by_run})

    _write_outputs(output_dir, rows, {**protocol, "metadata": metadata_by_run})
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
