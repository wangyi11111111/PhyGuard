# Reproduction Scripts

This folder contains lightweight scripts for reproducing the PhyGuard paper
protocol. The scripts are intentionally conservative: they run one dataset or a
small subset by default and require explicit arguments for larger protocols.

## Smoke Test

```bash
python reproduce/run_smoke.py
```

## Quick Single-Seed Protocol

```bash
python reproduce/run_protocol.py \
  --datasets PEMS08 \
  --scenarios random_missing_50 incident_perturbation \
  --seeds 1 \
  --epochs 20 \
  --guard-epochs 120 \
  --output-root results/reproduce_quick
```

Add `--include-imputeformer` to run the PyPOTS ImputeFormer baseline with the
same dataset/scenario/seed list.

## Anti-Leakage Validation Protocol

Use this protocol before making paper claims. For raw PEMS series, it splits
the continuous time series first, inserts a gap between train/validation/test,
and only then creates windows. The default `--stride 12 --gap 12` avoids the
heavy adjacent-window overlap used by quick debug runs.

```bash
python reproduce/run_antileakage_protocol.py \
  --datasets PEMS08 METR-LA \
  --scenarios random_missing_50 incident_perturbation sensor_failure_30 \
  --seeds 1 \
  --epochs 5 \
  --guard-epochs 20 \
  --output-dir results/antileakage_validation
```

For METR-LA, the script uses the Hugging Face predefined train/validation/test
files and subsamples timestamps by `--stride`; the original raw continuous
series is not reconstructed from those windowed parquet files. Treat PEMS
strict results as the cleaner anti-leakage check.

## Full Paper Protocol

Run the full protocol only when datasets and baseline dependencies are ready:

```bash
python reproduce/run_protocol.py \
  --datasets PEMS03 PEMS04 PEMS08 PEMS-BAY METR-LA \
  --scenarios random_missing_50 sensor_failure_30 incident_perturbation noise_random_missing \
  --seeds 1 2 3 \
  --include-imputeformer \
  --output-root results/reproduce_full
```

## Aggregate Results

```bash
python reproduce/aggregate_results.py \
  --input-root results/reproduce_full \
  --output-dir results/reproduce_full_tables
```

The aggregator writes `all_rows.csv`, `per_seed_pivot.csv`,
`main_by_scenario.csv`, `main_by_dataset.csv`, and `main_overall.csv`.
