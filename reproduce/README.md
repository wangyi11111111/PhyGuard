# Reproduction Scripts

This folder contains the scripts used for the final PhyPro experiments.

## Final Method

The main implementation is in:

```text
run_plugin_baseline_comparison.py
```

The paper-facing method is `PhyPro`. Internally, the final module class is
`ReliabilityConditionedPlugin`; old labels such as `PhyGuardRC` are development
artifacts and should not be used in final paper tables.

## Main Plug-in Comparison

Run a small version:

```bash
python reproduce/run_plugin_baseline_comparison.py \
  --datasets PEMS08 \
  --scenarios random_missing_50 \
  --seeds 1 \
  --backbones SAITSStrong MagiNetStrong \
  --epochs 20 \
  --plugin-epochs 20 \
  --train-samples 64 \
  --val-samples 16 \
  --test-samples 16 \
  --phypro-gate-floor 0.95 \
  --phypro-conflict-coef 0.75 \
  --output-dir results/phypro_smoke
```

The full paper protocol used:

```text
5 datasets x 3 scenarios x 3 seeds x 4 backbones
```

with backbones:

```text
BRITS, SAITS, ImputeFormer, MagiNet
```

## Ablation

Selected ablations for the manuscript are produced by:

```bash
python reproduce/run_phypro_ablation.py
```

The main-text ablation should keep only interpretable variants:

```text
Backbone
Generic correction only
Full PhyPro
w/o physics promotion
w/o residual/evidence bank
w/o conflict suppression
w/o failure evidence
```

## Visual Case

Generate the final PhyPro interpretability figure:

```bash
python reproduce/create_phypro_visual_case.py \
  --dataset PEMS-BAY \
  --scenario random_missing_50 \
  --seed 1 \
  --epochs 50 \
  --plugin-epochs 40 \
  --train-samples 64 \
  --val-samples 16 \
  --test-samples 16 \
  --phypro-gate-floor 0.95 \
  --phypro-conflict-coef 0.75 \
  --output-dir results/phypro_visual_case
```

Outputs:

```text
results/phypro_visual_case/figure_phypro_case.png
results/phypro_visual_case/figure_phypro_case.pdf
results/phypro_visual_case/case_arrays.npz
```

## Paper Evidence Files

Small CSV summaries used in the manuscript are under:

```text
results/phypro_paper_evidence/
results/phypro_missing_rate_robustness_quick/
```

Large raw experiment outputs are not required for the anonymous paper archive
unless the reviewer requests full logs.
