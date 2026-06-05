# PhyPro

PhyPro is a lightweight plug-in framework for robust traffic state
reconstruction under sparse and disrupted sensing. It is developed for the
paper:

> **PhyPro: Physics-Reliability Promoted Correction for Robust Sparse and
> Disrupted Traffic State Reconstruction**

The central idea is to use physics as local reliability and promotion evidence,
not as a fixed global loss. Given a reconstruction backbone output, PhyPro
learns a bounded generic correction, constructs a physics-aligned correction
direction, and uses local reliability, failure, and conflict evidence to decide
where physics should promote or suppress the correction.

Code link:

```text
https://anonymous.4open.science/r/PhyPRO
```

## Method

PhyPro applies the following correction form:

```text
x_hat = x0 + g(i,t) * Delta_g(i,t) + beta(i,t) * Delta_p(i,t)
```

where:

- `x0` is the reconstruction from a strong backbone.
- `Delta_g` is a data-driven residual proposal.
- `Delta_p` is a physics-aligned direction from graph and temporal residuals.
- `g(i,t)` controls the generic correction.
- `beta(i,t)` controls whether physics promotes the correction.

Physics is therefore not a direct predictor and not a globally enforced
constraint. It supplies a local direction and a reliability signal.

## Evidence Summary

The final paper evidence evaluates PhyPro on:

- 5 datasets: `PEMS03`, `PEMS04`, `PEMS08`, `PEMS-BAY`, `METR-LA`
- 3 scenarios: `random_missing_50`, `sensor_failure_30`,
  `incident_perturbation`
- 3 seeds
- 4 backbones: `BRITS`, `SAITS`, `ImputeFormer`, `MagiNet`

Across 180 paired runs, PhyPro reduces masked MAE from `0.3804` to `0.3441`,
an average improvement of `10.58%` over the corresponding backbones.

Compared with plug-in baselines, PhyPro improves over:

- `Generic Adapter`: `1.58%`
- `Calibration Guard`: `3.46%`
- `Failure/Anomaly Guard`: `3.29%`

The paired tests are significant under both paired t-test and Wilcoxon tests.

## Paper Figures

![PhyPro mechanism](paper/figures/figure_phypro_mechanism.png)

**Figure 1.** PhyPro treats physics as local reliability and promotion evidence
after a strong reconstruction backbone, rather than as a fixed global physics
loss.

![PhyPro visual case A](paper/figures/figure_phypro_case_a.png)

![PhyPro visual case B](paper/figures/figure_phypro_case_b.png)

![PhyPro visual case C](paper/figures/figure_phypro_case_c.png)

**Figure 2.** Representative PEMS-BAY random-missing case. The panels show the
observation pattern, reconstruction error reduction, learned correction,
physics direction, promotion weight, and failure evidence.

## Main Tables

### Backbone + PhyPro

| Backbone | Base masked MAE ↓ | +PhyPro masked MAE ↓ | Average reduction ↑ |
| --- | ---: | ---: | ---: |
| BRITS | 0.3874 ± 0.1254 | **0.3433 ± 0.1296** | 12.80% |
| ImputeFormer | 0.3969 ± 0.2534 | **0.3585 ± 0.2328** | 10.24% |
| MagiNet | 0.3610 ± 0.2236 | **0.3375 ± 0.2169** | 7.76% |
| SAITS | 0.3762 ± 0.1150 | **0.3372 ± 0.1191** | 11.51% |

### Plug-in Comparison

| Setting | Backbone ↓ | Generic Adapter ↓ | Calibration Guard ↓ | Failure/Anomaly Guard ↓ | PhyPro ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 0.3804 ± 0.1881 | 0.3496 ± 0.1830 | 0.3556 ± 0.1843 | 0.3550 ± 0.1841 | **0.3441 ± 0.1805** |
| Random missing | 0.2643 | 0.2309 | 0.2375 | 0.2372 | **0.2261** |
| Sensor failure | 0.5930 | 0.5661 | 0.5712 | 0.5698 | **0.5583** |
| Incident perturbation | 0.2839 | 0.2517 | 0.2582 | 0.2582 | **0.2479** |

### Missing-rate Robustness

| Missing rate | Backbone ↓ | Generic Adapter ↓ | Calibration Guard ↓ | Failure/Anomaly Guard ↓ | PhyPro ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30% | 0.3173 ± 0.1124 | 0.2960 ± 0.1020 | 0.3019 ± 0.1050 | 0.3051 ± 0.1072 | **0.2815 ± 0.0974** |
| 50% | 0.3317 ± 0.1140 | 0.3097 ± 0.1035 | 0.3147 ± 0.1065 | 0.3160 ± 0.1078 | **0.2988 ± 0.1002** |
| 70% | 0.3583 ± 0.1164 | 0.3372 ± 0.1085 | 0.3422 ± 0.1109 | 0.3427 ± 0.1113 | **0.3307 ± 0.1080** |

### Complexity

PhyPro adds `23,365` trainable parameters for all datasets.

| Dataset | Nodes | Extra forward time (ms) ↓ |
| --- | ---: | ---: |
| PEMS03 | 358 | 2.52 |
| PEMS04 | 307 | 1.93 |
| PEMS08 | 170 | **0.99** |
| PEMS-BAY | 325 | 2.11 |
| METR-LA | 207 | 1.21 |

## Lightweight Overhead

PhyPro adds `23,365` trainable parameters. The measured extra forward time is
about `0.99-2.52 ms` per batch on the tested traffic graphs with batch shape
`16 x 12 x N x 1`.

## Key Files

```text
reproduce/run_plugin_baseline_comparison.py   final PhyPro plug-in comparison
reproduce/run_phypro_ablation.py              selected ablation experiments
reproduce/create_phypro_visual_case.py        final Figure 2 visual case
METHOD_PHYPRO_FROZEN.md                       frozen method configuration
paper/main.tex                                manuscript draft
paper/figures/figure_phypro_mechanism.png     mechanism figure
paper/figures/figure_phypro_case_a.png        interpretability case figure A
paper/figures/figure_phypro_case_b.png        interpretability case figure B
paper/figures/figure_phypro_case_c.png        interpretability case figure C
```

Paper evidence files:

```text
results/phypro_paper_evidence/phypro_significance_tests_masked_mae.csv
results/phypro_paper_evidence/phypro_complexity_table.csv
results/phypro_paper_evidence/phypro_region_explainability_minimal_no_gain.csv
results/phypro_paper_evidence/phypro_param_sensitivity_no_gain.csv
results/phypro_paper_evidence/phypro_ablation_selected.csv
results/phypro_missing_rate_robustness_quick/missing_rate_robustness_pivot_masked_mae.csv
```

## Quick Reproduction

Run a small plug-in comparison:

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

Generate the visual case used in the manuscript:

```bash
python reproduce/create_phypro_visual_case.py \
  --dataset PEMS-BAY \
  --scenario random_missing_50 \
  --seed 1 \
  --epochs 50 \
  --plugin-epochs 40 \
  --phypro-gate-floor 0.95 \
  --phypro-conflict-coef 0.75 \
  --output-dir results/phypro_visual_case
```

## Datasets

The code uses public traffic benchmark sources. Raw datasets are not
redistributed in this repository. Loaders document the public sources used for:

- `PEMS03`, `PEMS04`, `PEMS08`
- `PEMS-BAY`
- `METR-LA`

Users should follow the licenses and terms of the original dataset providers.

## Notes

Use `PhyPro` in paper-facing text. Internal names such as `PhyGuardRC`, `V12`,
or old `PhyGuard` experiment labels are historical development artifacts and
should not be used as final method names.
