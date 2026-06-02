# Final Strict Experiment Report

This report records the final anti-inflation experiment used to decide whether
PhyGuard is ready for paper submission.

## Final Protocol

- Datasets: PEMS03, PEMS04, PEMS08, PEMS-BAY, METR-LA.
- Scenarios: random_missing_50, incident_perturbation, sensor_failure_30.
- Seeds: 1, 2, 3.
- Samples: train/validation/test = 64/16/16 windows.
- Window length: 12.
- Anti-leakage split: raw time split first, then windowing.
- Gap between raw train/validation/test segments: 12 time steps.
- Window stride: 12.
- Temporal evidence bank: disabled for the main result.
- Final correction key: fixed to `RegionAmplitudeScaled@1.50`.
- Training: epochs = 5, guard_epochs = 20.
- Main metric: target-region masked MAE.
- External baselines in the main comparison: KNN, GRINLite, MagiNet, SAITS, BRITS.

For METR-LA, the Hugging Face source provides predefined windowed parquet splits.
The script subsamples timestamps by stride, but it does not reconstruct the
original continuous raw series. Therefore, PEMS03/04/08/PEMS-BAY are the cleaner
raw anti-leakage checks; METR-LA is a predefined-split robustness check.

## Main Result

| Scope | Runs | PhyGuard | Best external | Gain vs best | Wins |
|---|---:|---:|---:|---:|---:|
| Overall | 45 | 0.3481 +/- 0.2235 | 0.4739 +/- 0.1346 | +28.88% | 36/45 |

Paired tests over all 45 runs:

- paired t-test p = 4.26e-05
- Wilcoxon signed-rank p = 4.70e-04

## Scenario Breakdown

| Scenario | Runs | PhyGuard | Best external | Gain vs best | Wins |
|---|---:|---:|---:|---:|---:|
| random_missing_50 | 15 | 0.1820 +/- 0.0587 | 0.4220 +/- 0.1329 | +56.27% | 15/15 |
| incident_perturbation | 15 | 0.2123 +/- 0.0549 | 0.4404 +/- 0.1278 | +50.82% | 15/15 |
| sensor_failure_30 | 15 | 0.6499 +/- 0.0601 | 0.5593 +/- 0.1050 | -20.46% | 6/15 |

Paired tests:

- random + incident, n = 30: paired t-test p = 4.98e-15; Wilcoxon p = 1.86e-09.
- sensor failure, n = 15: paired t-test p = 1.90e-02; Wilcoxon p = 3.02e-02, but in the wrong direction.

## Dataset Breakdown

| Dataset | Runs | PhyGuard | Best external | Gain vs best | Wins |
|---|---:|---:|---:|---:|---:|
| METR-LA | 9 | 0.4029 +/- 0.1563 | 0.6168 +/- 0.0423 | +35.61% | 9/9 |
| PEMS-BAY | 9 | 0.3435 +/- 0.2083 | 0.6107 +/- 0.0546 | +45.92% | 9/9 |
| PEMS03 | 9 | 0.3524 +/- 0.3039 | 0.3929 +/- 0.0971 | +21.16% | 6/9 |
| PEMS04 | 9 | 0.3223 +/- 0.2196 | 0.3786 +/- 0.0935 | +21.39% | 6/9 |
| PEMS08 | 9 | 0.3193 +/- 0.2475 | 0.3705 +/- 0.0554 | +20.31% | 6/9 |

## Failure Rows

PhyGuard loses only in sensor_failure_30 on PEMS03, PEMS04, and PEMS08.

| Dataset | Seed | Best external | Best external MAE | PhyGuard MAE | Gain |
|---|---:|---|---:|---:|---:|
| PEMS03 | 1 | SAITS | 0.548576 | 0.767210 | -39.85% |
| PEMS03 | 2 | SAITS | 0.498160 | 0.740618 | -48.67% |
| PEMS03 | 3 | SAITS | 0.514398 | 0.762443 | -48.22% |
| PEMS04 | 1 | SAITS | 0.548466 | 0.650257 | -18.56% |
| PEMS04 | 2 | SAITS | 0.404972 | 0.583257 | -44.02% |
| PEMS04 | 3 | SAITS | 0.521200 | 0.607137 | -16.49% |
| PEMS08 | 1 | SAITS | 0.423660 | 0.653335 | -54.21% |
| PEMS08 | 2 | SAITS | 0.458398 | 0.655762 | -43.06% |
| PEMS08 | 3 | SAITS | 0.444991 | 0.637446 | -43.25% |

## Decision

The final strict experiment supports the following claim:

> PhyGuard provides a robust physics-guided correction mechanism for random
> sparse missingness and incident/disruption scenarios under strict anti-leakage
> evaluation.

The final strict experiment does not support the stronger claim:

> PhyGuard is uniformly better than strong temporal imputation baselines under
> sensor failure.

For submission, the method is defensible only if the paper narrows the central
claim to sparse random missingness and incident/disruption robustness, and treats
sensor failure as a limitation or an auxiliary setting where temporal
self-attention baselines remain stronger.

## Files

- Raw run output: `results/antileakage_final_5x3x3/summary.csv`
- Protocol record: `results/antileakage_final_5x3x3/protocol.json`
- Aggregated tables: `results/antileakage_final_5x3x3_tables/`
