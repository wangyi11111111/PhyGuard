# LiteTrustGRIN Two-Stage 50-Epoch Debug Run

This run tests the method premise that physics should not be enforced before the imputer learns reconstruction.

Training:

1. Epochs 1-30: reconstruction-only GRIN-style pretraining.
2. Epochs 31-50: calibrated physics and trust/ranking regularization enabled.

## Results

| Scenario | GRINLite 50epoch | LiteTrustGRIN one-stage | LiteTrustGRIN two-stage |
|---|---:|---:|---:|
| random_missing_50 | 0.893680 | 0.923003 | 0.923208 |
| sensor_failure_30 | 1.445032 | 1.487384 | 1.492952 |

## Trust and Physics

| Scenario | physics_residual | trust_mean | trust_std |
|---|---:|---:|---:|
| random_missing_50 | 0.447956 | 0.165982 | 0.071621 |
| sensor_failure_30 | 0.259716 | 0.305830 | 0.116993 |

## Interpretation

Two-stage training did not close the masked-MAE gap to GRINLite. This suggests that the issue is not only the timing of physics loss. Directly applying physics regularization to the main prediction still trades reconstruction accuracy for physical consistency.

The next method change should decouple reconstruction from physics correction:

- main branch: GRIN-style imputation optimized for reconstruction;
- physics branch: predicts a correction or consistency score;
- trust gate: decides how much of the physics correction is allowed into the final prediction;
- physics loss: applies to the correction branch or consistency branch, not directly to the main imputer output.
