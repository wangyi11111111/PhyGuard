# LiteTrust-GRIN-Correction V1 50-Epoch Debug Run

This version decouples reconstruction from physics:

- `mu_data`: GRIN-style data imputation branch
- `delta_phys`: physics correction branch
- `trust`: local gate
- `mu_final = mu_data + trust * delta_phys`

Physics is no longer directly forced onto the main imputer output.

Training:

1. Epochs 1-30: reconstruction pretraining.
2. Epochs 31-50: correction branch plus trust-aware calibrated physics.

## Comparison

| Scenario | GRINLite 50epoch | LiteTrustGRIN one-stage | LiteTrustGRIN two-stage | LiteTrust-GRIN-Correction V1 |
|---|---:|---:|---:|---:|
| random_missing_50 | 0.893680 | 0.923003 | 0.923208 | 0.812169 |
| sensor_failure_30 | 1.445032 | 1.487384 | 1.492952 | 1.434401 |

## Trust and Physics

| Scenario | physics_residual | trust_mean | trust_std |
|---|---:|---:|---:|
| random_missing_50 | 0.510374 | 0.570476 | 0.201533 |
| sensor_failure_30 | 0.297646 | 0.371705 | 0.193053 |

## Interpretation

Correction V1 is the first version that beats the strong GRIN-style baseline on both debug scenarios. This supports the revised method design: physics should be a trust-gated correction, not a direct constraint on the reconstruction backbone.

This is still a small PEMS08 debug split, so it is not a final benchmark claim.
