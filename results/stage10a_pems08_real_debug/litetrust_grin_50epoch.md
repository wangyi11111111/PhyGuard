# LiteTrustGRIN 50-Epoch Debug Run

LiteTrustGRIN replaces the TCN reconstruction backbone with a GRIN-style bidirectional graph recurrent imputer, while preserving calibrated physics residuals and trust-weighted physics loss.

Data: real PEMS08 from ASTGNN zip, first 20 nodes, train/val/test samples 64/16/16.

## Comparison

| Scenario | GRINLite 50epoch | LiteTrustGRIN 50epoch | Current TCN LiteTrust 10epoch |
|---|---:|---:|---:|
| random_missing_50 | 0.893680 | 0.923003 | 1.040161 |
| sensor_failure_30 | 1.445032 | 1.487384 | 1.745230 |

## Physics and Trust

| Scenario | physics_residual | trust_mean | trust_std |
|---|---:|---:|---:|
| random_missing_50 | 0.449389 | 0.114975 | 0.108159 |
| sensor_failure_30 | 0.258571 | 0.245196 | 0.315374 |

## Interpretation

The backbone replacement is the right direction: LiteTrustGRIN is far stronger than the TCN-based LiteTrust. It is still behind GRINLite on masked MAE, which means the physics/trust losses are constraining the imputer before the reconstruction backbone fully converges.

The next structural method change should be two-stage training:

1. Pretrain the GRIN-style imputer with reconstruction losses only.
2. Enable calibrated physics and trust regularization after the imputer is already competitive.

This is a method change, not a hyperparameter-only tweak.
