# GRINLite 50-Epoch PEMS08 Debug Baseline

This is a compact GRIN-style graph recurrent imputation baseline, not the official GRIN implementation.

Data: real PEMS08 from the ASTGNN zip, first 20 nodes, train/val/test samples 64/16/16.

## Results

| Scenario | GRINLite masked MAE | LiteTrust current masked MAE | Gap |
|---|---:|---:|---:|
| random_missing_50 | 0.893680 | 1.040161 | -0.146481 |
| sensor_failure_30 | 1.445032 | 1.745230 | -0.300199 |

Negative gap means GRINLite is better.

## Interpretation

GRINLite is much stronger than the current LiteTrust implementation on this debug split. The gap suggests the current method is limited by its reconstruction backbone and temporal imputation mechanism, not only by the physics trust gate.

The next method change should move LiteTrust from a TCN reconstruction head toward a GRIN-style recurrent imputation backbone while preserving calibrated physics and trust weighting.
