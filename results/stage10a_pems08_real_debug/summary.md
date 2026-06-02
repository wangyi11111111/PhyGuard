# Stage 10A PEMS08 Real Debug

Real PEMS08 data was loaded directly from the ASTGNN zip:

`E:\ASTGNN-9c2e19b98c4cedf1f35214d8789685b6381b3aad.zip`

Data entry:

`ASTGNN-9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS08/PEMS08.npz`

Settings: PEMS08 only, first 20 nodes, two scenarios, three models, 10 epochs, train/val/test samples 64/16/16, CPU.

This run uses the corrected LiteTrust method:

- PEMS channel order: `flow, occupancy, speed`
- calibrated residual: `flow - alpha * occupancy * speed`
- trust gate extra features: temporal change, spatial deviation, local missing ratio, residual rank
- trust regularization: variance floor and high-conflict/low-conflict ranking loss from epoch 5

## Masked MAE

| Scenario | BaseTCN | FixedPhysics | LiteTrustPINN_full | Best |
|---|---:|---:|---:|---|
| random_missing_50 | 1.048213 | 1.048205 | 1.040161 | LiteTrustPINN_full |
| sensor_failure_30 | 1.748276 | 1.748282 | 1.745230 | LiteTrustPINN_full |

## Trust Statistics

| Scenario | trust_mean | trust_std |
|---|---:|---:|
| random_missing_50 | 0.374920 | 0.054123 |
| sensor_failure_30 | 0.443645 | 0.062284 |

## Full Rows

| Scenario | Model | MAE | RMSE | MAPE | Masked MAE | Physics residual | Trust mean |
|---|---|---:|---:|---:|---:|---:|---:|
| random_missing_50 | BaseTCN | 1.090079 | 1.624047 | 1.809226 | 1.048213 | 0.614179 |  |
| random_missing_50 | FixedPhysics | 1.089630 | 1.623619 | 1.808663 | 1.048205 | 0.612380 |  |
| random_missing_50 | LiteTrustPINN_full | 1.082374 | 1.616544 | 1.833887 | 1.040161 | 0.618427 | 0.374920 |
| sensor_failure_30 | BaseTCN | 1.621395 | 2.082538 | 2.467719 | 1.748276 | 0.614505 |  |
| sensor_failure_30 | FixedPhysics | 1.620608 | 2.081857 | 2.450078 | 1.748282 | 0.605253 |  |
| sensor_failure_30 | LiteTrustPINN_full | 1.643204 | 2.105353 | 2.608775 | 1.745230 | 0.739756 | 0.443645 |

## Data Status

- `real_data_used=true`
- `fallback_used=false`
- PEMS08 source shape after node/channel slice: `[17856, 20, 3]`
- adjacency was parsed from the ASTGNN PEMS08 CSV, not ring fallback
- These are still debug-subset results, not full PEMS08 benchmark numbers.
