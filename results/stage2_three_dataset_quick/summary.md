# Stage 2 Three-Dataset Quick Results

This is an ultra-quick debug run. Real PEMS08, PEMS04, and METR-LA data are not loaded in the current project, so every row uses synthetic toy fallback.

`fallback_used=true` and `formal_result=false` for every result below.

Settings: 3 datasets, 2 scenarios, 3 models, 10 epochs, train/val/test samples 16/8/8, CPU.

## Masked MAE

| Dataset | Scenario | BaseTCN | FixedPhysics | LiteTrustPINN_full | Best |
|---|---|---:|---:|---:|---|
| PEMS08 | random_missing_50 | 0.394156 | 0.394213 | 0.398860 | BaseTCN |
| PEMS08 | sensor_failure_30 | 0.726454 | 0.726408 | 0.727739 | FixedPhysics |
| PEMS04 | random_missing_50 | 0.380936 | 0.380974 | 0.385331 | BaseTCN |
| PEMS04 | sensor_failure_30 | 0.727949 | 0.727964 | 0.728419 | BaseTCN |
| METR-LA | random_missing_50 | 0.376169 | 0.376135 | 0.379832 | FixedPhysics |
| METR-LA | sensor_failure_30 | 0.678084 | 0.677758 | 0.675579 | LiteTrustPINN_full |

## Trust Mean

| Dataset | Scenario | LiteTrust trust mean | LiteTrust trust std |
|---|---|---:|---:|
| PEMS08 | random_missing_50 | 0.480159 | 0.124503 |
| PEMS08 | sensor_failure_30 | 0.480145 | 0.122944 |
| PEMS04 | random_missing_50 | 0.479634 | 0.123462 |
| PEMS04 | sensor_failure_30 | 0.478637 | 0.120080 |
| METR-LA | random_missing_50 | 0.479974 | 0.121312 |
| METR-LA | sensor_failure_30 | 0.478164 | 0.113412 |

## Notes

- FixedPhysics is almost tied with BaseTCN in most fallback settings.
- LiteTrustPINN_full only wins on METR-LA sensor_failure_30 in this ultra-quick fallback run.
- Physics residual is normalized, so values near 1.0 are expected and not directly comparable to raw physical units.
- These are not publishable benchmark numbers.
