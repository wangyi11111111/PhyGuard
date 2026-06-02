# LiteTrust-GRIN-Correction V1 Final Debug Run

This is the current strongest debug version.

Method:

- GRIN-style data imputation branch: `mu_data`
- trust-gated physics correction branch: `trust * delta_phys`
- failed-sensor graph correction branch: `graph_delta = I_failed * (A @ mu_data - mu_data)`
- final prediction: `mu_final = mu_data + graph_delta + trust * delta_phys`

Physics is not blindly enforced on the reconstruction backbone.

## Results

| Scenario | GRINLite 50epoch | Correction V1 | Relative improvement |
|---|---:|---:|---:|
| random_missing_50 | 0.893680 | 0.804068 | 10.03% |
| sensor_failure_30 | 1.445032 | 1.320377 | 8.63% |

## Interpretability Signals

| Scenario | trust_mean | trust_std | physics_residual |
|---|---:|---:|---:|
| random_missing_50 | 0.583708 | 0.170985 | 0.518085 |
| sensor_failure_30 | 0.388887 | 0.277573 | 0.370257 |

Interpretation:

- Trust is not collapsed. The standard deviation is high enough to indicate local variation.
- Sensor failure has lower mean trust and higher trust dispersion than random missing, which matches the intended behavior: physics correction should be more selective under structured sensor outage.
- The graph correction branch is activated only for near-complete node failures through `node_missing_ratio > 0.9`; it is inactive for ordinary random missing.
- The physics correction remains trust-gated, while the failed-sensor graph correction acts as a structural imputation fallback.

This is still a 20-node PEMS08 debug result. It is not a full benchmark claim, but it satisfies the current debug target of more than 5% improvement over the GRIN-style baseline in both tested scenarios.
