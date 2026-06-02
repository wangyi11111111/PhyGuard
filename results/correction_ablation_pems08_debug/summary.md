# Correction V1 Ablation on PEMS08 Debug

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, 50 epochs, seed 1. This is a method debug result, not a paper benchmark.

## Main MAE

| Scenario | Variant | Final masked MAE | vs GRINLite | Data-branch MAE | Correction gain vs data branch |
|---|---|---:|---:|---:|---:|
| random_missing_50 | GRINLite | 0.893680 | - | 0.893680 | 0.000000 |
| random_missing_50 | GRINLite + graph_delta | 0.893680 | 0.00% | 0.893680 | 0.000000 |
| random_missing_50 | GRINLite + trust * delta_phys | 0.796771 | +10.84% | 0.970345 | -0.173575 |
| random_missing_50 | Correction V1 full | 0.796771 | +10.84% | 0.970345 | -0.173575 |
| sensor_failure_30 | GRINLite | 1.445032 | - | 1.445032 | 0.000000 |
| sensor_failure_30 | GRINLite + graph_delta | 1.270554 | +12.07% | 1.445032 | -0.174477 |
| sensor_failure_30 | GRINLite + trust * delta_phys | 1.438463 | +0.45% | 1.477207 | -0.038743 |
| sensor_failure_30 | Correction V1 full | 1.308505 | +9.45% | 1.410982 | -0.102478 |

## Interpretability Outputs

| Scenario | Variant | Trust observed | Trust missing | Trust failed nodes | Trust normal nodes | graph_delta failed | delta_phys mean | delta_phys missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| random_missing_50 | GRINLite + trust * delta_phys | 0.563654 | 0.595353 | - | 0.579396 | - | 0.617923 | 0.556156 |
| random_missing_50 | Correction V1 full | 0.563654 | 0.595353 | - | 0.579396 | - | 0.617923 | 0.556156 |
| sensor_failure_30 | GRINLite + trust * delta_phys | 0.251896 | 0.630530 | 0.630530 | 0.251896 | 0.000000 | 0.330241 | 0.354626 |
| sensor_failure_30 | Correction V1 full | 0.240095 | 0.761414 | 0.761414 | 0.240095 | 0.145515 | 0.302309 | 0.349934 |

## Reading

- Random missing: `trust * delta_phys` is the useful component. `graph_delta` does nothing because there are no fully failed nodes. This supports the correction formulation for sparse random missing.
- Sensor failure: deterministic `graph_delta` is the strongest single component. Full Correction V1 improves over GRINLite by 9.45%, but is weaker than `GRINLite + graph_delta` by about 3.0% relative.
- Trust is not collapsed. In sensor failure, trust is much higher on failed/missing nodes than normal observed nodes. In this implementation, high trust means "use the learned correction more" rather than "blindly enforce physics".
- The current full method supports the disrupted-reconstruction claim better than the old PINN formulation, but it also shows a weakness: learned `trust * delta_phys` can dilute the clean graph_delta gain in sensor-failure cases.

## Method Implication

Do not present Correction V1 as simply "physics trust beats GRIN". The stronger claim is narrower:

1. Sparse random missing benefits from an adaptive residual correction.
2. Sensor failure benefits most from failed-node graph neighbor correction.
3. Trust provides explainable routing by assigning higher correction weight to missing/failed regions.
4. The next method change should make trust route between correction experts instead of only scaling `delta_phys`.
