# Correction V2 Routed on PEMS08 Debug

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, 50 epochs, seed 1. This is still a method debug result, not a paper benchmark.

## Result Table

| Scenario | Variant | Final masked MAE | vs GRINLite | Data-branch MAE | Correction gain |
|---|---|---:|---:|---:|---:|
| random_missing_50 | GRINLite | 0.893680 | - | 0.893680 | 0.000000 |
| random_missing_50 | GRINLite + graph_delta | 0.893680 | 0.00% | 0.893680 | 0.000000 |
| random_missing_50 | LiteTrust delta only | 0.804221 | +10.01% | 0.978636 | -0.174415 |
| random_missing_50 | Correction V2 routed | 0.800921 | +10.38% | 0.969526 | -0.168605 |
| sensor_failure_30 | GRINLite | 1.445032 | - | 1.445032 | 0.000000 |
| sensor_failure_30 | GRINLite + graph_delta | 1.270554 | +12.07% | 1.445032 | -0.174477 |
| sensor_failure_30 | LiteTrust delta only | 1.413402 | +2.19% | 1.450142 | -0.036740 |
| sensor_failure_30 | Correction V2 routed | 1.262174 | +12.65% | 1.455520 | -0.193347 |

## Routing / Interpretability

| Scenario | Variant | data weight | graph weight | physics weight | failed-node graph weight | failed-node physics weight | graph_delta failed | delta_phys missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| random_missing_50 | LiteTrust delta only | 0.400349 | 0.000001 | 0.599649 | - | - | - | 0.503928 |
| random_missing_50 | Correction V2 routed | 0.326559 | 0.079022 | 0.594419 | - | - | - | 0.528284 |
| sensor_failure_30 | LiteTrust delta only | 0.655575 | 0.000002 | 0.344423 | 0.000004 | 0.692663 | 0.000000 | 0.354266 |
| sensor_failure_30 | Correction V2 routed | 0.601408 | 0.381198 | 0.017394 | 1.000000 | 0.000000 | 0.813785 | 0.005718 |

## Reading

- Random missing is handled by the residual/physics correction route: physics weight on missing points is high, and both delta-only and routed variants beat GRINLite by about 10%.
- Sensor failure is handled by a structural graph route: failed-node graph weight is exactly 1.0 and failed-node physics weight is 0.0. This matches the claim that the model should not blindly trust physics under local failure.
- Correction V2 routed beats GRINLite in both current debug scenarios and now slightly beats `GRINLite + graph_delta` in sensor failure.
- The method statement should shift from "trust-weighted physics loss" to "trust-aware correction routing": physical residual correction is one expert, graph-neighbor correction is another expert, and the model/rule hybrid chooses where each is reliable.

## Current Method Name

Suggested name: LiteTrust-GRIN-Router.

Main equation:

`x_hat = x_data + g_graph(i,t) * delta_graph + g_phys(i,t) * delta_phys`

where `g_graph + g_phys + g_data = 1`; complete sensor failure hard-routes to graph correction, while sparse random missing is learned through the residual correction expert.
