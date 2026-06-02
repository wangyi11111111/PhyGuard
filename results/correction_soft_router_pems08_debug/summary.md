# Correction V2 Soft Router With Bounded Physics

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, 50 epochs, seed 1. This is a method-debug result, not a paper benchmark.

## Why This Run

The previous hard router made `physics_weight=0` on failed sensors. That gave good numbers but weakened the method claim. This run tests whether a small bounded physics expert has any marginal value:

- `CorrectionV2_soft_router_bounded`: failed sensors use graph prior `0.85`, physics prior `0.05`, data prior `0.10`.
- `CorrectionV2_no_physics`: same graph route, but physics expert disabled.
- `CorrectionV2_hard_router_bounded`: failed sensors hard-route to graph, physics weight `0`.
- `delta_phys` is bounded with `tanh` and `correction_clip=1.0` to prevent a tiny physics weight from multiplying an exploding correction.

## Main Results

| Scenario | Variant | Final masked MAE | vs GRINLite |
|---|---|---:|---:|
| random_missing_50 | GRINLite | 0.893680 | - |
| random_missing_50 | LiteTrust delta only bounded | 0.817050 | +8.57% |
| random_missing_50 | Correction V2 soft router bounded | 0.824955 | +7.69% |
| random_missing_50 | Correction V2 no physics | 0.896808 | -0.35% |
| random_missing_50 | Correction V2 hard router bounded | 0.824708 | +7.72% |
| sensor_failure_30 | GRINLite | 1.445032 | - |
| sensor_failure_30 | GRINLite + graph_delta | 1.267702 | +12.27% |
| sensor_failure_30 | LiteTrust delta only bounded | 1.431178 | +0.96% |
| sensor_failure_30 | Correction V2 soft router bounded | 1.257571 | +12.97% |
| sensor_failure_30 | Correction V2 no physics | 1.277794 | +11.57% |
| sensor_failure_30 | Correction V2 hard router bounded | 1.261546 | +12.70% |

## Interpretability

| Scenario | Variant | graph weight failed | physics weight failed | graph_delta failed | delta_phys missing |
|---|---|---:|---:|---:|---:|
| sensor_failure_30 | Correction V2 soft router bounded | 0.850000 | 0.050000 | 0.848872 | 0.601788 |
| sensor_failure_30 | Correction V2 no physics | 0.850000 | 0.000000 | 0.840858 | 0.005520 |
| sensor_failure_30 | Correction V2 hard router bounded | 1.000000 | 0.000000 | 0.813785 | 0.006328 |

## Reading

- Random missing still needs the physics/residual correction expert. Disabling physics gives `0.896808`, worse than GRINLite; bounded delta-only gives `0.817050`.
- Sensor failure is graph-dominant, but the bounded soft physics expert is not useless: soft router reaches `1.257571`, better than no-physics `1.277794` and hard router `1.261546`.
- The correct claim is not "physics solves sensor failure." The better claim is: under sensor failure, graph correction is primary, while a bounded low-weight physics/residual correction gives a small marginal gain after the router prevents it from dominating.
- The method is now more defensible: it learns/uses different correction experts by failure type instead of blindly enforcing physics everywhere.

## Current Method Statement

LiteTrust-GRIN-Router reconstructs sparse/disrupted traffic states through trust-aware correction routing:

`x_hat = x_data + g_graph(i,t) * delta_graph + g_phys(i,t) * delta_phys`

For random sparse missing, `g_phys` is high and residual correction is useful. For complete sensor failure, `g_graph` is dominant and `g_phys` is kept small and bounded, avoiding physics misguidance while retaining limited residual correction capacity.
