# LiteTrust-GRIN-ReliabilityRouter Debug

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, 50 epochs, seed 1. This is a method-debug result, not a paper benchmark.

## Method Change

This version replaces fixed failure priors with a lightweight reliability scorer. The scorer uses local evidence:

- observed ratio
- local missing ratio
- physics residual rank
- temporal change
- node missing ratio
- neighbor observed ratio
- low-residual evidence
- uncertainty proxy

It outputs data/graph/physics reliability weights. The coefficient map is fixed and monotonic; only a small bias vector is trainable, so the router stays lightweight and interpretable.

## Main Results

| Scenario | Variant | Masked MAE | Comment |
|---|---|---:|---|
| random_missing_50 | GRINLite | 0.893680 | baseline |
| random_missing_50 | LiteTrust delta only bounded | 0.817050 | strongest random-missing result so far |
| random_missing_50 | ReliabilityRouter + self-calibrated physics validity | 0.832526 | better than GRINLite and no-physics, close to signed-FD version |
| random_missing_50 | ReliabilityRouter no physics | 0.905522 | removing physics hurts |
| sensor_failure_30 | GRINLite | 1.445032 | baseline |
| sensor_failure_30 | GRINLite + graph_delta | 1.267702 | strong graph baseline |
| sensor_failure_30 | ReliabilityRouter + self-calibrated physics validity | 1.256050 | graph-dominant, still better than no-physics and graph-only |
| sensor_failure_30 | ReliabilityRouter no physics | 1.265352 | removing physics hurts slightly |

## Physics Contribution

| Scenario | Full ReliabilityRouter | No-physics | Physics gain |
|---|---:|---:|---:|
| random_missing_50 | 0.832526 | 0.905522 | +8.06% |
| sensor_failure_30 | 1.256050 | 1.265352 | +0.74% |

## Interpretability

| Scenario | phys weight mean | phys weight missing/failed | graph weight failed | delta_phys missing |
|---|---:|---:|---:|---:|
| random_missing_50 | 0.519217 | 0.609644 | - | 0.406617 |
| sensor_failure_30 | 0.138819 | 0.014439 | 0.974234 | 0.098932 |

| Scenario | physics validity mean | validity missing/failed |
|---|---:|---:|
| random_missing_50 | 0.564897 | 0.634525 |
| sensor_failure_30 | 0.299110 | 0.136263 |

## Reading

- The lightweight reliability router is computationally practical and removes the hard `0.85 / 0.05` rule from the main prediction path.
- Physics has measurable contribution versus the no-physics version in both scenarios, especially under random missing after suppressing graph routing outside node-failure cases.
- The physics expert now uses signed physical residuals and a lightweight FD projection term. Positive `q - rho*v` directly induces a negative flow correction, so the expert is no longer direction-agnostic.
- A hard closed-form projection was tested and rejected: it reduced physics residual but hurt random-missing MAE. The retained version uses residual-improvement regularization instead of forcing exact projection.
- A physics-validity gate now suppresses the physics expert when local evidence suggests residual correction is likely misleading. It is lightly self-calibrated on observed points by comparing data-expert and physics-expert errors.
- The graph expert dominates complete sensor failure, while physics is strongly down-weighted there.
- The random missing result is improved but still behind `LiteTrust delta only bounded` (`0.817050`). This means the reliability router is theoretically cleaner and now physics-centered, but the physical projection is still a weak first-order approximation.

## Current Judgment

This is a better methodological direction than the heavy risk router, but it is not yet an AAAI-level final method. The next improvement should evaluate the self-calibrated validity gate under noisy and incident perturbations, where physics misguidance should be more visible.
