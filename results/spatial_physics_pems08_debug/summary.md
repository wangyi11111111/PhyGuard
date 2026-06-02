# Spatial Physics Expert Debug

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, 50 epochs, seed 1. This is a method-debug result, not a paper benchmark.

## Method Tried

Added `SpatialConservationPhysicsExpert`, a lightweight optional physics expert that propagates neighboring FD residuals into failed nodes. It is different from `graph_delta`: graph_delta copies neighbor state context, while spatial physics only contributes through the physics expert path as `spatial_phys_delta`.

The module is now optional and disabled by default because the first debug result is mixed.

## Results

| Scenario | Variant | Masked MAE | Notes |
|---|---|---:|---|
| random_missing_50 | ReliabilityRouter without spatial physics | 0.855408 | spatial gate inactive |
| random_missing_50 | ReliabilityRouter with spatial physics | 0.855646 | no benefit, because no node-level failure exists |
| random_missing_50 | ReliabilityRouter no physics | 0.914612 | removing physics hurts |
| sensor_failure_30 | ReliabilityRouter without spatial physics | 1.282861 | graph-dominant, weak local physics |
| sensor_failure_30 | ReliabilityRouter with spatial physics | 1.279316 | spatial physics helps slightly over no-spatial |
| sensor_failure_30 | ReliabilityRouter no physics | 1.278230 | graph-only still slightly better |

## Interpretability

| Scenario | spatial gate missing/failed | spatial delta missing/failed | phys weight missing/failed | graph weight failed |
|---|---:|---:|---:|---:|
| random_missing_50 | 0.000000 | 0.000000 | 0.605626 | - |
| sensor_failure_30 | 0.330623 | 0.161436 | 0.017473 | 0.969395 |

## Reading

- The new spatial physics expert is actually active on failed nodes: failed-node spatial gate is `0.330623`, and failed-node spatial delta is `0.161436`.
- It improves sensor_failure_30 versus the same router without spatial physics: `1.282861 -> 1.279316`.
- It still does not beat no-physics graph routing in this debug run: `1.279316` vs `1.278230`.
- This means the current spatial physics formulation is not strong enough to be a main contribution. It should remain optional until redesigned.

## Decision

Do not make spatial physics the default main method yet. The result supports a narrower diagnosis: sensor failure needs cross-node physics, but a simple neighbor-residual propagation is too weak and too close to graph smoothing. A stronger version should use direction-aware conservation or learned upstream/downstream flow balance.
