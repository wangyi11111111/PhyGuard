# Directional Conservation Physics Debug

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, 50 epochs, seed 1. This is a method-debug result, not a paper benchmark.

## Method Tried

Added `DirectionalConservationPhysicsExpert`, an optional physics expert for node-level sensor failure.

Unlike `graph_delta`, it does not directly copy neighbor states. It builds direction-aware physics features:

- inferred upstream/downstream adjacency
- incoming flow `q_in`
- outgoing flow `q_out`
- flow balance residual
- density balance residual
- speed consistency residual
- local FD residual

If the adjacency is symmetric, the expert falls back to a weak node-order direction so the code remains runnable on the current debug graph.

## Results

| Scenario | Variant | Masked MAE | Notes |
|---|---|---:|---|
| sensor_failure_30 | ReliabilityRouter default | 1.267713 | graph-dominant baseline with local physics |
| sensor_failure_30 | ReliabilityRouter directional physics | 1.264947 | best of this directional check |
| sensor_failure_30 | ReliabilityRouter no physics | 1.267197 | graph-only routing |
| sensor_failure_30 | Directional physics with forced graph-to-physics shift | 1.290321 | rejected |

## Interpretability

| Variant | directional gate failed | directional delta failed | directional residual failed | graph weight failed | phys weight failed |
|---|---:|---:|---:|---:|---:|
| directional physics | 0.382585 | 0.176402 | 0.827670 | 0.971415 | 0.016267 |
| forced shift | 0.409875 | 0.368612 | 0.819587 | 0.912785 | 0.075576 |

## Reading

- Directional conservation helps slightly over both default physics and graph-only routing.
- The expert is active on failed nodes: failed-node directional gate is `0.382585`.
- Its actual final influence is still small because the router keeps failed-node physics weight low at `0.016267`.
- Forcing graph weight into physics lowers residual but worsens MAE, so that variant is rejected.

## Decision

Keep `DirectionalConservationPhysicsExpert` optional and disabled by default. It is a better theoretical direction than neighbor-residual spatial physics, but current debug evidence is still too small for a main claim. The useful claim remains: exact or over-weighted conservation can misguide reconstruction, so conservation must be validity-controlled.
