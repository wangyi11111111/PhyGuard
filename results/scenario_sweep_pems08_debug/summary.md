# PEMS08 Debug Scenario Sweep

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, seed 1, 30 epochs. This is a scenario diagnostic, not a paper benchmark.

## Scenarios

- `random_missing_50`: random sparse missing.
- `sensor_failure_30`: complete node-level sensor failure.
- `block_missing`: local spatial outage.
- `temporal_missing`: contiguous time-window missing.
- `noise_random_missing`: random missing plus noisy observations, clean target.
- `incident_perturbation`: random missing plus local speed perturbation, clean target.

## Main Table

| Scenario | GRINLite | ReliabilityRouter | Directional physics | No physics | Best |
|---|---:|---:|---:|---:|---|
| random_missing_50 | 0.983770 | 0.918611 | 0.918611 | 0.999535 | ReliabilityRouter |
| sensor_failure_30 | 1.446590 | 1.255010 | 1.254534 | 1.253872 | No physics |
| block_missing | 2.015473 | 1.696104 | 1.695704 | 1.696935 | Directional physics |
| temporal_missing | 1.005791 | 0.945727 | 0.945727 | 1.010718 | ReliabilityRouter |
| noise_random_missing | 0.984451 | 0.909122 | 0.909122 | 1.001749 | ReliabilityRouter |
| incident_perturbation | 0.982968 | 0.922968 | 0.922968 | 1.002917 | ReliabilityRouter |

## Physics Gain Versus No-Physics

| Scenario | Best physics-enabled | No physics | Relative gain |
|---|---:|---:|---:|
| random_missing_50 | 0.918611 | 0.999535 | +8.10% |
| sensor_failure_30 | 1.254534 | 1.253872 | -0.05% |
| block_missing | 1.695704 | 1.696935 | +0.07% |
| temporal_missing | 0.945727 | 1.010718 | +6.43% |
| noise_random_missing | 0.909122 | 1.001749 | +9.25% |
| incident_perturbation | 0.922968 | 1.002917 | +7.97% |

## Reading

- Physics is clearly useful in random missing, noisy random missing, temporal missing, and incident perturbation.
- Physics is not useful as the primary expert under complete sensor failure; graph routing already solves most of that case.
- Directional conservation has its largest relevance in node/block failures, but current gains are tiny: `sensor_failure_30` is slightly worse than no-physics, and `block_missing` gains only `0.07%`.
- Physics-enabled models often have a higher FD residual than no-physics in random/noise/incident, while MAE is lower. This supports the central claim that minimizing residual alone is not the goal; validity-controlled physical correction matters more than blindly lowering residual.
- The strongest current claim is not "physics solves every disruption." It is: physics helps sparse/noisy/temporal/local perturbation cases, while the router suppresses it when complete sensor failure is better handled by graph evidence.

## Decision

The scenario sweep supports keeping the current ReliabilityRouter as the main lightweight method. Directional conservation should remain optional until it shows larger gains on directed real graphs or with better upstream/downstream metadata.
