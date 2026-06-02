# Physics Validity Gate Ablation

Scope: real PEMS08 data from the ASTGNN zip, first 20 nodes, seed 1, 30 epochs. This is a method diagnostic, not a paper benchmark.

## Compared Variants

- `ReliabilityRouter with validity`: the previous self-calibrated physics-validity gate.
- `ReliabilityRouter no validity`: same router and physics correction, but no validity damping.
- `No physics`: graph/data routing without the physics expert.

## Results

| Scenario | With validity | No validity | No physics | Best |
|---|---:|---:|---:|---|
| random_missing_50 | 0.918611 | 0.907277 | 0.999535 | no validity |
| sensor_failure_30 | 1.255010 | 1.255813 | 1.253872 | no physics |
| block_missing | 1.696104 | 1.695540 | 1.696935 | no validity |
| temporal_missing | 0.945727 | 0.944070 | 1.010718 | no validity |
| noise_random_missing | 0.909122 | 0.901495 | 1.001749 | no validity |
| incident_perturbation | 0.922968 | 0.908902 | 1.002917 | no validity |

## Reading

- The self-calibrated validity gate is not helping in the current implementation.
- It slightly helps suppress physics under sensor failure, but sensor failure is graph-dominant and should not be the main comparison point.
- In the important sparse/noisy/incident/temporal settings, validity damping makes the model too conservative and weakens the physics correction.
- The main method should therefore be `ReliabilityRouter + physics correction`, with the validity gate kept as an optional diagnostic module rather than a default component.

## Decision

Default `LiteTrustGRINReliabilityRouter(use_validity_gate=False)` from now on. The paper method should not claim self-calibrated validity as a core contribution unless it is redesigned.
