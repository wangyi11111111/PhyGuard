# Gated Generic-vs-Physics Expert Ablation

Scope: real PEMS08 debug from ASTGNN zip, first 20 nodes, seed 1.

Training protocol:

- official GRIN pretrain: 30 epochs
- correction/router training: 5 epochs
- scenarios: `random_missing_50`, `noise_random_missing`, `incident_perturbation`

## Compared Variants

- `Official GRIN`: official GRINet only.
- `Generic`: official GRIN + generic residual adapter.
- `Physics`: official GRIN + constrained physics correction expert.
- `Gated`: official GRIN + reliability gate over generic and physics experts.

## Results

| Scenario | Official GRIN | Generic | Physics | Gated |
|---|---:|---:|---:|---:|
| random_missing_50 | 0.682696 | 0.672306 | 0.676001 | 0.669968 |
| noise_random_missing | 0.683350 | 0.672115 | 0.673673 | 0.668720 |
| incident_perturbation | 0.679828 | 0.670578 | 0.674999 | 0.668542 |

## Relative Gain

| Scenario | Gated vs Official GRIN | Gated vs Generic | Gated vs Physics |
|---|---:|---:|---:|
| random_missing_50 | +1.86% | +0.35% | +0.89% |
| noise_random_missing | +2.14% | +0.51% | +0.74% |
| incident_perturbation | +1.66% | +0.30% | +0.96% |

## Interpretation

This is the first official-GRIN experiment where the improvement can be attributed to the trust-routing structure rather than to a generic post-processing adapter.

- `Generic` beats official GRIN, so post-GRIN correction is useful.
- `Physics` also beats official GRIN, but is weaker than generic in these quick runs.
- `Gated` beats both `Generic` and `Physics` in all three scenarios.

This supports the narrower LiteTrust claim:

```text
Physics alone is not universally best, and generic correction alone is also incomplete.
The useful mechanism is reliability routing between a data-driven correction expert and a physics-constrained expert.
```

The gain is still small, around `1.66%` to `2.14%` over official GRIN and below the desired `5%` target. This should be recorded as a promising structural signal, not a final paper-level result.

## Next Improvement

To push toward paper-level gains, the next structural step should make the gate region-aware:

- output expert utility by missing/noisy/incident region, not only global node-time features
- add a contrastive utility loss: gated expert should beat both generic and physics experts on the same masked points
- record where the gate chooses physics versus generic

## Contrastive Utility Gate Check

Implemented:

```text
err_generic = |x_generic - y|
err_phys    = |x_phys - y|
utility_target = sigmoid((err_generic - err_phys) / tau)
L_gate = BCE(phys_weight, utility_target)
L_harm = relu(|x_gated - y| - min(err_generic, err_phys))
```

Quick result on `random_missing_50`:

| Variant | Masked MAE |
|---|---:|
| Gated without contrastive utility | 0.669968 |
| Gated with contrastive utility | 0.669914 |

The contrastive loss is directionally positive but the gain is only `0.000054`, which is not meaningful. Do not expand this exact loss to all scenarios yet.

Next experiment should change the gate structure, not just the loss weight: make the gate region-aware and report where physics is chosen over generic correction.

## Region-Aware Gate Check

Implemented:

```text
Official GRIN
+ generic correction expert
+ physics correction expert
+ region-aware gate
```

The gate now receives explicit region evidence:

```text
observed ratio, local missing ratio, node-failure score,
neighbor-missing ratio, temporal-change score, spatial-deviation score,
physics residual rank
```

It also writes gate statistics by region:

- missing region: target mask
- noisy region: observed points with injected Gaussian noise
- incident region: local node-time perturbation mask

Quick result, same `30e GRIN pretrain + 5e correction/router` setting:

| Scenario | Official GRIN | Previous Gated | Region-Aware Gated |
|---|---:|---:|---:|
| random_missing_50 | 0.682696 | 0.669968 | 0.669702 |
| noise_random_missing | 0.683350 | 0.668720 | 0.668535 |
| incident_perturbation | 0.679828 | 0.668542 | 0.668249 |

Gate choice by region:

| Scenario / region | Physics weight | Generic weight | Interpretation |
|---|---:|---:|---|
| random missing / missing | 0.530048 | 0.469952 | Slightly prefers physics on sparse unobserved points. |
| noisy / missing | 0.531199 | 0.468801 | Still leans physics where values are missing. |
| noisy / noisy observed | 0.404896 | 0.595104 | Prefers generic correction where observations are corrupted. |
| incident / all incident | 0.466888 | 0.533112 | Prefers generic correction in perturbed local regions. |
| incident / incident + missing | 0.528842 | 0.471158 | Uses more physics when the incident region is also unobserved. |

Interpretation:

The region-aware gate gives the right qualitative behavior: physics is used more on missing sparse points, while generic correction is selected more on noisy or locally perturbed observed points. This supports the local trust-routing story, but the quantitative gain is still small: only about `0.00019` to `0.00029` MAE over the previous gated version. This is useful as an interpretability mechanism, not yet a paper-level performance jump.

Next structural direction should strengthen the physics expert itself, because the gate is already learning reasonable local choices but the physics candidate is still weaker than the generic candidate in these quick runs.

## Physics Candidate V2

Implemented a stronger physics candidate instead of only adjusting the gate loss.

V1 physics expert:

```text
delta_phys = learned_delta + small_gain * FD_residual_delta
```

V2 physics expert:

```text
explicit_phys_delta =
  FD flow projection
  + graph projection for occupancy/speed
  + temporal smoothing projection for speed

delta_phys =
  learned_phys_delta
  + projection_strength * explicit_phys_delta
  + small_gain * FD_residual_delta
```

The final model still uses physics-generic fusion:

```text
x_final = (1 - w_phys) * x_generic + w_phys * x_phys
```

### V2 Results

Same setting: `30e official GRIN pretrain + 5e correction/router`.

| Scenario | Official GRIN | Old Physics Only | V2 Physics Only | Old Region-Aware Gated | V2 Physics-Guided Fusion |
|---|---:|---:|---:|---:|---:|
| random_missing_50 | 0.682696 | 0.676001 | 0.671009 | 0.669702 | 0.663284 |
| noise_random_missing | 0.683350 | 0.673673 | 0.671675 | 0.668535 | 0.663839 |
| incident_perturbation | 0.679828 | 0.674999 | 0.669710 | 0.668249 | 0.661747 |

### Relative Gain

| Scenario | V2 Fusion vs Official | V2 Fusion vs Old Region-Aware Gated | V2 Fusion vs V2 Physics Only |
|---|---:|---:|---:|
| random_missing_50 | +2.84% | +0.96% | +1.15% |
| noise_random_missing | +2.86% | +0.70% | +1.17% |
| incident_perturbation | +2.66% | +0.97% | +1.19% |

### V2 Gate Statistics

| Scenario / region | Physics weight | Generic weight |
|---|---:|---:|
| random missing / missing | 0.703211 | 0.296789 |
| noisy / missing | 0.703767 | 0.296233 |
| noisy / noisy observed | 0.596235 | 0.403765 |
| incident / all incident | 0.654108 | 0.345892 |
| incident / incident+missing | 0.706451 | 0.293549 |

Interpretation:

This is a real structural improvement. The physics-only candidate is now stronger than the old physics-only candidate in all three scenarios, and the fusion result is better than both official GRIN and V2 physics-only. The gain is still below the desired `5%` target, but the method claim is stronger because physics is now a useful candidate rather than just an auxiliary feature.

The current limitation is that V2 physics has become broadly useful, so the gate leans toward physics in noisy and incident regions too. This is acceptable for reconstruction accuracy, but the paper framing should emphasize adaptive fusion rather than a binary rule that noisy/incident regions must always reject physics.

## Channel-Wise Physics-Generic Fusion V3

Implemented two structural changes:

```text
1. channel-wise physics-generic fusion
   w_phys: [B, T, N, C]
   instead of one scalar weight shared by flow/occupancy/speed

2. component-wise physics projection
   delta_phys =
     alpha_fd       * FD flow projection
   + alpha_graph    * graph projection
   + alpha_temporal * temporal speed projection
   + learned_phys_delta
```

This makes the method more fine-grained: flow, occupancy, and speed no longer have to trust physics equally, and FD/graph/temporal physics components can have different local strengths.

### V3 Results

Same setting: `30e official GRIN pretrain + 5e correction/router`.

| Scenario | Official GRIN | V2 Physics Only | V3 Physics Only | V2 Fusion | V3 Channel-Wise Fusion |
|---|---:|---:|---:|---:|---:|
| random_missing_50 | 0.682696 | 0.671009 | 0.666338 | 0.663284 | 0.662996 |
| noise_random_missing | 0.683350 | 0.671675 | 0.666577 | 0.663839 | 0.662900 |
| incident_perturbation | 0.679828 | 0.669710 | 0.665267 | 0.661747 | 0.661534 |

### Relative Gain

| Scenario | V3 Fusion vs Official | V3 Fusion vs V2 Fusion | V3 Fusion vs V3 Physics Only |
|---|---:|---:|---:|
| random_missing_50 | +2.89% | +0.04% | +0.50% |
| noise_random_missing | +2.99% | +0.14% | +0.55% |
| incident_perturbation | +2.69% | +0.03% | +0.56% |

### V3 Region Behavior

| Scenario / region | Physics weight | Generic weight |
|---|---:|---:|
| random missing / missing | 0.600093 | 0.399907 |
| noisy / missing | 0.600415 | 0.399585 |
| noisy / noisy observed | 0.482380 | 0.517620 |
| incident / all incident | 0.547907 | 0.452093 |
| incident / incident+missing | 0.610640 | 0.389360 |

### V3 Physics Component Strength

| Scenario | FD strength | Graph strength | Temporal strength |
|---|---:|---:|---:|
| random_missing_50 | 0.522194 | 0.542859 | 0.516085 |
| noise_random_missing | 0.521141 | 0.543177 | 0.516668 |
| incident_perturbation | 0.522085 | 0.542907 | 0.516156 |

Interpretation:

V3 confirms that the physics candidate can be strengthened structurally: V3 physics-only improves over V2 physics-only in all three scenarios. The final fusion also improves over V2 fusion, but only slightly. The main value of V3 is methodological: it removes the overly coarse scalar fusion and gives each traffic variable and physics component its own local role.

The limitation is still clear: the best gain over official GRIN is about `2.7%` to `3.0%`, not the desired `5%`. The next improvement should not add more gate complexity. It should improve the correction target, especially by learning a residual-to-error mapping that optimizes reconstruction utility rather than only reducing physical residual.
