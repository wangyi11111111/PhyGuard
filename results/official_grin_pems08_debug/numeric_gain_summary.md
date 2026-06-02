# Numeric Gain Summary

Protocol:

```text
official GRIN cache: 30 epochs
LiteTrust head: numeric-gain correction search
correction epochs: 1
dataset: PEMS08 debug, 20 nodes
scenarios: random_missing_50 / noise_random_missing / incident_perturbation
```

Paired baseline is `mu_data_masked_mae` from the same LiteTrust forward pass.

## Current Best Official Outputs

| Scenario | Official GRIN | Best LiteTrust output | Preset | Relative Gain |
|---|---:|---:|---:|
| random_missing_50 | 0.646528 | 0.638678 | router + target-only + clip=1.0 | +1.21% |
| noise_random_missing | 0.655596 | 0.643784 | router + target-only + clip=1.0 | +1.80% |
| incident_perturbation | 0.658474 | 0.638893 | verified + target-only + clip=1.0 | +2.97% |
| Average | 0.653532 | 0.640452 | scenario-selected official outputs | +2.00% |

The strongest single preset across all three scenarios is:

```text
--utility-router-correction --target-only-loss --correction-clip 1.0
```

| Scenario | Official GRIN | Single-Preset LiteTrust | Relative Gain |
|---|---:|---:|---:|
| random_missing_50 | 0.646528 | 0.638678 | +1.21% |
| noise_random_missing | 0.655596 | 0.643784 | +1.80% |
| incident_perturbation | 0.658474 | 0.640422 | +2.74% |
| Average | 0.653532 | 0.640962 | +1.92% |

## Earlier Generic-v3 Baseline

| Scenario | Generic-v3 1e | Generic-v3 3e | Better |
|---|---:|---:|---|
| random_missing_50 | 0.639490 | 0.639482 | 3e by 0.000008 |
| noise_random_missing | 0.645067 | 0.645448 | 1e |
| incident_perturbation | 0.641924 | 0.642087 | 1e |

Decision:

- Replace the old `generic-v3 correction, 1 epoch` preset with `utility-router + target-only + clip=1.0` as the current single-preset numeric baseline.
- If scenario-specific selection is allowed, use `verified + target-only + clip=1.0` for incident.
- The gain improved from the earlier `+1.74%` average to `+1.92%` as a single preset, or `+2.00%` with scenario-selected official outputs.
- This still does not reach the desired `5%`; the next numeric step must improve candidate selection / router supervision rather than only changing clip values.

Artifacts:

```text
C:/tmp/numeric_gain_router_targetonly_clip100_30p1
C:/tmp/numeric_gain_verified_targetonly_clip100_incident_30p1
```
