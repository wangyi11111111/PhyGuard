# PhyPro Frozen Method Configuration

Frozen date: 2026-06-05

This document fixes the current PhyPro mainline before the next full experiment
run. The purpose is to stop structure changes during the following evaluation
cycle and keep the experimental evidence interpretable.

## Method Name

**PhyPro**: Physics-Reliability Promoted Correction.

The method treats physics as local promotion evidence rather than a fixed global
constraint. A data-driven correction proposal provides the main reconstruction
capacity, while local physics, temporal, graph, and failure evidence decide
where an additional physics-aligned promotion should be applied or suppressed.

## Frozen Inference Form

```text
x_hat = x0
      + gate(i,t) * Delta_generic(i,t)
      + beta(i,t) * Delta_physics_aligned(i,t)
```

Where:

- `x0` is the backbone reconstruction.
- `Delta_generic` is a data-driven residual proposal.
- `gate(i,t)` keeps the generic correction active while allowing mild local
  reliability conditioning.
- `Delta_physics_aligned` is a graph-temporal consistency promotion direction.
- `beta(i,t)` controls whether physics-aligned promotion is activated.

## Frozen Architecture

Implementation entry point:

```text
reproduce/run_plugin_baseline_comparison.py
```

Frozen module:

```text
ReliabilityConditionedPlugin
```

Default configuration:

```text
hidden_dim = 99
correction_clip = 0.20
gate_floor = 0.95
generic_features = [base, local_missing, temporal gaps, graph_delta,
                    temporal_delta, observed_error_proxy]
reliability_features = all 12 local evidence features
Delta_physics_aligned = 0.6 * graph_delta + 0.4 * temporal_delta
```

Conflict suppression:

```text
conflict = clip(
    0.35 * failure_score
  + 0.35 * physics_residual_rank
  + 0.20 * observed_error_proxy
  + 0.10 * spatial_gap_rank,
  0, 1
)

beta = beta_raw * (1 - 0.75 * conflict)
```

The default `gate_floor` was selected by a small sensitivity check on PEMS08
with `random_missing_50` and `incident_perturbation`, two backbones, and seed 1.
The tested pairs were `(0.90, 0.75)`, `(0.95, 0.75)`, `(0.90, 0.50)`, and
`(0.95, 0.50)` for `(gate_floor, conflict_coef)`. The `(0.95, 0.75)` setting
gave the lowest mean PhyPro MAE in this check.

Training loss:

```text
L = reconstruction_loss
  + 0.05 * utility_gate_loss
  + 0.05 * promotion_utility_loss
  + conflict_aware_harm_loss
  + delta_shrinkage_loss
```

## Frozen Main Comparison Protocol

The default plug-in comparison should use:

```text
Backbone
Backbone + GenericAdapter
Backbone + CalibrationGuard
Backbone + FailureAnomalyGuard
Backbone + PhyPro
```

Default backbones for the next quick run:

```text
SAITS
MagiNet
```

Default datasets and scenarios:

```text
PEMS03, PEMS04, PEMS08, PEMS-BAY, METR-LA
random_missing_50, sensor_failure_30, incident_perturbation
```

Default first run:

```text
seed = 1
epochs = 20 or 30 for quick validation
plugin_epochs = 20 or 30 for quick validation
```

If the single-seed trend is stable, the same frozen method should be rerun with
three seeds. Method structure and hyperparameters should not be changed between
the single-seed and three-seed runs unless the single-seed result clearly fails
and a new frozen version is declared.

## Reporting Notes

Use `PhyPro` in paper-facing text. Avoid internal version names such as
`PhyGuardRC`, `V12`, or `promotion quick`.

The central claim should be:

```text
PhyPro turns physics from a fixed global constraint into a local reliability
promotion signal: physics promotes correction when local evidence is reliable
and is suppressed when local conflict suggests physical misguidance.
```
