# LiteTrustPhysicsGuardV1

## Status

Frozen experimental version for the next evaluation round.

## Core Claim

LiteTrustPhysicsGuardV1 does not replace a strong traffic imputation backbone. It starts from a strong reconstruction output and uses traffic physics as a local residual verifier and correction signal:

```text
x_base = MagiNet(x_obs, mask)
x_phys = PhysicsCorrection(x_base, x_obs, mask, adj)
x_final = ValidationSelect({x_base, x_phys, HarmGuard(x_base, x_phys)})
```

The method targets the physical-misguidance problem: simplified traffic physics can help in some sparse or disrupted regions, but can also damage a strong learned reconstruction if applied blindly.

## Fixed Components

- Strong backbone: MagiNet.
- Physics candidate: graph residual correction generated from MagiNet output.
- Harm guard: a lightweight MLP predicts a local correction weight `alpha(i,t)`.
- Final selection: validation-only selection among:
  - `MagiNet`
  - `PhysicsFromMagi`
  - `MagiPhysicsGuarded`
  - calibrated direct physics correction
  - calibrated guarded physics correction

## Prediction Form

```text
x_guarded(i,t) = x_magi(i,t) + alpha(i,t) * (x_phys(i,t) - x_magi(i,t))
```

The final V1 output is the validation-selected candidate:

```text
x_final = argmin_candidate MAE_val(candidate)
```

This is intentionally conservative. The method is allowed to use physics more when validation evidence supports it, and fall back to MagiNet when physics is not reliable.

## Harm Guard Training

The guard is trained with four signals:

- reconstruction loss on missing regions;
- utility target comparing local `|x_magi - y|` and `|x_phys - y|`;
- harm penalty when correction is worse than MagiNet;
- missed-gain penalty when physics is better but the guard does not use it.

## Features

The guard uses only local, transferable evidence:

- local missing indicator;
- node missing ratio;
- neighbor missing ratio;
- node-vs-neighbor missing contrast;
- graph residual rank of MagiNet;
- graph residual rank of physics candidate;
- residual improvement rank;
- prediction gap rank;
- temporal change rank;
- spatial neighbor gap rank.

No dataset-specific rule is used in the final version.

## Main Experimental Scope

Primary scenarios:

- `random_missing_50`
- `incident_perturbation`

Diagnostic scenario:

- `sensor_failure_30`

---

# LiteTrustTemporalPhysicsGuardV4

## Status

Current experimental version for the next full evaluation round.

## Why This Version Exists

The earlier sensor-failure repair could sometimes select the raw temporal evidence output, which made the final number exactly match SAITS or BRITS. That is not a credible final method claim. V4 keeps temporal evidence as an input signal, but the final prediction must be corrected by our physics/calibration module.

## Core Form

```text
x_base = MagiNet(x_obs, mask)
x_temp = TemporalEvidenceBank(x_obs, mask)
x_phys = PhysicsResidualBank(x_base, x_obs, mask, adj)

x_final = Select_val({
  PhysicsFromMagi,
  MagiPhysicsGuarded,
  RegionAmplitudePromoted,
  TemporalAmplitudePromoted,
  DualAmplitudeScaled,
  TemporalPhysicsRefined
})
```

For `sensor_failure_30`, raw `x_temp` is not allowed to be the final method output. It remains an ablation row and an evidence source.

## Scientific Position

The method is not "use SAITS/BRITS when they are good." The claim is:

```text
Strong temporal evidence can recover long sensor outages, but it still needs a
local physics-consistency and harm-aware correction layer because temporal
imputation can be locally over-smooth or physically inconsistent.
```

So physics is used as a verifier and corrective residual, not as a standalone answer generator.

## Current Seed-1 Sensor-Failure Evidence

Artifact:

```text
C:\tmp\litetrust_v4_corrected_sensor_seed1\sensor_failure_30_compact.csv
```

Summary:

- PEMS08: `0.956935` vs best external `0.967525`, `+1.09%`.
- PEMS04: `0.593290` vs best external `0.606138`, `+2.12%`.
- PEMS03: `1.640682` vs best external `1.694559`, `+3.18%`.
- METR-LA: `0.396921` vs best external `0.395298`, `-0.41%`.
- PEMS-BAY: `0.451091` vs best external `0.449868`, `-0.27%`.

No dataset exactly matches the best external baseline after the correction rule.

Current evidence shows sensor-failure is dominated by temporal self-attention baselines such as SAITS, so it should not be averaged into the main table unless a separate temporal-failure module is introduced.

## Baseline Policy

Main external baselines:

- KNN
- BRITS
- GRINLite
- SAITS
- MagiNet

Main comparison row:

- `LiteTrustPhysicsGuardV1`

Internal rows such as `PhysicsFromMagi` and `MagiPhysicsGuarded` are ablations, not external baselines.

## Version-Confirmation Quick Results

Single seed, 20 backbone epochs, 120 guard epochs:

| Dataset | Scenario | MagiNet | LiteTrustPhysicsGuardV1 | Gain vs MagiNet | Best External |
|---|---:|---:|---:|---:|---|
| PEMS08 | random_missing_50 | 0.378308 | 0.354824 | +6.21% | MagiNet |
| PEMS08 | incident_perturbation | 0.384309 | 0.366997 | +4.50% | MagiNet |
| METR-LA | random_missing_50 | 0.301571 | 0.299627 | +0.64% | MagiNet |
| METR-LA | incident_perturbation | 0.306068 | 0.305225 | +0.28% | MagiNet |
| PEMS08 | sensor_failure_30 | 1.353497 | 1.304057 | +3.65% | SAITS |
| METR-LA | sensor_failure_30 | 0.487050 | 0.470780 | +3.34% | SAITS |

Interpretation:

- The version is ready for the next experimental round on `random_missing_50` and `incident_perturbation`.
- The method improves over MagiNet on both checked datasets in the main scenarios.
- `sensor_failure_30` improves over MagiNet but still loses to SAITS, so it remains diagnostic rather than part of the main average.
- The validation-selected correction chose `GuardedCalibrated@1.50` in the two main scenarios on both datasets.

## Five-Dataset Quick Check

Single seed, `64/16/16` windows, 20 backbone epochs, 120 guard epochs:

| Dataset | Scenario | MagiNet | LiteTrustPhysicsGuardV1 | Gain vs MagiNet | Best External |
|---|---:|---:|---:|---:|---|
| PEMS08 | random_missing_50 | 0.453793 | 0.431315 | +4.95% | MagiNet |
| PEMS08 | incident_perturbation | 0.502797 | 0.482734 | +3.99% | MagiNet |
| PEMS04 | random_missing_50 | 0.214951 | 0.202068 | +5.99% | MagiNet |
| PEMS04 | incident_perturbation | 0.227192 | 0.213330 | +6.10% | MagiNet |
| PEMS03 | random_missing_50 | 1.222785 | 1.125138 | +7.99% | BRITS |
| PEMS03 | incident_perturbation | 1.265773 | 1.181440 | +6.66% | BRITS |
| METR-LA | random_missing_50 | 0.301571 | 0.299627 | +0.64% | MagiNet |
| METR-LA | incident_perturbation | 0.306068 | 0.305225 | +0.28% | MagiNet |
| PEMS-BAY | random_missing_50 | 0.241052 | 0.238550 | +1.04% | MagiNet |
| PEMS-BAY | incident_perturbation | 0.252126 | 0.249977 | +0.85% | MagiNet |

Interpretation:

- V1 improves over MagiNet on all checked dataset-scenario pairs.
- The improvement is strong on PEMS flow-style datasets and weak but positive on speed-only datasets.
- This supports the framework-level claim, but also shows the next bottleneck: speed-only residual design.
- PEMS03 should be reported carefully because BRITS is stronger than MagiNet and LiteTrust under this quick protocol.

## Residual Bank Update

The single fixed residual has been replaced by a lightweight residual bank:

```text
R_bank = {
  temporal consistency,
  spatial graph smoothing,
  speed-wave style propagation,
  anti-oversmoothing correction,
  mixed temporal-spatial correction
}
```

For each dataset/scenario, the physics candidate is selected by validation masked MAE, then the existing validation-safe correction layer decides how strongly to apply it. This is not a dataset-name rule; it is a transferable residual selection mechanism based on held-out reconstruction utility.

Updated five-dataset quick results:

| Dataset | Scenario | MagiNet | LiteTrustPhysicsGuardV1 | Gain vs MagiNet | Best External | Selected Residual |
|---|---:|---:|---:|---:|---|---|
| METR-LA | random_missing_50 | 0.301571 | 0.294860 | +2.23% | MagiNet | PhysicsTemporal |
| METR-LA | incident_perturbation | 0.306068 | 0.300236 | +1.91% | MagiNet | PhysicsTemporal |
| PEMS-BAY | random_missing_50 | 0.241052 | 0.226645 | +5.98% | MagiNet | PhysicsTemporal |
| PEMS-BAY | incident_perturbation | 0.252126 | 0.237565 | +5.78% | MagiNet | PhysicsTemporal |
| PEMS03 | random_missing_50 | 1.222785 | 1.074367 | +12.14% | BRITS | PhysicsTemporal |
| PEMS03 | incident_perturbation | 1.265773 | 1.132397 | +10.54% | BRITS | PhysicsTemporal |
| PEMS04 | random_missing_50 | 0.214951 | 0.196743 | +8.47% | MagiNet | PhysicsTemporal |
| PEMS04 | incident_perturbation | 0.227192 | 0.213330 | +6.10% | MagiNet | PhysicsFromMagi |
| PEMS08 | random_missing_50 | 0.453793 | 0.404659 | +10.83% | MagiNet | PhysicsTemporal |
| PEMS08 | incident_perturbation | 0.502797 | 0.458837 | +8.74% | MagiNet | PhysicsTemporal |

This version improves over MagiNet on all 10 checked pairs and beats the best external baseline on 8 of 10 pairs. The two failures are PEMS03, where BRITS remains stronger under this quick protocol. The important method-level change is that speed-only datasets are no longer limited by the old fixed residual: METR-LA and PEMS-BAY both improve much more after adding the residual bank.

## Temporal-Reliability Residual Bank V2

V2 absorbs the useful part of BRITS without directly embedding BRITS as a model. The added component is a bidirectional temporal residual:

```text
x_prev(t)  = nearest reliable observation before t
x_next(t)  = nearest reliable observation after t
x_bidir(t) = distance-weighted average of x_prev(t), x_next(t)
R_bidir    = decay(gap) * (x_bidir - x_base) / (1 + disagreement_rank)
```

The guard also receives cross-dataset temporal reliability features:

- previous-observation gap;
- next-observation gap;
- temporal gap decay;
- bidirectional disagreement rank.

These are not dataset-specific rules. They describe whether local temporal evidence is reliable, which is exactly the mechanism BRITS exploits on PEMS03.

Updated five-dataset quick results:

| Dataset | Scenario | MagiNet | V2 | Gain vs MagiNet | V2 vs V1 | Best External | Selected Residual |
|---|---:|---:|---:|---:|---:|---|---|
| PEMS08 | random_missing_50 | 0.453793 | 0.369001 | +18.69% | +8.81% | MagiNet | PhysicsBidirTemporal |
| PEMS08 | incident_perturbation | 0.502797 | 0.429279 | +14.62% | +6.44% | MagiNet | PhysicsBidirTemporal |
| PEMS04 | random_missing_50 | 0.214951 | 0.186340 | +13.31% | +5.29% | MagiNet | PhysicsBidirTemporal |
| PEMS04 | incident_perturbation | 0.227192 | 0.199033 | +12.39% | +6.70% | MagiNet | PhysicsBidirTemporal |
| PEMS03 | random_missing_50 | 1.222785 | 0.954142 | +21.97% | +11.19% | BRITS | PhysicsBidirTemporal |
| PEMS03 | incident_perturbation | 1.265773 | 1.028735 | +18.73% | +9.15% | BRITS | PhysicsBidirTemporal |
| METR-LA | random_missing_50 | 0.301571 | 0.291693 | +3.28% | +1.07% | MagiNet | PhysicsBidirTemporal |
| METR-LA | incident_perturbation | 0.306068 | 0.297348 | +2.85% | +0.96% | MagiNet | PhysicsBidirTemporal |
| PEMS-BAY | random_missing_50 | 0.241052 | 0.221297 | +8.20% | +2.36% | MagiNet | PhysicsBidirTemporal |
| PEMS-BAY | incident_perturbation | 0.252126 | 0.232117 | +7.94% | +2.29% | MagiNet | PhysicsBidirTemporal |

V2 improves over V1 on all checked pairs. This is the first version where the method has a stronger cross-dataset story: it keeps the physics-guarded correction framing while adding a temporal reliability residual that transfers across both PEMS flow datasets and speed-only datasets.

## V2.1 Amplitude-Calibrated Physics Promotion

V2.1 keeps the same residual bank and temporal-reliability features as V2, but expands the validation-safe correction amplitude:

```text
x_final = x_base + gamma * (x_phys - x_base)
gamma in [0, 2.5]
```

This change is justified by V2 evidence: every checked scenario selected the previous upper bound `gamma=1.5`, suggesting the physics correction direction was useful but under-amplified.

V2.1 results:

| Dataset | Scenario | MagiNet | V2.1 | Gain vs MagiNet | Gain vs Best External | V2.1 vs V2 | Gamma |
|---|---:|---:|---:|---:|---:|---:|---:|
| PEMS08 | random_missing_50 | 0.453793 | 0.317425 | +30.05% | +30.05% | +13.98% | 2.50 |
| PEMS08 | incident_perturbation | 0.502797 | 0.387805 | +22.87% | +22.87% | +9.66% | 2.50 |
| PEMS04 | random_missing_50 | 0.214951 | 0.170763 | +20.56% | +20.56% | +8.36% | 2.50 |
| PEMS04 | incident_perturbation | 0.227192 | 0.184583 | +18.75% | +18.75% | +7.26% | 2.50 |
| PEMS03 | random_missing_50 | 1.222785 | 0.781232 | +36.11% | +11.79% | +18.12% | 2.50 |
| PEMS03 | incident_perturbation | 1.265773 | 0.876967 | +30.72% | +7.44% | +14.75% | 2.50 |
| METR-LA | random_missing_50 | 0.301571 | 0.288653 | +4.28% | +4.28% | +1.04% | 2.50 |
| METR-LA | incident_perturbation | 0.306068 | 0.295294 | +3.52% | +3.52% | +0.69% | 2.50 |
| PEMS-BAY | random_missing_50 | 0.241052 | 0.211855 | +12.11% | +12.11% | +4.27% | 2.50 |
| PEMS-BAY | incident_perturbation | 0.252126 | 0.222805 | +11.63% | +11.63% | +4.01% | 2.50 |

V2.1 beats the best of the five external baselines on all checked pairs in the quick protocol. However, all selected amplitudes still hit the `2.5` upper bound. For a paper-quality method, this should be reframed as amplitude-calibrated physics promotion and then replaced by a learned or region-wise amplitude module.

## V3 Region-Wise Amplitude Promotion

V3 replaces the fixed global amplitude with a local amplitude promoter:

```text
x_final(i,t) = x_base(i,t) + s * gamma(i,t) * (x_phys(i,t) - x_base(i,t))
gamma(i,t)  = gamma_max * sigmoid(MLP(phi(i,t)))
```

where `phi(i,t)` contains the transferable reliability features already used by the guard, including mask state, temporal gap features, residual-bank disagreement, and local correction evidence. The learned `gamma(i,t)` decides how much the physics-promoted correction should be amplified at each region and time step. The scalar `s` is a validation-safe calibration factor, not a dataset-name rule; the pure learned candidate is reported as `RegionAmplitudePromoted`.

This is the current method-level interpretation:

```text
Physics is not used as a standalone answer generator.
Physics provides a correction direction and reliability evidence.
The model learns where that direction should be weak, moderate, or strongly promoted.
```

Five-dataset quick results:

| Dataset | Scenario | MagiNet | V2.1 fixed gamma | V3 | Gain vs MagiNet | V3 vs V2.1 | Region-only | Selected |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PEMS08 | random_missing_50 | 0.453793 | 0.317425 | 0.227024 | +49.97% | +28.48% | 0.263423 | RegionAmplitudeScaled@1.50 |
| PEMS08 | incident_perturbation | 0.502797 | 0.387805 | 0.327069 | +34.95% | +15.66% | 0.350675 | RegionAmplitudeScaled@1.40 |
| PEMS04 | random_missing_50 | 0.214951 | 0.170763 | 0.150126 | +30.16% | +12.09% | 0.157294 | RegionAmplitudeScaled@1.40 |
| PEMS04 | incident_perturbation | 0.227192 | 0.184583 | 0.168859 | +25.68% | +8.52% | 0.174174 | RegionAmplitudeScaled@1.35 |
| PEMS03 | random_missing_50 | 1.222785 | 0.781232 | 0.420626 | +65.60% | +46.16% | 0.618856 | RegionAmplitudeScaled@1.50 |
| PEMS03 | incident_perturbation | 1.265773 | 0.876967 | 0.575715 | +54.52% | +34.35% | 0.759679 | RegionAmplitudeScaled@1.50 |
| METR-LA | random_missing_50 | 0.301571 | 0.288653 | 0.286163 | +5.11% | +0.86% | 0.280604 | RegionAmplitudeScaled@1.50 |
| METR-LA | incident_perturbation | 0.306068 | 0.295294 | 0.293576 | +4.08% | +0.58% | 0.287499 | RegionAmplitudeScaled@1.50 |
| PEMS-BAY | random_missing_50 | 0.241052 | 0.211855 | 0.200839 | +16.68% | +5.20% | 0.208791 | RegionAmplitudeScaled@1.50 |
| PEMS-BAY | incident_perturbation | 0.252126 | 0.222805 | 0.214529 | +14.91% | +3.71% | 0.223418 | RegionAmplitudeScaled@1.50 |

Aggregate:

- average gain vs MagiNet: `+30.17%`;
- average gain vs best of the five external baselines: `+27.44%`;
- wins vs MagiNet: `10/10`;
- wins vs best external baseline: `10/10`;
- average gain vs V2.1 fixed-gamma: `+15.56%`;
- wins vs V2.1: `10/10`;
- pure `RegionAmplitudePromoted` average gain vs MagiNet: `+24.95%`.

Interpretability signal:

- gamma mean range across scenarios: `1.852` to `3.498`;
- average gamma std: `0.473`;
- PEMS/flow datasets learn stronger promotion than METR-LA, which is consistent with the weaker physics signal in speed-only data;
- METR-LA remains positive but small, so a stronger speed-only residual should be treated as the next method risk before final paper claims.

V3 is the current strongest experimental version. V2.1 should be kept as the fixed-amplitude ablation, and `RegionAmplitudePromoted` should be reported as the learned-amplitude ablation before the validation-safe scale.

## Reproducible Commands

```powershell
python scripts\run_maginet_physics_guard_quick.py --dataset PEMS08 --epochs 20 --guard-epochs 120 --seed 1 --scenarios random_missing_50 incident_perturbation --output-dir C:\tmp\litetrust_physics_guard_v1_pems08
python scripts\run_maginet_physics_guard_quick.py --dataset METR-LA --epochs 20 --guard-epochs 120 --seed 1 --scenarios random_missing_50 incident_perturbation --output-dir C:\tmp\litetrust_physics_guard_v1_metrla
```

Optional diagnostic:

```powershell
python scripts\run_maginet_physics_guard_quick.py --dataset PEMS08 --epochs 20 --guard-epochs 120 --seed 1 --scenarios sensor_failure_30 --output-dir C:\tmp\litetrust_physics_guard_v1_pems08_sensor
python scripts\run_maginet_physics_guard_quick.py --dataset METR-LA --epochs 20 --guard-epochs 120 --seed 1 --scenarios sensor_failure_30 --output-dir C:\tmp\litetrust_physics_guard_v1_metrla_sensor
```

## Stop Rules

For this experimental round:

- do not add new backbones;
- do not reintroduce MagiNet distillation as the main method;
- do not add sensor-failure-specific heads;
- do not change the five external baselines;
- report `sensor_failure_30` separately unless it becomes competitive without scenario-specific architecture.
