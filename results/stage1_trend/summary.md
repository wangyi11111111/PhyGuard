# Stage 1 Single-Dataset Trend Summary

Dataset: `toy`

Scenarios: `random_missing_50`, `sensor_failure_30`, `incident_perturbation`

Models: `V0_BaseTCN`, `V1_FixedPhysics`, `V2_TrustPhysics`, `V3_TrustPhysics_Uncertainty`, `V4_ConflictAware_LiteTrust`

Training: 20 epochs, seed 1, hidden dim 32, batch size 16, CPU.

## Main Results

| Scenario | Best overall masked MAE | Best missing-region MAE | Best incident-region MAE |
|---|---:|---:|---:|
| random_missing_50 | V3, 0.115574 | V3, 0.115568 | n/a |
| sensor_failure_30 | V2, 0.370712 | V2, 0.370712 | n/a |
| incident_perturbation | V3, 0.165967 | V3, 0.165964 | V2, 1.002399 |

## Required Checks

- Overall MAE best model: V3 is best on random missing and incident; V2 is best on sensor failure.
- Missing-region MAE best model: V3 on random missing, V2 on sensor failure, V3 on incident missing mask.
- Incident-region MAE best model: V2, not V4. V4 is slightly better than V3 in incident region but not best overall.
- Fixed physics helpful: yes, but the effect is small. V1 improves over V0 on sensor failure and is nearly tied elsewhere.
- Trust physics better than fixed physics: yes on sensor failure and incident-region MAE, but random missing is weaker.
- Uncertainty helpful: yes. V3 gives the best overall masked MAE in random missing and incident scenarios.
- Conflict regularization lowers incident trust: yes. V4 trust mean normal is 0.236556 and incident is 0.229077.
- Trust collapse: no. Trust means stay away from exactly 0 or 1, and trust std is positive.
- Recommend entering next quick benchmark: yes, with caution that toy-data physics residual remains weak.

## Gate Criteria

| Criterion | Result |
|---|---|
| V4 beats V0 in at least two scenarios | pass |
| V4 beats V1 in incident or sensor failure | pass |
| trust mean not collapsed to 0 or 1 | pass |
| training loss stable | pass |
| no NaN | pass |
| training time acceptable | pass |

## Decision

Stage 9 passes as a single-dataset trend gate. The next step can be the Stage 10 three-dataset quick benchmark, but it should remain small and should not include seven baselines or multi-seed full runs yet.
