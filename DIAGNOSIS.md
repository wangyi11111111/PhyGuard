# LiteTrust-PINN Diagnosis

## Status

Stage 7 initially failed, then passed after a focused protocol and conflict-score fix. Do not treat this as final robustness evidence yet; confirm with the Stage 9 single-dataset trend suite before scaling.

## Problems Found

1. Conflict regularization did not lower incident-region trust.
2. The current conflict score reduced normal-region trust more strongly than incident-region trust.
3. Incident-region MAE was not worse, but the intended trust behavior was wrong.

## Evidence

| Scenario | Model | Overall masked MAE | Incident MAE | Trust normal | Trust incident |
| --- | --- | ---: | ---: | ---: | ---: |
| random_missing_50 | V3 without conflict | 0.115506 | n/a | 0.273966 | n/a |
| random_missing_50 | V4 with conflict | 0.115610 | n/a | 0.087100 | n/a |
| incident_perturbation | V3 without conflict | 0.156647 | 0.854222 | 0.252953 | 0.656491 |
| incident_perturbation | V4 with conflict | 0.156631 | 0.852412 | 0.077534 | 0.686298 |

## Focused Retry

I tried the next minimal fix: conflict score = residual rank + observed temporal-change rank, rerunning only `incident_perturbation`.

| Model | Overall masked MAE | Incident MAE | Trust normal | Trust incident |
| --- | ---: | ---: | ---: | ---: |
| V3 without conflict | 0.156647 | 0.854222 | 0.252953 | 0.656491 |
| V4 rank + temporal conflict | 0.156653 | 0.853284 | 0.083556 | 0.676658 |

This retry still failed. Incident MAE improved slightly, but incident trust stayed much higher than normal trust.

## Final Fix

The successful minimal fix changed two things:

1. Incident protocol: use speed-only perturbation (`flow_drop_ratio=0.0`, `speed_drop_ratio=0.5`) so the toy incident actually creates physical inconsistency.
2. Conflict score: use only the high-anomaly tail from `rank(temporal_change + spatial_deviation)` multiplied by residual rank. This avoids globally suppressing trust in normal regions.

| Model | Overall masked MAE | Incident MAE | Trust normal | Trust incident |
| --- | ---: | ---: | ---: | ---: |
| V3 without conflict | 0.165967 | 1.029268 | 0.269839 | 0.324121 |
| V4 fixed conflict | 0.165968 | 1.029160 | 0.236556 | 0.229077 |

This passes the focused Stage 7 criterion on the test incident region: incident trust is lower than normal trust, and incident MAE does not worsen.

## Interpretation

V4 slightly improved incident-region MAE, but it failed the main criterion: incident trust should be lower than normal trust. In the current toy protocol, the residual-based conflict score appears to penalize broad normal regions more than the incident region. The incident region also has lower measured physics residual than normal, so raw residual magnitude is not a reliable conflict signal here.

## Suggested Fixes

1. Keep the speed-only incident protocol for conflict-aware testing, because the original flow+speed drop hid the physical inconsistency.
2. Keep temporal/spatial anomaly features in the trust gate.
3. Keep conflict regularization focused on the high-anomaly tail rather than applying it broadly.
4. Monitor validation behavior: in the quick run, test-side trust ordering passed, but validation-side incident trust was still higher than normal trust.
5. Confirm with the Stage 9 single-dataset trend suite before moving to multi-dataset experiments.

## Next Minimal Experiment

The next experiment can be Stage 9 single-dataset trend confirmation. Keep it small:

- toy or PEMS08 debug fallback
- `random_missing_50`, `sensor_failure_30`, `incident_perturbation`
- V0/V1/V2/V3/V4
- 20 epochs
- seed 1

Do not enter Stage 10 or baselines until Stage 9 passes.

## Stage Decision

Stage 7 is now acceptable as a quick trend gate. Stage 8 remains optional; the more useful next step is Stage 9 single-dataset trend confirmation.
