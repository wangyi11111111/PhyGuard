# Official GRIN Repo Integration Check

Scope: official GRIN repository at `C:/Users/21329/grin_official_cache`.

Implemented inside the official repo:

- `lib/nn/models/litetrust_grin.py`
- export `LiteTrustGRINet` from `lib/nn/models/__init__.py`
- register `model_name='litetrust_grin'` in `scripts/run_imputation.py`

The official Lightning entry currently cannot run in this environment because `pytorch_lightning` / `torchmetrics` / `onnxruntime` hit a NumPy 2.x ABI error. Therefore, the quick check uses a lightweight PyTorch runner:

```text
scripts/run_official_repo_litetrust_quick.py
```

This runner directly imports the official repo classes:

```text
official_repo_grin        = lib.nn.models.GRINet
official_repo_litetrust   = lib.nn.models.LiteTrustGRINet
```

Dataset: real PEMS08 debug from the ASTGNN zip, first 20 nodes, seed 1.

## Training Protocol

Bad setting:

```text
joint training, 30 epochs
```

Result:

| Scenario | Official GRIN | LiteTrust joint | Result |
|---|---:|---:|---|
| random_missing_50 | 0.646528 | 0.656502 | worse |

Interpretation: adding correction/fusion from the start interferes with GRIN optimization.

Working setting:

```text
Official GRIN: 30 epochs
LiteTrust-GRIN: 30 epochs GRIN pretrain + freeze GRIN + 5 epochs correction/fusion
```

## Results

| Scenario | Official Repo GRIN | Official Repo LiteTrust-GRIN | Relative Gain |
|---|---:|---:|---:|
| random_missing_50 | 0.646528 | 0.640533 | +0.93% |
| noise_random_missing | 0.655596 | 0.649023 | +1.00% |
| incident_perturbation | 0.658474 | 0.644692 | +2.09% |

## Interpretation

The official-repo integrated LiteTrust head brings positive improvement in all three quick scenarios, but the gain is smaller than the earlier custom-pipeline V3 result (`+2.69%` to `+2.99%`).

This is still useful evidence:

- the gain does not disappear after moving the method into the official GRIN codebase;
- the strongest improvement appears in the disrupted incident scenario;
- two-stage training is necessary;
- the current implementation is not yet paper-level strong, because the official-repo gain is mostly around `1%`, not `5%`.

Next structural target:

```text
Train the correction/fusion head with an explicit utility objective after GRIN pretraining:
the head should optimize reconstruction improvement over frozen GRIN,
not only produce physically smoother candidates.
```

## Utility-Aware Correction Objective

Implemented:

```text
return_details=True in LiteTrustGRINet
```

The correction head can now expose:

```text
mu_data, x_generic, x_phys, final_delta, phys_weight
```

Added a utility-aware correction loss in the quick runner:

```text
L = L_reconstruction
  + 0.5 * SmoothL1(final_delta, clamp(y - mu_data))
  + 0.2 * relu(err_final - min(err_grin, err_generic, err_phys))
  + 0.05 * BCE(phys_weight, utility_target)

utility_target = sigmoid((err_generic - err_phys) / tau)
```

Purpose:

- train the head to correct frozen GRIN only when correction is useful;
- penalize harmful corrections;
- supervise fusion toward the locally better expert.

### Quick Check

Protocol:

```text
10e GRIN pretrain + freeze GRIN + 3e correction
random_missing_50
```

| Variant | Masked MAE |
|---|---:|
| Official GRIN 10e | 1.009418 |
| LiteTrust correction, no utility | 1.008032 |
| LiteTrust correction, utility-aware | 1.007394 |

Interpretation:

The utility-aware objective is directionally positive, but the gain over the no-utility correction is only `0.000638`. This is too small to justify a large experiment. The idea is valid as a diagnostic objective, but not yet a paper-level structural jump.

The 30e+5e utility run exceeded the local 20-minute timeout, so it was stopped. Do not expand this exact utility loss before making it cheaper or more targeted.

Next target:

```text
Replace generic utility regularization with a more direct residual-to-error correction head:
learn a calibrated error-correction map for frozen GRIN, then route physics only where it improves that map.
```

## Residual-to-Error Calibrator Trial

Implemented a new optional candidate inside `LiteTrustGRINet`:

```text
error_delta = Calibrator([mu_data, physics_residual, region_features])
x_error = mu_data + error_delta
x_final = (1 - w_phys) * x_error + w_phys * x_phys
```

The goal was to learn where frozen GRIN is wrong directly, instead of using a generic residual adapter.

Result on `random_missing_50`, quick protocol:

```text
10e GRIN pretrain + freeze GRIN + 3e correction
```

| Variant | Masked MAE |
|---|---:|
| Official GRIN 10e | 1.009418 |
| Previous LiteTrust no utility | 1.008032 |
| Residual-to-error calibrator, no utility | 1.017177 |
| Residual-to-error calibrator, utility-aware | 1.017980 |

Interpretation:

This is a failed structural direction in the current form. The direct residual-to-error calibrator over-corrects frozen GRIN and worsens reconstruction. The utility loss does not fix it; it slightly worsens the calibrator trial.

Decision:

- `use_error_calibrator` is now an explicit optional flag and defaults to `False`.
- The working default returns to generic-vs-physics fusion.
- Do not expand this calibrator unless it is redesigned with a safer residual correction parameterization, such as a shrinkage residual map:

```text
x_final = mu_data + gamma(i,t,c) * bounded_error_delta
gamma constrained near 0 at initialization
```

## Shrinkage Calibrator Fix

The failed calibrator was revised into a conservative residual refinement:

```text
x_error = x_generic + gamma(i,t,c) * bounded_error_delta
gamma = 0.3 * sigmoid(region_head(region_features))
gamma initialized near 0
```

This preserves the working generic candidate and only allows a small learned residual refinement.

Quick result on `random_missing_50`:

```text
10e GRIN pretrain + freeze GRIN + 3e correction
```

| Variant | Masked MAE |
|---|---:|
| Official GRIN 10e | 1.009418 |
| Previous LiteTrust no utility | 1.008032 |
| Failed direct calibrator + utility | 1.017980 |
| Shrinkage calibrator + utility | 1.008142 |

Interpretation:

The shrinkage fix solves the over-correction problem, recovering from `1.017980` to `1.008142`. However, it still does not beat the previous working LiteTrust head (`1.008032`). Therefore, the shrinkage calibrator remains optional and disabled by default.

Decision:

- keep `--use-error-calibrator` available for future diagnostics;
- keep default LiteTrust as generic-vs-physics fusion;
- do not use the shrinkage calibrator in the main result table unless later scenarios show a clear advantage.

## Scenario-Aware Fusion Trial

Implemented `Scenario-Aware Generic-vs-Physics Fusion` with an explicit scenario token for the three main table scenarios:

```text
random_missing_50      -> [1, 0, 0]
noise_random_missing   -> [0, 1, 0]
incident_perturbation  -> [0, 0, 1]
```

`sensor_failure_30` is intentionally excluded from the main scenario token set and should not be included in the main average.

First attempt:

```text
concatenate scenario token into generic / physics / gate heads
```

Result:

| Protocol | Scenario | Masked MAE |
|---|---|---:|
| 10e GRIN + 3e correction | random_missing_50 | 1.025068 |

This is worse than both official GRIN 10e (`1.009418`) and previous LiteTrust (`1.008032`). The cause is likely that direct concatenation changes all head input dimensions and initialization behavior.

Fix:

```text
zero-initialized scenario adapters
```

The default heads are preserved, and scenario-specific effects are added only as small residual adapters:

```text
generic_delta += scenario_generic_adapter(token)
phys_delta    += scenario_physics_adapter(token)
gate_logit    += scenario_gate_adapter(token)
projection    += scenario_projection_adapter(token)
```

All scenario adapters are zero-initialized, so the model is initially equivalent to the previous default.

Status:

- scenario-aware smoke passed;
- longer `10e+3e` and `5e+1e` quick checks exceeded local timeouts under the current CPU-contended environment;
- no complete positive result yet.

Decision:

- keep the zero-initialized scenario adapter implementation;
- do not report scenario-aware fusion as a result yet;
- next run should be done after clearing CPU contention, or with a cached GRIN checkpoint so only the correction head is trained.

## Cached Frozen-GRIN Workflow

Implemented cache support in:

```text
scripts/run_official_repo_litetrust_quick.py
```

New flags:

```text
--save-grin-cache
--load-grin-cache
--cache-dir
--litetrust-only
```

Purpose:

```text
1. train GRIN once
2. save frozen GRIN weights
3. reuse the same GRIN weights for default LiteTrust, scenario-aware LiteTrust, utility variants, and calibrator variants
4. train only correction/fusion heads
```

Smoke checks:

| Check | Command shape | Result |
|---|---|---|
| save cache | `--pretrain-epochs 1 --epochs 1 --save-grin-cache` | passed |
| load cache | `--pretrain-epochs 1 --epochs 1 --load-grin-cache` | passed |
| LiteTrust-only load | `--pretrain-epochs 1 --epochs 1 --load-grin-cache --litetrust-only` | passed |

Observed runtime under current CPU contention:

- save-cache smoke: about 150s
- load-cache smoke with GRIN baseline included: about 87s
- load-cache LiteTrust-only: about 19s

Recommended next command sequence after CPU contention is cleared:

```text
# cache GRIN once per main scenario
python scripts/run_official_repo_litetrust_quick.py \
  --pretrain-epochs 30 --epochs 0 --save-grin-cache \
  --scenarios random_missing_50 noise_random_missing incident_perturbation

# default LiteTrust correction only
python scripts/run_official_repo_litetrust_quick.py \
  --pretrain-epochs 30 --epochs 5 --load-grin-cache --litetrust-only \
  --scenarios random_missing_50 noise_random_missing incident_perturbation

# scenario-aware LiteTrust correction only
python scripts/run_official_repo_litetrust_quick.py \
  --pretrain-epochs 30 --epochs 5 --load-grin-cache --litetrust-only --scenario-aware \
  --scenarios random_missing_50 noise_random_missing incident_perturbation
```

Main average should use only:

```text
random_missing_50
noise_random_missing
incident_perturbation
```

Do not include `sensor_failure_30` in the main average.

## Cached 30e GRIN Main-Scenario Results

The empty-log failure in pure GRIN cache evaluation was fixed by skipping `train_log.csv` when no training epochs are run.

Pure frozen GRIN cache evaluation:

| Scenario | GRIN masked MAE |
|---|---:|
| random_missing_50 | 0.646528 |
| noise_random_missing | 0.655596 |
| incident_perturbation | 0.658474 |

Default LiteTrust correction-only result using the same frozen GRIN cache:

| Scenario | LiteTrust masked MAE | Gain vs GRIN |
|---|---:|---:|
| random_missing_50 | 0.640533 | 0.93% |
| noise_random_missing | 0.649023 | 1.00% |
| incident_perturbation | 0.644692 | 2.09% |
| main-scenario mean | 0.644749 | 1.34% |

Zero-initialized scenario-aware adapter using the same frozen GRIN cache:

| Scenario | Scenario-aware masked MAE | Gain vs GRIN |
|---|---:|---:|
| random_missing_50 | 0.640685 | 0.90% |
| noise_random_missing | 0.649245 | 0.97% |
| incident_perturbation | 0.644782 | 2.08% |
| main-scenario mean | 0.644904 | 1.32% |

Decision:

- scenario-aware adapter is executable, but it is slightly worse than default LiteTrust in all three main scenarios;
- do not use scenario-aware fusion as the main method in its current form;
- current official-repo LiteTrust evidence supports a modest improvement over GRIN, especially incident perturbation, but the main-scenario average gain is still below the desired 2% target;
- the next structural change should improve random/noise sparse reconstruction rather than adding more scenario tokens.

## Selective / Utility-Routed Correction Trial

New implementation switches added to the official-repo wrapper:

```text
--selective-correction
--physics-vetted-correction
--generic-only-correction
--utility-router-correction
--diagnostics
```

The diagnostic path now reports candidate MAE for:

```text
mu_data / x_generic / x_vetted / x_phys / x_fused / x_router / oracle_best
```

Main-scenario results with the same cached 30e GRIN backbone and 5e correction training:

| Method | Random | Noise | Incident | Mean gain vs GRIN |
|---|---:|---:|---:|---:|
| default | 0.640533 | 0.649023 | 0.644692 | 1.34% |
| selective_harm | 0.641283 | 0.650070 | 0.647179 | 1.12% |
| physics_vetted | 0.659235 | 0.670424 | 0.666385 | -1.81% |
| generic_only | 0.639645 | 0.645759 | 0.642016 | 1.69% |
| generic_error_calibrated | 0.639623 | 0.645684 | 0.641894 | 1.70% |
| utility_router | 0.640383 | 0.648423 | 0.644868 | 1.37% |

Utility-router interpretability:

| Scenario | GRIN weight | Generic weight | Physics weight | Fused weight | Harm rate vs GRIN |
|---|---:|---:|---:|---:|---:|
| random_missing_50 | 0.224 | 0.546 | 0.030 | 0.200 | 0.435 |
| noise_random_missing | 0.225 | 0.545 | 0.030 | 0.200 | 0.426 |
| incident_perturbation | 0.222 | 0.550 | 0.030 | 0.198 | 0.382 |

Decision:

- `selective_harm` reduced neither MAE nor average gain, so it is rejected as currently implemented.
- `physics_vetted` is rejected; a new residual-conditioned head is too unstable under 5e correction training.
- `generic_error_calibrated` is the best current numerical variant, with `1.70%` main-scenario average gain.
- `utility_router` is methodologically useful because it lowers harm rate and learns to suppress the weak direct physics candidate to about `3%` average weight, but it does not improve enough yet.
- The key evidence is that `oracle_best` remains far lower than all realized outputs, e.g. random `0.5542` versus router `0.6404`; the bottleneck is local candidate selection/calibration, not absence of an upper bound.

Next structural direction:

```text
Do not use direct physics candidate as a high-weight expert.
Use physics as a verifier/regularizer for correction utility, and improve the router toward the oracle gap.
```

## Physics-Verified Generic Correction Trial

Implemented:

```text
--physics-verified-correction
```

Prediction form:

```text
x_verified = x_grin + verifier_gate(i,t,c) * (x_generic_calibrated - x_grin)
```

The verifier does not output a physics candidate. It uses physics residual before/after the generic correction, residual improvement, missing-pattern features, and correction magnitude to decide whether to keep or suppress the learned generic correction.

Training signals:

```text
L_verify = BCE(verifier_gate, sigmoid((err_grin - err_generic) / tau))
L_harm   = relu(|x_verified - y| - |x_grin - y|)
L_phys_verify = relu(|R_verified| - |R_grin|)
```

Two versions were checked:

| Method | Random | Noise | Incident | Mean gain vs GRIN | Gate mean | Harm rate |
|---|---:|---:|---:|---:|---:|---:|
| generic_error_calibrated | 0.639623 | 0.645684 | 0.641894 | 1.70% | - | 0.440 |
| physics_verified_v1 | 0.640389 | 0.647325 | 0.644983 | 1.42% | 0.806 | 0.429 |
| physics_verified_v1b | 0.639830 | 0.646277 | 0.643353 | 1.58% | 0.949 | 0.434 |

Interpretation:

- Physics verification lowers harm rate slightly, so the direction is meaningful.
- It does not beat `generic_error_calibrated`; suppressing corrections based on residual evidence loses more MAE than it saves.
- Residual diagnostics show `residual_after_data` is often larger than `residual_before`, even when generic correction improves MAE. This is direct evidence that residual monotonicity is not aligned with task error in this setting.
- Current conclusion: physics should be used as a learned utility feature and explanation signal, not as a direct residual-decrease constraint.

Next method implication:

```text
Keep the generic calibrated correction as the prediction backbone.
Use physics residual features in a contrastive utility model, but remove or heavily weaken direct residual-decrease penalties.
The next useful target is not lower residual everywhere; it is better calibration of when generic correction improves MAE.
```

## Contrastive Utility Verifier Trial

Implemented:

```text
--contrastive-utility-verifier
```

This version uses the same prediction form as physics-verified correction:

```text
x_verified = x_grin + verifier_gate * (x_generic_calibrated - x_grin)
```

but removes direct residual-decrease supervision. Physics residuals are only utility features and diagnostics.

Two utility targets were tested:

```text
V1: soft target = sigmoid((err_grin - err_generic) / tau)
V2: hard target = 1[err_generic < err_grin]
```

Results:

| Method | Random | Noise | Incident | Mean gain vs GRIN | Gate mean | Harm rate |
|---|---:|---:|---:|---:|---:|---:|
| generic_error_calibrated | 0.639623 | 0.645684 | 0.641894 | 1.70% | - | 0.440 |
| physics_verified_v1b | 0.639830 | 0.646277 | 0.643353 | 1.58% | 0.949 | 0.434 |
| contrastive_utility_verifier_v1 | 0.639810 | 0.646233 | 0.643383 | 1.58% | 0.949 | 0.435 |
| contrastive_utility_verifier_v2 | 0.639810 | 0.646233 | 0.643382 | 1.58% | 0.949 | 0.435 |

Interpretation:

- Removing the residual-decrease penalty prevents the verifier from getting worse, but it still does not beat the generic calibrated correction.
- Hard contrastive labels do not change the result under the current 5e correction budget because the verifier gate remains close to its high initialization (`~0.949`).
- The current verifier mainly acts as a mild shrinkage layer over generic correction; it is not yet a strong local selector.

Decision:

- Do not claim the contrastive verifier as a successful performance module yet.
- The strongest empirical baseline remains `generic_error_calibrated`.
- The method claim should be narrowed: physics residual is useful as a diagnostic/utility feature, but current gains come mostly from calibrated correction on top of GRIN.
- To make the verifier central, the next structural step must improve gate learnability, likely by freezing the generic correction after training and training the verifier as a second-stage classifier, or by adding explicit region-balanced utility supervision.

## Two-Stage Utility Verifier Trial

Implemented:

```text
--two-stage-verifier
--verifier-epochs
--verifier-min-gate
```

Training protocol:

```text
Stage A:
  freeze GRIN
  train generic calibrated correction only

Stage B:
  freeze GRIN and generic correction
  re-initialize verifier head
  train only verifier_gate with balanced utility BCE
```

Verifier target:

```text
target = 1[err_generic < err_grin]
```

Balanced verifier loss:

```text
L = balanced_BCE(verifier_gate, target)
  + hard_negative_BCE
  + harm_loss
```

Results:

| Method | Random | Noise | Incident | Mean gain vs GRIN | Gate mean | Gate pos | Gate neg |
|---|---:|---:|---:|---:|---:|---:|---:|
| generic_error_calibrated | 0.639623 | 0.645684 | 0.641894 | 1.70% | - | - | - |
| two_stage_p1p1 | 0.638607 | 0.646040 | 0.646250 | 1.51% | 0.453 | 0.476 | 0.424 |
| two_stage_p5p1 | 0.643895 | 0.651794 | 0.651701 | 0.67% | 0.399 | 0.395 | 0.405 |
| two_stage_p5p5 | 0.643871 | 0.651863 | 0.651917 | 0.66% | 0.381 | 0.377 | 0.386 |
| two_stage_floor06_p1p1 | 0.638627 | 0.644920 | 0.643094 | 1.73% | 0.781 | 0.790 | 0.769 |
| two_stage_floor08_p1p1 | 0.638987 | 0.644911 | 0.642410 | 1.75% | 0.891 | 0.895 | 0.885 |
| two_stage_floor09_p1p1 | 0.639225 | 0.644969 | 0.642136 | 1.74% | 0.945 | 0.948 | 0.942 |

Interpretation:

- Two-stage training makes the verifier learnable: in the no-floor `1e+1e` check, gate on positive utility points is higher than on negative points (`0.476` vs `0.424`).
- Without a gate floor, the verifier over-suppresses useful correction and hurts incident reconstruction.
- With a conservative floor, the verifier becomes a light utility modulator instead of a hard suppressor.
- Best current variant is `two_stage_floor08_p1p1`, with main-scenario average gain `1.75%`, slightly above `generic_error_calibrated` (`1.70%`).
- This is a real structural improvement, but still below the desired `2%` average target.

Current method claim supported by data:

```text
Physics residual should not directly correct traffic states.
It is useful as part of a utility verifier that lightly modulates a learned correction.
Hard suppression is harmful; conservative, floor-bounded verification is more stable.
```

Next recommendation:

```text
Use two_stage_floor08_p1p1 as the current main LiteTrust variant.
Do not increase verifier epochs without a better calibration objective.
The next improvement should target the remaining oracle gap with better verifier features or explicit hard-negative mining, not stronger direct physics projection.
```

## Hard-Negative Utility Verifier V2

Implemented:

```text
--hard-negative-verifier
--hard-negative-margin
```

Training target:

```text
hard_negative = err_generic > err_grin + margin
target = 0 for hard_negative
target = 1 otherwise
```

The goal is to make the verifier block only clearly harmful corrections, instead of suppressing all cases where generic is marginally worse.

Results with `verifier_min_gate=0.8`, `1e` generic correction, and `1e` verifier:

| Method | Random | Noise | Incident | Mean gain vs GRIN | Hard-neg gate | Safe gate |
|---|---:|---:|---:|---:|---:|---:|
| two_stage_floor08_p1p1 | 0.638987 | 0.644911 | 0.642410 | 1.75% | - | - |
| hardneg_floor08_m002 | 0.638987 | 0.644917 | 0.642418 | 1.75% | 0.884 | 0.894 |
| hardneg_floor08_m005 | 0.638987 | 0.644917 | 0.642417 | 1.75% | 0.884 | 0.894 |

Interpretation:

- Hard-negative verifier behaves as intended: hard-negative gate is lower than safe gate.
- The numerical effect is almost identical to the previous floor-bounded verifier.
- Increasing the hard-negative margin from `0.02` to `0.05` does not materially change the outcome under one verifier epoch.
- This gives a cleaner explanation, but it does not move the average above the current `1.75%`.

Decision:

- Keep `two_stage_floor08_p1p1` as the current main result.
- Keep hard-negative diagnostics for interpretability, but do not claim it as an additional performance gain.
- The next performance bottleneck is likely the generic correction candidate, not verifier calibration.

## Generic Correction Head Redesign

Motivation:

```text
Verifier/gate variants are capped around 1.75% average gain.
The next structural bottleneck is the quality and harm rate of the generic correction candidate.
```

Implemented:

```text
--generic-v2-correction
--generic-v3-correction
--generic-v4-correction
```

Designs tested:

| Variant | Design | Decision |
|---|---|---|
| Generic V2 | Replace generic delta with local / graph / temporal branch mixture | Rejected: branch mixture diluted the effective local correction |
| Generic V3 | Keep generic/error-calibrated correction as main path, add bounded residual refinement | Keep as ablation: safe and slightly improves random, but mean gain is tiny |
| Generic V4 | Region-aware scale over generic/error-calibrated delta | Rejected: scale stays near 1.0 and does not improve MAE |

Main three-scenario results:

| Method | Random | Noise | Incident | Mean | Mean gain vs GRIN |
|---|---:|---:|---:|---:|---:|
| GRIN cached | 0.646528 | 0.655596 | 0.658474 | 0.653532 | - |
| two_stage_floor08_p1p1 | 0.638987 | 0.644911 | 0.642410 | 0.642103 | 1.75% |
| Generic V2, 1e | 0.641320 | 0.650095 | 0.651398 | 0.647604 | 0.91% |
| Generic V3 + utility, 1e | 0.639402 | 0.644994 | 0.641900 | 0.642099 | 1.75% |
| Generic V3 + utility, 3e | 0.639281 | 0.645276 | 0.642007 | 0.642188 | 1.74% |
| Generic V4 + utility, 1e | 0.639509 | 0.645115 | 0.641984 | 0.642203 | 1.73% |

Diagnostics:

- Generic V2 branch weights stayed close to uniform and hurt random missing.
- Generic V3 is conservative: `generic_v3_gain_mean` is about `0.025`, and `generic_v3_refine_abs_mean` is below `0.004` in the best 1e run.
- Generic V4 learned `generic_v4_scale_mean` around `0.997`, so it mostly recovered the original correction and added no useful structure.

Interpretation:

```text
The correction head should not be replaced by a multi-expert mixture unless the expert routing is strongly supervised.
The best new candidate is Generic V3 + utility loss, but its gain over two_stage_floor08_p1p1 is negligible.
This confirms that the current local correction is already close to the useful regime; the remaining oracle gap is mostly about detecting harmful correction regions, not adding more correction magnitude.
```

Current decision:

- Keep `two_stage_floor08_p1p1` as the main LiteTrust variant for now.
- Keep `Generic V3 + utility` as a correction-head ablation because it is lightweight and slightly improves random/noise.
- Do not use Generic V2 or Generic V4 as the main method.

## Physics-Informed Harm Verifier

Motivation:

```text
Generic correction is near its current useful limit.
The remaining problem is not stronger correction, but identifying where correction harms the frozen GRIN prediction.
```

Implemented:

```text
--physics-harm-verifier
--two-stage-harm-verifier
--harm-keep-min
--sparse-harm-verifier
--harm-threshold
--harm-temperature
```

Prediction form:

```text
x_generic = x_grin + delta_generic
harm_prob = H(region evidence, residual_before, residual_after_generic, delta evidence)
keep = 1 - harm_prob
x_final = x_grin + keep * delta_generic
```

For the sparse version:

```text
harm_block = sigmoid((harm_prob - threshold) / temperature)
keep = keep_min + (1 - keep_min) * (1 - harm_block)
```

Training target:

```text
hard_negative = 1[ |x_generic - y| > |x_grin - y| + margin ]
L_harm = BCE(harm_prob, hard_negative)
```

Results:

| Method | Random | Noise | Incident | Mean | Harm prob hard | Harm prob safe | Recall@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| generic/error-calibrated reference | 0.639488 | 0.645055 | 0.641910 | 0.642151 | - | - | - |
| harm floor08, 1e+1e | 0.639458 | 0.645275 | 0.642659 | 0.642464 | 0.454 | 0.446 | 0.08-0.09 |
| harm floor095, 1e+1e | 0.639514 | 0.645126 | 0.642097 | 0.642246 | 0.454 | 0.446 | 0.08-0.09 |
| sparse harm floor095, 1e+3e | 0.639468 | 0.645048 | 0.641999 | 0.642171 | 0.475 | 0.468 | 0.20-0.26 |

Interpretation:

- The harm verifier learns the intended direction: hard-negative regions receive higher `harm_prob` than safe regions.
- The separation is still weak: hard-negative and safe probabilities differ by only about `0.006-0.007`.
- Continuous shrinkage hurts incident because many useful corrections are also slightly suppressed.
- Sparse harm verification is safer and improves recall, but it still does not beat the generic/error-calibrated reference or the current `two_stage_floor08_p1p1` main result.

Decision:

- Keep the harm verifier implementation and diagnostics because it is methodologically aligned with the paper claim.
- Do not use the current harm verifier as the main result.
- Next structural fix should improve harmful-region features, not increase correction capacity:
  `local error proxy`, `neighbor disagreement`, `temporal residual jump`, and region-balanced hard-negative sampling by scenario.

## Physics-Informed Harm Verifier V2

Implemented evidence expansion:

```text
generic spatial disagreement
generic temporal disagreement
correction-vs-graph-prior gap
correction-vs-temporal-prior gap
residual increase rank
```

Implemented training controls:

```text
--harm-hard-weight
--harm-safe-weight
--harm-utility-target
```

Results:

| Method | Random | Noise | Incident | Mean | Harm hard | Harm safe | Recall@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 sparse, hard weight 2.0 | 0.639444 | 0.645057 | 0.642067 | 0.642189 | 0.538 | 0.544 | 0.83-0.85 |
| V2 sparse, balanced | 0.639441 | 0.645056 | 0.642062 | 0.642186 | 0.536 | 0.543 | 0.83-0.86 |
| V2 utility keep target | 0.641550 | 0.648706 | 0.648747 | 0.646335 | 0.526-0.530 | 0.530-0.536 | 0.74-0.82 |

Interpretation:

- Adding stronger local evidence improves recall, but the learned ordering is wrong: `harm_prob_safe` is still higher than `harm_prob_hard_negative`.
- Increasing hard-negative loss weight does not fix the ordering.
- Utility keep regression collapses toward excessive shrinkage (`keep_mean` around `0.47`) and harms all three scenarios.
- The best predictions observed in this run are still from the generic/error-calibrated candidate, not from the harm-filtered output.

Decision:

- Do not use the harm verifier as a direct prediction gate.
- Keep it only as an auxiliary diagnostic/regularizer candidate.
- The next method revision should make physics verifier regularize the generic correction representation or loss, not directly shrink the output.

Updated method direction:

```text
Physics as utility regularizer:
  train generic correction normally
  train a verifier to detect harmful corrections
  use verifier loss/features to shape correction learning
  final output remains the calibrated generic correction unless verifier confidence is very high
```

## Harm-Regularized Generic Correction

Implemented:

```text
--harm-regularized-correction
```

This mode keeps the final prediction as calibrated generic correction:

```text
x_final = x_grin + delta_generic_calibrated
```

The physics-informed harm head is used only during training:

```text
L = L_mae
  + 0.35 * L_harm_bce
  + 0.75 * relu(|x_final - y| - |x_grin - y| - margin)
  + 0.03 * harm_prob_detached * |delta|
```

Results:

| Method | Random | Noise | Incident | Mean | Decision |
|---|---:|---:|---:|---:|---|
| two_stage_floor08_p1p1 | 0.638987 | 0.644911 | 0.642410 | 0.642103 | current main |
| harm_regularized, 1e | 0.639455 | 0.645029 | 0.641855 | 0.642113 | safe but not better |
| harm_regularized, 3e | 0.639520 | 0.645674 | 0.642451 | 0.642549 | over-regularized |

Interpretation:

- Moving harm verification from output gating to training regularization is safer.
- The 1e version improves incident over `two_stage_floor08`, but loses slightly on random/noise.
- Longer training over-regularizes the correction and degrades all main scenarios.
- Harm probabilities remain near their low initialization in the 1e run, so most of the effect comes from the explicit GRIN-relative harm penalty rather than a strong learned verifier.

Decision:

- Keep `--harm-regularized-correction` as an auxiliary method branch.
- Do not replace `two_stage_floor08_p1p1` as the main result.
- The useful component is the GRIN-relative harm penalty; the learned harm classifier still needs a better target or a separate calibration set.

## Two-Stage Binary Region-Balanced Verifier

Implemented:

```text
--two-stage-harm-verifier
--harm-utility-target
```

This version freezes the generic calibrated correction first, then trains the verifier as a second-stage utility classifier with region-balanced hard/safe supervision. The final prediction remains the generic correction; the verifier is diagnostic / auxiliary.

Best short check:

| Method | Random | Harm Prob Hard | Harm Prob Safe | Keep Hard | Keep Safe | Verifier Candidate |
|---|---:|---:|---:|---:|---:|---:|
| 5e verifier | 1.749483 | 0.532269 | 0.506712 | 0.467731 | 0.493288 | `x_harm_verified_masked_mae 1.763682` |

Interpretation:

- The verifier does move in the right direction after a few epochs.
- The separation is still small, and the verifier candidate is worse than the generic correction.
- This is useful as a utility regularizer / diagnostic head, not as a direct output gate.

Decision:

- Keep the verifier auxiliary.
- Do not route final prediction through `x_harm_verified`.
- The next useful step is a better utility target or a calibration set with clearer hard/safe labels.

## Quantile-Balanced Verifier Sweep

This version uses batch-wise quantiles to define hard/safe regions, plus explicit ranking losses on both `harm_prob` and `harm_keep`.

Results:

| Scenario | Harm Hard | Harm Safe | Keep Hard | Keep Safe | Note |
|---|---:|---:|---:|---:|---|
| random_missing_50 | 0.524178 | 0.504913 | 0.475822 | 0.495087 | weak separation |
| sensor_failure_30 | 0.749544 | 0.490486 | 0.250456 | 0.509514 | clear separation |
| incident_perturbation | 0.522750 | 0.504900 | 0.477250 | 0.495100 | weak separation |

Interpretation:

- The verifier is not uniformly useful.
- It becomes informative on sensor failure, where hard regions get much higher harm scores and much lower keep scores.
- On random missing and incident perturbation, the signal is still near the boundary, so the head should stay auxiliary.

Decision:

- Keep the verifier scenario-sensitive.
- Do not promote it to the main prediction path.
- Next step should be either scenario-aware calibration or a separate sensor-failure-specific utility head, not a universal direct gate.

## Two-Expert Harm Verifier

Implemented:

```text
physics_harm_head
physics_harm_sensor_head
physics_harm_gate_head
```

The verifier is now split into a general expert and a sensor-failure expert, mixed by a small region gate. This is trained with separate losses for the gate, the general expert, and the sensor expert.

Best short check:

| Scenario | Harm Hard | Harm Safe | Gate Mean | General Prob | Sensor Prob | Decision |
|---|---:|---:|---:|---:|---:|---|
| random_missing_50 | 0.418959 | 0.398301 | 0.278031 | 0.539217 | 0.050797 | weak but usable |
| sensor_failure_30 | 0.425072 | 0.316673 | 0.278482 | 0.427971 | 0.049935 | best separation so far |

Interpretation:

- The new heads are no longer frozen at initialization.
- Sensor failure gets the clearest harm separation.
- The sensor expert still stays conservative, so the final output should remain the generic correction path.

Decision:

- Keep the two-expert verifier auxiliary.
- Do not use `x_harm_verified` as the final model output.
- The next useful step is to freeze generic correction during verifier calibration if we want cleaner expert separation without chasing direct MAE gains.

## Frozen-Generic Verifier Calibration

This variant freezes the generic correction branch in stage B and only trains the verifier heads:

```text
physics_harm_head
physics_harm_sensor_head
physics_harm_gate_head
```

Best sensor-failure check:

| Scenario | Harm Hard | Harm Safe | Keep Hard | Keep Safe | Final MAE | Verifier Candidate |
|---|---:|---:|---:|---:|---:|---:|
| sensor_failure_30 | 0.540669 | 0.313030 | 0.459331 | 0.686970 | 1.650539 | `x_harm_verified_masked_mae 1.668704` |

Interpretation:

- Freezing the generic correction makes the verifier signal much cleaner.
- The sensor-failure split is now obvious enough to support the claim that the verifier learns when the physics-harm signal matters.
- The verifier candidate still does not beat the generic path, so it should remain a utility/calibration module, not the final output.

Decision:

- Keep generic correction frozen during verifier calibration.
- Keep the verifier auxiliary and scenario-sensitive.
- Do not route inference through `x_harm_verified` as the default output.

## Frozen-Generic Follow-up on Random / Incident

This check used the same verifier-only calibration but ran on `random_missing_50` and `incident_perturbation` after fixing the local NumPy compatibility issue.

Results:

| Scenario | Harm Hard | Harm Safe | Final MAE | Verifier Candidate |
|---|---:|---:|---:|---:|
| random_missing_50 | 0.411777 | 0.394514 | 1.783437 | `x_harm_verified_masked_mae 1.779849` |
| incident_perturbation | 0.411019 | 0.394026 | 1.784619 | `x_harm_verified_masked_mae 1.780751` |

Interpretation:

- The verifier signal remains only mildly separated outside sensor failure.
- The verifier candidate is still worse than the frozen generic correction path.
- This supports the narrower claim that the module is a scenario-sensitive utility checker, not a general replacement for generic correction.

Decision:

- Keep this branch as calibration / diagnostic only.
- Do not promote it to the final output path.

## Bounded Suppression Branch

This branch uses a bounded allowance:

```text
x_final = mu_data + (0.7 + 0.3 * raw_allowance) * delta_generic
```

The idea is to prevent physics from erasing generic correction entirely.

Quick result:

| Scenario | x_generic | x_harm_suppressed | Note |
|---|---:|---:|---|
| random_missing_50 | 1.783403 | 1.783476 | no gain |
| incident_perturbation | 1.784579 | 1.784599 | no gain |
| sensor_failure_30 | 1.672545 | 1.674183 | worse |

Interpretation:

- Bounded suppression is stable.
- It does not yet improve reconstruction over the generic path.
- This means the remaining gap is not just “too much correction”; the model still needs a more region-specific rule for when correction should be allowed.

Decision:

- Keep the branch as an exploratory variant.
- Do not use it as the main result.

## Physics Candidate Promotion

Since `x_phys` was consistently stronger than the generic/verifier paths, it was promoted from a diagnostic candidate into explicit output branches.

Implemented:

```text
--physics-candidate-correction
--physics-promoted-correction
```

The best current branch is `physics_promoted_correction`:

```text
normal sparse / incident regions -> x_fused
sensor-failure regions           -> x_phys
```

Quick result:

| Scenario | mu_data | x_generic | x_phys | x_fused | x_physics_promoted | vs mu_data | vs x_generic |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_missing_50 | 1.777746 | 1.740173 | 1.717409 | 1.712812 | 1.712812 | +3.653% | +1.572% |
| incident_perturbation | 1.778187 | 1.741185 | 1.719612 | 1.714888 | 1.714888 | +3.560% | +1.510% |
| sensor_failure_30 | 1.713007 | 1.647043 | 1.600010 | 1.632767 | 1.600010 | +6.596% | +2.856% |

Interpretation:

- Physics is no longer just a verifier or diagnostic signal.
- The strongest current path uses physics as a promoted correction candidate.
- Training the physics head further was not helpful; the strong behavior comes from the explicit physics/graph/time correction structure.

Decision:

- Promote `physics_promoted_correction` as the current main branch.
- Keep verifier/suppression as auxiliary interpretability modules.

## Learned Physics Promotion

The rule-based promotion was then turned into a learned gate:

```text
--learned-physics-promotion
```

The gate predicts `physics_promotion_score` from region features and the relative utility of `x_phys` versus `x_fused`.

Quick check:

| Run | promotion_mean | phys_better_mean | fused_better_mean | Note |
|---|---:|---:|---:|---|
| smoke | 0.620 | 0.619 | 0.622 | stable but blunt |
| 10e suite | 0.602 | 0.593 | 0.613 | still coarse |

Interpretation:

- The learned gate runs correctly.
- It does not yet learn a sharp separation between physics-better and fused-better regions.
- The method direction is right, but the supervision is still too soft.

Decision:

- Keep `physics_promoted_correction` as the main structural idea.
- Treat `learned_physics_promotion` as the next refinement target, not the finished result.

## Hard-Region Promotion Supervision

The promotion gate supervision was hardened to batch-wise quantile regions:

```text
physics-better -> hard region
fused-better   -> safe region
ambiguous      -> low-weight margin region
```

Quick result:

| Run | promotion_mean | phys_better_mean | fused_better_mean | MAE note |
|---|---:|---:|---:|---|
| smoke | 0.621 | 0.620 | 0.622 | still coarse |
| 10e suite | 0.606 | 0.601 | 0.611 | no sharp separation |

Interpretation:

- Hard supervision is cleaner than the earlier sigmoid target.
- The gate still does not become a strong region separator.
- Performance remains close to the earlier physics-promoted branch, so the gain is methodological, not yet numerical.

Decision:

- Keep the hard-region supervision as the current learned-promotion formulation.
- The next step should be stronger region sampling or an explicit failure-mode feature, not another softening of the gate.

## Discrete Physics Router

The continuous promotion score was replaced by a discrete router:

```text
--discrete-physics-promotion
```

It chooses among:

```text
fused
physics
generic
```

Quick result:

| Scenario | selected mode pattern | output note |
|---|---|---|
| random_missing_50 | mostly physics | did not exploit fused advantage |
| incident_perturbation | mostly physics | still coarse |
| sensor_failure_30 | almost all physics | consistent but not selective |

Interpretation:

- This is structurally cleaner than a scalar gate.
- The current training still collapses toward physics, so the router is not yet selective enough.
- The gain is architectural, not yet numerical.

Decision:

- Keep discrete routing as the method form.
- The next refinement should be better mode labels or explicit failure-mode sampling, not more scalar gate tuning.

## Discrete Router Target Rework

The router supervision was changed again so that `fused` is the safe default:

```text
target = fused
physics/generic override only if they beat fused by a margin
```

Two supporting changes were added:

- batch-wise class balancing to reduce majority-class collapse;
- failure-mode upweighting so specialist wins are not ignored.

Also, the `physics_promotion_mode_head` initialization was neutralized so the router no longer starts with a physics prior.

### Smoke Check

Protocol:

```text
random_missing_50
--hard-negative-margin 0.10
1e pretrain + 1e router
```

Target mass:

| fused | physics | generic |
|---|---:|---:|---:|
| 0.751 | 0.098 | 0.151 |

### Quick Suite

Protocol:

```text
10e router
--hard-negative-margin 0.10
random_missing_50 / noise_random_missing / incident_perturbation
```

| Scenario | x_fused | x_discrete | Result |
|---|---:|---:|---|
| random_missing_50 | 1.715475 | 1.739013 | worse |
| noise_random_missing | 1.715588 | 1.724584 | worse |
| incident_perturbation | 1.717704 | 1.722331 | worse |

Interpretation:

- The router no longer collapses to physics.
- The target is now logically cleaner: fused is the default, specialists need to earn the override.
- Numerically, the hard router still does not beat the fused candidate, so this is a supervision correction, not a benchmark win.
- The next refinement should likely be region-aware failure sampling from observed masks, not another round of generic class weighting.

## Numeric-Gain Preset

Because the discrete router currently hurts MAE, the numeric table should use the strongest stable head instead of the most recent router experiment.

Current best stable preset:

```text
official GRIN 30e cache
generic-v3 correction
correction epochs = 1
```

Paired baseline is the same forward pass `mu_data_masked_mae`.

| Scenario | Official GRIN | Generic-v3 LiteTrust | Relative Gain |
|---|---:|---:|---:|
| random_missing_50 | 0.646528 | 0.639490 | +1.09% |
| noise_random_missing | 0.655596 | 0.645067 | +1.61% |
| incident_perturbation | 0.658474 | 0.641924 | +2.51% |
| Average | 0.653532 | 0.642160 | +1.74% |

Decision:

- Use `generic-v3 correction, 1e` as the current main numeric result.
- Do not use the discrete router in the main table yet.
- The next gain-oriented target is to close the gap to the `oracle_best` candidate selection, not to tune the hard router.
