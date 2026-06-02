# Official GRIN Integration

Official repo: `https://github.com/Graph-Machine-Learning-Group/grin`

Local clone: `C:/Users/21329/grin_official_cache`

Output root: `C:/Users/21329/litetrust_official_grin_outputs/official_grin_pems08_debug`

## Integration Mode

The original official runner depends on the old paper environment (`python 3.8`, `torch==1.8`, `pytorch-lightning==1.4`, `torchmetrics==0.5`, `h5py/tables`). The current workspace uses a newer NumPy/PyTorch stack, so the old Lightning runner is not used directly.

Instead, `models/official_grin_wrapper.py` imports the official `GRINet` model class and trains it with the current LiteTrust data pipeline and unified evaluator. This keeps the baseline model official while avoiding environment breakage from the old runner.

## Files Added

- `models/official_grin_wrapper.py`
- `scripts/run_official_grin_pems08_debug.py`
- `scripts/evaluate_external_outputs.py`

## Output Format

Each official GRIN run writes:

- `pred.npy`
- `true.npy`
- `mask.npy`
- `metrics.json`

This is the same format expected by the unified external evaluator.

## Smoke Result

Command:

```bash
python scripts/run_official_grin_pems08_debug.py --epochs 1 --scenarios random_missing_50
python scripts/evaluate_external_outputs.py --run-dir C:/Users/21329/litetrust_official_grin_outputs/official_grin_pems08_debug/random_missing_50
```

Result:

| Scenario | Epochs | Masked MAE | Note |
|---|---:|---:|---|
| random_missing_50 | 1 | 1.780388 | smoke only, not a comparison result |

## 30-Epoch Quick Result

Command:

```bash
python scripts/run_official_grin_pems08_debug.py --epochs 30 --scenarios random_missing_50 noise_random_missing incident_perturbation
```

| Scenario | Official GRINet masked MAE |
|---|---:|
| random_missing_50 | 0.682696 |
| noise_random_missing | 0.683350 |
| incident_perturbation | 0.679828 |

For reference, the current LiteTrust ReliabilityRouter 30-epoch debug results on the same scenarios were:

| Scenario | LiteTrust no-validity masked MAE | Official GRINet masked MAE | Gap |
|---|---:|---:|---:|
| random_missing_50 | 0.907277 | 0.682696 | +0.224581 |
| noise_random_missing | 0.901495 | 0.683350 | +0.218145 |
| incident_perturbation | 0.908902 | 0.679828 | +0.229074 |

## Reading After Quick Result

Official GRINet is substantially stronger than the current LiteTrust debug implementation on the three main sparse/noisy/incident scenarios. This means the next research step should not be more tuning of the lightweight backbone. If the goal is to beat or improve a strong baseline, LiteTrust should be implemented as a correction/router module on top of official GRINet outputs or hidden states.

## LiteTrust Correction Smoke

Added `OfficialGRIN_LiteTrustCorrection`:

```text
Official GRIN prediction + physics residual correction + reliability weighting
```

Same 20-epoch `random_missing_50` comparison:

| Model | Masked MAE |
|---|---:|
| Official GRINet | 0.768159 |
| Official GRIN + LiteTrust correction | 0.765457 |

The gain is real but tiny: about `0.35%`. This is not enough as a paper result. The next better implementation is two-stage training: train official GRIN first, freeze it, then train only the correction/router head.

Two-stage quick check:

| Model | Setting | Masked MAE |
|---|---|---:|
| Official GRIN + LiteTrust correction | 10 epoch GRIN pretrain + 10 epoch frozen correction | 1.018319 |

This is worse than joint training. Current evidence says the correction module is not yet strong enough on top of official GRIN; more useful work would redesign the correction target rather than continue small training schedule changes.

## Improvement-Aware Gate V2

Implemented an experimental improvement-aware gate loss:

```text
gate target = sigmoid((|x_grin - y| - |x_final - y|) / tau)
harm penalty = gate * relu(|x_final - y| - |x_grin - y|)
```

20-epoch `random_missing_50` result:

| Model | Masked MAE |
|---|---:|
| Official GRINet | 0.768159 |
| Official GRIN + LiteTrust correction V1 | 0.765457 |
| Official GRIN + improvement-aware V2 | 0.773720 |

V2 is worse. The likely issue is that the gate target is too noisy early in training and over-regularizes the correction. The runner keeps this loss behind `--improvement-gate-loss`; default training is reverted to V1.

## Strong-GRIN Correction Check

After training official GRIN for 30 epochs, freezing/continuing with a 5-epoch LiteTrust correction gives:

| Scenario | Official GRIN 30e | GRIN + correction 30e+5e | Gain |
|---|---:|---:|---:|
| random_missing_50 | 0.682696 | 0.675658 | +1.03% |
| noise_random_missing | 0.683350 | 0.675943 | +1.08% |
| incident_perturbation | 0.679828 | 0.673213 | +0.97% |

This is a real but small improvement.

## Physics Feature Ablation

To verify whether the gain comes from physics, the same correction module was run with the physics residual features disabled.

| Scenario | GRIN + physics correction | GRIN + no-physics correction | Physics-specific gain |
|---|---:|---:|---:|
| random_missing_50 | 0.675658 | 0.675825 | +0.000167 |
| noise_random_missing | 0.675943 | 0.675840 | -0.000103 |
| incident_perturbation | 0.673213 | 0.673309 | +0.000096 |

This does not support a physics-specific contribution. The improvement mainly comes from adding a small post-GRIN correction module.

## Decision After Ablation

Do not claim this version as LiteTrust-PINN. It improves official GRIN slightly, but the ablation shows the physics residual is not the driver.

Next structural change should remove the generic correction shortcut:

```text
Official GRIN output
+ constrained physics projection candidate
+ no-physics residual adapter as a separate rival expert
+ reliability gate trained by contrastive utility:
   trust physics only when physics candidate beats both GRIN and no-physics adapter
```

The next experiment must compare:

1. `Official GRIN`
2. `Official GRIN + generic correction`
3. `Official GRIN + constrained physics correction`
4. `Official GRIN + reliability gate over generic vs physics experts`

Only if the constrained physics or gated physics expert beats the generic correction can we claim the improvement is due to the proposed pain point.

## Gated Expert Ablation

The required structural ablation was run with:

```text
Official GRIN 30 epochs
+ correction/router 5 epochs
```

Detailed file: `results/official_grin_pems08_debug/gated_expert_ablation.md`

| Scenario | Official GRIN | Generic | Physics | Gated |
|---|---:|---:|---:|---:|
| random_missing_50 | 0.682696 | 0.672306 | 0.676001 | 0.669968 |
| noise_random_missing | 0.683350 | 0.672115 | 0.673673 | 0.668720 |
| incident_perturbation | 0.679828 | 0.670578 | 0.674999 | 0.668542 |

This is the first result where the gated method beats both single experts in every tested scenario. The improvement is still modest, but it is now attributable to the trust-routing structure rather than merely adding a generic post-processing adapter.

## Next Run

Use the official GRIN wrapper on the three scenarios that currently support the LiteTrust claim:

```bash
python scripts/run_official_grin_pems08_debug.py --epochs 30 --scenarios random_missing_50 noise_random_missing incident_perturbation
```

Do not treat `sensor_failure_30` as a primary win condition unless it improves; prior diagnostics show it is graph-dominant.
