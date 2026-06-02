# Anti-Leakage Validation Notes

This document records the first strict anti-inflation checks for PhyGuard. The
goal is not to replace the full paper protocol yet, but to prevent the current
debug table from being overclaimed.

## What Was Tightened

- Raw PEMS time series are split before windowing.
- A raw gap is inserted between train, validation, and test.
- Windows use `stride=12` by default instead of adjacent `stride=1` windows.
- A stricter ablation mode can disable the temporal evidence bank:
  `--ablation no_temporal_evidence_bank`.

For METR-LA, the Hugging Face source provides predefined windowed parquet
splits. The strict script subsamples timestamps by stride, but it does not
reconstruct the original continuous raw series. METR-LA strict results should
therefore be treated as a HF-split robustness check, while PEMS strict results
are the cleaner anti-leakage check.

## Smoke Results

All rows below use `epochs=1`, `guard_epochs=2`, `seed=1`,
`train/val/test=64/16/16`, `stride=12`, and `gap=12`. These are smoke results,
not final paper numbers.

| Dataset | Scenario | Strict mode | Best external | Best external MAE | PhyGuard MAE | Gain | Selected physics bank | Selected correction |
|---|---|---|---|---:|---:|---:|---|---|
| PEMS08 | random_missing_50 | full | SAITS | 0.573487 | 0.308621 | +46.19% | PhysicsBidirTemporal | RegionAmplitudeScaled@1.50 |
| PEMS08 | random_missing_50 | no temporal bank | SAITS | 0.573487 | 0.308621 | +46.19% | PhysicsBidirTemporal | RegionAmplitudeScaled@1.50 |
| METR-LA | random_missing_50 | no temporal bank | SAITS | 0.760459 | 0.490289 | +35.53% | PhysicsBidirTemporal | RegionAmplitudeScaled@1.50 |

## Interpretation

The strict PEMS08 run no longer has train/validation/test adjacent-window
overlap. Under this setting, PhyGuard still improves over the best external
baseline in the smoke run.

The full run exposed one important risk: `TemporalBidirObs` can be extremely
strong under random missing because it uses within-window bidirectional observed
values. This candidate is useful as an internal diagnostic, but it should not be
the main source of paper claims unless the protocol explicitly allows
transductive within-window imputation. The stricter `no_temporal_evidence_bank`
run keeps the same PhyGuard score on PEMS08, which suggests the smoke gain is
mainly from the physics residual bank and region amplitude promotion, not from
the temporal interpolation candidate.

The remaining risk is validation-time correction selection. The current smoke
chooses `RegionAmplitudeScaled@1.50` by validation MAE. For final claims, this
should be controlled with either a fixed preregistered correction setting,
nested validation, or a small sensitivity table showing that the gain is not a
single validation-selected accident.

## Required Before Paper Claims

1. Run strict anti-leakage with 3 seeds on PEMS08 and at least one more raw PEMS
   dataset.
2. Report a strict table with and without the temporal evidence bank.
3. Add a sensitivity table for the correction amplitude or freeze the amplitude
   policy before test evaluation.
4. Keep the old quick table as debug evidence only, not as the final claimed
   result.
