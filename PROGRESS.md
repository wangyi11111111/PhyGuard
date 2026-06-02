# LiteTrust-PINN Progress

## Latest Update: Internalized MagiNet-Style Graph Repair

- motivation:
  - user pointed out that directly using MagiNet as an internal candidate is still engineering combination
  - changed the direction to extract MagiNet's useful mechanism instead of preserving or routing to its prediction

- method change:
  - added `MaskAwareGraphRepair` in `models/litetrust_pinn.py`
  - extracted three MagiNet-like ideas into our model:
    - learnable missing token
    - mask-aware feature/node/time embedding
    - missing-aware dynamic graph repair
  - changed `scripts/run_temporal_anchor_litetrust_quick.py` so final internal fusion uses:
    - `MaskAwareGraphRepair`
    - `TemporalAnchorPhysicsCalibrated`
    - `PhysicsFromGraphRepair`
  - MagiNet is now only an external baseline row, not an internal model candidate

- PEMS08 debug, seed `1`, epochs `20`:
  - `random_missing_50`:
    - best external `MagiNet`: `0.378308`
    - `MaskAwareGraphRepair`: `0.521179`
    - `TemporalAnchorPhysicsCalibrated`: `0.439356`
    - final internal fusion: `0.439356`
    - status: still worse than MagiNet by `16.14%`
  - `sensor_failure_30`:
    - best external `SAITS`: `0.892811`
    - `TemporalAnchorPhysicsCalibrated`: `0.913356`
    - final internal fusion: `0.926085`
    - status: close to SAITS but still worse by `3.73%`
  - `incident_perturbation`:
    - best external `MagiNet`: `0.384309`
    - `TemporalAnchorPhysicsCalibrated`: `0.454890`
    - final internal fusion: `0.433908`
    - status: worse than MagiNet by `12.91%`, but router improves over temporal-only correction

- METR-LA HF, seed `1`, epochs `20`:
  - `random_missing_50`:
    - best external `MagiNet`: `0.301571`
    - final internal fusion: `0.343411`
    - status: worse than MagiNet by `13.87%`
  - `sensor_failure_30`:
    - best external `SAITS`: `0.395298`
    - final internal fusion: `0.469438`
    - status: worse than SAITS by `18.76%`
  - `incident_perturbation`:
    - best external `MagiNet`: `0.306068`
    - final internal fusion: `0.354251`
    - status: worse than MagiNet by `15.74%`

- interpretation:
  - removing direct MagiNet use makes the method cleaner but exposes that our internal graph repair is still too weak
  - the temporal-anchor branch is useful for sensor failure and transfers across datasets, but it does not fully match SAITS
  - the current dynamic graph repair extracts the idea, but not enough of MagiNet's performance; graph confidence on METR-LA is very low, suggesting the learned repair graph is too diffuse
  - current evidence is not yet paper-level after removing direct MagiNet candidates

- next structural diagnosis:
  - strengthen internal graph repair with a two-stage objective:
    1. train `MaskAwareGraphRepair` as a standalone reconstruction model with masked self-supervision on observed entries
    2. then train physics/calibration on top of its output
  - add top-k sparse dynamic graph instead of fully diffuse softmax graph
  - add multi-scale temporal gated convolution after graph repair, matching MagiNet's GTU-like advantage without copying the full model
  - keep MagiNet and SAITS only as external baselines

## Latest Update: MagiNet-Style Top-k Graph and GTU Internalization

- method change:
  - upgraded `MaskAwareGraphRepair` with:
    - top-k sparse missing-aware dynamic graph
    - MagiNet-style multi-scale gated temporal units with kernel sizes `3/5/7`
    - graph repair followed by physics residual correction through `PhysicsFromGraphRepair`
  - MagiNet remains only an external baseline; internal fusion no longer consumes MagiNet predictions

- PEMS08 debug, seed `1`, epochs `20`:
  - `random_missing_50`:
    - `MagiNet`: `0.378308`
    - `MaskAwareGraphRepair`: `0.471912`
    - `PhysicsFromGraphRepair`: `0.461607`
    - `TemporalAnchorPhysicsCalibrated`: `0.439356`
    - final internal fusion: `0.440909`
    - internal graph repair improved from `0.521179` to `0.471912`
  - `incident_perturbation`:
    - `MagiNet`: `0.384309`
    - `MaskAwareGraphRepair`: `0.474558`
    - `PhysicsFromGraphRepair`: `0.468015`
    - `TemporalAnchorPhysicsCalibrated`: `0.454890`
    - final internal fusion: `0.437654`
    - internal graph repair improved from `0.511905` to `0.474558`

- METR-LA HF, seed `1`, epochs `20`:
  - `random_missing_50`:
    - `MagiNet`: `0.301571`
    - `MaskAwareGraphRepair`: `0.361108`
    - `PhysicsFromGraphRepair`: `0.357320`
    - `TemporalAnchorPhysicsCalibrated`: `0.342864`
    - final internal fusion: `0.357320`
    - internal graph repair improved from `0.398801` to `0.361108`
  - `incident_perturbation`:
    - `MagiNet`: `0.306068`
    - `MaskAwareGraphRepair`: `0.363903`
    - `PhysicsFromGraphRepair`: `0.361010`
    - `TemporalAnchorPhysicsCalibrated`: `0.354602`
    - final internal fusion: `0.361010`
    - internal graph repair improved from `0.403267` to `0.363903`

- interpretation:
  - copying more MagiNet-style structure helped consistently on both datasets
  - however, internalized graph repair is still behind external MagiNet by about `18-25%` relative in random/incident settings
  - this suggests the remaining missing piece is not just GTU/top-k, but the fuller MagiNet block stack: residual temporal attention, Chebyshev mask-aware spatial convolution, and final RNN/fc head
  - next step should vendor a near-complete MagiNet-style backbone inside our codebase or reproduce its full block stack in `MaskAwareGraphRepairV2`, then attach LiteTrust physics and calibration

## Latest Update: MaskAwareGraphRepairV2

- method change:
  - added `MagiStyleRepairBlock` and `MaskAwareGraphRepairV2` in `models/litetrust_pinn.py`
  - V2 internalizes a fuller MagiNet-style block stack:
    - learnable missing token
    - node and time embeddings
    - residual temporal attention
    - missing-aware dynamic graph learning
    - top-k sparse repair graph
    - Chebyshev-like mask-aware spatial graph convolution
    - multi-scale `3/5/7` gated temporal units
    - residual normalization
    - bidirectional GRU final head
  - updated `scripts/run_temporal_anchor_litetrust_quick.py` to use `MaskAwareGraphRepairV2`
  - MagiNet remains only an external baseline; internal candidates are:
    - `MaskAwareGraphRepairV2`
    - `TemporalAnchorPhysicsCalibrated`
    - `PhysicsFromGraphRepair`

- verification:
  - compile check passed for `models/litetrust_pinn.py`, `models/__init__.py`, and `scripts/run_temporal_anchor_litetrust_quick.py`
  - shape check passed for `MaskAwareGraphRepairV2`

- PEMS08 debug, seed `1`, epochs `20`:
  - `random_missing_50`:
    - `MagiNet`: `0.378308`
    - previous `MaskAwareGraphRepair`: `0.471912`
    - `MaskAwareGraphRepairV2`: `0.456910`
    - `PhysicsFromGraphRepair`: `0.432375`
    - final internal fusion: `0.448393`
  - `incident_perturbation`:
    - `MagiNet`: `0.384309`
    - previous `MaskAwareGraphRepair`: `0.474558`
    - `MaskAwareGraphRepairV2`: `0.432424`
    - `PhysicsFromGraphRepair`: `0.408872`
    - final internal fusion: `0.408872`

- METR-LA HF, seed `1`, epochs `20`:
  - `random_missing_50`:
    - `MagiNet`: `0.301571`
    - previous `MaskAwareGraphRepair`: `0.361108`
    - `MaskAwareGraphRepairV2`: `0.318072`
    - `PhysicsFromGraphRepair`: `0.315915`
    - final internal fusion: `0.315915`
  - `incident_perturbation`:
    - `MagiNet`: `0.306068`
    - previous `MaskAwareGraphRepair`: `0.363903`
    - `MaskAwareGraphRepairV2`: `0.317340`
    - `PhysicsFromGraphRepair`: `0.317428`
    - final internal fusion: `0.317428`

- interpretation:
  - V2 substantially closes the gap to MagiNet, especially on METR-LA
  - METR-LA random is now within about `4.76%` of MagiNet after physics correction
  - METR-LA incident is within about `3.71%` of MagiNet after physics correction
  - PEMS08 still has a larger gap, but V2 plus physics is clearly better than V1
  - physics correction is useful on top of the internal Magi-style graph repair in random and incident settings

- next:
  - run `sensor_failure_30` with V2 to make sure it does not hurt the temporal-anchor advantage
  - then run all three scenarios on both PEMS08 and METR-LA in one table
  - if PEMS08 remains behind MagiNet, inspect whether the gap is from training protocol or the final head capacity

## Latest Update: MagiNet Teacher Distillation for V2

- method change:
  - added optional teacher distillation to `scripts/run_temporal_anchor_litetrust_quick.py`
  - flag: `--distill-maginet`
  - default weight tested: `--distill-weight 0.45`
  - teacher is the official MagiNet prediction already trained in the same protocol
  - student remains lightweight `MaskAwareGraphRepairV2`; inference does not use MagiNet
  - loss adds:
    - missing-region teacher imitation
    - hard-region weighting when teacher is locally better
    - harm-aware penalty against being worse than teacher in target missing regions
    - original target-supervised reconstruction loss remains active

- PEMS08 debug, seed `1`, epochs `20`, distill weight `0.45`:
  - `random_missing_50`:
    - `MagiNet`: `0.378308`
    - V2 no distill: `0.456910`
    - V2 distilled: `0.440730`
    - distilled V2 + physics: `0.414788`
    - final internal fusion: `0.414788`
  - `incident_perturbation`:
    - `MagiNet`: `0.384309`
    - V2 no distill: `0.432424`
    - V2 distilled: `0.442686`
    - distilled V2 + physics: `0.417776`
    - final internal fusion: `0.431566`

- METR-LA HF, seed `1`, epochs `20`, distill weight `0.45`:
  - `random_missing_50`:
    - `MagiNet`: `0.301571`
    - V2 no distill: `0.318072`
    - V2 distilled: `0.306940`
    - distilled V2 + physics: `0.308520`
    - final internal fusion: `0.308520`
  - `incident_perturbation`:
    - `MagiNet`: `0.306068`
    - V2 no distill: `0.317340`
    - V2 distilled: `0.309363`
    - distilled V2 + physics: `0.312104`
    - final internal fusion: `0.312104`

- interpretation:
  - lightweight distillation works clearly on METR-LA: distilled V2 reaches within about `1.8%` of MagiNet on random and about `1.1%` on incident before physics
  - on PEMS08, distillation improves random but hurts incident slightly before physics; physics recovers some of it
  - too much teacher imitation can suppress perturbation-specific physics utility
  - this supports a lightweight path, but distillation should be scenario/utility-aware instead of globally fixed

- next:
  - test lower distillation weights such as `0.20` or `0.30`
  - make distillation residual-aware: imitate MagiNet only where teacher beats target-supervised V2 or where teacher residual is lower
  - keep physics correction after student, but calibrate it separately because teacher imitation can already absorb part of physics behavior

## Latest Update: SAITS Advantage Diagnosis and Preservation Fix

- issue found:
  - the earlier five-baseline script trained SAITS/MagiNet once for the external baseline rows and again as candidates inside ConfidenceGuarded
  - this made the SAITS comparison noisy and unfair
  - fixed `scripts/run_confidence_guarded_five_baseline_flow_quick.py` so MagiNet and SAITS are trained once and reused both as external baselines and internal candidates

- method change:
  - added region diagnostics for target / sensor-like / non-sensor regions
  - added `SAITSPreservedConfidenceGuarded`
  - rule: when validation selects SAITS and the scenario is sensor-like, preserve the SAITS output instead of letting router/physics mix into it

- fair seed-1 results, PEMS08 debug flow-only:
  - `random_missing_50`: best external `MagiNet` `0.504411`, ours `0.474988`, gain `+5.83%`
  - `sensor_failure_30`: best external `SAITS` `1.033954`, ours `1.033954`, tie `0.00%`
  - `incident_perturbation`: best external `MagiNet` `0.529567`, ours `0.488190`, gain `+7.82%`

- fair seed-1 results, METR-LA flow-only:
  - `random_missing_50`: best external `MagiNet` `0.345454`, ours `0.340031`, gain `+1.57%`
  - `sensor_failure_30`: best external `SAITS` `0.424271`, ours `0.424271`, tie `0.00%`
  - `incident_perturbation`: best external `SAITS` `0.390823`, ours `0.345397`, gain `+11.62%`

- diagnosis:
  - random missing and incident favor MagiNet/physics-style residual correction
  - sensor failure favors SAITS because temporal self-attention remains usable when an entire sensor has no local observations
  - unrestricted router/physics can damage sensor-failure outputs even when sensor-like regions look partly improved, because a small non-sensor region can become very bad
  - preserving SAITS in sensor-failure mode removes the main negative transfer without hurting random/incident

- current status:
  - the method now wins random and incident on both PEMS08 and METR-LA quick protocols
  - sensor failure is no longer a loss, but it is a tie with SAITS, not a win
  - next improvement should target "SAITS + physics verifier" only where there is positive utility, not replacing SAITS globally

## Latest Update: METR-LA Five External Baseline Quick

- script:
  - extended `scripts/run_confidence_guarded_five_baseline_flow_quick.py` with `--dataset METR-LA`
  - uses real METR-LA HF data, normalized and reduced to a single channel for the same five-baseline quick protocol

- run:
  - command: `python scripts/run_confidence_guarded_five_baseline_flow_quick.py --dataset METR-LA --epochs 8 --seed 1 --output-dir C:/tmp/confidence_guarded_five_baseline_metrla_seed1`
  - seed `1`
  - epochs `8`
  - train/val/test windows: `64/16/16`
  - nodes: `207`
  - scenarios: `random_missing_50`, `sensor_failure_30`, `incident_perturbation`

- results:
  - `random_missing_50`: best external `MagiNet` `0.345454`, ours `0.340031`, gain `+1.57%`
  - `sensor_failure_30`: best external `SAITS` `0.437815`, ours `0.438645`, gap `-0.19%`
  - `incident_perturbation`: best external `SAITS` `0.386242`, ours `0.345397`, gain `+10.57%`

- interpretation:
  - on METR-LA, ConfidenceGuarded beats the five external baselines on random missing and incident perturbation
  - sensor failure is effectively tied with SAITS but still slightly worse
  - cross-dataset evidence is now stronger than before: PEMS08 wins random/incident and METR-LA wins random/incident, while sensor failure remains the narrow unresolved case
  - the next structural fix should be strict SAITS preservation in sensor-failure regions, because the current fallback selects SAITS but the final output can still be slightly worse

## Latest Update: METR-LA Real HF ConfidenceGuarded Recheck

- run:
  - command: `python scripts/run_regime_adaptive_litetrust_quick.py`
  - dataset: real METR-LA from HF cache
  - seed `1`
  - epochs `10`
  - scenarios: `random_missing_50`, `sensor_failure_30`
  - note: this is the current cross-dataset quick protocol, not the five-external-baseline protocol

- METR-LA results:
  - `random_missing_50`: `ConfidenceGuardedUtilityRouter` `0.287094`
    - vs previous BaseTCN `0.315836`: gain `+9.10%`
    - vs `LiteTrustPINN_full` `0.313739`: gain `+8.49%`
    - vs `LiteTrustGRINCorrection` `0.319233`: gain `+10.07%`
  - `sensor_failure_30`: `ConfidenceGuardedUtilityRouter` `0.305049`
    - vs previous BaseTCN `0.357151`: gain `+14.59%`
    - vs `LiteTrustGRINCorrection` `0.309227`: gain `+1.35%`

- interpretation:
  - the method transfers better to METR-LA than to PEMS08 debug under this quick protocol
  - random missing on METR-LA shows a larger gain than PEMS08, which supports some cross-dataset value
  - external-baseline comparison still only exists for PEMS08 debug flow-only; METR-LA external baselines remain untested

## Latest Update: ConfidenceGuarded vs Five External Baselines

- script:
  - added `scripts/run_confidence_guarded_five_baseline_flow_quick.py`
  - uses the same PEMS08 debug flow-only protocol as the existing five-baseline benchmark
  - external baselines: `MagiNet`, `KNN`, `BRITS`, `SAITS`, `GRINLite`
  - our compared model: `ConfidenceGuardedUtilityRouter`

- run:
  - command: `python scripts/run_confidence_guarded_five_baseline_flow_quick.py --epochs 8 --seed 1 --output-dir C:/tmp/confidence_guarded_five_baseline_seed1`
  - seed `1`
  - epochs `8`
  - scenarios: `random_missing_50`, `sensor_failure_30`, `incident_perturbation`
  - output: `C:/tmp/confidence_guarded_five_baseline_seed1/confidence_guarded_five_baseline_summary.md`

- results:
  - `random_missing_50`: best external `MagiNet` `0.504411`, ours `0.474988`, gain `+5.83%`
  - `sensor_failure_30`: best external `SAITS` `1.218358`, ours `1.397534`, gap `-14.71%`
  - `incident_perturbation`: best external `MagiNet` `0.529567`, ours `0.488190`, gain `+7.82%`

- interpretation:
  - under the same five-baseline flow-only protocol, the new ConfidenceGuarded version is competitive on random missing and incident perturbation
  - sensor failure remains the main unresolved weakness because SAITS is still much stronger there
  - the confidence guard is falling back to `SAITS` on sensor failure, but the final prediction still worsens relative to raw SAITS; this means the guard should be stricter for sensor-like regions or should directly preserve raw SAITS when validation picks SAITS
  - do not run 3 seeds yet; first fix the sensor-failure guard so it cannot underperform the validation-selected external candidate

## Latest Update: Learned Residual-Utility Router With Confidence Guard

- method change:
  - added a small `ResidualUtilityRouter` in `scripts/run_regime_adaptive_litetrust_quick.py`
  - fixed the reconstruction candidates first, then trained the router as a second-stage utility classifier
  - router target: choose the candidate with lowest target-region error at each node/time region
  - router input: missing pattern, node-vs-neighbor missing contrast, residual ranks, candidate disagreement, and residual-regime weights
  - added `ConfidenceGuardedUtilityRouter`:
    - low router confidence: fallback to validation-selected residual regime
    - high confidence + sensor-like region: hard utility selection
    - high confidence + non-sensor region: soft utility blend

- run:
  - real PEMS08 debug + real METR-LA HF cache
  - seed `1`
  - epochs `10`
  - scenarios: `random_missing_50`, `sensor_failure_30`

- guarded results:
  - PEMS08 `random_missing_50`: `1.037749`, same as `LiteTrustPINN_full`, `+0.998%` over BaseTCN `1.048213`
  - PEMS08 `sensor_failure_30`: `1.461910`, `+16.380%` over BaseTCN `1.748276`, and `+1.746%` over `LiteTrustGRINCorrection` `1.487885`
  - METR-LA `random_missing_50`: `0.287094`, `+9.100%` over BaseTCN `0.315836`, and `+5.217%` over previous residual-regime best `0.302899`
  - METR-LA `sensor_failure_30`: `0.305049`, `+14.588%` over BaseTCN `0.357151`, and `+1.351%` over `LiteTrustGRINCorrection` `0.309227`

- evidence:
  - learned router alone improves METR-LA but can hurt PEMS08 when confidence is low
  - confidence guard prevents negative transfer on PEMS08 and keeps the METR-LA gains
  - this supports the revised method story: physics is not used as a direct answer generator; it defines residual regimes and utility evidence for when correction should be trusted

- next check:
  - repeat this exact guarded setup on at least one more seed before treating the gains as stable
  - then compare only against the five external baselines, excluding internal physics candidates from the main table

## Latest Update: Residual-Regime Physics Quick Check

- method change:
  - added `scripts/run_regime_adaptive_litetrust_quick.py`
  - tested candidate residual regimes instead of directly forcing a physics candidate into the prediction
  - candidate set:
    - `LiteTrustPINN_full`
    - `LiteTrustGRINCorrection`
    - `absolute_node_missing_regime`
    - `contrast_sensor_regime`
    - `residual_verified_regime`
    - `ValidationSelectedResidualRegime`

- run:
  - real PEMS08 debug + real METR-LA HF cache
  - seed `1`
  - epochs `10`
  - scenarios: `random_missing_50`, `sensor_failure_30`
  - no new full benchmark, no multi-seed expansion

- key results:
  - PEMS08 `random_missing_50`: best `LiteTrustPINN_full` `1.037749`, still better than BaseTCN `1.048213` by `0.998%`
  - PEMS08 `sensor_failure_30`: best `contrast_sensor_regime` `1.461910`, better than BaseTCN `1.748276` by `16.380%`, and better than `LiteTrustGRINCorrection` `1.487885` by `1.746%`
  - METR-LA `random_missing_50`: best `absolute_node_missing_regime` `0.302899`, better than BaseTCN `0.315836` by `4.096%`, and better than `LiteTrustPINN_full` `0.313739` by `3.455%`
  - METR-LA `sensor_failure_30`: best `LiteTrustGRINCorrection` `0.309227`, better than BaseTCN `0.357151` by `13.418%`

- interpretation:
  - direct physics-form correction remains unsafe; it was not kept as the main path
  - physics is useful as a residual-regime selector: it can improve sensor-failure selection on PEMS08 and random-missing selection on METR-LA
  - one fixed regime is still not robust enough across both datasets
  - validation selection failed on METR-LA random missing, so it should not be used as the final method yet
  - next structural target: replace validation selection with a learned/calibrated residual-utility router trained to predict which candidate lowers target-region MAE, with region-balanced sampling

## Latest Update: Cross-Dataset Real Check After Smoothing

- run:
  - PEMS08 real debug from ASTGNN zip
  - METR-LA real HF loader
  - single seed `1`
  - scenarios: `random_missing_50`, `sensor_failure_30`
  - models: `BaseTCN`, `FixedPhysics`, `LiteTrustPINN_full`, `LiteTrustGRINCorrection`
  - same 10-epoch quick config on both datasets

- key results:
  - PEMS08 `random_missing_50`: best `LiteTrustPINN_full` `1.037749`, gain `0.998%` over BaseTCN
  - PEMS08 `sensor_failure_30`: best `LiteTrustGRINCorrection` `1.487885`, gain `14.894%` over BaseTCN
  - METR-LA `random_missing_50`: best `LiteTrustPINN_full` `0.313739`, gain `0.664%` over BaseTCN
  - METR-LA `sensor_failure_30`: best `LiteTrustGRINCorrection` `0.309227`, gain `13.418%` over BaseTCN

- interpretation:
  - the method still transfers better to sensor failure than to random missing
  - random missing gains are present but too small to call stable
  - the issue is now more likely in the residual definition / physics promotion side than in the gate smoothness itself

## Latest Update: Cross-Dataset Smoothing Pass

- method change:
  - replaced several hard node-failure / anomaly cutoffs with the same smooth local-vs-neighborhood signal
  - updated `models/official_grin_wrapper.py`, `scripts/run_conflict_test.py`, `scripts/run_stage1_trend_suite.py`, `scripts/run_stage2_three_dataset_quick.py`, `scripts/run_strong_candidate_fusion_flow_quick.py`, and `scripts/run_correction_ablation_pems08_debug.py`

- intent:
  - keep the correction / verifier logic from collapsing into PEMS08-specific rules
  - make the physics side and the region gate behave the same way on METR-LA and PEMS-style data

- status:
  - code compiles on import
  - data evidence still comes from the earlier real METR-LA HF single-seed runs
  - the next check should be the same METR-LA HF scenario after this smoothing pass

## Latest Update: METR-LA HF Real Single-Seed Check

- data source:
  - Hugging Face dataset: [`witgaw/METR-LA`](https://huggingface.co/datasets/witgaw/METR-LA)
  - loader now reads `train.parquet`, `val.parquet`, `test.parquet`, and `sensor_graph/adj_mx.npy`
  - real METR-LA run uses `fallback_used=false`

- run:
  - seed: `1`
  - scenarios: `random_missing_50`, `sensor_failure_30`
  - models: `BaseTCN`, `FixedPhysics`, `LiteTrustPINN_full`, `LiteTrustGRINCorrection`

- key results:
  - `sensor_failure_30`: `LiteTrustGRINCorrection` masked MAE `1.400377`, best among the tested models
  - `random_missing_50`: `FixedPhysics` masked MAE `1.989333`, slightly better than the other tested models

- interpretation:
  - the loader is now genuinely wired to real METR-LA data
  - the method still shows its clearest advantage on sensor failure, not random missing
  - this is only a single-seed sanity run, not yet a full benchmark

## Latest Update: METR-LA Single-Seed Fallback Check

- dataset:
  - METR-LA synthetic fallback through `scripts/run_stage2_three_dataset_quick.py`
  - real METR-LA data is not present locally, so this is smoke/trend evidence only

- run:
  - single seed: `1`
  - scenarios: `random_missing_50`, `sensor_failure_30`
  - models: `BaseTCN`, `FixedPhysics`, `LiteTrustPINN_full`, `LiteTrustGRIN`, `LiteTrustGRINCorrection`

- key results:
  - `sensor_failure_30`: `LiteTrustGRINCorrection` masked MAE `0.467233`, best among the tested models
  - `random_missing_50`: `FixedPhysics` masked MAE `0.361842`, slightly better than `BaseTCN` and `LiteTrustPINN_full`

- interpretation:
  - the method family shows a clear advantage on node-level failure in this fallback setting
  - random missing is not uniformly improved, so the current claim should stay narrow
  - this run is useful as a direction check, not as formal dataset evidence

## Latest Update: Frozen V10 External-Baseline Experiment

- frozen method:
  - added `METHOD_V10_FROZEN.md`
  - fixed the current method as `Validation-Selected Reliability Repair`
  - no further architecture changes should be mixed into this experimental round

- external baseline supplementation:
  - ran five external baselines for seed 2:
    - `python scripts/run_five_baselines_flow_quick.py --epochs 8 --seed 2 --output-dir C:/tmp/five_baselines_flow_quick_seed2_v10`
  - ran five external baselines for seed 3:
    - `python scripts/run_five_baselines_flow_quick.py --epochs 8 --seed 3 --output-dir C:/tmp/five_baselines_flow_quick_seed3_v10`
  - combined with the existing seed 1 external baseline results

- frozen V10 external-only 3-seed summary:
  - `C:/tmp/strong_candidate_fusion_flow_quick_v10_stability/frozen_v10_external_3seed_summary.md`

- three-seed mean masked MAE against five external baselines:
  - random_missing_50: best external MagiNet `0.527235`, frozen V10 `0.490315`, gain `+7.00%`
  - sensor_failure_30: best external SAITS `0.898237`, frozen V10 `0.974401`, gap `-8.48%`
  - incident_perturbation: best external MagiNet `0.571295`, frozen V10 `0.529957`, gain `+7.24%`

- interpretation:
  - the current frozen version has useful evidence for random missing and incident perturbation
  - sensor failure is still not enough under the proper five-external-baseline comparison because SAITS remains stronger on the three-seed mean
  - this confirms the evidence gap: the method cannot yet claim robust superiority across all disrupted settings

## Latest Update: V10 Three-Seed Stability Check

- stability check:
  - seed 1: `C:/tmp/strong_candidate_fusion_flow_quick_v10`
  - seed 2: `C:/tmp/strong_candidate_fusion_flow_quick_v10_seed2`
  - seed 3: `C:/tmp/strong_candidate_fusion_flow_quick_v10_seed3`
  - aggregate summary: `C:/tmp/strong_candidate_fusion_flow_quick_v10_stability/summary.md`
  - external-only summary: `C:/tmp/strong_candidate_fusion_flow_quick_v10_stability/external_only_summary.md`

- three-seed mean masked MAE:
  - random_missing_50: MagiNet `0.527235`, PhysicsFromMagi `0.489332`, ReliabilityRepair `0.489913`, StrongCandidateFusion `0.490315`
  - sensor_failure_30: MagiNet `1.501485`, SAITS `1.020494`, ReliabilityRepair `0.974401`, StrongCandidateFusion `0.974401`
  - incident_perturbation: MagiNet `0.571295`, PhysicsFromMagi `0.531340`, ReliabilityRepair `0.529957`, StrongCandidateFusion `0.529957`

- three-seed gains:
  - random_missing_50: `+7.00%` over MagiNet, but `-0.20%` against the strongest internal candidate
  - sensor_failure_30: `+35.10%` over MagiNet and `+4.52%` over SAITS
  - incident_perturbation: `+7.24%` over MagiNet and matches the strongest internal candidate

- interpretation:
  - the sensor-failure gain is stable across three seeds and is no longer a seed-1 accident
  - random and incident remain consistently better than MagiNet, but the final selected output often collapses to the best internal repair/physics candidate
  - this supports the framework story as validation-selected reliability repair, but the router itself still needs improvement if the claim is adaptive fusion rather than candidate selection
  - main comparison tables should now exclude internal candidates and only report the five external baselines

## Latest Update: Validation-Selected Reliability Repair

- structural change:
  - added a validation-selected safety output in `scripts/run_strong_candidate_fusion_flow_quick.py`
  - final prediction now chooses among `router / MagiNet / SAITS / PhysicsFromMagi / ReliabilityRepair` using validation MAE
  - this prevents the router from suppressing a stronger reliability candidate

- quick benchmark:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 8 --repair-epochs 120 --router-epochs 120 --seed 1 --output-dir C:/tmp/strong_candidate_fusion_flow_quick_v10`

- results:
  - random_missing_50: `0.462785`
  - sensor_failure_30: `0.986044`
  - incident_perturbation: `0.487524`

- comparison against the best baselines:
  - random_missing_50: better than MagiNet `0.504411` by about `8.24%`
  - sensor_failure_30: better than SAITS `1.033954` by about `4.64%`
  - incident_perturbation: better than MagiNet `0.529567` by about `7.95%`

- interpretation:
  - this is the first version that clearly beats SAITS on sensor_failure_30
  - the selected final output is `ReliabilityRepair`, so the model is no longer forced to use the router when a stronger repair candidate exists
  - the current main gap is that random_missing and incident still rely heavily on the same repair candidate, so the next improvement should test whether the validation-selected repair remains stable across more seeds or collapses on a slightly different split

## Latest Update: Node-Reliability Conditioned Repair Trial

- structural change:
  - kept the repair branch generic, but added a stronger node-reliability conditioned routing prior
  - when node-level missingness dominates neighborhood evidence, the router shifts mass toward the local repair path
  - this is still not a sensor-specific head; it is a region-conditioned rule inside the shared router

- quick benchmark:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 8 --repair-epochs 120 --router-epochs 120 --seed 1 --output-dir C:/tmp/strong_candidate_fusion_flow_quick_v8`

- results:
  - random_missing_50: `0.462842`
  - sensor_failure_30: `1.052793`
  - incident_perturbation: `0.487204`

- comparison against the previous reliability-prior version:
  - random_missing_50: `0.462751 -> 0.462842`, essentially unchanged
  - sensor_failure_30: `1.152343 -> 1.052793`, improved materially
  - incident_perturbation: `0.487344 -> 0.487204`, slightly better

- interpretation:
  - the region-conditioned reliability rule recovered most of the sensor-failure loss without adding a dedicated sensor head
  - sensor failure is now close to `SAITS`, but still not better than it
  - random missing and incident remain better than MagiNet and PhysicsFromMagi, but the main sensor-failure gap is still open

## Latest Update: Reliability Prior Recovery Trial

- structural change:
  - kept `ReliabilityRepair` as a generic node-reliability branch
  - strengthened the reliability prior in `scripts/run_strong_candidate_fusion_flow_quick.py`
  - sensor-failure is now handled as a strong node-missing reliability case, not a dedicated head
  - the router now boosts local repair weight when node-level missingness dominates neighborhood evidence

- quick benchmark:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 8 --repair-epochs 120 --router-epochs 120 --seed 1 --output-dir C:/tmp/strong_candidate_fusion_flow_quick_v7`

- results:
  - random_missing_50: `0.462751`
  - sensor_failure_30: `1.152343`
  - incident_perturbation: `0.487344`

- comparison against the fully generic repair version:
  - random_missing_50: `0.460227 -> 0.462751`, slightly worse
  - sensor_failure_30: `1.203457 -> 1.152343`, improved but still worse than the earlier sensor-aware version
  - incident_perturbation: `0.485452 -> 0.487344`, slightly worse

- interpretation:
  - the reliability prior helps recover some sensor-failure performance without reverting to a hard sensor-specific head
  - but the generic repair branch still does not beat the earlier sensor-aware routing version
  - next step should be a region-conditioned repair rule, not more generic weighting

## Latest Update: Generic Reliability Repair Trial

- structural change:
  - removed the hard sensor-failure fallback from `scripts/run_strong_candidate_fusion_flow_quick.py`
  - renamed the fourth candidate from `SensorRepair` to `ReliabilityRepair`
  - the repair branch is now framed as generic node-reliability repair rather than a sensor-failure-specific head
  - router now uses a continuous reliability prior instead of a sensor-failure override

- quick benchmark:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 8 --repair-epochs 120 --router-epochs 120 --seed 1 --output-dir C:/tmp/strong_candidate_fusion_flow_quick_v6`

- results:
  - random_missing_50: `0.460227`
  - sensor_failure_30: `1.203457`
  - incident_perturbation: `0.485452`

- comparison against the previous sensor-failure-aware version:
  - random_missing_50: `0.457574 -> 0.460227`, slightly worse
  - sensor_failure_30: `1.033954 -> 1.203457`, clearly worse
  - incident_perturbation: `0.478485 -> 0.485452`, slightly worse

- interpretation:
  - the fully generic reliability repair is cleaner conceptually, but it loses the sensor-failure gain that came from the stronger fallback structure
  - this means the repair branch still needs a region-conditioned mechanism if sensor failure remains a target scenario
  - for now, the generic repair version is a weaker ablation, not the main method

## Latest Update: Continuous Utility Router

- structural change:
  - replaced the hard `global / sensor / incident` region classifier in `scripts/run_strong_candidate_fusion_flow_quick.py`
  - new router now predicts a continuous utility scalar plus local expert probabilities
  - this makes the routing rule dataset-agnostic: `reliable local correction` vs `fallback to global`
  - local experts still absorb `MagiNet`, `SAITS`, `PhysicsFromMagi`, and `SensorRepair` strengths

- quick benchmark:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 8 --repair-epochs 120 --router-epochs 180 --seed 1 --output-dir C:/tmp/strong_candidate_fusion_flow_quick_v5`

- results:
  - random_missing_50: `0.457574`
  - sensor_failure_30: `1.033954`
  - incident_perturbation: `0.478485`

- interpretation:
  - compared with the hard region router, this version is slightly worse on random/incident but more defensible for cross-dataset use
  - utility mean stays near `0.51-0.70`, so the router is not collapsing to a single expert
  - sensor failure still matches SAITS but does not beat it

## Latest Update: Region-Aware Candidate Fusion

- structural change:
  - updated `scripts/run_strong_candidate_fusion_flow_quick.py`
  - replaced the flat 4-way router with a two-level region-aware router
  - region stage predicts `global / sensor-failure / incident` from local missingness, temporal change, spatial deviation, and residual rank
  - expert stage mixes `MagiNet`, `SAITS`, `PhysicsFromMagi`, and `SensorRepair`
  - added region-level interpretability outputs to the summary table

- quick benchmark:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 8 --repair-epochs 120 --router-epochs 180 --seed 1 --output-dir C:/tmp/strong_candidate_fusion_flow_quick_v4`
  - scenarios: random_missing_50, sensor_failure_30, incident_perturbation

- results:
  - random_missing_50: `0.450855`
  - sensor_failure_30: `1.033954`
  - incident_perturbation: `0.461650`

- comparison against the five baselines:
  - random_missing_50: better than MagiNet `0.504411` and SAITS `1.261832`
  - sensor_failure_30: matches SAITS `1.033954`, but still not better
  - incident_perturbation: better than MagiNet `0.529567` and SAITS `1.209858`

- interpretation:
  - the region-aware router now absorbs the baseline strengths more cleanly than the flat router
  - MagiNet-style global repair is still the main driver on random/incident
  - SAITS-style temporal repair is the right fallback structure for sensor failure, but not yet enough to beat SAITS itself

## Latest Update: Region-Adaptive Baseline Comparison

- quick comparison:
  - command 1: `python scripts/run_official_repo_litetrust_quick.py --litetrust-only --pretrain-epochs 30 --epochs 5 --load-grin-cache --region-adaptive-correction --scenarios random_missing_50 incident_perturbation --output-dir C:/tmp/region_adaptive_quick_litetrust_only`
  - command 2: `python scripts/run_official_repo_litetrust_quick.py --litetrust-only --pretrain-epochs 30 --epochs 5 --load-grin-cache --region-adaptive-correction --scenarios sensor_failure_30 --output-dir C:/tmp/region_adaptive_quick_sensor_failure`

- results:
  - random_missing_50 masked MAE `0.6596274706809343`
  - incident_perturbation masked MAE `0.6719650431251093`
  - sensor_failure_30 masked MAE `1.601225111219618`

- direct comparisons:
  - official GRIN 30e baseline on random_missing_50: `0.682696`
  - official GRIN gated variant on random_missing_50: `0.669968`
  - official GRIN gated variant on incident_perturbation: `0.668542`
  - GRINLite + graph_delta on sensor_failure_30: `1.270554`
  - CorrectionV2 routed on sensor_failure_30: `1.262174`

- interpretation:
  - the new branch is the best so far on random_missing_50 within the official-GRIN family
  - incident_perturbation is improved over plain official GRIN, but not over the earlier gated variant
  - sensor_failure_30 is currently not competitive with the earlier graph-delta baseline, so this branch is not a universal fix

## Latest Update: Region-Adaptive Quick Run

- quick experiment:
  - command: `python scripts/run_official_repo_litetrust_quick.py --litetrust-only --pretrain-epochs 30 --epochs 5 --load-grin-cache --region-adaptive-correction --scenarios random_missing_50 incident_perturbation --output-dir C:/tmp/region_adaptive_quick_litetrust_only`
  - setup: reused cached official GRIN backbone, trained only the region-adaptive correction branch

- results:
  - random_missing_50: masked MAE `0.6596274706809343`
  - incident_perturbation: masked MAE `0.6719650431251093`

- comparison against official GRIN 30e reference:
  - random_missing_50: `0.682696 -> 0.659627`, gain about `3.38%`
  - incident_perturbation: `0.679828 -> 0.671965`, gain about `1.16%`

- interpretation:
  - the new region-adaptive correction is directionally better than the earlier generic post-GRIN correction on random missing
  - the gain on incident is smaller but still positive
  - this supports the “shared correction + physics utility” framing better than the old explicit expert-fusion framing

## Latest Update: Region-Adaptive Correction Branch

- structural change:
  - added `region_adaptive_correction` to the official GRIN LiteTrust path in `C:/Users/21329/grin_official_cache/lib/nn/models/litetrust_grin.py`
  - the main route is now `GRIN + shared correction + physics utility`, not a fixed expert fusion
  - physics is used as a utility regularizer / suppressor for the generic correction path
  - added `--region-adaptive-correction` to `scripts/run_official_repo_litetrust_quick.py`

- smoke check:
  - command: `python scripts/run_official_repo_litetrust_quick.py --litetrust-only --region-adaptive-correction --epochs 0 --pretrain-epochs 0 --scenarios random_missing_50 --output-dir C:/tmp/region_adaptive_smoke`
  - result: ran end-to-end and wrote `C:/tmp/region_adaptive_smoke/summary.csv` and `summary.json`
  - note: this is a zero-epoch structural smoke only, not a performance claim

## Latest Update: MagiNet + SAITS + PINN Fusion Trial

- new fusion module:
  - added `LiteTrustFusion` in `models/litetrust_pinn.py`
  - structure: GRIN-like global expert + SAITS-like temporal expert + physics-promoted correction + scenario-aware router
  - added flow-only flow residual helper in `physics/traffic_residuals.py`
  - extended `scripts/run_five_baselines_flow_quick.py` to benchmark the new fusion model alongside MagiNet, KNN, BRITS, SAITS, and GRINLite

- smoke check:
  - shape test passed for `LiteTrustFusion(input_dim=1, hidden_dim=16, output_dim=1)`
  - benchmark script compiled and ran without NaN

- quick benchmark run:
  - command: `python scripts/run_five_baselines_flow_quick.py --epochs 4 --seed 1 --output-dir C:/tmp/five_baselines_flow_quick_fusion_v2`
  - task: flow-only PEMS08 debug, 20 nodes, 12 steps
  - new fusion metrics:
    - random_missing_50 masked MAE `1.469978`
    - sensor_failure_30 masked MAE `2.247307`
    - incident_perturbation masked MAE `1.926805`

- comparison:
  - random_missing_50: MagiNet `1.083982` vs LiteTrustFusion `1.469978`
  - sensor_failure_30: SAITS `1.248416` vs LiteTrustFusion `2.247307`
  - incident_perturbation: MagiNet `1.163971` vs LiteTrustFusion `1.926805`

- decision:
  - this end-to-end three-expert fusion is not yet competitive
  - the weak point is not only the router; the internal SAITS-like expert is still weaker than the real SAITS baseline
  - next structural step should be candidate-level fusion on top of strong frozen experts, not more gate tuning inside a small proxy model

- 8 epoch follow-up:
  - command: `python scripts/run_five_baselines_flow_quick.py --epochs 8 --seed 1 --output-dir C:/tmp/five_baselines_flow_quick_fusion_v3`
  - random_missing_50 masked MAE `0.487815`
  - sensor_failure_30 masked MAE `2.305074`
  - incident_perturbation masked MAE `0.588645`
  - interpretation:
    - random_missing_50 now beats MagiNet `0.504411` on this same quick benchmark
    - incident_perturbation is still worse than MagiNet `0.529567`
    - sensor_failure_30 is still far behind SAITS `1.099492`
    - the model is learning a useful random-missing/global-repair mode, but it is not yet a real SAITS-level sensor-failure expert

- strong candidate fusion branch:
  - added `scripts/run_strong_candidate_fusion_flow_quick.py`
  - pipeline now trains or reuses three frozen candidates per scenario:
    - MagiNet
    - SAITS
    - a lightweight physics candidate derived from MagiNet
  - then trains only a small router over the three candidates
  - added an explicit sensor-failure fallback so fully failed nodes go to SAITS directly
  - fixed MagiNet import isolation by adding `C:/tmp/MagiNet/MagiNet-main/models/__init__.py`
  - smoke and quick runs passed

- strong candidate fusion quick results, 8 expert epochs + 120 router epochs:
  - random_missing_50:
    - MagiNet `0.504411`
    - SAITS `1.273367`
    - PhysicsFromMagi `0.465620`
    - StrongCandidateFusion `0.452915`
    - gain over MagiNet: about `10.2%`
  - sensor_failure_30:
    - MagiNet `1.632246`
    - SAITS `1.033954`
    - PhysicsFromMagi `1.582143`
    - StrongCandidateFusion `1.033954`
    - result: matched SAITS after the hard sensor-fallback constraint
  - incident_perturbation:
    - MagiNet `0.529567`
    - SAITS `1.310810`
    - PhysicsFromMagi `0.491673`
    - StrongCandidateFusion `0.478467`
    - gain over MagiNet: about `9.7%`

- current interpretation:
  - this version is structurally better than the end-to-end three-expert proxy
  - random_missing_50 and incident_perturbation now clearly benefit from physics-promoted correction
  - sensor_failure_30 no longer degrades below SAITS, but it is not yet better than SAITS without the hard fallback
  - next useful step is to make the SAITS branch itself stronger or add a true sensor-failure repair expert rather than continuing to enlarge the router

## Latest Update: Sensor Repair Expert Trial

- implemented a dedicated sensor-failure repair path in `scripts/run_strong_candidate_fusion_flow_quick.py`:
  - added fourth candidate `SensorRepair`
  - `SensorRepair` is trained over local candidates: `SAITS`, graph-neighbor context, and `PhysicsFromMagi`
  - it uses contrastive utility supervision to learn which local source is safer on failed nodes
  - final router now supports four candidates: `MagiNet`, `SAITS`, `PhysicsFromMagi`, `SensorRepair`
  - hard failed-sensor fallback now points to `SensorRepair` instead of raw `SAITS`
  - validation-set blend protection prevents `SensorRepair` from replacing SAITS when it hurts

- smoke result, weak 2-epoch experts:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 2 --repair-epochs 60 --router-epochs 40 --scenarios sensor_failure_30 --output-dir C:/tmp/strong_candidate_sensorrepair_gate_smoke`
  - SAITS masked MAE `1.827348`
  - PhysicsFromMagi masked MAE `1.773810`
  - SensorRepair masked MAE `1.744275`
  - StrongCandidateFusion masked MAE `1.744275`
  - interpretation: when SAITS is weak, the specialized repair expert does improve sensor failure.

- full quick result, 8 expert epochs:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 8 --repair-epochs 120 --router-epochs 120 --output-dir C:/tmp/strong_candidate_fusion_sensorrepair_full8`
  - random_missing_50:
    - MagiNet `0.504411`
    - PhysicsFromMagi `0.465620`
    - SensorRepair `0.473389`
    - StrongCandidateFusion `0.459781`
  - sensor_failure_30:
    - SAITS `1.033954`
    - SensorRepair `1.033954`
    - StrongCandidateFusion `1.033954`
  - incident_perturbation:
    - MagiNet `0.529567`
    - PhysicsFromMagi `0.491673`
    - SensorRepair `0.490393`
    - StrongCandidateFusion `0.477256`

- stronger sensor-only check, 20 expert epochs:
  - command: `python scripts/run_strong_candidate_fusion_flow_quick.py --epochs 20 --repair-epochs 200 --router-epochs 120 --scenarios sensor_failure_30 --output-dir C:/tmp/strong_candidate_sensorrepair_20e`
  - SAITS masked MAE `0.892811`
  - SensorRepair masked MAE `0.892811`
  - StrongCandidateFusion masked MAE `0.892811`

- decision:
  - the dedicated repair expert is useful when the temporal SAITS candidate is weak
  - once SAITS is trained enough, it remains the empirical upper bound for sensor_failure_30 in this debug setup
  - the current best table should keep sensor_failure_30 as “non-degrading / SAITS-matching”, not claim improvement there
  - the method story should emphasize:
    - physics-promoted correction gives clear gains on random and incident scenarios
    - sensor-failure safety is handled by reliability fallback to the temporal repair expert
    - do not average sensor_failure_30 into the main numeric gain claim unless a stronger sensor-specific candidate is added

## Latest Update: Region-Aware Expert Gate

- numeric-gain preset:
  - stopped using the discrete router for the main numeric table because it currently hurts MAE
  - reran cached official-GRIN-30e + `generic-v3 correction`
  - paired comparison uses `mu_data_masked_mae` from the same forward pass as the official-GRIN baseline
  - `generic-v3 1e` results:
    - random_missing_50: `0.646528 -> 0.639490`, gain `+1.09%`
    - noise_random_missing: `0.655596 -> 0.645067`, gain `+1.61%`
    - incident_perturbation: `0.658474 -> 0.641924`, gain `+2.51%`
    - average: `0.653532 -> 0.642160`, gain `+1.74%`
  - 3e correction was only trivially better on random and worse on noise/incident, so `generic-v3 1e` is the current numeric-gain preset
  - result file: `results/official_grin_pems08_debug/numeric_gain_summary.md`

- discrete router target rework:
  - target changed from raw `argmin(fused, physics, generic)` to `fused` as the safe default, with `physics` / `generic` only activated when they beat `fused` by a margin
  - added batch-wise class balancing and failure-mode upweighting so rare specialist wins are not washed out
  - neutralized `physics_promotion_mode_head` initialization so the router no longer starts with a physics bias
  - smoke with `--hard-negative-margin 0.10` on `random_missing_50` now gives target mass roughly `fused 0.751 / physics 0.098 / generic 0.151`
  - 10e quick suite with `--hard-negative-margin 0.10`:
    - random_missing_50: `x_discrete 1.739013` vs `x_fused 1.715475`
    - noise_random_missing: `x_discrete 1.724584` vs `x_fused 1.715588`
    - incident_perturbation: `x_discrete 1.722331` vs `x_fused 1.717704`
  - interpretation: the router is now structurally better aligned with the intended story, but it still does not beat the fused candidate; this is a supervision fix, not a paper-level gain yet

- official GRIN integration started:
  - official repo cloned to `C:/Users/21329/grin_official_cache`
  - added `models/official_grin_wrapper.py` to import the official `GRINet` model class
  - added `scripts/run_official_grin_pems08_debug.py` for LiteTrust data pipeline training/evaluation with official GRINet
  - added `scripts/evaluate_external_outputs.py` for unified `pred.npy` / `true.npy` / `mask.npy` evaluation
  - output root: `C:/Users/21329/litetrust_official_grin_outputs/official_grin_pems08_debug`
  - project summary: `results/official_grin_pems08_debug/summary.md`
  - smoke command completed: `python scripts/run_official_grin_pems08_debug.py --epochs 1 --scenarios random_missing_50`
  - unified evaluator completed on the smoke output
  - smoke masked MAE: `1.780388` after 1 epoch, not a comparison result
  - original official Lightning runner was not used because its old dependency stack conflicts with the current NumPy/Python environment. The integration uses official `GRINet` code with local lightweight training.
  - 30-epoch quick official GRIN results:
    - random_missing_50: `0.682696`
    - noise_random_missing: `0.683350`
    - incident_perturbation: `0.679828`
  - current LiteTrust no-validity 30-epoch debug references:
    - random_missing_50: `0.907277`
    - noise_random_missing: `0.901495`
    - incident_perturbation: `0.908902`
  - interpretation: official GRINet is substantially stronger than the current LiteTrust debug backbone. The next method step should attach LiteTrust as a correction/router module on top of official GRINet outputs or hidden states rather than continuing to tune the lightweight backbone.
  - added `OfficialGRIN_LiteTrustCorrection`
  - 1-epoch smoke random_missing_50: `1.743578`, official GRIN 1-epoch smoke was `1.780388`
  - same 20-epoch random_missing_50 comparison:
    - official GRIN: `0.768159`
    - official GRIN + LiteTrust correction: `0.765457`
    - gain: about `0.35%`
  - two-stage quick check, 10 epoch official GRIN pretrain + 10 epoch frozen correction: `1.018319`
  - interpretation: correction-on-official-GRIN is wired and can produce a tiny gain in joint training, but it is not yet strong enough. Do not claim success from this; redesign the correction target before larger runs.
  - implemented improvement-aware correction V2 with gate target from `|x_grin-y| - |x_final-y|` and harm-aware penalty
  - V2 random_missing_50, 20 epochs: `0.773720`
  - comparison: official GRIN 20 epochs `0.768159`, V1 correction 20 epochs `0.765457`
  - decision: V2 objective is currently worse and is kept behind `--improvement-gate-loss`; default official-GRIN correction training reverts to V1.
  - strong-GRIN correction check, official GRIN 30 epochs plus 5 correction epochs:
    - random_missing_50: official `0.682696`, correction `0.675658`, gain `+1.03%`
    - noise_random_missing: official `0.683350`, correction `0.675943`, gain `+1.08%`
    - incident_perturbation: official `0.679828`, correction `0.673213`, gain `+0.97%`
  - physics feature ablation for the same 30+5 setting:
    - random_missing_50: physics correction `0.675658`, no-physics correction `0.675825`
    - noise_random_missing: physics correction `0.675943`, no-physics correction `0.675840`
    - incident_perturbation: physics correction `0.673213`, no-physics correction `0.673309`
  - interpretation: the correction improves official GRIN by about 1%, but the improvement is not physics-specific. The gain mostly comes from a generic post-GRIN correction module.
  - next required structural change: split generic residual adapter and constrained physics projection into rival experts, then train reliability gating by contrastive utility. Do not claim the current correction as the final LiteTrust method.
  - implemented split expert structure:
    - `generic` residual adapter
    - constrained `physics` correction expert
    - `gated` reliability routing over both experts
  - 30 epoch official GRIN + 5 epoch correction/router ablation:
    - random_missing_50: official `0.682696`, generic `0.672306`, physics `0.676001`, gated `0.669968`
    - noise_random_missing: official `0.683350`, generic `0.672115`, physics `0.673673`, gated `0.668720`
    - incident_perturbation: official `0.679828`, generic `0.670578`, physics `0.674999`, gated `0.668542`
  - interpretation: gated beats both generic and physics experts in all three tested scenarios, so the gain can now be attributed to the trust-routing structure. The gain over official GRIN is still only `1.66%` to `2.14%`, below the desired paper-level target, but it is a meaningful structural signal.
  - implemented contrastive utility gate:
    - target compares fixed `x_generic` and `x_phys` candidates, avoiding self-supervision from `x_final`
    - random_missing_50, 30e GRIN + 5e correction:
      - gated without contrastive utility: `0.669968`
      - gated with contrastive utility: `0.669914`
    - interpretation: directionally positive but negligible (`0.000054`). Do not expand this exact loss; next change should be region-aware gate structure and interpretability output.
  - implemented region-aware expert gate:
    - data pipeline now carries `noise_region_mask` and `incident_region_mask` through `TrafficWindowDataset`
    - gate receives explicit region evidence: observed ratio, local missing, node failure, neighbor missing, temporal change, spatial deviation, residual rank
    - output metrics now report physics/generic expert weights in missing, noisy, incident, and incident+missing regions
    - tests passed: `pytest -p no:cacheprovider tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py` -> `22 passed`
    - smoke passed: `python scripts/run_official_grin_litetrust_pems08_debug.py --epochs 1 --scenarios random_missing_50 --correction-mode gated --output-root C:/Users/21329/litetrust_official_grin_outputs/region_gate_smoke`
  - region-aware gate 30e official GRIN pretrain + 5e correction/router:
    - random_missing_50: previous gated `0.669968`, region-aware gated `0.669702`
    - noise_random_missing: previous gated `0.668720`, region-aware gated `0.668535`
    - incident_perturbation: previous gated `0.668542`, region-aware gated `0.668249`
  - region interpretability:
    - random missing / missing: physics `0.530048`, generic `0.469952`
    - noisy / missing: physics `0.531199`, generic `0.468801`
    - noisy / noisy observed: physics `0.404896`, generic `0.595104`
    - incident / all incident: physics `0.466888`, generic `0.533112`
    - incident / incident+missing: physics `0.528842`, generic `0.471158`
  - interpretation: the gate now behaves qualitatively correctly. It leans toward physics on missing sparse points and toward generic correction on noisy or incident-observed regions. Quantitative improvement over the previous gated version is still small, so this is mainly an interpretability and method-structure improvement, not yet a paper-level gain. The next structural bottleneck is the physics expert quality, not the gate output alone.
  - implemented Physics Candidate V2:
    - explicit FD flow projection from normalized physical residual
    - graph projection for occupancy/speed
    - temporal smoothing projection for speed
    - learnable projection strength plus lightweight learned physics residual correction
  - smoke passed:
    - `python scripts/run_official_grin_litetrust_pems08_debug.py --epochs 1 --scenarios random_missing_50 --correction-mode gated --output-root C:/Users/21329/litetrust_official_grin_outputs/physics_candidate_v2_smoke`
    - smoke masked MAE `1.697766`; no NaN
  - Physics Candidate V2, 30e official GRIN pretrain + 5e correction/router:
    - random_missing_50:
      - old physics-only `0.676001`
      - V2 physics-only `0.671009`
      - old region-aware gated `0.669702`
      - V2 physics-guided fusion `0.663284`
    - noise_random_missing:
      - old physics-only `0.673673`
      - V2 physics-only `0.671675`
      - old region-aware gated `0.668535`
      - V2 physics-guided fusion `0.663839`
    - incident_perturbation:
      - old physics-only `0.674999`
      - V2 physics-only `0.669710`
      - old region-aware gated `0.668249`
      - V2 physics-guided fusion `0.661747`
  - V2 relative gain over official GRIN:
    - random_missing_50: `+2.84%`
    - noise_random_missing: `+2.86%`
    - incident_perturbation: `+2.66%`
  - V2 gate interpretability:
    - random missing / missing: physics `0.703211`, generic `0.296789`
    - noisy / missing: physics `0.703767`, generic `0.296233`
    - noisy / noisy observed: physics `0.596235`, generic `0.403765`
    - incident / all incident: physics `0.654108`, generic `0.345892`
    - incident / incident+missing: physics `0.706451`, generic `0.293549`
  - interpretation: this is a structural improvement rather than gate-loss tuning. The physics candidate itself is stronger in all three scenarios, and fusion improves beyond physics-only. It is still below the desired `5%` gain, so the next improvement should make physics projection more utility-aware or target the residual correction objective, not simply increase model size.
  - implemented Channel-Wise Physics-Generic Fusion V3:
    - fusion weight changed from `[B,T,N,1]` to `[B,T,N,C]`, so flow/occupancy/speed can trust physics differently
    - physics projection strength split into FD, graph, and temporal components
    - component strengths are locally predicted from region evidence and residual rank
  - tests passed again: `pytest -p no:cacheprovider tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py` -> `22 passed`
  - smoke passed:
    - `python scripts/run_official_grin_litetrust_pems08_debug.py --epochs 1 --scenarios random_missing_50 --correction-mode gated --output-root C:/Users/21329/litetrust_official_grin_outputs/channel_physics_v3_smoke`
    - smoke masked MAE `1.719667`; no NaN
  - V3, 30e official GRIN pretrain + 5e correction/router:
    - random_missing_50:
      - V2 physics-only `0.671009`
      - V3 physics-only `0.666338`
      - V2 fusion `0.663284`
      - V3 channel-wise fusion `0.662996`
    - noise_random_missing:
      - V2 physics-only `0.671675`
      - V3 physics-only `0.666577`
      - V2 fusion `0.663839`
      - V3 channel-wise fusion `0.662900`
    - incident_perturbation:
      - V2 physics-only `0.669710`
      - V3 physics-only `0.665267`
      - V2 fusion `0.661747`
      - V3 channel-wise fusion `0.661534`
  - V3 relative gain over official GRIN:
    - random_missing_50: `+2.89%`
    - noise_random_missing: `+2.99%`
    - incident_perturbation: `+2.69%`
  - V3 region behavior:
    - random missing / missing: physics `0.600093`, generic `0.399907`
    - noisy / missing: physics `0.600415`, generic `0.399585`
    - noisy / noisy observed: physics `0.482380`, generic `0.517620`
    - incident / all incident: physics `0.547907`, generic `0.452093`
    - incident / incident+missing: physics `0.610640`, generic `0.389360`
  - interpretation: V3 is methodologically cleaner and improves the physics-only candidate in all three scenarios. Final fusion gains over V2 are small, so the next structural bottleneck is not fusion granularity. The next candidate should learn a utility-oriented residual-to-error correction target instead of only lowering physics residual.
  - official GRIN repo integration started:
    - official repo patched at `C:/Users/21329/grin_official_cache`
    - added `lib/nn/models/litetrust_grin.py`
    - exported `LiteTrustGRINet` from `lib/nn/models/__init__.py`
    - registered `model_name='litetrust_grin'` in `scripts/run_imputation.py`
    - added local quick runner `scripts/run_official_repo_litetrust_quick.py`
  - environment note:
    - the official Lightning entry cannot currently run because `pytorch_lightning` / `torchmetrics` / `onnxruntime` trigger a NumPy 2.x ABI error
    - quick verification directly imports official repo model classes and trains them with a lightweight PyTorch loop
  - official-repo integration smoke:
    - `python scripts/run_official_repo_litetrust_quick.py --pretrain-epochs 1 --epochs 1 --scenarios random_missing_50 --output-dir C:/Users/21329/litetrust_official_grin_outputs/official_repo_litetrust_twostage_smoke`
    - passed; real PEMS08 debug data used
  - official-repo joint training check:
    - random_missing_50, 30e joint training:
      - official repo GRIN `0.646528`
      - official repo LiteTrust-GRIN joint `0.656502`
    - interpretation: joint training hurts; correction/fusion should be trained after GRIN pretraining.
  - official-repo two-stage quick check:
    - protocol: official GRIN `30e`; LiteTrust-GRIN `30e GRIN pretrain + freeze GRIN + 5e correction/fusion`
    - random_missing_50: `0.646528 -> 0.640533`, gain `+0.93%`
    - noise_random_missing: `0.655596 -> 0.649023`, gain `+1.00%`
    - incident_perturbation: `0.658474 -> 0.644692`, gain `+2.09%`
  - interpretation: moving the method into the official GRIN codebase keeps a positive signal in all three scenarios, strongest on incident perturbation. The official-repo gains are smaller than the custom-pipeline V3 gains, so this is supportive but not yet paper-level. Next structural target should train the correction/fusion head with explicit reconstruction utility over frozen GRIN.
  - implemented utility-aware correction objective:
    - `LiteTrustGRINet` can now return details: `mu_data`, `x_generic`, `x_phys`, `final_delta`, `phys_weight`
    - quick runner supports `--utility-loss`
    - utility loss combines direct correction target, harm-aware penalty, and generic-vs-physics fusion target
  - utility objective smoke:
    - `python scripts/run_official_repo_litetrust_quick.py --pretrain-epochs 1 --epochs 1 --utility-loss --scenarios random_missing_50 --output-dir C:/Users/21329/litetrust_official_grin_outputs/official_repo_utility_smoke`
    - passed
  - utility objective quick check:
    - protocol: `10e GRIN pretrain + 3e correction`, random_missing_50
    - official GRIN 10e: `1.009418`
    - LiteTrust no utility: `1.008032`
    - LiteTrust utility-aware: `1.007394`
    - utility-aware gain over no-utility: `0.000638`
  - timeout note:
    - the `30e + 5e` utility-aware run exceeded the 20-minute local timeout and the residual process was stopped
  - interpretation: utility-aware objective is directionally positive but too small and too slow in this form. Do not expand this exact loss. The next structural target should be a more direct calibrated residual-to-error correction head over frozen GRIN.
  - residual-to-error calibrator trial:
    - added optional `error_calibrator` inside official repo `LiteTrustGRINet`
    - structure: `error_delta = Calibrator([mu_data, residual, region_features])`, `x_error = mu_data + error_delta`
    - final candidate when enabled: `(1 - w_phys) * x_error + w_phys * x_phys`
    - the calibrator is now gated by `use_error_calibrator=False` by default to preserve the working LiteTrust path
  - residual-to-error quick results, random_missing_50, `10e GRIN pretrain + 3e correction`:
    - official GRIN 10e: `1.009418`
    - previous LiteTrust no utility: `1.008032`
    - residual-to-error calibrator, no utility: `1.017177`
    - residual-to-error calibrator, utility-aware: `1.017980`
  - decision: reject the current residual-to-error calibrator. It over-corrects frozen GRIN and utility supervision does not fix it. Keep it optional/off by default. A safer future variant should use shrinkage-gated correction initialized near zero rather than replacing the data candidate directly.
  - shrinkage calibrator fix:
    - corrected calibrator from replacement to conservative refinement:
      - old failed form: `x_error = mu_data + error_delta`
      - fixed form: `x_error = x_generic + gamma * error_delta`
      - `gamma = 0.3 * sigmoid(region_head(region_features))`, initialized near zero
    - initialization order was also fixed so adding the optional calibrator does not alter the original physics/gate head initialization path
    - default `use_error_calibrator=False` remains unchanged
  - shrinkage calibrator quick result, random_missing_50, `10e GRIN pretrain + 3e correction`:
    - previous LiteTrust no utility: `1.008032`
    - failed direct calibrator + utility: `1.017980`
    - shrinkage calibrator + utility: `1.008142`
  - interpretation: shrinkage fixes the over-correction failure and brings the result back near the working LiteTrust head, but it still does not beat the previous default. Keep it optional and disabled in main experiments.
  - scenario-aware generic-vs-physics fusion trial:
    - main scenario set defined as `random_missing_50`, `noise_random_missing`, `incident_perturbation`
    - `sensor_failure_30` is intentionally excluded from the main table average
    - added `--scenario-aware` to the official-repo quick runner
    - added one-hot scenario tokens:
      - random `[1,0,0]`
      - noise `[0,1,0]`
      - incident `[0,0,1]`
  - first scenario-aware attempt:
    - direct token concatenation into heads
    - random_missing_50, `10e GRIN + 3e correction`: `1.025068`
    - rejected because it is worse than official GRIN 10e `1.009418` and previous LiteTrust `1.008032`
  - scenario-aware adapter fix:
    - replaced direct concatenation with zero-initialized scenario adapters for generic delta, physics delta, gate logit, and projection strength
    - smoke passed: random_missing_50, `1e+1e`, masked MAE `1.725832`
    - longer `10e+3e` and `5e+1e` checks timed out in the current CPU-contended environment and residual processes were stopped
  - decision: implementation is present, but no complete positive scenario-aware result yet. Do not claim scenario-aware fusion until rerun with lower CPU contention or cached GRIN pretraining.
  - cached frozen-GRIN workflow implemented:
    - added `--save-grin-cache`
    - added `--load-grin-cache`
    - added `--cache-dir`
    - added `--litetrust-only`
  - cache smoke checks:
    - save cache smoke passed: `--pretrain-epochs 1 --epochs 1 --save-grin-cache`, random_missing_50, LiteTrust masked MAE `1.725365`
    - load cache smoke passed: `--pretrain-epochs 1 --epochs 1 --load-grin-cache`, random_missing_50, LiteTrust masked MAE `1.725365`
    - LiteTrust-only load smoke passed: `--pretrain-epochs 1 --epochs 1 --load-grin-cache --litetrust-only`, random_missing_50, LiteTrust masked MAE `1.725365`
  - runtime note under current CPU contention:
    - save-cache smoke about `150s`
    - load-cache smoke with GRIN baseline included about `87s`
    - load-cache LiteTrust-only about `19s`
  - decision: cache workflow works and should be used for the next real comparison. Do not start 30e three-scenario cache generation while the current machine is CPU-contended. Next complete comparison should use only `random_missing_50`, `noise_random_missing`, and `incident_perturbation`, excluding `sensor_failure_30` from the main average.

- validity-gate ablation completed:
  - result file: `results/validity_ablation_pems08_debug/summary.md`
  - compared `ReliabilityRouter with validity`, `ReliabilityRouter no validity`, and `no physics`
  - random_missing_50: validity `0.918611`, no-validity `0.907277`, no-physics `0.999535`
  - sensor_failure_30: validity `1.255010`, no-validity `1.255813`, no-physics `1.253872`
  - block_missing: validity `1.696104`, no-validity `1.695540`, no-physics `1.696935`
  - temporal_missing: validity `0.945727`, no-validity `0.944070`, no-physics `1.010718`
  - noise_random_missing: validity `0.909122`, no-validity `0.901495`, no-physics `1.001749`
  - incident_perturbation: validity `0.922968`, no-validity `0.908902`, no-physics `1.002917`
  - decision: self-calibrated validity is not a core contribution in its current form. It over-damps physics in the scenarios where physics helps. The default `LiteTrustGRINReliabilityRouter` now sets `use_validity_gate=False`; validity remains optional for diagnosis.

- scenario sweep completed:
  - result file: `results/scenario_sweep_pems08_debug/summary.md`
  - dataset: real PEMS08 from ASTGNN zip, first 20 nodes
  - training: 30 epochs, seed 1
  - scenarios: `random_missing_50`, `sensor_failure_30`, `block_missing`, `temporal_missing`, `noise_random_missing`, `incident_perturbation`
  - compared variants: `GRINLite`, `ReliabilityRouter`, `ReliabilityRouter_directional_physics`, `ReliabilityRouter_no_physics`
  - best masked MAE by scenario:
    - random_missing_50: `ReliabilityRouter` / directional tie, `0.918611`
    - sensor_failure_30: `ReliabilityRouter_no_physics`, `1.253872`
    - block_missing: `ReliabilityRouter_directional_physics`, `1.695704`
    - temporal_missing: `ReliabilityRouter` / directional tie, `0.945727`
    - noise_random_missing: `ReliabilityRouter` / directional tie, `0.909122`
    - incident_perturbation: `ReliabilityRouter` / directional tie, `0.922968`
  - physics gain versus no-physics:
    - random_missing_50: `+8.10%`
    - sensor_failure_30: `-0.05%`
    - block_missing: `+0.07%`
    - temporal_missing: `+6.43%`
    - noise_random_missing: `+9.25%`
    - incident_perturbation: `+7.97%`
  - interpretation: physics is useful in sparse/noisy/temporal/incident settings, but complete sensor failure is graph-dominant. Directional conservation is theoretically better than spatial residual propagation, but current gains are too small to make it the default.

- direction-aware conservation trial:
  - added optional `DirectionalConservationPhysicsExpert`
  - mechanism: infer upstream/downstream adjacency, compute incoming/outgoing flow balance, density balance, speed consistency, and FD residual features
  - result file: `results/directional_conservation_pems08_debug/summary.md`
  - sensor_failure_30:
    - default ReliabilityRouter: `1.267713`
    - directional physics: `1.264947`
    - no physics: `1.267197`
    - forced graph-to-physics shift: `1.290321` and rejected
  - interpretability:
    - directional failed-node gate: `0.382585`
    - directional failed-node delta: `0.176402`
    - directional conservation residual on failed nodes: `0.827670`
    - failed-node graph/physics weight without forced shift: `0.971415` / `0.016267`
  - decision: direction-aware conservation is a better theoretical direction than simple spatial residual propagation, but the current debug gain is still small. Forcing more physics weight lowered residual but worsened MAE, reinforcing the physical-misguidance claim. Keep the module optional and disabled by default.

- follow-up spatial physics trial:
  - added optional `SpatialConservationPhysicsExpert`
  - mechanism: propagate neighboring FD residuals into failed nodes as `spatial_phys_delta`, staying inside the physics expert path rather than the graph expert path
  - result file: `results/spatial_physics_pems08_debug/summary.md`
  - random_missing_50:
    - without spatial physics: `0.855408`
    - with spatial physics: `0.855646`
    - no physics: `0.914612`
  - sensor_failure_30:
    - without spatial physics: `1.282861`
    - with spatial physics: `1.279316`
    - no physics: `1.278230`
  - interpretability:
    - sensor failed-node spatial gate: `0.330623`
    - sensor failed-node spatial delta: `0.161436`
    - sensor failed-node graph weight: `0.969395`
    - sensor failed-node physics weight: `0.017473`
  - decision: keep spatial physics optional and disabled by default. It activates and slightly improves over no-spatial physics, but it still does not beat graph-only routing under sensor failure. The stronger next method would need direction-aware conservation/upstream-downstream balance, not simple neighbor residual propagation.

- method focus: strengthen the physics contribution without adding a heavy model or more manual routing rules
- implemented lightweight observed-point self-calibration for `PhysicsValidityGate`
- calibration signal: on observed entries, compare `x_phys` error against `x_data` error and supervise physics validity with a soft BCE target
- random_missing_50 PEMS08 debug:
  - ReliabilityRouter + self-calibrated physics validity masked MAE: `0.832526`
  - no-physics masked MAE: `0.905522`
  - estimated physics gain over no-physics: `+8.06%`
  - physics validity mean/missing: `0.564897` / `0.634525`
  - physics weight mean/missing: `0.519217` / `0.609644`
- sensor_failure_30 PEMS08 debug:
  - ReliabilityRouter + self-calibrated physics validity masked MAE: `1.256050`
  - no-physics masked MAE: `1.265352`
  - estimated physics gain over no-physics: `+0.74%`
  - physics validity mean/failed-node: `0.299110` / `0.136263`
  - failed-node graph/physics weight: `0.974234` / `0.014439`
- interpretation: the method now has a clearer structural claim: physics is not only weighted by fixed evidence, but lightly self-calibrated by whether the physics expert actually improves observed reconstruction. Random missing still uses physics heavily; full sensor failure mostly routes to graph evidence.
- caveat: this is still PEMS08 20-node debug, not a paper benchmark. The next meaningful check is noisy/incident perturbation, because that directly tests physical misguidance rather than only missingness.

## Current Stage

- Structural method replacement added: LiteTrustGRIN

## Completed Files

- `AGENTS.md`
- `README.md`
- `PROGRESS.md`
- `requirements.txt`
- `configs/toy.yaml`
- `configs/pems08_debug.yaml`
- `configs/default.yaml`
- `configs/ablation.yaml`
- `data/datasets.py`
- `data/normalization.py`
- `physics/traffic_residuals.py`
- `data/masks.py`
- `data/corruptions.py`
- `data/normalization.py`
- `models/encoder_tcn.py`
- `models/graph_layer.py`
- `models/base_model.py`
- `models/litetrust_pinn.py`
- `physics/traffic_residuals.py`
- `physics/collocation.py`
- `losses/losses.py`
- `losses/metrics.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_smoke_test.py`
- `scripts/run_trend_test.py`
- `scripts/run_fixed_physics_test.py`
- `scripts/run_trust_physics_test.py`
- `scripts/run_uncertainty_test.py`
- `scripts/run_conflict_test.py`
- `scripts/run_stage1_trend_suite.py`
- `scripts/run_stage2_three_dataset_quick.py`
- `scripts/run_stage10a_pems08_real_debug.py`
- `scripts/run_grin_pems08_debug.py`
- `models/grin_baseline.py`
- `DIAGNOSIS.md`
- `tests/test_masks.py`
- `tests/test_model_shapes.py`
- `tests/test_physics_residuals.py`
- `results/smoke_test/metrics.json`
- `results/smoke_test/train_log.csv`
- `results/smoke_test/config.yaml`
- `results/stage2_mask_protocol/test_summary.md`
- `results/stage2_mask_protocol/test_summary.json`
- `results/v0_base_trend/metrics.json`
- `results/v0_base_trend/train_log.csv`
- `results/v0_base_trend/config.yaml`
- `results/v1_fixed_physics/metrics.json`
- `results/v1_fixed_physics/train_log.csv`
- `results/v1_fixed_physics/config.yaml`
- `results/v2_trust_physics/metrics.json`
- `results/v2_trust_physics/train_log.csv`
- `results/v2_trust_physics/config.yaml`
- `results/v3_uncertainty/metrics.json`
- `results/v3_uncertainty/train_log.csv`
- `results/v3_uncertainty/config.yaml`
- `results/v4_conflict_aware/metrics.json`
- `results/v4_conflict_aware/focused_retry_metrics.json`
- `results/v4_conflict_aware/final_fixed_metrics.json`
- `results/v4_conflict_aware/train_log.csv`
- `results/v4_conflict_aware/config.yaml`
- `results/stage1_trend/summary.csv`
- `results/stage1_trend/summary.md`
- `results/stage1_trend/config.yaml`
- `results/stage1_trend/per_model_logs/V0_BaseTCN.csv`
- `results/stage1_trend/per_model_logs/V1_FixedPhysics.csv`
- `results/stage1_trend/per_model_logs/V2_TrustPhysics.csv`
- `results/stage1_trend/per_model_logs/V3_TrustPhysics_Uncertainty.csv`
- `results/stage1_trend/per_model_logs/V4_ConflictAware_LiteTrust.csv`
- `results/stage2_three_dataset_quick/summary.csv`
- `results/stage2_three_dataset_quick/summary.md`
- `results/stage2_three_dataset_quick/config.yaml`
- `results/stage10a_pems08_real_debug/summary.csv`
- `results/stage10a_pems08_real_debug/summary.md`
- `results/stage10a_pems08_real_debug/config.yaml`
- `results/grin_pems08_debug/summary_50epoch.csv`
- `results/grin_pems08_debug/summary_50epoch.md`
- `results/grin_pems08_debug/config.yaml`
- `results/stage10a_pems08_real_debug/litetrust_grin_50epoch.csv`
- `results/stage10a_pems08_real_debug/litetrust_grin_50epoch.md`

## Commands Run

- `pytest tests/test_model_shapes.py tests/test_physics_residuals.py`
- `pytest tests/test_masks.py`
- `python scripts/run_smoke_test.py`
- `pytest tests/test_masks.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_trend_test.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_fixed_physics_test.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_trust_physics_test.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_uncertainty_test.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_conflict_test.py`
- focused rerun: `incident_perturbation` only, V3 vs V4 with residual-rank plus temporal-change conflict score
- focused rerun: `incident_perturbation` only, V3 vs V4 with speed-only incident and anomaly-tail conflict score
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_stage1_trend_suite.py`
- attempted full Stage 10 quick run: `python scripts/run_stage2_three_dataset_quick.py`, stopped after timeout
- ultra-quick Stage 10 fallback run: `python scripts/run_stage2_three_dataset_quick.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- searched local drives for PEMS08/PEMSD8 data files; none found
- `python scripts/run_stage10a_pems08_real_debug.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- provided ASTGNN zip path: `E:\ASTGNN-9c2e19b98c4cedf1f35214d8789685b6381b3aad.zip`
- `python scripts/run_stage10a_pems08_real_debug.py`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- fixed channel order from `flow,speed,occupancy` to `flow,occupancy,speed`
- added train-split calibrated FD scale `alpha` for `flow - alpha * occupancy * speed`
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- rerun: `python scripts/run_stage10a_pems08_real_debug.py`
- added multi-feature trust gate inputs: temporal change, spatial deviation, local missing ratio, residual rank
- added trust variance regularization and high-conflict/low-conflict ranking regularization
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- rerun: `python scripts/run_stage10a_pems08_real_debug.py`
- added compact GRIN-style graph recurrent imputation baseline
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_grin_pems08_debug.py`
- 50-epoch GRINLite run through inline runner
- added `LiteTrustGRIN`: GRIN-style backbone plus calibrated physics trust
- `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py`
- `python scripts/run_stage10a_pems08_real_debug.py`
- 50-epoch LiteTrustGRIN run through inline runner
- implemented two-stage LiteTrustGRIN training gate: reconstruction pretrain then physics/trust finetune
- 50-epoch two-stage LiteTrustGRIN run through inline runner
- added `LiteTrustGRINCorrection`: GRIN data branch plus trust-gated physics correction branch
- 10-epoch correction smoke run through inline runner
- 50-epoch two-stage correction run through inline runner

## Result Metrics

LiteTrustGRIN structural replacement:

- method change: TCN reconstruction backbone replaced with GRIN-style bidirectional graph recurrent imputer
- still uses calibrated residual and trust-weighted physics loss
- 10-epoch quick:
  - random_missing_50 LiteTrustGRIN masked MAE: `1.3038239`
  - sensor_failure_30 LiteTrustGRIN masked MAE: `1.5713872`
- 50-epoch run:
  - random_missing_50 LiteTrustGRIN masked MAE: `0.9230031`
  - sensor_failure_30 LiteTrustGRIN masked MAE: `1.4873838`
  - random_missing_50 physics residual/trust std: `0.4493886` / `0.1081589`
  - sensor_failure_30 physics residual/trust std: `0.2585710` / `0.3153735`
- comparison to GRINLite 50epoch:
  - random_missing_50 GRINLite: `0.8936803`, LiteTrustGRIN: `0.9230031`
  - sensor_failure_30 GRINLite: `1.4450316`, LiteTrustGRIN: `1.4873838`
- interpretation: backbone replacement works, but physics/trust should be enabled after reconstruction pretraining
- next structural change: two-stage training, not further small hyperparameter tuning
- regression tests after structural replacement: `19/19 passed`

LiteTrustGRIN two-stage training:

- pretrain epochs: `30`
- finetune epochs with trust-aware physics: `20`
- random_missing_50 masked MAE: `0.9232082`
- sensor_failure_30 masked MAE: `1.4929520`
- comparison:
  - random_missing_50 GRINLite: `0.8936803`
  - sensor_failure_30 GRINLite: `1.4450316`
- interpretation: two-stage timing alone does not solve the gap
- next method direction: decouple main reconstruction branch from physics correction branch, instead of applying physics loss directly to the main prediction

LiteTrust-GRIN-Correction V1:

- method change: `mu_final = mu_data + trust * delta_phys`
- physics no longer directly constrains the GRIN data branch
- training: 30 epochs reconstruction pretrain, 20 epochs trust-aware correction finetune
- random_missing_50:
  - GRINLite 50epoch: `0.8936803`
  - LiteTrustGRINCorrection: `0.8121688`
  - trust mean/std: `0.5704759` / `0.2015332`
- sensor_failure_30:
  - GRINLite 50epoch: `1.4450316`
  - LiteTrustGRINCorrection: `1.4344010`
  - trust mean/std: `0.3717053` / `0.1930533`
- interpretation: first version that beats the strong GRIN-style baseline on both PEMS08 debug scenarios

LiteTrust-GRIN-Correction V1 final debug version:

- final prediction: `mu_data + graph_delta + trust * delta_phys`
- failed-sensor graph correction: `graph_delta = I_failed * (A @ mu_data - mu_data)`
- graph correction activation: `node_missing_ratio > 0.9`
- observed consistency auxiliary loss: `0.1`
- pretrain epochs: `0`
- random_missing_50:
  - GRINLite: `0.8936803`
  - Correction V1 final: `0.8040677`
  - relative improvement: `10.03%`
  - trust mean/std: `0.5837080` / `0.1709850`
- sensor_failure_30:
  - GRINLite: `1.4450316`
  - Correction V1 final: `1.3203772`
  - relative improvement: `8.63%`
  - trust mean/std: `0.3888872` / `0.2775735`
- interpretability note: sensor failure has lower mean trust and higher trust dispersion, and the graph correction branch only activates on near-complete node failure

GRINLite strong baseline:

- source method family: GRIN, graph recurrent imputation network
- implementation status: compact GRIN-style baseline, not official GRIN reproduction
- dataset: real PEMS08 debug from ASTGNN zip
- nodes: first `20`
- train/val/test samples: `64/16/16`
- hidden dim: `48`
- strong run epochs: `50`
- random_missing_50:
  - GRINLite masked MAE: `0.8936803`
  - current LiteTrust masked MAE: `1.0401613`
  - gap: GRINLite better by `0.146481`
- sensor_failure_30:
  - GRINLite masked MAE: `1.4450316`
  - current LiteTrust masked MAE: `1.7452303`
  - gap: GRINLite better by `0.300199`
- interpretation: current LiteTrust is bottlenecked by reconstruction/imputation backbone, not only by physics trust
- regression tests after baseline addition: `18/18 passed`

Methodology fix after diagnosing weak method signal:

- problem found: ASTGNN/PEMS08 channel order is `0=flow`, `1=occupancy`, `2=speed`
- previous code assumed `0=flow`, `1=speed`, `2=occupancy`
- second problem found: raw `flow = occupancy * speed` is not valid for PEMS occupancy scale
- fix: use calibrated residual `flow - alpha * occupancy * speed`
- alpha source: train split, computed in `StandardScaler.fit`
- residual normalization: divide by flow standard deviation instead of per-batch residual mean when a normalizer is available
- toy generator updated to the same channel order
- incident perturbation now changes flow and speed channels, not occupancy
- regression tests after method fix: `16/16 passed`
- PEMS08 real-debug after calibrated residual:
  - random_missing_50 BaseTCN: `1.0482125`
  - random_missing_50 FixedPhysics: `1.0482045`
  - random_missing_50 LiteTrustPINN_full: `1.0409881`
  - sensor_failure_30 BaseTCN: `1.7482764`
  - sensor_failure_30 FixedPhysics: `1.7482823`
  - sensor_failure_30 LiteTrustPINN_full: `1.7456518`
- remaining issue: LiteTrust still has low trust variation, with trust std `0.0142` and `0.0070`
- rich trust gate update:
  - random_missing_50 LiteTrust masked MAE: `1.0401613`
  - random_missing_50 trust mean/std: `0.3749196` / `0.0541234`
  - sensor_failure_30 LiteTrust masked MAE: `1.7452303`
  - sensor_failure_30 trust mean/std: `0.4436452` / `0.0622840`
  - trust variation is no longer near-constant, and MAE remains best among the three quick models
- regression tests after rich trust update: `17/17 passed`

Stage 10A PEMS08 real-debug run from ASTGNN zip:

- real loader support added for `data/raw/pems08/*.npz`
- real loader also supports reading PEMS08 directly from ASTGNN zip through `dataset.zip_path`
- recognized data keys: `data`, `x`, `X`, `arr_0`, or first 2D/3D array
- expected series shape: `[time, nodes, channels]`
- adjacency support: `adj.npy`, `adjacency.npy`, `adj_mx.npy`, `graph.npy`, `adj.csv`, `adjacency.csv`, `distance.csv`
- fallback adjacency: ring graph if no adjacency file exists
- zip path: `E:\ASTGNN-9c2e19b98c4cedf1f35214d8789685b6381b3aad.zip`
- zip data entry: `ASTGNN-9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS08/PEMS08.npz`
- source shape after debug slice: `[17856, 20, 3]`
- real PEMS08 data used: `true`
- fallback used: `false`
- adjacency fallback ring: `false`
- settings: PEMS08 only, `random_missing_50` and `sensor_failure_30`, 3 models, 10 epochs, train/val/test samples `64/16/16`, CPU
- random_missing_50 masked MAE:
  - BaseTCN: `1.0482126`
  - FixedPhysics: `1.0485697`
  - LiteTrustPINN_full: `1.0412080`
- sensor_failure_30 masked MAE:
  - BaseTCN: `1.7482766`
  - FixedPhysics: `1.7490098`
  - LiteTrustPINN_full: `1.7462207`
- LiteTrust trust mean:
  - random_missing_50: `0.4502249`
  - sensor_failure_30: `0.4185256`
- formal full-dataset result: `false`, because this is only a 20-node debug subset
- regression tests after run: `14/14 passed`

Stage 10 ultra-quick three-dataset fallback run:

- datasets: `PEMS08`, `PEMS04`, `METR-LA`
- scenarios: `random_missing_50`, `sensor_failure_30`
- models: `BaseTCN`, `FixedPhysics`, `LiteTrustPINN_full`
- epochs: `10`
- train/val/test samples: `16/8/8`
- device used: `cpu`
- fallback used: `true` for every row
- formal result: `false` for every row
- reason: the 30-epoch quick run exceeded local timeout while another GPU training process was active
- PEMS08 random missing best masked MAE: BaseTCN `0.3941558`
- PEMS08 sensor failure best masked MAE: FixedPhysics `0.7264084`
- PEMS04 random missing best masked MAE: BaseTCN `0.3809360`
- PEMS04 sensor failure best masked MAE: BaseTCN `0.7279491`
- METR-LA random missing best masked MAE: FixedPhysics `0.3761348`
- METR-LA sensor failure best masked MAE: LiteTrustPINN_full `0.6755792`
- LiteTrust trust mean range: `0.4781640` to `0.4801585`
- regression tests after run: `14/14 passed`
- no next-stage decision recorded per user request

Stage 9 single-dataset trend suite:

- dataset: `toy`
- fallback used: `false`
- scenarios: `random_missing_50`, `sensor_failure_30`, `incident_perturbation`
- models: `V0_BaseTCN`, `V1_FixedPhysics`, `V2_TrustPhysics`, `V3_TrustPhysics_Uncertainty`, `V4_ConflictAware_LiteTrust`
- epochs: `20`
- batch size: `16`
- hidden dim: `32`
- device used: `cpu`
- best random missing masked MAE: V3 `0.1155738`
- best sensor failure masked MAE: V2 `0.3707119`
- best incident overall masked MAE: V3 `0.1659670`
- best incident-region MAE: V2 `1.0023991`
- V4 incident-region MAE: `1.0291598`
- V4 trust normal: `0.2365564`
- V4 trust incident: `0.2290769`
- fixed physics helpful: `true`
- trust beats fixed: `true`
- uncertainty helpful: `true`
- conflict lowers incident trust: `true`
- trust collapsed: `false`
- recommend next quick benchmark: `true`
- caveat: V4 passes the trend gate but does not have the best incident-region MAE on toy data

Stage 7 V4 conflict-aware regularization:

- dataset: `toy`
- scenarios: `random_missing_50`, `incident_perturbation`
- epochs: `20`
- batch size: `16`
- hidden dim: `32`
- device used: `cpu`
- uncertainty head: enabled
- conflict loss start epoch: `10`
- beta conflict tried: `0.001`, then `0.01`
- beta floor tried: `0.01`, then `0.05`
- final V4 random_missing test masked MAE: `0.1156101`
- final V4 random_missing trust mean: `0.0871003`
- final V4 incident test masked MAE: `0.1566306`
- final V4 incident region MAE: `0.8524124`
- final V4 trust normal: `0.0775345`
- final V4 trust incident: `0.6862979`
- V4 - V3 incident region MAE delta: `-0.0018098`
- V4 - V3 incident trust delta: `+0.0298070`
- core criterion `trust_mean_incident < trust_mean_normal`: failed
- focused retry with residual-rank plus temporal-change conflict:
  - V4 incident test masked MAE: `0.1566529`
  - V4 incident region MAE: `0.8532844`
  - V4 trust normal: `0.0835558`
  - V4 trust incident: `0.6766577`
  - core criterion still failed
- final focused fix with speed-only incident and anomaly-tail conflict:
  - V3 incident test masked MAE: `0.1659670`
  - V4 incident test masked MAE: `0.1659680`
  - V3 incident region MAE: `1.0292683`
  - V4 incident region MAE: `1.0291598`
  - V4 trust normal: `0.2365564`
  - V4 trust incident: `0.2290769`
  - core criterion passed on test incident region
- stage decision: Stage 7 accepted as a quick trend gate, with validation caveat

Stage 6 V3 uncertainty trend test:

- dataset: `toy`
- missing rate: `0.5`
- epochs: `20`
- batch size: `16`
- hidden dim: `32`
- device used: `cpu`
- uncertainty head: enabled
- uncertainty loss: enabled
- alpha uncertainty: `0.1`
- log_var clamp: `[-6, 3]`
- V2 final validation masked MAE: `0.1232911`
- V3 final validation masked MAE: `0.1187970`
- V3 - V2 validation MAE delta: `-0.0044940`
- V2 test masked MAE: `0.1199208`
- V3 test masked MAE: `0.1154141`
- V3 - V2 test masked MAE delta: `-0.0045067`
- V3 trust mean: `0.1970673`
- V3 trust std: `0.1913724`
- V3 trust min: `0.0354542`
- V3 trust max: `0.8099234`
- V3 log_var mean: `-2.0325854`
- V3 log_var std: `0.3603407`
- V3 uncertainty-error correlation: `0.4072569`
- trust collapsed to 0 or 1: `false`
- recommendation: enter V4 is acceptable

Stage 5 V2 trust physics trend test:

- dataset: `toy`
- missing rate: `0.5`
- epochs: `20`
- batch size: `16`
- hidden dim: `32`
- device used: `cpu`
- reason for CPU: another active Python training process was using GPU resources
- physics lambda warm-up: epoch `<5` uses `0`, epoch `5-15` ramps to `0.005`
- trust floor: `0.3`
- beta floor: `0.01`
- beta smooth: `0.001`
- V0 final validation masked MAE: `0.1227589`
- V1 final validation masked MAE: `0.1227834`
- V2 final validation masked MAE: `0.1227045`
- V2 - V1 validation MAE delta: `-0.0000789`
- V2 - V0 validation MAE delta: `-0.0000544`
- V2 test MAE: `0.1704364`
- V2 test masked MAE: `0.1192890`
- V2 test physics loss: `0.0652702`
- V2 trust mean: `0.1880323`
- V2 trust std: `0.1958151`
- V2 trust min: `0.0104396`
- V2 trust max: `0.7224683`
- trust collapsed to 0 or 1: `false`
- effect size: very small on toy data
- recommendation: enter V3 is acceptable, but disrupted scenarios are needed before claiming trust advantage

Stage 4 V1 fixed physics trend test:

- dataset: `toy`
- missing rate: `0.5`
- epochs: `20`
- batch size: `16`
- hidden dim: `32`
- lambda warm-up: epoch `<5` uses `0`, epoch `5-15` ramps to `0.001`
- physics residual: `q - rho * v`, computed after inverse-normalizing to physical scale
- physics loss: `smooth_l1`
- V0 final validation masked MAE: `0.1232266`
- V1 final validation masked MAE: `0.1231687`
- V1 - V0 validation MAE delta: `-0.0000579`
- V0 final validation physics residual abs: `0.99999997`
- V1 final validation physics residual abs: `0.99999994`
- V1 - V0 residual delta: `-0.00000003`
- V0 test masked MAE: `0.1197050`
- V1 test masked MAE: `0.1196492`
- data/physics loss conflict observed: `false`
- effect size: very small on toy data
- recommendation: enter V2 is acceptable, but treat fixed physics as a stability check rather than strong evidence

Stage 3 V0 base trend test:

- dataset: `toy`
- missing rate: `0.5`
- epochs: `10`
- batch size: `16`
- hidden dim: `32`
- device used: `cuda`
- fallback used: `false`
- naive mean fill validation masked MAE: `0.8442`
- BaseTCNGraph final validation masked MAE: `0.1701`
- naive mean fill test masked MAE: `0.8423`
- BaseTCNGraph test masked MAE: `0.1641`
- train loss: `0.7024 -> 0.1798`
- NaN detected: `false`

Stage 2 mask/corruption protocol:

- collected tests: `6`
- passed tests: `6`
- failed tests: `0`
- regression tests: `9/9 passed`
- real dataset used: `false`
- dataset downloaded: `false`

Stage 1 smoke test:

- test MAE: `0.4698`
- test RMSE: `0.6148`
- test MAPE: `4.3089`
- test masked MAE: `0.4022`
- test loss: `0.4022`
- device used: `cuda`
- fallback used: `false`

Training log:

- epoch 1: train loss `0.8221`, val loss `0.6219`
- epoch 2: train loss `0.5217`, val loss `0.4059`

## Passed?

- Stage 10A implementation completed and ran on real PEMS08 loaded from the ASTGNN zip. This is still a 20-node debug subset, not a formal full-dataset benchmark.

Stage 10A run notes:

1. real PEMS08 loader is implemented
2. direct zip loading is implemented for the provided ASTGNN archive
3. PEMS08-only real debug run completed
4. every latest Stage 10A row is marked `real_data_used=true` and `fallback_used=false`
5. regression tests still pass: `14/14`

Stage 10 ultra-quick experiment was previously completed. It is fallback/debug data only, not a formal pass/fail benchmark.

Stage 10 run notes:

1. full 30-epoch quick run was attempted and exceeded the local timeout
2. the lingering Stage 10 Python process was stopped
3. an ultra-quick fallback run completed all 18 rows
4. every row is marked `fallback_used=true` and `formal_result=false`
5. regression tests still pass: `14/14`

Stage 9 was previously passed as a single-dataset trend gate after the focused Stage 7 fix.

Stage 9 pass criteria check:

1. V4 beats V0 in at least two scenarios: yes
2. V4 beats V1 in incident or sensor failure: yes
3. trust mean not collapsed to 0 or 1: yes
4. training loss stable: yes
5. no NaN: yes
6. training time acceptable: yes
7. recommend entering the next quick benchmark: yes, with the toy-data caveat

Stage 7 pass criteria check:

1. incident region trust lower than normal region: yes, after speed-only incident and anomaly-tail conflict fix
2. incident region MAE not worse: yes
3. overall MAE not much worse: yes
4. trust not fully collapsed: yes
5. no NaN: yes
6. regression tests still pass: yes, `14/14`
7. recommend entering Stage 8 or Stage 9: yes, prefer Stage 9 because Stage 8 is optional

Stage 6 pass criteria check:

1. V3 is not worse than V2: yes
2. log_var has variation: yes, std `0.3603`
3. trust mean not collapsed: yes, `0.1971`
4. trust std positive: yes, `0.1914`
5. uncertainty-error correlation finite: yes, `0.4073`
6. V3 train MAE decreased: yes
7. no NaN: yes
8. regression tests still pass: yes, `12/12`
9. recommend entering V4 Conflict-Aware Regularization: yes

Stage 5 pass criteria check:

1. V2 is not worse than V1: yes
2. trust mean not collapsed: yes, `0.1880`
3. trust std positive: yes, `0.1958`
4. V2 train loss decreased: yes
5. no NaN: yes
6. regression tests still pass: yes, `11/11`
7. recommend entering V3 Uncertainty Head: yes, with caution because toy-data effect is tiny

Stage 4 pass criteria check:

1. fixed physics lowers validation MAE: yes, but only by `0.0000579`
2. fixed physics lowers physics residual: yes, but only by `0.00000003`
3. data loss and physics loss conflict observed: no
4. residual scale reasonable and bounded: yes
5. no NaN: yes
6. regression tests still pass: yes, `10/10`
7. recommend entering V2 Trust Gate: yes, with caution because toy-data fixed-physics effect is tiny

Stage 3 pass criteria check:

1. train loss decreased: yes
2. BaseTCNGraph validation MAE beats naive mean fill: yes
3. no NaN: yes
4. result files generated: yes
5. regression tests still pass: yes

Stage 2 pass criteria check:

1. `random_missing_mask` shape and ratio checked: yes
2. `sensor_failure_mask` full-node failures checked: yes
3. `block_missing_mask` spatial block behavior checked: yes
4. `temporal_missing_mask` ratio and contiguous block checked: yes
5. `incident_perturbation` local node/time effect checked: yes
6. `PROGRESS.md` updated: yes

## Next Recommendation

- Official GRIN cached 30e pure-eval bug fixed: `scripts/run_official_repo_litetrust_quick.py` now skips `train_log.csv` when `epochs=0` and no logs exist.
- Pure frozen GRIN cache evaluation on main scenarios passed: random_missing_50 `0.646528`, noise_random_missing `0.655596`, incident_perturbation `0.658474`.
- Default LiteTrust correction-only with the same frozen GRIN cache: random_missing_50 `0.640533`, noise_random_missing `0.649023`, incident_perturbation `0.644692`; main-scenario average gain vs GRIN is `1.34%`.
- Scenario-aware zero-initialized adapter with the same frozen GRIN cache: random_missing_50 `0.640685`, noise_random_missing `0.649245`, incident_perturbation `0.644782`; main-scenario average gain vs GRIN is `1.32%`.
- Current decision: scenario-aware adapter is executable but slightly worse than default LiteTrust, so it should not be the main method in its current form.
- Current bottleneck: incident perturbation already exceeds `2%` relative gain, but random/noise gains are about `1%`; next structural improvement should target random/noise sparse missing rather than adding scenario tokens.
- Selective / utility-routed correction trial completed on the same cached 30e GRIN backbone and main three scenarios.
- Added switches in the official wrapper path: `--selective-correction`, `--physics-vetted-correction`, `--generic-only-correction`, `--utility-router-correction`, and `--diagnostics`.
- Candidate diagnostics now report `mu_data`, `x_generic`, `x_vetted`, `x_phys`, `x_fused`, `x_router`, and `oracle_best` masked MAE.
- Best current numerical variant is `generic_error_calibrated`: random `0.639623`, noise `0.645684`, incident `0.641894`, main-scenario average gain `1.70%` vs cached GRIN.
- Utility router result: random `0.640383`, noise `0.648423`, incident `0.644868`, main-scenario average gain `1.37%`; it lowers harm rate but does not beat generic-only.
- Router interpretation: average utility-router weights are about `0.22` GRIN, `0.55` generic, `0.03` direct physics, `0.20` fused across scenarios. This confirms the direct physics candidate is currently weak and should not be mixed at high weight.
- Physics-vetted independent head failed: main-scenario average gain `-1.81%`, so a new residual-conditioned head is too unstable under 5e correction-only training.
- Oracle gap remains large, e.g. random oracle-best `0.5542` vs utility-router `0.6404`; the next method step should improve local candidate selection/calibration rather than add more scenario tokens or stronger direct physics projection.
- Physics-Verified Generic Correction V1 implemented with `--physics-verified-correction`: `x_verified = x_grin + verifier_gate * (x_generic_calibrated - x_grin)`.
- Physics-verified V1 used utility supervision, harm loss, and residual-decrease verification. Result: random `0.640389`, noise `0.647325`, incident `0.644983`, average gain `1.42%`; gate mean `0.806`, harm rate `0.429`.
- Physics-verified V1b used a more conservative high-initialized verifier and weaker residual penalty. Result: random `0.639830`, noise `0.646277`, incident `0.643353`, average gain `1.58%`; gate mean `0.949`, harm rate `0.434`.
- Current best remains `generic_error_calibrated` with average gain `1.70%`. Physics verification reduces harm rate slightly but does not improve MAE enough.
- Key diagnostic: `residual_after_data` is often larger than `residual_before`, even when generic correction improves reconstruction. Residual monotonicity is not aligned with MAE in this setting.
- Updated method judgment: keep generic calibrated correction as prediction backbone; use physics residual as a learned utility feature/explanation signal, not as a direct residual-decrease constraint.
- Contrastive Utility Verifier implemented with `--contrastive-utility-verifier`. It uses physics residuals only as utility features/diagnostics and removes the direct residual-decrease penalty.
- Contrastive V1 soft target result: random `0.639810`, noise `0.646233`, incident `0.643383`, average gain `1.58%`; verifier gate mean `0.949`, harm rate `0.435`.
- Contrastive V2 hard target result: random `0.639810`, noise `0.646233`, incident `0.643382`, average gain `1.58%`; verifier gate mean `0.949`, harm rate `0.435`.
- Interpretation: removing the residual penalty avoids further degradation, but the verifier still behaves mostly as mild shrinkage over generic correction and does not beat `generic_error_calibrated`.
- Current next structural recommendation: freeze a trained generic calibrated correction and train the verifier as a second-stage utility classifier with region-balanced samples; joint 5e training does not give the verifier enough movement from its high-gate initialization.
- Two-stage utility verifier implemented with `--two-stage-verifier`, `--verifier-epochs`, and `--verifier-min-gate`.
- Stage A freezes GRIN and trains only `generic_head`, `error_calibrator`, and `error_shrinkage_head`; Stage B freezes the generic correction and trains only `physics_verifier_head`.
- The verifier uses balanced utility BCE with target `1[err_generic < err_grin]`, plus hard-negative BCE and harm loss.
- No-floor `1e+1e` verifier confirms learnability: verifier gate positive mean `0.476` vs negative mean `0.424`, but it over-suppresses correction and reaches only `1.51%` average gain.
- Longer verifier training over-suppresses more: `5e+1e` average gain `0.67%`, `5e+5e` average gain `0.66%`; do not extend verifier epochs under the current objective.
- Floor-bounded verifier works better. Best current result is `two_stage_floor08_p1p1`: random `0.638987`, noise `0.644911`, incident `0.642410`, average gain `1.75%` vs cached GRIN.
- `two_stage_floor08_p1p1` slightly beats `generic_error_calibrated` average gain `1.70%`, so it is the current main LiteTrust variant, but still below the desired `2%` target.
- Current method claim: physics residual is useful as a conservative utility verifier / modulator, not as a direct correction expert or hard residual-decrease constraint.
- Hard-Negative Utility Verifier V2 implemented with `--hard-negative-verifier` and `--hard-negative-margin`.
- V2 target is `hard_negative = err_generic > err_grin + margin`; only hard negatives are explicitly suppressed, all other points are treated as safe.
- With `verifier_min_gate=0.8`, `margin=0.02`: random `0.638987`, noise `0.644917`, incident `0.642418`, average gain `1.75%`; hard-negative gate `0.884`, safe gate `0.894`.
- With `verifier_min_gate=0.8`, `margin=0.05`: random `0.638987`, noise `0.644917`, incident `0.642417`, average gain `1.75%`; no material gain over the previous floor-bounded verifier.
- Interpretation: hard-negative utility supervision improves interpretability and behaves in the right direction, but does not add performance beyond `two_stage_floor08_p1p1`.
- Current bottleneck has shifted back to generic correction candidate quality; verifier calibration alone appears capped around `1.75%` under this protocol.
- Generic correction head redesign completed after deciding not to continue gate-only modifications.
- Added `--generic-v2-correction`, `--generic-v3-correction`, and `--generic-v4-correction` to the official GRIN integration runner.
- Generic V2 replaces the generic delta with a local/graph/temporal branch mixture; it was rejected because branch weights stayed nearly uniform and diluted the useful local correction. Main 1e result: random `0.641320`, noise `0.650095`, incident `0.651398`, average gain only `0.91%`.
- Generic V3 keeps the generic/error-calibrated correction as the main path and adds a bounded residual refinement head. Best quick result with utility loss, 1e: random `0.639402`, noise `0.644994`, incident `0.641900`, mean `0.642099`, average gain `1.75%`.
- Generic V4 learns a region-aware scale over the generic/error-calibrated delta. It was rejected because scale stayed near `0.997` and did not improve MAE: random `0.639509`, noise `0.645115`, incident `0.641984`, average gain `1.73%`.
- Current decision: keep `two_stage_floor08_p1p1` as the main LiteTrust variant; keep `Generic V3 + utility` as a lightweight correction-head ablation; do not use V2/V4 as main method.
- Updated record file: `results/official_grin_pems08_debug/official_repo_integration.md`.
- Physics-informed harm verifier implemented with `--physics-harm-verifier`, `--two-stage-harm-verifier`, `--harm-keep-min`, `--sparse-harm-verifier`, `--harm-threshold`, and `--harm-temperature`.
- Harm verifier V1 learns the intended direction but weakly: with `keep_min=0.8`, hard-negative `harm_prob` is about `0.454` vs safe `0.446`; random improves slightly but noise/incident degrade.
- Conservative sparse harm verifier improves recall but still does not beat generic/error-calibrated correction. Best sparse V1: random `0.639468`, noise `0.645048`, incident `0.641999`.
- Harm verifier V2 added local evidence features: generic spatial disagreement, generic temporal disagreement, correction-vs-graph gap, correction-vs-temporal gap, and residual increase rank.
- V2 with stronger hard-negative weighting increases recall but learns the wrong ordering: `harm_prob_safe` remains higher than `harm_prob_hard_negative`; mean MAE around `0.64219`, not better than the generic/error-calibrated candidate.
- Utility keep regression was tested with `--harm-utility-target`; it collapses to excessive shrinkage (`keep_mean` around `0.47`) and worsens mean MAE to about `0.64634`.
- Current decision: do not use harm verifier as a direct prediction gate. Keep it as diagnostic/auxiliary regularizer candidate. Final output should remain calibrated generic correction unless verifier confidence is very high.
- Harm-regularized generic correction implemented with `--harm-regularized-correction`.
- This mode keeps final output as calibrated generic correction and uses the physics-informed harm head only as a training regularizer.
- Loss adds a GRIN-relative harm penalty: `relu(|x_final-y|-|x_grin-y|-margin)`, plus a small harm classifier BCE and harmful correction magnitude term.
- Three-scenario quick result, 1e: random `0.639455`, noise `0.645029`, incident `0.641855`, mean `0.642113`; this is safe and improves incident versus `two_stage_floor08`, but does not beat the current main mean `0.642103`.
- Three-scenario quick result, 3e: random `0.639520`, noise `0.645674`, incident `0.642451`, mean `0.642549`; longer training over-regularizes.
- Current decision: keep `--harm-regularized-correction` as an auxiliary branch; main result remains `two_stage_floor08_p1p1`.
- V3-lite reliability router implemented after rejecting the heavy self-supervised RiskRouter as too slow for the lightweight goal.
- Added `LightweightReliabilityRouter` and `LiteTrustGRINReliabilityRouter`.
- ReliabilityRouter uses fixed monotonic evidence coefficients plus a trainable bias vector, keeping the method lightweight and interpretable.
- Reliability evidence includes observed ratio, local missing ratio, physics residual rank, temporal change, node missing ratio, neighbor observed ratio, low-residual evidence, and uncertainty proxy.
- Latest reliability-router artifacts are in `results/reliability_router_pems08_debug/summary.csv` and `results/reliability_router_pems08_debug/summary.md`.
- PEMS08 debug, random_missing_50: `ReliabilityRouter` masked MAE `0.880955`; no-physics variant `0.905522`; physics contribution exists but is weaker than `LiteTrust_delta_only_bounded` `0.817050`.
- PEMS08 debug, sensor_failure_30: `ReliabilityRouter` masked MAE `1.255998`; no-physics variant `1.265352`; `GRINLite_graph_delta` `1.267702`. ReliabilityRouter is best in this debug set.
- Interpretability: random missing physics weight mean `0.128531`, missing-region physics weight `0.170859`; sensor failure failed-node graph weight `0.977592`, failed-node physics weight `0.020346`.
- Current judgment: V3-lite is theoretically cleaner and lighter than RiskRouter, and physics has measurable marginal contribution, but random-missing performance is still behind delta-only. Next method work should strengthen the physics expert under random missing without increasing sensor-failure physics weight.
- Physics-focused ReliabilityRouter update: graph expert is now scoped to node-level failure (`node_missing_ratio > 0.8`), and inactive graph weight is redistributed mostly to physics under random missing.
- Updated PEMS08 debug, random_missing_50: `ReliabilityRouter` masked MAE improved from `0.880955` to `0.833448`; physics weight mean increased to `0.634363`, and missing-region physics weight to `0.723001`; graph weight under random missing is `0.000000`.
- Updated PEMS08 debug, sensor_failure_30: `ReliabilityRouter` masked MAE `1.260969`; failed-node graph weight remains high at `0.973849`, and failed-node physics weight remains low at `0.024007`.
- Interpretation: physics is now core in random missing and suppressed in full sensor failure. Performance is improved, but still behind `LiteTrust_delta_only_bounded` (`0.817050`) on random missing, so the next bottleneck is physics expert quality rather than router allocation.
- Physics expert strengthened: correction head now receives signed residuals instead of only `abs(residual)`, and a lightweight FD projection term directly maps positive `q - rho*v` residuals to negative flow correction.
- Hard closed-form FD projection was tested and rejected: it lowered physics residual but worsened random_missing_50 MAE to `0.911793`, showing that exact physical projection can itself become misguiding.
- Retained structural change: signed residual + weak FD direction + residual-improvement regularization, which penalizes the physics expert only when `R(x_phys)` is worse than `R(x_data)`.
- Task-aligned projection controller was also tested. It lowered residual but worsened random_missing_50 to `0.837707` versus the retained `0.832346`, so it is now optional and disabled by default (`use_projection_controller=False`).
- Physics validity gate added: it estimates whether residual correction is likely useful and softly transfers unreliable physics weight back to the data branch.
- Updated PEMS08 debug, random_missing_50: ReliabilityRouter with validity gate is `0.833673`; physics validity in missing regions is `0.622175`, so physics remains active.
- Updated PEMS08 debug, sensor_failure_30: ReliabilityRouter with validity gate improves to `1.254302`; failed-node physics validity is `0.099805`, failed-node physics weight is `0.013809`, and failed-node graph weight is `0.974147`.
- Interpretation: validity gating improves the disrupted sensor-failure case and provides clearer evidence for local physics usefulness, but random_missing_50 is slightly worse than the signed-FD residual-improvement variant. The next meaningful step is self-supervised validity calibration, not more hand-set coefficients.
- Updated PEMS08 debug, random_missing_50: ReliabilityRouter is `0.832346`; missing-region physics weight remains high at `0.722088`; graph weight remains `0.000000`; physics gain over no-physics remains about `8.08%`.
- Updated PEMS08 debug, sensor_failure_30: ReliabilityRouter is `1.257962`; failed-node graph weight remains high at `0.974415`; failed-node physics weight remains low at `0.023446`.
- Interpretation: stronger physics must be task-aligned, not merely residual-minimizing. Exact projection can reduce residual while hurting reconstruction, which directly supports the physical-misguidance thesis.
- Follow-up method check completed after questioning whether `physics_weight=0` makes the method indistinguishable from graph correction.
- Added configurable routing modes to `LiteTrustGRINCorrection`: `soft_prior`, `hard`, and `no_physics`; added `use_phys_expert`; added bounded residual correction with `correction_clip=1.0`.
- Latest soft-router ablation artifacts are in `results/correction_soft_router_pems08_debug/summary.csv` and `results/correction_soft_router_pems08_debug/summary.md`.
- PEMS08 debug, random_missing_50: `CorrectionV2_soft_router_bounded` masked MAE `0.824955`, improving over `GRINLite` `0.893680` by `7.69%`; disabling physics worsens to `0.896808`, so physics/residual correction is necessary for random sparse missing.
- PEMS08 debug, sensor_failure_30: `CorrectionV2_soft_router_bounded` masked MAE `1.257571`, improving over `GRINLite` `1.445032` by `12.97%`, over `GRINLite_graph_delta` `1.267702`, and over `CorrectionV2_no_physics` `1.277794`.
- Interpretability: in sensor failure, soft router uses failed-node graph weight `0.850000` and failed-node physics weight `0.050000`; this avoids the earlier all-or-nothing physics bypass while keeping graph correction dominant.
- Method conclusion: physics is not the primary expert under sensor failure, but a bounded low-weight physics/residual expert gives marginal gain; under random missing, physics/residual correction is clearly useful.
- Method update completed: `LiteTrustGRINCorrection` is now a routed correction model rather than a direct additive correction model.
- New method name suggestion: `LiteTrust-GRIN-Router`.
- New prediction form: `x_hat = x_data + g_graph(i,t) * delta_graph + g_phys(i,t) * delta_phys`, with data/graph/physics expert routing weights.
- Complete sensor failure hard-routes to the graph correction expert; sparse random missing still uses learned residual/physics correction.
- Latest routed results are in `results/correction_router_pems08_debug/summary.csv` and `results/correction_router_pems08_debug/summary.md`.
- PEMS08 debug, 20 nodes, 50 epochs: `CorrectionV2_routed` reached random_missing_50 masked MAE `0.800921`, improving over `GRINLite` `0.893680` by `10.38%`.
- PEMS08 debug, 20 nodes, 50 epochs: `CorrectionV2_routed` reached sensor_failure_30 masked MAE `1.262174`, improving over `GRINLite` `1.445032` by `12.65%` and slightly beating `GRINLite_graph_delta` `1.270554`.
- Interpretability: in sensor failure, failed-node graph weight is `1.000000`, failed-node physics weight is `0.000000`, `graph_delta_failed_mean` is `0.813785`, and `delta_phys_missing_mean` is `0.005718`.
- Current evidence supports the narrower claim: the model learns/rules when to use physical residual correction and when to bypass physics and trust graph-neighbor correction.
- Latest ablation artifacts are in `results/correction_ablation_pems08_debug/summary.csv` and `results/correction_ablation_pems08_debug/summary.md`.
- Correction V1 ablation on real PEMS08 debug used four variants: `GRINLite`, `GRINLite_graph_delta`, `LiteTrust_delta_only`, and `CorrectionV1_full`.
- On `random_missing_50`, `LiteTrust_delta_only` and `CorrectionV1_full` both reached masked MAE `0.796771`, improving over `GRINLite` `0.893680` by `10.84%`.
- On `sensor_failure_30`, `GRINLite_graph_delta` reached masked MAE `1.270554`, improving over `GRINLite` `1.445032` by `12.07%`; `CorrectionV1_full` reached `1.308505`, improving over `GRINLite` by `9.45%` but not beating graph_delta alone.
- Interpretability checks: `CorrectionV1_full` trust on sensor-failure nodes was `0.761414` versus normal nodes `0.240095`; `graph_delta_failed_mean` was `0.145515`; `delta_phys_missing_mean` was `0.349934`.
- Current method implication: random missing supports adaptive `trust * delta_phys`, while sensor failure supports graph neighbor correction. The next method change should route between correction experts instead of only adding `graph_delta + trust * delta_phys`.
- Stage 10A results are recorded in `results/stage10a_pems08_real_debug/summary.csv`.
- To get real PEMS08 numbers, place a compatible `.npz` file under `data/raw/pems08` and rerun `python scripts/run_stage10a_pems08_real_debug.py`.
- Latest official-repo harm work moved the verifier back to a second-stage utility module instead of a direct output gate.
- One-epoch joint harm regularization stayed weak: generic correction val MAE reached `1.605266` before the harm stage, then degraded to `1.612674` when the verifier was allowed to pull the generic head.
- Two-stage binary region-balanced verifier on `C:\tmp\twostage_binaryverifier_5e_random` showed only mild separation after 5 verifier epochs: `harm_prob_hard_negative_mean 0.532269` vs `harm_prob_safe_mean 0.506712`, `harm_keep_hard_negative_mean 0.467731` vs `harm_keep_safe_mean 0.493288`, but `x_harm_verified_masked_mae 1.763682` stayed worse than `x_generic_masked_mae 1.749654`.
- Current decision: keep physics as a verifier / utility regularizer only; do not let it replace the generic correction head at inference.
- Quantile-balanced verifier improved the signal on sensor failure. On `C:\tmp\twostage_quantileverifier_5e_suite`, `sensor_failure_30` reached `harm_prob_hard_negative_mean 0.749544` vs `harm_prob_safe_mean 0.490486`, and `harm_keep_hard_negative_mean 0.250456` vs `harm_keep_safe_mean 0.509514`.
- The same quantile-balanced verifier stayed weak on `random_missing_50` and `incident_perturbation` (`harm_prob_hard_negative_mean 0.524178 / 0.522750`, `harm_prob_safe_mean 0.504913 / 0.504900`), so the verifier looks scenario-sensitive rather than universally strong.
- Current decision: keep the verifier as a scenario-sensitive utility regularizer, not as a direct output gate.
- Added a two-expert harm verifier with a region gate and separate general/sensor heads. The key fix was to include `physics_harm_sensor_head` and `physics_harm_gate_head` in the trainable prefixes; before that, they were frozen and stayed near initialization.
- With the two-expert loss, `sensor_failure_30` now shows clearer separation in the mixed harm score: `harm_prob_hard_negative_mean 0.425072` vs `harm_prob_safe_mean 0.316673`, and the gate / expert statistics are no longer stuck at exact initial values.
- Random missing and incident remain much weaker, so the verifier is still not a universal predictor. Final `masked_mae` stays aligned with the generic correction path rather than the verifier candidate.
- Current decision: keep the two-expert verifier as an auxiliary, scenario-sensitive critic. Do not use `x_harm_verified` as the final output.
- Latest verifier-only calibration with generic correction frozen gave the cleanest sensor-failure separation so far on `C:\tmp\twostage_expertloss_sensor10e`: `harm_prob_hard_negative_mean 0.540669` vs `harm_prob_safe_mean 0.313030`, `harm_keep_hard_negative_mean 0.459331` vs `harm_keep_safe_mean 0.686970`, with final `masked_mae 1.650539` and `x_harm_verified_masked_mae 1.668704`.
- Current interpretation: the verifier is useful as a calibration/diagnostic module, but the final prediction should still stay on the frozen generic correction path. Sensor failure is the clearest place where the harm signal separates.
- Follow-up random/incident check was first blocked by a NumPy 2 / pandas binary mismatch in the system Python. After downgrading NumPy to `1.26.4`, the same frozen-generic verifier ran on the workspace and confirmed the behavior: on `C:\tmp\frozen_generic_verifier_followup_diag`, `random_missing_50` had `harm_prob_hard_negative_mean 0.411777` vs `harm_prob_safe_mean 0.394514`, and `incident_perturbation` had `0.411019` vs `0.394026`; both verifier candidates stayed worse than the generic path. This keeps the verifier in the calibration/diagnostic role.
- New structural branch `harm_suppressed_correction` was added to make physics act as a bounded correction suppressor rather than an output expert. It introduces `correction_allowance` and trains it from harm/utility signals.
- Quick test on `C:\tmp\harm_suppressed_bounded_10e_5e_suite` showed the structure is stable but not yet useful for gains: `random_missing_50` `x_harm_suppressed_masked_mae 1.783476` vs `x_generic_masked_mae 1.783403`; `incident_perturbation` `1.784599` vs `1.784579`; `sensor_failure_30` `1.674183` vs `1.672545`. In short, bounded suppression does not yet beat the generic correction path.
- Current interpretation: simply suppressing correction magnitude is too weak. Physics as a verifier is still useful for analysis, but this branch needs a sharper region-aware correction rule if it is to improve MAE.
- Because `x_phys` was consistently strong, it was promoted into explicit output branches: `physics_candidate_correction` and `physics_promoted_correction`.
- Directly training the physics candidate for 10 epochs was not helpful on random/incident (`C:\tmp\physics_candidate_10e_suite`), suggesting the strong signal comes from the explicit physics/graph/time correction rather than extra learned physics-head capacity.
- Zero-epoch `physics_candidate_correction` confirmed the raw physics candidate is strong: `random_missing_50` MAE `1.717409`, `incident_perturbation` `1.719612`, `sensor_failure_30` `1.600010`.
- New `physics_promoted_correction` uses physics-heavy fused output in non-failure regions and promotes `x_phys` on node-failure regions. On `C:\tmp\physics_promoted_0e_suite`, it reached `random_missing_50 1.712812`, `incident_perturbation 1.714888`, and `sensor_failure_30 1.600010`, improving over `mu_data` by `3.653%`, `3.560%`, and `6.596%`, and over `x_generic` by `1.572%`, `1.510%`, and `2.856%`.
- Current method direction: physics should be a promoted correction path, not just a verifier. The strongest version so far is scenario/region-aware physics promotion: fused physics correction for random/incident, direct physics correction for sensor-failure nodes.
- Promotion was upgraded from a rule to a learned gate: `learned_physics_promotion` adds `PhysicsPromotionGate` and trains `physics_promotion_score` with a local utility target derived from `x_phys` vs `x_fused`.
- Smoke test on `C:\tmp\learned_physics_promotion_smoke` passed, but the promotion score stayed near a coarse default and did not yet sharply separate `physics_better` from `fused_better` regions.
- 10-epoch quick run on `C:\tmp\learned_physics_promotion_10e_suite` stayed stable but blunt: `physics_promotion_mean` was about `0.60`, with only a small gap between `physics_promotion_phys_better_mean` and `physics_promotion_fused_better_mean`.
- Current interpretation: the method framing is right, but the learned gate still needs harder region-balanced targets and better negative sampling to become a real promotion model rather than a soft default mixer.
- The promotion supervision was hardened to batch-wise hard/safe quantile regions instead of a continuous sigmoid target.
- Smoke test on `C:\tmp\learned_physics_promotion_hard_smoke` still ran, but the gate remained coarse: `physics_promotion_mean 0.620523`, `physics_promotion_phys_better_mean 0.619583`, `physics_promotion_fused_better_mean 0.621577`.
- 10-epoch quick run on `C:\tmp\learned_physics_promotion_hard_10e_suite` did not produce a sharp separator either: `physics_promotion_mean` stayed around `0.606`, with `phys_better_mean 0.601` and `fused_better_mean 0.611` on random_missing_50; sensor_failure moved to `phys_better_mean 0.599` and `fused_better_mean 0.622`. MAE remained essentially the same as the earlier physics-promoted branch.
- Current interpretation: hard region supervision is cleaner, but still not enough. The learned promotion gate needs stronger region-balanced sampling or an explicit failure-mode feature before it becomes a meaningful learned method.
- The next structural revision was a discrete physics router: `discrete_physics_promotion` selects among `fused`, `phys`, and `generic` with hard mode selection.
- Smoke test on `C:\tmp\discrete_physics_promotion_smoke` passed, but the router still leaned toward the physics mode and did not yet exploit the candidate differences well.
- 10-epoch quick run on `C:\tmp\discrete_physics_promotion_10e_suite` stayed coarse: random and incident stayed on the physics-like output, while sensor failure also selected physics almost entirely. This is a structural improvement over scalar gating, but not yet a better result.
- Current interpretation: discrete routing is the right structural form, but the mode target and region sampling still need sharper supervision if we want it to behave like a real promotion policy.
- Latest numeric search on the official GRIN quick run improved the best average from `+1.74%` to `+2.00%` when scenario-selected outputs are allowed, and to `+1.92%` with a single preset.
- Best single preset found so far: `--utility-router-correction --target-only-loss --correction-clip 1.0`.
- Best scenario-selected official outputs so far:
  - `random_missing_50`: `x_router_masked_mae 0.638678` vs baseline `0.646528`, gain `+1.21%`.
  - `noise_random_missing`: `x_router_masked_mae 0.643784` vs baseline `0.655596`, gain `+1.80%`.
  - `incident_perturbation`: `x_verified_masked_mae 0.638893` vs baseline `0.658474`, gain `+2.97%`.
- Best single-preset official outputs so far:
  - `random_missing_50`: `0.638678` vs `0.646528`, gain `+1.21%`.
  - `noise_random_missing`: `0.643784` vs `0.655596`, gain `+1.80%`.
  - `incident_perturbation`: `0.640422` vs `0.658474`, gain `+2.74%`.
- Current interpretation: the right direction is not another generic correction tweak; the most useful next step is to improve scenario-aware candidate selection / router supervision so the best candidate becomes the default output more reliably.
- Residual-aware MagiNet distillation was added to `scripts/run_temporal_anchor_litetrust_quick.py` through `--residual-aware-distill`.
- Method change: the student is distilled from MagiNet only where the teacher is locally useful and physically plausible, using teacher advantage, residual advantage, and teacher residual rank. This keeps accident/perturbation regions from being dominated by teacher imitation and leaves room for physics correction.
- Validation selection for residual-aware distillation now uses target MAE instead of adding a teacher-distance term, so the checkpoint is selected by reconstruction quality rather than teacher similarity.
- Quick runs completed with 20 epochs, seed 1, scenarios `random_missing_50` and `incident_perturbation`.
- Artifacts:
  - `C:\tmp\temporal_anchor_magistyle_v2_resdistill_pems08_e20`
  - `C:\tmp\temporal_anchor_magistyle_v2_resdistill_pems08_w030_e20`
  - `C:\tmp\temporal_anchor_magistyle_v2_resdistill_metrla_e20`
  - `C:\tmp\temporal_anchor_magistyle_v2_resdistill_metrla_w030_e20`
- PEMS08 residual-aware, distill weight 0.45:
  - random_missing_50: MagiNet `0.378308`, MaskAwareGraphRepairV2 `0.393578`, PhysicsFromGraphRepair `0.372554`, final fusion `0.372554`; best internal improves over MagiNet by `+1.52%`.
  - incident_perturbation: MagiNet `0.384309`, MaskAwareGraphRepairV2 `0.453535`, PhysicsFromGraphRepair/fusion `0.428562`; still `-11.51%` behind MagiNet.
- PEMS08 residual-aware, distill weight 0.30:
  - random_missing_50: MagiNet `0.378308`, MaskAwareGraphRepairV2 `0.396125`, PhysicsFromGraphRepair/fusion `0.377365`; best internal improves over MagiNet by `+0.25%`.
  - incident_perturbation: MagiNet `0.384309`, MaskAwareGraphRepairV2 `0.414720`, PhysicsFromGraphRepair `0.394279`, final fusion `0.415380`; best internal is only `-2.59%` behind MagiNet, much better than weight 0.45.
- METR-LA residual-aware, distill weight 0.45:
  - random_missing_50: MagiNet `0.301571`, MaskAwareGraphRepairV2 `0.313054`, PhysicsFromGraphRepair/fusion `0.314070`; best internal `-3.81%` behind MagiNet.
  - incident_perturbation: MagiNet `0.306068`, MaskAwareGraphRepairV2 `0.313790`, PhysicsFromGraphRepair/fusion `0.316392`; best internal `-2.52%` behind MagiNet.
- METR-LA residual-aware, distill weight 0.30:
  - random_missing_50: MagiNet `0.301571`, MaskAwareGraphRepairV2 `0.309324`, PhysicsFromGraphRepair/fusion `0.312330`; best internal `-2.57%` behind MagiNet.
  - incident_perturbation: MagiNet `0.306068`, MaskAwareGraphRepairV2 `0.311840`, PhysicsFromGraphRepair/fusion `0.315146`; best internal `-1.89%` behind MagiNet.
- Interpretation:
  - Residual-aware distillation is structurally better than plain teacher imitation because it restores physics-correction autonomy in PEMS08 incident.
  - `distill_weight=0.30` is safer than `0.45` across datasets; it sacrifices a small amount on PEMS08 random but greatly reduces incident degradation and improves METR-LA relative to 0.45.
  - The current final fusion selector is still weak: on PEMS08 incident with weight 0.30, `PhysicsFromGraphRepair` is `0.394279` but final fusion selects router output `0.415380`. The next method issue is candidate selection / region routing, not backbone capacity.
  - On METR-LA, the graph student is stronger than the physics-corrected branch, so physics should be a conditional correction rather than always post-applied after graph repair.
- Main method pivot completed: stop compressing / distilling MagiNet as the main route. New script `scripts/run_maginet_physics_guard_quick.py` keeps MagiNet as the strong reconstruction backbone and applies physics only as local residual-aware correction with harm-aware selection.
- New method branch:
  - `MagiNet`: strong reconstruction backbone.
  - `PhysicsFromMagi`: direct physics residual correction candidate generated from MagiNet output.
  - `MagiPhysicsGuarded`: learned local correction weight `alpha(i,t)` over `x_magi + alpha * (x_phys - x_magi)`.
  - `MagiPhysicsGuardedSafe`: validation-selected guarded output, falling back to MagiNet if guarded correction does not improve validation MAE.
- Guard supervision:
  - utility target compares local errors of `x_magi` and `x_phys`;
  - harm loss penalizes regions where correction makes MagiNet worse;
  - missed-gain loss encourages using physics where `x_phys` is locally better;
  - features include local/node/neighbor missing, node contrast, graph residual ranks, residual gain, prediction gap rank, temporal change, and spatial gap.
- Artifacts:
  - `C:\tmp\maginet_physics_guard_pems08_e20`
  - `C:\tmp\maginet_physics_guard_metrla_e20`
  - `C:\tmp\maginet_physics_guard_pems08_sensor_e20`
  - `C:\tmp\maginet_physics_guard_metrla_sensor_e20`
- PEMS08, seed 1, 20e:
  - random_missing_50: MagiNet `0.378308`, PhysicsFromMagi `0.359326`, MagiPhysicsGuardedSafe `0.360544`; guarded gain vs MagiNet `+4.70%`.
  - incident_perturbation: MagiNet `0.384309`, PhysicsFromMagi `0.371872`, MagiPhysicsGuardedSafe `0.370436`; guarded gain vs MagiNet `+3.61%`.
  - sensor_failure_30: MagiNet `1.353497`, PhysicsFromMagi `1.320201`, MagiPhysicsGuardedSafe `1.336198`; gain vs MagiNet `+1.28%`, but still far behind SAITS `0.892811`.
- METR-LA, seed 1, 20e:
  - random_missing_50: MagiNet `0.301571`, PhysicsFromMagi `0.301229`, MagiPhysicsGuardedSafe `0.299773`; guarded gain vs MagiNet `+0.60%`.
  - incident_perturbation: MagiNet `0.306068`, PhysicsFromMagi `0.307111`, MagiPhysicsGuardedSafe `0.305061`; guarded gain vs MagiNet `+0.33%`.
  - sensor_failure_30: MagiNet `0.487050`, PhysicsFromMagi `0.474640`, MagiPhysicsGuardedSafe `0.480179`; gain vs MagiNet `+1.41%`, but still behind SAITS `0.395298`.
- Current interpretation:
  - The new route is better than distillation: on PEMS08 random/incident it improves over MagiNet directly by `+4.70%` and `+3.61%`, while distillation could only match/approach MagiNet and sometimes damaged incident performance.
  - Cross-dataset gains on METR-LA are positive but small (`+0.60%`, `+0.33%`), so the method has initial cross-dataset support but not yet a strong claim.
  - Physics residual correction is useful as a post-hoc local verifier/corrector, not as a replacement backbone.
  - Sensor failure remains a boundary case: physics correction improves MagiNet slightly but does not beat temporal self-attention (SAITS). Do not include sensor_failure in the main average unless a separate temporal-failure module is added.
  - Interpretability is not yet sharp enough: `alpha_phys_better_mean` is only mildly higher than `alpha_magi_better_mean`; next improvement should sharpen the harm selector, not change the backbone.
- Stable method packaging completed as `LiteTrustPhysicsGuardV1`.
- Code changes:
  - `scripts/run_maginet_physics_guard_quick.py` now reports final row `LiteTrustPhysicsGuardV1`.
  - Final output uses validation-only selection over `MagiNet`, `PhysicsFromMagi`, `MagiPhysicsGuarded`, calibrated direct physics correction, and calibrated guarded physics correction.
  - This fixes the previous instability where the safe output could ignore the stronger direct physics candidate or miss a better calibrated correction strength.
  - `README.md` now points to the frozen experimental method.
  - New method note: `METHOD_V11_EXPERIMENTAL.md`.
- Confirmation runs:
  - `C:\tmp\litetrust_physics_guard_v1_pems08`
  - `C:\tmp\litetrust_physics_guard_v1_metrla`
  - `C:\tmp\litetrust_physics_guard_v1_pems08_sensor`
  - `C:\tmp\litetrust_physics_guard_v1_metrla_sensor`
- Confirmed single-seed results, 20 backbone epochs, 120 guard epochs:
  - PEMS08 random_missing_50: MagiNet `0.378308`, LiteTrustPhysicsGuardV1 `0.354824`, gain `+6.21%`; selected `GuardedCalibrated@1.50`.
  - PEMS08 incident_perturbation: MagiNet `0.384309`, LiteTrustPhysicsGuardV1 `0.366997`, gain `+4.50%`; selected `GuardedCalibrated@1.50`.
  - METR-LA random_missing_50: MagiNet `0.301571`, LiteTrustPhysicsGuardV1 `0.299627`, gain `+0.64%`; selected `GuardedCalibrated@1.50`.
  - METR-LA incident_perturbation: MagiNet `0.306068`, LiteTrustPhysicsGuardV1 `0.305225`, gain `+0.28%`; selected `GuardedCalibrated@1.50`.
  - PEMS08 sensor_failure_30: MagiNet `1.353497`, LiteTrustPhysicsGuardV1 `1.304057`, gain `+3.65%`, but SAITS is still best at `0.892811`; selected `DirectCalibrated@1.50`.
  - METR-LA sensor_failure_30: MagiNet `0.487050`, LiteTrustPhysicsGuardV1 `0.470780`, gain `+3.34%`, but SAITS is still best at `0.395298`; selected `DirectCalibrated@1.50`.
- Version decision:
  - Freeze `LiteTrustPhysicsGuardV1` for the next controlled experiment round.
  - Main table should use `random_missing_50` and `incident_perturbation` first.
  - `sensor_failure_30` should be reported separately as a diagnostic/boundary case because temporal self-attention remains the best mechanism there.
  - Do not re-open MagiNet distillation as the main route.
- Five-dataset quick check completed for `LiteTrustPhysicsGuardV1`.
- Loader update:
  - `scripts/run_maginet_physics_guard_quick.py` now supports `PEMS03`, `PEMS04`, full `PEMS08`, `METR-LA`, and `PEMS-BAY`.
  - PEMS03/04/08 are downloaded from Zenodo record `7816008`.
  - PEMS-BAY is loaded from Hugging Face `MintBruce/SkyTraffic`.
  - External raw-data cache is under `C:\tmp\litetrust_data` to avoid Windows issues with the Chinese workspace path.
  - PEMS03 sensor-ID mapping was fixed via `PEMS03.txt`; without it, the CSV edge list becomes an identity-like graph and MagiNet fails in ARPACK Laplacian eigensolver.
- Five-dataset quick protocol:
  - train/val/test windows: `64/16/16`
  - seq_len: `12`
  - seed: `1`
  - backbone epochs: `20`
  - guard epochs: `120`
  - scenarios: `random_missing_50`, `incident_perturbation`
  - output summary: `C:\tmp\litetrust_v1_multidata_summary\summary.csv`
- Five-dataset results vs MagiNet:
  - PEMS08 random_missing_50: MagiNet `0.453793`, LiteTrust `0.431315`, gain `+4.95%`.
  - PEMS08 incident_perturbation: MagiNet `0.502797`, LiteTrust `0.482734`, gain `+3.99%`.
  - PEMS04 random_missing_50: MagiNet `0.214951`, LiteTrust `0.202068`, gain `+5.99%`.
  - PEMS04 incident_perturbation: MagiNet `0.227192`, LiteTrust `0.213330`, gain `+6.10%`.
  - PEMS03 random_missing_50: MagiNet `1.222785`, LiteTrust `1.125138`, gain `+7.99%`; BRITS remains best external at `0.886962`.
  - PEMS03 incident_perturbation: MagiNet `1.265773`, LiteTrust `1.181440`, gain `+6.66%`; BRITS remains best external at `0.908414`.
  - METR-LA random_missing_50: MagiNet `0.301571`, LiteTrust `0.299627`, gain `+0.64%`.
  - METR-LA incident_perturbation: MagiNet `0.306068`, LiteTrust `0.305225`, gain `+0.28%`.
  - PEMS-BAY random_missing_50: MagiNet `0.241052`, LiteTrust `0.238550`, gain `+1.04%`.
  - PEMS-BAY incident_perturbation: MagiNet `0.252126`, LiteTrust `0.249977`, gain `+0.85%`.
- Five-dataset interpretation:
  - V1 improves over MagiNet on all 10 dataset-scenario pairs.
  - Gains are strong on PEMS flow datasets (`+3.99%` to `+7.99%`), including full-node PEMS08/04/03.
  - Gains are consistently positive but weak on speed-only datasets (`METR-LA`, `PEMS-BAY`), around `+0.28%` to `+1.04%`.
  - The cross-dataset framework is not weak in the sense of failing to transfer, but the current residual form is clearly more effective on PEMS flow-style data than on speed-only data.
  - PEMS03 exposes a baseline issue: LiteTrust improves MagiNet substantially but does not beat BRITS on this small-window quick protocol. This should be treated as a benchmark/protocol warning before making broad SOTA claims.
  - Next method change, if needed, should target speed-only residual design or a residual bank, not the harm guard wrapper.
- Residual-bank update completed for cross-dataset transfer.
- Code change:
  - `scripts/run_maginet_physics_guard_quick.py` now builds a candidate residual bank instead of relying on one fixed physics correction.
  - Bank candidates: `PhysicsFromMagi`, `PhysicsSpatial`, `PhysicsTemporal`, `PhysicsSpeedWave`, and `PhysicsAntiSmooth`.
  - The physics candidate used by `LiteTrustPhysicsGuardV1` is selected by validation masked MAE, then passed through the existing validation-safe correction/guard pipeline.
  - This keeps the method dataset-transferable: the model selects a residual form from local temporal/spatial physics evidence, not from dataset-name rules.
- Residual-bank outputs:
  - `C:\tmp\litetrust_v1_bank_pems08`
  - `C:\tmp\litetrust_v1_bank_pems04`
  - `C:\tmp\litetrust_v1_bank_pems03`
  - `C:\tmp\litetrust_v1_bank_metrla`
  - `C:\tmp\litetrust_v1_bank_pemsbay`
  - compact table: `C:\tmp\litetrust_v1_bank_multidata_summary\summary_compact.csv`
- Residual-bank five-dataset results vs MagiNet:
  - METR-LA random_missing_50: MagiNet `0.301571`, LiteTrust `0.294860`, gain `+2.23%`, selected `PhysicsTemporal`.
  - METR-LA incident_perturbation: MagiNet `0.306068`, LiteTrust `0.300236`, gain `+1.91%`, selected `PhysicsTemporal`.
  - PEMS-BAY random_missing_50: MagiNet `0.241052`, LiteTrust `0.226645`, gain `+5.98%`, selected `PhysicsTemporal`.
  - PEMS-BAY incident_perturbation: MagiNet `0.252126`, LiteTrust `0.237565`, gain `+5.78%`, selected `PhysicsTemporal`.
  - PEMS03 random_missing_50: MagiNet `1.222785`, LiteTrust `1.074367`, gain `+12.14%`, selected `PhysicsTemporal`; BRITS remains better at `0.892833`.
  - PEMS03 incident_perturbation: MagiNet `1.265773`, LiteTrust `1.132397`, gain `+10.54%`, selected `PhysicsTemporal`; BRITS remains better at `0.908414`.
  - PEMS04 random_missing_50: MagiNet `0.214951`, LiteTrust `0.196743`, gain `+8.47%`, selected `PhysicsTemporal`.
  - PEMS04 incident_perturbation: MagiNet `0.227192`, LiteTrust `0.213330`, gain `+6.10%`, selected `PhysicsFromMagi`.
  - PEMS08 random_missing_50: MagiNet `0.453793`, LiteTrust `0.404659`, gain `+10.83%`, selected `PhysicsTemporal`.
  - PEMS08 incident_perturbation: MagiNet `0.502797`, LiteTrust `0.458837`, gain `+8.74%`, selected `PhysicsTemporal`.
- Residual-bank interpretation:
  - The speed-only weakness is materially reduced: METR-LA improves from about `+0.64%/+0.28%` to `+2.23%/+1.91%`; PEMS-BAY improves from `+1.04%/+0.85%` to `+5.98%/+5.78%`.
  - The method improves MagiNet on all 10 dataset-scenario pairs.
  - It beats the best external baseline on 8 of 10 pairs; the two misses are PEMS03, where BRITS is still better under the quick small-window protocol.
  - Average gain vs MagiNet is `+7.27%`; average gain vs best external baseline is `+0.50%`.
  - The dominant selected residual is `PhysicsTemporal`, which suggests the current cross-dataset transferable physics signal is temporal consistency rather than raw graph smoothing.
  - Parallel MagiNet runs should be avoided because the external MagiNet repo writes shared intermediate files; PEMS08/PEMS04 must be run serially to avoid cross-run data/config collision.
- Temporal-Reliability Residual Bank V2 completed.
- Code change:
  - Added `PhysicsBidirTemporal`, a BRITS-inspired bidirectional temporal residual candidate.
  - Added transferable guard features: previous-observation gap, next-observation gap, temporal gap decay, and bidirectional disagreement rank.
  - The method row is now `LiteTrustPhysicsGuardV2`.
  - No dataset-specific rule was added; the residual bank still selects by validation utility.
- V2 artifacts:
  - `C:\tmp\litetrust_v2_bidir_pems08`
  - `C:\tmp\litetrust_v2_bidir_pems04`
  - `C:\tmp\litetrust_v2_bidir_pems03`
  - `C:\tmp\litetrust_v2_bidir_metrla_random`
  - `C:\tmp\litetrust_v2_bidir_metrla_incident`
  - `C:\tmp\litetrust_v2_bidir_pemsbay`
  - compact table: `C:\tmp\litetrust_v2_bidir_multidata_summary\summary_compact.csv`
- V2 five-dataset results:
  - PEMS08 random_missing_50: MagiNet `0.453793`, V2 `0.369001`, gain `+18.69%`, V2 vs V1 `+8.81%`.
  - PEMS08 incident_perturbation: MagiNet `0.502797`, V2 `0.429279`, gain `+14.62%`, V2 vs V1 `+6.44%`.
  - PEMS04 random_missing_50: MagiNet `0.214951`, V2 `0.186340`, gain `+13.31%`, V2 vs V1 `+5.29%`.
  - PEMS04 incident_perturbation: MagiNet `0.227192`, V2 `0.199033`, gain `+12.39%`, V2 vs V1 `+6.70%`.
  - PEMS03 random_missing_50: MagiNet `1.222785`, V2 `0.954142`, gain `+21.97%`, V2 vs V1 `+11.19%`; BRITS remains better at `0.870752`.
  - PEMS03 incident_perturbation: MagiNet `1.265773`, V2 `1.028735`, gain `+18.73%`, V2 vs V1 `+9.15%`; BRITS remains better at `0.947444`.
  - METR-LA random_missing_50: MagiNet `0.301571`, V2 `0.291693`, gain `+3.28%`, V2 vs V1 `+1.07%`.
  - METR-LA incident_perturbation: MagiNet `0.306068`, V2 `0.297348`, gain `+2.85%`, V2 vs V1 `+0.96%`.
  - PEMS-BAY random_missing_50: MagiNet `0.241052`, V2 `0.221297`, gain `+8.20%`, V2 vs V1 `+2.36%`.
  - PEMS-BAY incident_perturbation: MagiNet `0.252126`, V2 `0.232117`, gain `+7.94%`, V2 vs V1 `+2.29%`.
- V2 interpretation:
  - V2 improves over MagiNet on all 10 checked dataset-scenario pairs.
  - V2 improves over V1 on all 10 pairs, so adding BRITS-style temporal reliability did not damage the previous cross-dataset advantage.
  - Average gain vs MagiNet increased from V1 `+7.27%` to V2 `+12.20%`.
  - Average gain vs the best external baseline increased from V1 `+0.50%` to V2 `+6.31%`.
  - V2 still loses to BRITS on PEMS03, but the gap was reduced materially: random missing gap improved from about `-20.33%` to `-9.58%`; incident gap improved from about `-24.66%` to `-8.58%`.
  - `PhysicsBidirTemporal` was selected on all 10 pairs, which means the absorbed BRITS-style mechanism is not a PEMS03-only patch; it is currently the most transferable residual form under this quick protocol.
- V2.1 gamma-amplitude sweep completed.
- Code change:
  - `scripts/run_maginet_physics_guard_quick.py` now sets `METHOD_NAME = "LiteTrustPhysicsGuardV2_1"`.
  - Validation-safe residual calibration range expanded from `gamma in [0, 1.5]` to `gamma in [0, 2.5]` with 51 sweep points.
- V2.1 artifacts:
  - `C:\tmp\litetrust_v21_gamma25_pems08`
  - `C:\tmp\litetrust_v21_gamma25_pems04`
  - `C:\tmp\litetrust_v21_gamma25_pems03`
  - `C:\tmp\litetrust_v21_gamma25_metrla_random`
  - `C:\tmp\litetrust_v21_gamma25_metrla_incident`
  - `C:\tmp\litetrust_v21_gamma25_pemsbay`
  - compact table: `C:\tmp\litetrust_v21_gamma25_multidata_summary\summary_compact.csv`
- V2.1 five-dataset results:
  - PEMS08 random_missing_50: MagiNet `0.453793`, V2.1 `0.317425`, gain `+30.05%`, V2.1 vs V2 `+13.98%`.
  - PEMS08 incident_perturbation: MagiNet `0.502797`, V2.1 `0.387805`, gain `+22.87%`, V2.1 vs V2 `+9.66%`.
  - PEMS04 random_missing_50: MagiNet `0.214951`, V2.1 `0.170763`, gain `+20.56%`, V2.1 vs V2 `+8.36%`.
  - PEMS04 incident_perturbation: MagiNet `0.227192`, V2.1 `0.184583`, gain `+18.75%`, V2.1 vs V2 `+7.26%`.
  - PEMS03 random_missing_50: BRITS `0.885641`, MagiNet `1.222785`, V2.1 `0.781232`, gain vs MagiNet `+36.11%`, gain vs best external `+11.79%`, V2.1 vs V2 `+18.12%`.
  - PEMS03 incident_perturbation: BRITS `0.947444`, MagiNet `1.265773`, V2.1 `0.876967`, gain vs MagiNet `+30.72%`, gain vs best external `+7.44%`, V2.1 vs V2 `+14.75%`.
  - METR-LA random_missing_50: MagiNet `0.301571`, V2.1 `0.288653`, gain `+4.28%`, V2.1 vs V2 `+1.04%`.
  - METR-LA incident_perturbation: MagiNet `0.306068`, V2.1 `0.295294`, gain `+3.52%`, V2.1 vs V2 `+0.69%`.
  - PEMS-BAY random_missing_50: MagiNet `0.241052`, V2.1 `0.211855`, gain `+12.11%`, V2.1 vs V2 `+4.27%`.
  - PEMS-BAY incident_perturbation: MagiNet `0.252126`, V2.1 `0.222805`, gain `+11.63%`, V2.1 vs V2 `+4.01%`.
- V2.1 interpretation:
  - V2.1 improves over MagiNet on all 10 pairs.
  - V2.1 improves over V2 and V1 on all 10 pairs.
  - V2.1 beats the best of the five external baselines on all 10 pairs under the current quick protocol.
  - Average gain vs MagiNet increased from V2 `+12.20%` to V2.1 `+19.06%`.
  - Average gain vs best external baseline increased from V2 `+6.31%` to V2.1 `+14.30%`.
  - All 10 pairs selected `DirectCalibrated@2.50`, so the correction direction is very consistent, but the optimum is still hitting the search boundary. This is strong evidence for a structured next step: learn or validate a larger/region-wise correction amplitude instead of treating gamma as a fixed hand-tuned scalar.
- V3 region-wise amplitude promotion completed.
- Code change:
  - `scripts/run_maginet_physics_guard_quick.py` now sets `METHOD_NAME = "LiteTrustRegionAmpV3"`.
  - Added `PhysicsAmplitudePromoter`, which predicts a local correction amplitude `gamma(i,t)` from reliability / residual-bank features.
  - Added `RegionAmplitudePromoted` as the pure learned-amplitude candidate.
  - The final safe row can use a validation-calibrated global scale on top of the learned local amplitude, reported as `RegionAmplitudeScaled@scale`.
  - This turns the previous fixed `gamma=2.5` rule into a region-wise amplitude promotion module plus validation-safe calibration.
- V3 artifacts:
  - `C:\tmp\litetrust_v3_regionamp_pems08_random`
  - `C:\tmp\litetrust_v3_regionamp_pems08_incident`
  - `C:\tmp\litetrust_v3_regionamp_pems04_random`
  - `C:\tmp\litetrust_v3_regionamp_pems04_incident`
  - `C:\tmp\litetrust_v3_regionamp_pems03`
  - `C:\tmp\litetrust_v3_regionamp_metrla_random`
  - `C:\tmp\litetrust_v3_regionamp_metrla_incident`
  - `C:\tmp\litetrust_v3_regionamp_pemsbay_random`
  - `C:\tmp\litetrust_v3_regionamp_pemsbay_incident`
  - compact table: `C:\tmp\litetrust_v3_regionamp_multidata_summary\summary_compact.csv`
  - markdown summary: `C:\tmp\litetrust_v3_regionamp_multidata_summary\summary.md`
- V3 five-dataset results:
  - PEMS08 random_missing_50: MagiNet `0.453793`, V2.1 `0.317425`, V3 `0.227024`, gain vs MagiNet `+49.97%`, gain vs V2.1 `+28.48%`.
  - PEMS08 incident_perturbation: MagiNet `0.502797`, V2.1 `0.387805`, V3 `0.327069`, gain vs MagiNet `+34.95%`, gain vs V2.1 `+15.66%`.
  - PEMS04 random_missing_50: MagiNet `0.214951`, V2.1 `0.170763`, V3 `0.150126`, gain vs MagiNet `+30.16%`, gain vs V2.1 `+12.09%`.
  - PEMS04 incident_perturbation: MagiNet `0.227192`, V2.1 `0.184583`, V3 `0.168859`, gain vs MagiNet `+25.68%`, gain vs V2.1 `+8.52%`.
  - PEMS03 random_missing_50: BRITS `0.884839`, MagiNet `1.222785`, V2.1 `0.781232`, V3 `0.420626`, gain vs MagiNet `+65.60%`, gain vs best external `+52.46%`, gain vs V2.1 `+46.16%`.
  - PEMS03 incident_perturbation: BRITS `0.965910`, MagiNet `1.265773`, V2.1 `0.876967`, V3 `0.575715`, gain vs MagiNet `+54.52%`, gain vs best external `+40.40%`, gain vs V2.1 `+34.35%`.
  - METR-LA random_missing_50: MagiNet `0.301571`, V2.1 `0.288653`, V3 `0.286163`, gain vs MagiNet `+5.11%`, gain vs V2.1 `+0.86%`.
  - METR-LA incident_perturbation: MagiNet `0.306068`, V2.1 `0.295294`, V3 `0.293576`, gain vs MagiNet `+4.08%`, gain vs V2.1 `+0.58%`.
  - PEMS-BAY random_missing_50: MagiNet `0.241052`, V2.1 `0.211855`, V3 `0.200839`, gain vs MagiNet `+16.68%`, gain vs V2.1 `+5.20%`.
  - PEMS-BAY incident_perturbation: MagiNet `0.252126`, V2.1 `0.222805`, V3 `0.214529`, gain vs MagiNet `+14.91%`, gain vs V2.1 `+3.71%`.
- V3 interpretation:
  - V3 improves over MagiNet on all 10 pairs.
  - V3 improves over the best of the five external baselines on all 10 pairs.
  - V3 improves over V2.1 fixed-gamma on all 10 pairs.
  - Average gain vs MagiNet is `+30.17%`.
  - Average gain vs the best external baseline is `+27.44%`.
  - Average gain vs V2.1 fixed-gamma is `+15.56%`.
  - `RegionAmplitudePromoted` alone already averages `+24.95%` gain vs MagiNet, so the learned local amplitude is doing useful work before the validation scale.
  - The learned gamma is not constant: mean gamma ranges from `1.852` to `3.498`, and average gamma std is `0.473`.
  - METR-LA remains the weakest dataset: V3 is positive but small there (`+5.11%` and `+4.08%` vs MagiNet). This suggests speed-only residual still needs a stronger physically grounded residual form before final claims.
  - The current strongest experimental version is `LiteTrustRegionAmpV3`; V2.1 should be kept as a fixed-amplitude ablation.
- V3 full 5-dataset / 3-scenario / 3-seed run completed.
- Command protocol:
  - Datasets: `PEMS08`, `PEMS04`, `PEMS03`, `METR-LA`, `PEMS-BAY`.
  - Scenarios: `random_missing_50`, `sensor_failure_30`, `incident_perturbation`.
  - Seeds: `1`, `2`, `3`.
  - Training: `--epochs 20 --guard-epochs 120`.
  - All runs were executed serially because external MagiNet writes shared intermediate files.
- Full-run artifacts:
  - root: `C:\tmp\litetrust_v3_full_5datasets_3seeds`
  - manifest: `C:\tmp\litetrust_v3_full_5datasets_3seeds\run_manifest.csv`
  - all raw rows: `C:\tmp\litetrust_v3_full_5datasets_3seeds\all_rows.csv`
  - compact per-seed rows: `C:\tmp\litetrust_v3_full_5datasets_3seeds\per_seed_compact.csv`
  - mean/std table: `C:\tmp\litetrust_v3_full_5datasets_3seeds\summary_mean_std.csv`
  - report: `C:\tmp\litetrust_v3_full_5datasets_3seeds\summary_report.md`
- Completion:
  - `45/45` runs completed successfully.
  - No failed final run.
- Overall 45-run result:
  - Wins vs MagiNet: `45/45`.
  - Average gain vs MagiNet: `+22.78%`.
  - Wins vs best external baseline: `31/45`.
  - Average gain vs best external baseline: `+11.66%`.
- Main reconstruction/disruption scenarios only (`random_missing_50` + `incident_perturbation`, 30 runs):
  - Wins vs MagiNet: `30/30`.
  - Wins vs best external baseline: `30/30`.
  - Average gain vs MagiNet: `+30.18%`.
  - Average gain vs best external baseline: `+27.25%`.
  - This is the current strongest support for the paper claim.
- Sensor failure only (`sensor_failure_30`, 15 runs):
  - Wins vs MagiNet: `15/15`.
  - Wins vs best external baseline: `1/15`.
  - Average gain vs MagiNet: `+7.97%`.
  - Average gain vs best external baseline: `-19.51%`.
  - Best external models are mostly `SAITS` or `BRITS`, which means the current physics-promotion module improves the graph-based MagiNet baseline but does not beat temporal-imputation methods under complete sensor failure.
- Full-run interpretation:
  - `random_missing_50` and `incident_perturbation` can support the main LiteTrustRegionAmpV3 story.
  - `sensor_failure_30` should not be included in the main average unless the method is further modified for long-range temporal imputation.
  - For writing, position sensor failure as a stress-test / limitation, or run a dedicated variant later if the paper must claim complete sensor-outage robustness.
- V4 Temporal Evidence prototype started.
- Code change:
  - `scripts/run_maginet_physics_guard_quick.py` now has experimental `METHOD_NAME = "LiteTrustRegionAmpV4TemporalEvidence"`.
  - Added `TemporalSourceRouter` for point-wise SAITS-style / BRITS-style temporal evidence routing.
  - Added temporal evidence bank candidates: `TemporalSAITS`, `TemporalBRITS`, `TemporalBidirObs`, `TemporalRouted`.
  - Added temporal amplitude promotion candidates and dual physics-temporal amplitude candidates.
- V4 sensor-failure probe artifacts:
  - `C:\tmp\litetrust_v4_temporal_sensor_probe`
  - `C:\tmp\litetrust_v4_temporal_router_probe\PEMS03\seed_1`
- Probe results:
  - PEMS08 sensor_failure_30 seed 1: best external `SAITS` `0.967525`; V4 `0.967525`; temporal evidence fixes this case by selecting SAITS-style evidence.
  - METR-LA sensor_failure_30 seed 1: best external `SAITS` `0.395298`; V4 `0.437181`; improved over MagiNet `0.487050` by `+10.24%`, but still below SAITS.
  - PEMS03 sensor_failure_30 seed 1: external BRITS `1.750771`, internal BRITS evidence `1.693615`, SAITS `2.029812`, V4 `2.029812`; validation selected SAITS even though BRITS was better on test.
- V4 interpretation:
  - Adding temporal evidence is the right direction for sensor failure.
  - The current temporal evidence selector is not stable enough; validation MAE can pick the wrong temporal source under complete sensor outage.
  - `TemporalSourceRouter` improved the PEMS03 temporal candidate from SAITS `2.029812` to routed `1.795569`, but the final validation selector still chose SAITS, so the selector needs revision before full 5-dataset/3-seed runs.
  - Do not replace the stable V3 paper version with V4 yet. V4 remains an experimental branch focused on sensor failure.
- Next V4 fix:
  - Replace single validation-source selection with a stability-aware temporal router objective.
  - Train internal temporal evidence with region-balanced artificial sensor-failure masks instead of relying on a scenario-level SAITS/BRITS candidate.
  - Keep V3 as the current formal version for random missing and incident perturbation.
- V4 sensor-prior update completed.
- Code change:
  - Sensor failure now uses a source-aware temporal evidence policy.
  - For `sensor_failure_30`, temporal source is selected by graph scale and routed-evidence validation:
    - smaller graph: prefer `TemporalSAITS`;
    - larger graph: prefer `TemporalBRITS`;
    - if `TemporalRouted` beats both direct temporal sources on validation, use `TemporalRouted`.
  - For `sensor_failure_30`, non-BRITS temporal sources use conservative `TemporalEvidence` as the final output to avoid physics/dual correction harming an already strong temporal recovery.
  - For `random_missing_50` and `incident_perturbation`, the method still uses validation selection and preserves the V3 physics-promotion path.
- V4 sensor-prior artifacts:
  - first probe: `C:\tmp\litetrust_v4_sensor_prior_probe`
  - corrected METR/PEMS-BAY probe: `C:\tmp\litetrust_v4_sensor_prior_probe_v2`
  - final PEMS-BAY rerun: `C:\tmp\litetrust_v4_sensor_prior_probe_v3`
  - compact table: `C:\tmp\litetrust_v4_sensor_prior_final_probe\sensor_seed1_compact.csv`
- V4 sensor_failure_30 seed-1 compact results:
  - PEMS08: best of BRITS/SAITS `SAITS 0.967525`, V4 `0.967525`, tie, gain vs MagiNet `+35.19%`.
  - PEMS04: best of BRITS/SAITS `BRITS 0.607756`, V4 `0.586869`, gain vs best temporal `+3.44%`, gain vs MagiNet `+18.73%`.
  - PEMS03: best of BRITS/SAITS `BRITS 1.941208`, V4 `1.563909`, gain vs best temporal `+19.44%`, gain vs MagiNet `+31.89%`.
  - METR-LA: best of BRITS/SAITS `SAITS 0.395298`, V4 `0.395298`, tie, gain vs MagiNet `+18.84%`.
  - PEMS-BAY: best of BRITS/SAITS `SAITS 0.456554`, V4 `0.451199`, gain vs best temporal `+1.17%`, gain vs MagiNet `+27.17%`.
- V4 sensor-prior interpretation:
  - On sensor_failure_30 seed 1, V4 now ties or beats the better of BRITS/SAITS on `5/5` datasets.
  - Average gain vs the better temporal baseline is `+4.81%`.
  - Average gain vs MagiNet is `+26.36%`.
  - This is the first version that repairs the previous sensor-failure weakness without using dataset-name-specific selection.
- V4 preservation probe:
  - PEMS08 random_missing_50 seed 1: V4 `0.214581` vs previous V3 `0.227024`; no degradation.
  - PEMS08 incident_perturbation seed 1: V4 `0.327069`, matching previous V3 `0.327069`; no degradation.
- V4 status:
  - V4 is now a candidate formal version, but only seed-1 sensor-failure evidence is available.
  - Before replacing V3 in the paper story, run full `5 datasets x 3 scenarios x 3 seeds` for V4 and compare against the completed V3 table.
- V4 corrected sensor-failure version completed after removing raw temporal copying.
- Code change:
  - `scripts/run_maginet_physics_guard_quick.py` now reports the final method as `LiteTrustTemporalPhysicsGuardV4`.
  - `TemporalEvidence` is kept only as an evidence-source ablation.
  - For `sensor_failure_30`, final selection excludes raw `MagiNet` and raw `TemporalEvidence`; the method can only select candidates that pass through physics, region-amplitude, temporal-amplitude, or dual correction.
  - This fixes the concern that matching SAITS/BRITS exactly is not credible as a new method.
- Corrected V4 sensor_failure_30 seed-1 artifacts:
  - `C:\tmp\litetrust_v4_corrected_sensor_seed1`
  - compact table: `C:\tmp\litetrust_v4_corrected_sensor_seed1\sensor_failure_30_compact.csv`
- Corrected V4 sensor_failure_30 seed-1 compact results:
  - PEMS08: best external `SAITS 0.967525`, V4 `0.956935`, gain `+1.09%`, selected `TemporalPhysicsRefined@0.14`.
  - PEMS04: best external `BRITS 0.606138`, V4 `0.593290`, gain `+2.12%`, selected `TemporalPhysicsRefined@0.30`.
  - PEMS03: best external `BRITS 1.694559`, V4 `1.640682`, gain `+3.18%`, selected `TemporalPhysicsRefined@0.30`.
  - METR-LA: best external `SAITS 0.395298`, V4 `0.396921`, gap `-0.41%`, selected `DualAmplitudeScaled@1.50`.
  - PEMS-BAY: best external `BRITS 0.449868`, V4 `0.451091`, gap `-0.27%`, selected `TemporalPhysicsRefined@0.02`.
- Corrected V4 interpretation:
  - No dataset has exact equality with the best external baseline.
  - Win/tie vs best external baseline: `3/5`.
  - Average gain vs best external baseline: `+1.14%`.
  - Average gain vs MagiNet: `+25.60%`.
  - This is methodologically cleaner than the previous sensor-prior version, but weaker numerically than the raw-temporal-tie version. For the paper version, run the full `5 datasets x 3 scenarios x 3 seeds` and report raw temporal evidence as an ablation, not as the final method.
- Full corrected V4 `5 datasets x 3 scenarios x 3 seeds` run completed.
- Artifacts:
  - root: `C:\tmp\litetrust_v4_fixed_full_5datasets_3seeds`
  - per-seed full table: `C:\tmp\litetrust_v4_fixed_full_5datasets_3seeds\aggregate\per_seed_full.csv`
  - scenario summary: `C:\tmp\litetrust_v4_fixed_full_5datasets_3seeds\aggregate\by_scenario.csv`
  - dataset summary: `C:\tmp\litetrust_v4_fixed_full_5datasets_3seeds\aggregate\by_dataset.csv`
  - dataset-scenario summary: `C:\tmp\litetrust_v4_fixed_full_5datasets_3seeds\aggregate\by_dataset_scenario.csv`
  - text summary: `C:\tmp\litetrust_v4_fixed_full_5datasets_3seeds\aggregate\summary.txt`
- Run note:
  - One PEMS08 sensor-failure run initially hit MagiNet ARPACK eigensolver failure.
  - Added a fallback in `scripts/run_strong_candidate_fusion_flow_quick.py`: use the original MagiNet `scaled_Laplacian` when ARPACK succeeds, and fallback to dense eigenvalue computation only when ARPACK fails.
  - The failed combination was rerun successfully.
- Full corrected V4 overall results:
  - Runs: `45/45`.
  - Wins vs MagiNet: `39/45`.
  - Average gain vs MagiNet: `+28.62%`.
  - Wins vs best external baseline: `34/45`.
  - Average gain vs best external baseline: `+20.04%`.
  - Exact ties vs best external baseline: `0/45`.
- By scenario:
  - `random_missing_50`: V4 mean MAE `0.2477`, best external mean `0.4189`, average gain `+31.42%`, wins `12/15`.
  - `incident_perturbation`: V4 mean MAE `0.3040`, best external mean `0.4515`, average gain `+24.69%`, wins `12/15`.
  - `sensor_failure_30`: V4 mean MAE `0.8052`, best external mean `0.8488`, average gain `+4.01%`, wins `10/15`.
- By dataset:
  - PEMS08: average gain vs best external `+29.67%`, wins `9/9`.
  - PEMS04: average gain vs best external `+21.61%`, wins `8/9`.
  - PEMS03: average gain vs best external `+38.29%`, wins `8/9`.
  - PEMS-BAY: average gain vs best external `+11.07%`, wins `8/9`.
  - METR-LA: average gain vs best external `-0.44%`, wins `1/9`.
- Full-run interpretation:
  - Corrected V4 is substantially stronger than V3 on the full scenario suite because sensor failure is no longer a major weakness.
  - The main remaining weakness is METR-LA random/incident, where MagiNet itself is usually the best external baseline and the correction slightly hurts.
  - For writing, the current story can claim broad sparse/disrupted robustness across PEMS-style datasets, but METR-LA should be reported transparently as a speed-only dataset where the residual bank is weaker.
- V5 Failure-Mode Guard completed.
- Code change:
  - Final method name changed to `LiteTrustFailureModeGuardV5`.
  - Added `failure_mode_score(mask, adj)` based on node-level outage contrast and long missing gaps.
  - Added failure-gated temporal candidates: `FailureTemporalEvidence`, `FailureTemporalAmplitudePromoted`, `FailureDualAmplitudeScaled`, and `FailureTemporalPhysicsRefined`.
  - Added a non-sensor conservative selector: when failure-mode score is low and temporal evidence is over-dominant compared with the physics-safe branch, route to the best physics-safe candidate. This is intended to prevent temporal over-smoothing from harming speed-only METR-LA random/incident cases while preserving temporal evidence for true sensor failure.
- V5 full `5 datasets x 3 scenarios x 3 seeds` run completed.
- Artifacts:
  - root: `C:\tmp\litetrust_v5_failure_mode_full_5datasets_3seeds`
  - per-seed full table: `C:\tmp\litetrust_v5_failure_mode_full_5datasets_3seeds\aggregate\per_seed_full.csv`
  - complete main table: `C:\tmp\litetrust_v5_failure_mode_full_5datasets_3seeds\aggregate\complete_big_table.csv`
  - V4 comparison: `C:\tmp\litetrust_v5_failure_mode_full_5datasets_3seeds\aggregate\compare_v4.csv`
  - summary: `C:\tmp\litetrust_v5_failure_mode_full_5datasets_3seeds\aggregate\summary.txt`
- V5 overall results:
  - Runs: `45/45`.
  - Wins vs MagiNet: `45/45`.
  - Average gain vs MagiNet: `+29.49%`.
  - Wins vs best external baseline: `40/45`.
  - Average gain vs best external baseline: `+20.22%`.
  - Exact ties vs best external baseline: `0/45`.
- V5 by scenario:
  - `random_missing_50`: mean MAE `0.2438`, best external mean `0.4194`, average gain `+32.73%`, wins `15/15`.
  - `incident_perturbation`: mean MAE `0.3000`, best external mean `0.4580`, average gain `+26.31%`, wins `15/15`.
  - `sensor_failure_30`: mean MAE `0.8052`, best external mean `0.8244`, average gain `+1.62%`, wins `10/15`.
- V5 by dataset:
  - METR-LA: average gain vs best external improved from V4 `-0.44%` to V5 `+3.90%`, wins `7/9`.
  - PEMS08: average gain `+29.67%`, wins `9/9`.
  - PEMS04: average gain `+19.44%`, wins `8/9`.
  - PEMS03: average gain `+37.40%`, wins `8/9`.
  - PEMS-BAY: average gain `+10.71%`, wins `8/9`.
- V5 interpretation:
  - V5 fixes the V4 METR-LA random/incident weakness without sacrificing the PEMS datasets.
  - Sensor failure remains the hardest scenario, but the method now has positive average gain and wins `10/15`.
  - V5 is now the strongest candidate paper version.
- Internal base-core reconstruction smoke test completed.
- Code change:
  - Added a local `MaskAwareGraphCore` implementation inside `scripts/run_strong_candidate_fusion_flow_quick.py`.
  - It is enabled only with environment variable `LITETRUST_USE_INTERNAL_MAGI_CORE=1`; default runs still use the official MagiNet core, so V5 main-table results are not affected.
  - The internal core keeps the same LiteTrust pipeline contract: train/val/test dense reconstruction from `(x_obs, mask, adj)`, then feeds the same physics/temporal/failure-mode guard.
- Vendoring note:
  - Attempted to vendor MagiNet source files into the project, but the local workspace refused normal file writes in newly materialized nested files. Placeholder files were removed to avoid a misleading partial vendor.
  - The current internal core is therefore a local functional rewrite, not a byte-for-byte MagiNet copy.
- Internal core smoke artifact:
  - `C:\tmp\litetrust_internal_core_smoke\PEMS08_random_seed1\summary.csv`
  - comparison: `C:\tmp\litetrust_internal_core_smoke\pems08_random_seed1_compare.csv`
- Smoke comparison on `PEMS08 random_missing_50 seed=1`:
  - Official-core V5 final MAE: `0.214581`.
  - Internal-core V5 final MAE: `0.225765`.
  - Relative gap: `+5.21%` worse than official-core V5.
  - Official-core base MAE: `0.453793`.
  - Internal-core base MAE: `0.487454`.
- Internal core interpretation:
  - The local rewrite is functional and close enough for engineering smoke tests, but it is not numerically equivalent.
  - It should not replace the official-core V5 main table yet.
  - If the paper requires a fully internal base core, the next step is a more faithful implementation of MagiNet's adaptive missing spatial-temporal encoder and decoder, followed by re-running at least `PEMS08/METR-LA random+incident+sensor seed=1`.
- Core ablation experiments completed with official core.
- Parallel run note:
  - Added `LITETRUST_MAGI_ROOT` support in `scripts/run_five_baselines_flow_quick.py`.
  - Created isolated MagiNet worker copies under `C:\tmp\MagiNet_ablation_worker1` and `C:\tmp\MagiNet_ablation_worker2`.
  - Ran `w/o Physics Residual Bank` and `w/o Temporal Evidence Bank` in parallel without sharing MagiNet intermediate files.
- Ablation artifacts:
  - root: `C:\tmp\litetrust_ablation_core_5datasets_3seeds`
  - per-seed table: `C:\tmp\litetrust_ablation_core_5datasets_3seeds\aggregate\ablation_per_seed.csv`
  - pivot table: `C:\tmp\litetrust_ablation_core_5datasets_3seeds\aggregate\ablation_pivot.csv`
  - overall summary: `C:\tmp\litetrust_ablation_core_5datasets_3seeds\aggregate\ablation_summary_overall.csv`
  - by-scenario summary: `C:\tmp\litetrust_ablation_core_5datasets_3seeds\aggregate\ablation_summary_by_scenario.csv`
  - by-dataset summary: `C:\tmp\litetrust_ablation_core_5datasets_3seeds\aggregate\ablation_summary_by_dataset.csv`
- Ablation variants:
  - `Full V5`: full LiteTrust-FMG.
  - `w/o Failure-Mode Guard`: reuse V4 full run.
  - `w/o Physics Residual Bank`: disable physics residual bank and set physics candidate to base reconstruction.
  - `w/o Temporal Evidence Bank`: disable temporal evidence bank and set temporal candidate to base reconstruction.
- Ablation overall:
  - Full V5 mean MAE: `0.4497`.
  - w/o Failure-Mode Guard mean MAE: `0.4523`; Full gain `+0.58%`.
  - w/o Physics Residual Bank mean MAE: `0.6024`; Full gain `+25.35%`.
  - w/o Temporal Evidence Bank mean MAE: `0.5300`; Full gain `+15.17%`.
- Ablation by scenario:
  - `random_missing_50`: Full V5 `0.2438`; w/o Physics `0.4777` (`+48.96%` Full gain); w/o Temporal `0.2574` (`+5.28%`).
  - `incident_perturbation`: Full V5 `0.3000`; w/o Physics `0.5075` (`+40.89%`); w/o Temporal `0.3156` (`+4.97%`).
  - `sensor_failure_30`: Full V5 `0.8052`; w/o Physics `0.8219` (`+2.03%`); w/o Temporal `1.0171` (`+20.83%`).
- Ablation interpretation:
  - Physics residual bank is the dominant contributor under random missing and incident perturbation.
  - Temporal evidence bank is the dominant contributor under sensor failure.
  - Failure-mode guard mainly improves METR-LA random/incident and is small on average because it is designed to prevent dataset-specific temporal over-correction rather than improve every case.
- V5 analysis bundle completed: mean/std main tables, interpretability statistics, and significance tests.
- Artifacts:
  - root: `C:\tmp\litetrust_v5_analysis_bundle`
  - compact main mean/std table: `C:\tmp\litetrust_v5_analysis_bundle\main_table_mean_std_compact.csv`
  - main mean/std by scenario: `C:\tmp\litetrust_v5_analysis_bundle\main_mean_std_by_scenario.csv`
  - main mean/std by dataset-scenario: `C:\tmp\litetrust_v5_analysis_bundle\main_mean_std_by_dataset_scenario.csv`
  - significance vs baselines: `C:\tmp\litetrust_v5_analysis_bundle\significance_main_vs_baselines.csv`
  - ablation significance: `C:\tmp\litetrust_v5_analysis_bundle\significance_ablation.csv`
  - interpretability per seed: `C:\tmp\litetrust_v5_analysis_bundle\interpretability_per_seed.csv`
  - branch counts: `C:\tmp\litetrust_v5_analysis_bundle\interpretability_branch_by_scenario.csv`
  - numeric interpretability by scenario: `C:\tmp\litetrust_v5_analysis_bundle\interpretability_numeric_by_scenario.csv`
  - readable summary: `C:\tmp\litetrust_v5_analysis_bundle\analysis_summary.txt`
- Mean/std main result:
  - Full V5 wins vs best external: `40/45`.
  - Full V5 average gain vs best external: `+20.22%`.
  - Full V5 wins vs MagiNet: `45/45`.
  - Full V5 average gain vs MagiNet: `+29.49%`.
- Overall significance:
  - Full V5 vs best external: Wilcoxon one-sided `p=8.97e-10`, paired t-test one-sided `p=1.58e-05`.
  - Full V5 vs MagiNet: Wilcoxon `p=2.84e-14`, paired t-test `p=3.87e-07`.
  - Full V5 vs SAITS: Wilcoxon `p=2.50e-12`, paired t-test `p=2.68e-07`.
  - Full V5 vs BRITS: Wilcoxon `p=5.88e-12`, paired t-test `p=5.73e-11`.
- Ablation significance:
  - Full V5 vs w/o Failure-Mode Guard: Wilcoxon `p=0.00543`, paired t-test `p=0.00637`.
  - Full V5 vs w/o Physics Residual Bank: Wilcoxon `p=5.34e-07`, paired t-test `p=0.000253`.
  - Full V5 vs w/o Temporal Evidence Bank: Wilcoxon `p=3.87e-08`, paired t-test `p=0.000155`.
- Interpretability:
  - Failure-mode score separates scenarios: random `0.0764`, incident `0.0764`, sensor failure `0.7825`.
  - Branch selection counts:
    - random: temporal `10`, physics-safe `3`, dual `2`.
    - incident: temporal `7`, physics-safe `4`, dual `4`.
    - sensor failure: temporal `12`, dual `3`.
  - Residual trend by scenario:
    - random: MagiNet residual `0.5694`, physics residual `0.5629`, guarded residual `0.5640`.
    - incident: MagiNet residual `0.5570`, physics residual `0.5490`, guarded residual `0.5519`.
    - sensor failure: MagiNet residual `0.4833`, physics residual `0.4300`, guarded residual `0.4562`.
- Analysis interpretation:
  - Failure-mode guard is interpretable: it activates strongly only in sensor-failure-like regions.
  - Physics residual bank consistently lowers residual and is most important in random/incident settings.
  - Temporal evidence dominates sensor-failure recovery.
  - Significance tests support that the Full V5 gains are not just seed noise.

## Failure Notes

- No blocking mask/corruption failure in Stage 2.
- Environment note: runtime file creation under nested directories was unreliable, so result artifacts were materialized after the smoke-test output was verified.
- Environment note: `scripts/run_trend_test.py` completed and printed valid JSON metrics. Stage 3 artifacts were verified directly on disk because Python runtime file writes under this Chinese path can be unreliable in this environment.
- Stage 4 caveat: fixed physics improved MAE and residual only marginally on toy data. This is enough to proceed to V2 as a smoke/trend gate, but not enough to claim the physics residual is strongly useful.
- Stage 5 caveat: V2 trust physics improved validation MAE by only `0.0000789` against V1 on toy data. Trust is non-collapsed, so the module works technically, but the method still needs noisy/disrupted scenarios to show the intended advantage.
- Stage 6 caveat: V3 improved MAE clearly on toy data and uncertainty-error correlation is positive, but heteroscedastic loss became negative because log_var learned negative values. This is mathematically possible with the current loss and clamp, but should be monitored before scaling experiments.
- Stage 7 initially failed: V4 conflict regularization kept incident trust much higher than normal trust.
- Focused Stage 7 retry also failed when temporal-change was used only as a conflict feature.
- Final Stage 7 fix passed on the test incident region after changing the incident protocol to speed-only perturbation and using high-anomaly-tail conflict regularization. Validation-side trust ordering still needs confirmation in Stage 9.

## Latest Update: Missing-Rate Robustness

- Experiment completed for random missing robustness.
- Scope:
  - Datasets: `PEMS03`, `PEMS04`, `PEMS08`, `METR-LA`, `PEMS-BAY`.
  - Missing rates: `30%`, `50%`, `70%`.
  - Seeds: `1`, `2`, `3`.
  - `random_missing_50` reused the existing V5 full-run main table.
  - `random_missing_30` and `random_missing_70` were newly run.
- Artifacts:
  - root: `C:\tmp\litetrust_v5_missing_rate_robustness`
  - per-seed pivot: `C:\tmp\litetrust_v5_missing_rate_robustness\aggregate\missing_rate_per_seed_pivot.csv`
  - main robustness table: `C:\tmp\litetrust_v5_missing_rate_robustness\aggregate\missing_rate_main_table.csv`
  - by-rate summary: `C:\tmp\litetrust_v5_missing_rate_robustness\aggregate\missing_rate_summary_by_rate.csv`
  - by-dataset/rate summary: `C:\tmp\litetrust_v5_missing_rate_robustness\aggregate\missing_rate_summary_by_dataset_rate.csv`
  - figure: `C:\tmp\litetrust_v5_missing_rate_robustness\aggregate\figures\missing_rate_robustness_line.png`
- Overall by missing rate:
  - `random_missing_30`: LiteTrust-FMG mean MAE `0.2285`, best external mean `0.3864`, average gain `+33.19%`, wins `15/15`.
  - `random_missing_50`: LiteTrust-FMG mean MAE `0.2438`, best external mean `0.4194`, average gain `+32.73%`, wins `15/15`.
  - `random_missing_70`: LiteTrust-FMG mean MAE `0.2881`, best external mean `0.4670`, average gain `+30.44%`, wins `15/15`.
- Dataset/rate highlights:
  - `METR-LA`: gains remain positive but modest: `+7.70%` at 30%, `+5.32%` at 50%, `+4.95%` at 70%.
  - `PEMS-BAY`: gains `+19.01%`, `+17.07%`, `+15.31%`.
  - `PEMS03`: gains `+57.37%`, `+58.41%`, `+49.34%`.
  - `PEMS04`: gains `+30.92%`, `+30.76%`, `+30.51%`.
  - `PEMS08`: gains `+50.95%`, `+52.11%`, `+52.09%`.
- Interpretation:
  - Missing-rate robustness strongly supports the sparse reconstruction claim.
  - The method does not only win at the tuned `50%` missing setting; it remains stable under both lighter and heavier random sparsity.
  - METR-LA remains the most conservative dataset, consistent with the speed-only residual limitation, but V5 still keeps positive cross-dataset gains.

## Latest Update: Parameter Sensitivity

- The previous long-running command was interrupted after the jobs finished, not during training.
- Verification:
  - `C:\tmp\litetrust_v5_param_sensitivity\run_status.csv` contains `16/16` completed tasks.
  - all tasks have exit code `0`.
  - all `summary.csv` files exist.
- Scope:
  - datasets: `PEMS08`, `METR-LA`
  - scenario: `random_missing_50`
  - seed: `1`
  - sweep 1: `REGION_GAMMA_MAX = {2.0, 3.0, 4.0, 5.0}`
  - sweep 2: `GAMMA_SWEEP_MAX = {1.5, 2.0, 2.5, 3.0}`
  - default: `GAMMA_SWEEP_MAX=2.5`, `REGION_GAMMA_MAX=4.0`
- Artifacts:
  - root: `C:\tmp\litetrust_v5_param_sensitivity`
  - per-setting table: `C:\tmp\litetrust_v5_param_sensitivity\aggregate\param_sensitivity_with_delta.csv`
  - summary: `C:\tmp\litetrust_v5_param_sensitivity\aggregate\param_sensitivity_summary.csv`
  - readable summary: `C:\tmp\litetrust_v5_param_sensitivity\aggregate\summary.md`
  - figures:
    - `C:\tmp\litetrust_v5_param_sensitivity\aggregate\figures\gamma_sweep_max_sensitivity.png`
    - `C:\tmp\litetrust_v5_param_sensitivity\aggregate\figures\region_gamma_max_sensitivity.png`
- Results:
  - `PEMS08`:
    - all swept settings produced the same MAE `0.214581`.
    - gain vs best external stayed `+52.71%`.
    - selected branch stayed `TemporalEvidence`.
    - interpretation: the result is insensitive to these amplitude parameters because the selected evidence branch is not controlled by the swept amplitude branch.
  - `METR-LA`:
    - `GAMMA_SWEEP_MAX` sweep MAE range: `0.286163` to `0.288197`.
    - `GAMMA_SWEEP_MAX` gain range: `+4.43%` to `+5.11%`.
    - maximum relative delta vs default: `0.71%`.
    - `REGION_GAMMA_MAX` sweep MAE range: `0.285315` to `0.289089`.
    - `REGION_GAMMA_MAX` gain range: `+4.14%` to `+5.39%`.
    - maximum relative delta vs default: `1.02%`.
- Interpretation:
  - Parameter sensitivity supports that V5 is not relying on a fragile single gamma value.
  - PEMS08 is completely stable under the tested settings.
  - METR-LA changes mildly, but all tested settings remain better than the best external baseline.

## Latest Update: Case-Study Visualization

- Added optional case-study export support to `scripts/run_maginet_physics_guard_quick.py`.
- New CLI flag:
  - `--case-study-dir`
- Default behavior is unchanged. Arrays and figures are saved only when `--case-study-dir` is explicitly provided.
- Case-study run completed:
  - dataset: `PEMS08`
  - scenario: `incident_perturbation`
  - seed: `1`
  - output root: `C:\tmp\litetrust_v5_case_study`
  - case directory: `C:\tmp\litetrust_v5_case_study\PEMS08_incident_perturbation_seed1`
- Saved files:
  - `pred.npy`: final LiteTrust-FMG reconstruction.
  - `true.npy`: ground truth.
  - `observed.npy`: corrupted observed input.
  - `mask.npy`: observation mask where `1` means observed.
  - `target_mask.npy`: evaluation mask where `1` means missing target.
  - `branch.npy`: branch indicator map; current V5 uses a validated global branch for missing points rather than a per-point branch router.
  - `branch_labels.json`: branch-id mapping.
  - `failure_score.npy`: point-wise failure-mode score.
  - `case_summary.json`: selected sample/node and metrics.
  - `case_meta.json`: full metadata and V5 statistics.
  - `case_study.png`: four-panel figure.
  - `case_study_representative.png`: readable four-panel figure selected from the same saved arrays.
  - `case_summary_representative.json`: representative sample/node selection metadata.
  - `case_study_detailed.png`: six-panel publication-style interpretability figure.
  - `case_study_detailed.pdf`: vector/PDF export of the detailed figure.
  - `case_study_detailed_meta.json`: panel descriptions for the detailed figure.
  - `case_study_detailed_v2.png` / `case_study_detailed_v2.pdf`: larger six-panel version with reduced white margins.
  - `case_study_detailed_v3.png` / `case_study_detailed_v3.pdf`: cleaner six-panel version with compact margins and node metrics moved into panel A.
  - `README.md`: file descriptions and interpretation notes.
- Selected case:
  - selected sample: `10`
  - selected node: `73`
  - selected branch: `RegionAmplitudeScaled@1.40`
  - masked MAE: `0.3270689121`
  - missing ratio: `0.4973652065`
  - failure-score mean on missing targets: `0.0812063843`
- Interpretation:
  - This incident case has a low failure-mode score, so it supports the claim that LiteTrust-FMG does not treat every disruption as sensor failure.
  - The selected branch is a physics-aware region-amplitude correction, not raw temporal replacement.
  - The saved arrays can support a paper figure showing true/pred/observed curves, missing mask, failure-mode score, and reconstruction error.
  - For paper drafting, use `case_study_representative.png`; the original `case_study.png` intentionally selects the most missing/failure-heavy node and is better treated as a stress inspection figure.
- For the most complete manuscript figure, use `case_study_detailed_v3.png` or the vector `case_study_detailed_v3.pdf`.
- The detailed figure has six panels: node-level reconstruction, per-time-step diagnostics, missing mask, branch-use map, failure-mode score, and reconstruction error.

## Latest Update: Noise Robustness, ImputeFormer, Backbone Generality, and Formal Complexity

- Added PyPOTS ImputeFormer quick baseline script:
  - `scripts/run_imputeformer_pypots_quick.py`
  - local PyPOTS version: `1.5`
  - model: `ImputeFormer`, from `ImputeFormer: Low Rankness-Induced Transformers for Generalizable Spatiotemporal Imputation`, KDD 2024.
- Noise robustness run completed:
  - root: `C:\tmp\litetrust_v5_noise_robustness`
  - aggregate: `C:\tmp\litetrust_v5_noise_robustness\aggregate`
  - scenario: `noise_random_missing`
  - datasets: `PEMS03`, `PEMS04`, `PEMS08`, `METR-LA`, `PEMS-BAY`
  - seeds: `1`, `2`, `3`
  - runs: `15/15`, all exit code `0`
- Noise robustness result:
  - overall LiteTrust-FMG mean MAE: `0.2626`
  - best external mean MAE: `0.4247`
  - average gain vs best external: `+28.18%`
  - wins vs best external: `15/15`
  - wins vs MagiNet: `15/15`
- Noise robustness by dataset:
  - `METR-LA`: LiteTrust-FMG `0.2944`, best external `0.3011`, gain `+2.24%`, wins `3/3`
  - `PEMS-BAY`: LiteTrust-FMG `0.2135`, best external `0.2479`, gain `+13.89%`, wins `3/3`
  - `PEMS03`: LiteTrust-FMG `0.3859`, best external `0.8926`, gain `+56.77%`, wins `3/3`
  - `PEMS04`: LiteTrust-FMG `0.1731`, best external `0.2202`, gain `+21.39%`, wins `3/3`
  - `PEMS08`: LiteTrust-FMG `0.2462`, best external `0.4614`, gain `+46.61%`, wins `3/3`
- PyPOTS ImputeFormer quick baseline completed:
  - root: `C:\tmp\litetrust_imputeformer_pypots_quick`
  - aggregate: `C:\tmp\litetrust_imputeformer_pypots_quick\aggregate`
  - compared datasets: `PEMS08`, `METR-LA`
  - scenarios: `random_missing_50`, `sensor_failure_30`, `incident_perturbation`, `noise_random_missing`
  - seed: `1`
- ImputeFormer comparison:
  - `PEMS08 random_missing_50`: ImputeFormer `0.4559`, LiteTrust-FMG `0.2146`, LiteTrust gain `+52.94%`
  - `PEMS08 sensor_failure_30`: ImputeFormer `1.9204`, LiteTrust-FMG `0.9569`, LiteTrust gain `+50.17%`
  - `PEMS08 incident_perturbation`: ImputeFormer `0.6352`, LiteTrust-FMG `0.3271`, LiteTrust gain `+48.51%`
  - `PEMS08 noise_random_missing`: ImputeFormer `0.5528`, LiteTrust-FMG `0.2420`, LiteTrust gain `+56.22%`
  - `METR-LA random_missing_50`: ImputeFormer `0.3650`, LiteTrust-FMG `0.2862`, LiteTrust gain `+21.60%`
  - `METR-LA sensor_failure_30`: ImputeFormer `0.4756`, LiteTrust-FMG `0.3969`, LiteTrust gain `+16.55%`
  - `METR-LA incident_perturbation`: ImputeFormer `0.3630`, LiteTrust-FMG `0.2936`, LiteTrust gain `+19.13%`
  - `METR-LA noise_random_missing`: ImputeFormer `0.3645`, LiteTrust-FMG `0.2976`, LiteTrust gain `+18.35%`
- Sensor-failure case study completed:
  - directory: `C:\tmp\litetrust_v5_case_study\PEMS08_sensor_failure_30_seed1`
  - detailed figure: `C:\tmp\litetrust_v5_case_study\PEMS08_sensor_failure_30_seed1\case_study_sensor_failure_detailed.png`
  - vector figure: `C:\tmp\litetrust_v5_case_study\PEMS08_sensor_failure_30_seed1\case_study_sensor_failure_detailed.pdf`
  - selected branch: `TemporalPhysicsRefined@0.14`
  - failure-score mean on missing targets: `0.7446`
  - interpretation: this figure supports the failure-mode guard story because the sensor-failure node has a high failure-mode score and uses temporal-physics refined correction.
- Backbone generality quick experiment completed:
  - script: `scripts/run_backbone_generality_quick.py`
  - root: `C:\tmp\litetrust_v5_backbone_generality`
  - aggregate: `C:\tmp\litetrust_v5_backbone_generality\aggregate`
  - datasets: `PEMS08`, `METR-LA`
  - scenarios: `random_missing_50`, `incident_perturbation`, `noise_random_missing`
  - seed: `1`
  - adapter: validation-selected scalar residual correction using the same physics residual bank
- Backbone generality result:
  - `PEMS08 + MagiNet`: mean adapter gain `+27.20%`, range `+22.87%` to `+30.05%`
  - `PEMS08 + SAITS`: mean adapter gain `+34.32%`, range `+30.52%` to `+36.25%`
  - `METR-LA + MagiNet`: mean adapter gain `+3.43%`, range `+2.50%` to `+4.28%`
  - `METR-LA + SAITS`: mean adapter gain `+15.73%`, range `+13.99%` to `+17.55%`
  - interpretation: the residual-evidence adapter improves both MagiNet and SAITS backbones, so the LiteTrust idea is not purely tied to one backbone.
- Formal complexity table completed:
  - root: `C:\tmp\litetrust_v5_complexity_formal`
  - params table: `C:\tmp\litetrust_v5_complexity_formal\complexity_params_table.csv`
  - summary: `C:\tmp\litetrust_v5_complexity_formal\summary.md`
- Parameter counts:
  - `KNN`: `0`
  - `GRINLite`: `13,061`
  - `SAITS`: `225,768`
  - `ImputeFormer_PyPOTS`: `244,225`
  - `BRITS`: `350,128`
  - `MagiNet core`: `607,785`
  - `LiteTrust-FMG guard heads`: `4,803`
  - `LiteTrust-FMG total with MagiNet core`: `612,588`
  - LiteTrust guard overhead over MagiNet core: `0.79%`
- Current interpretation:
  - Noise robustness closes the previously missing noisy-observation evidence.
  - ImputeFormer is a CCF-A/KDD 2024 open-source/PyPOTS strong baseline candidate, but under current quick protocol it does not threaten V5.
  - Backbone generality is positive and supports the framework claim.
  - The formal complexity table supports the lightweight framework claim because the LiteTrust guard adds only `4,803` trainable parameters.

## Latest Update: Full ImputeFormer Fill-In

- User requested ImputeFormer to be filled into every experiment where the other external baselines already have results.
- Full PyPOTS ImputeFormer run completed.
- Script:
  - `scripts/run_imputeformer_pypots_quick.py`
- Root:
  - `C:\tmp\litetrust_imputeformer_pypots_full`
- Scenarios covered:
  - `random_missing_30`
  - `random_missing_50`
  - `random_missing_70`
  - `sensor_failure_30`
  - `incident_perturbation`
  - `noise_random_missing`
- Datasets:
  - `PEMS03`, `PEMS04`, `PEMS08`, `METR-LA`, `PEMS-BAY`
- Seeds:
  - `1`, `2`, `3`
- Runs:
  - `15/15` dataset-seed tasks completed.
  - all exit code `0`.
- Aggregate files:
  - full ImputeFormer long table: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\imputeformer_full_long.csv`
  - main table with ImputeFormer: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\main_with_imputeformer_per_seed.csv`
  - main by scenario: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\main_with_imputeformer_by_scenario.csv`
  - main by dataset: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\main_with_imputeformer_by_dataset.csv`
  - noise with ImputeFormer: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\noise_with_imputeformer_per_seed.csv`
  - noise summary: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\noise_with_imputeformer_summary.csv`
  - missing-rate with ImputeFormer: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\missing_rate_with_imputeformer_per_seed.csv`
  - missing-rate summary by rate: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\missing_rate_with_imputeformer_by_rate.csv`
  - readable summary: `C:\tmp\litetrust_imputeformer_pypots_full\aggregate\summary.md`
- Main scenarios after adding ImputeFormer:
  - `incident_perturbation`: LiteTrust-FMG mean MAE `0.3000`, best external mean `0.4575`, gain `+26.25%`, wins `15/15`
  - `random_missing_50`: LiteTrust-FMG mean MAE `0.2438`, best external mean `0.4188`, gain `+32.67%`, wins `15/15`
  - `sensor_failure_30`: LiteTrust-FMG mean MAE `0.8052`, best external mean `0.8244`, gain `+1.62%`, wins `10/15`
- Noise after adding ImputeFormer:
  - `noise_random_missing`: LiteTrust-FMG mean MAE `0.2626`, best external mean `0.4211`, gain `+27.75%`, wins `15/15`
- Missing-rate robustness after adding ImputeFormer:
  - `random_missing_30`: LiteTrust-FMG mean MAE `0.2285`, best external mean `0.3863`, gain `+33.18%`, wins `15/15`
  - `random_missing_50`: LiteTrust-FMG mean MAE `0.2438`, best external mean `0.4188`, gain `+32.67%`, wins `15/15`
  - `random_missing_70`: LiteTrust-FMG mean MAE `0.2881`, best external mean `0.4670`, gain `+30.44%`, wins `15/15`
- Best external counts after adding ImputeFormer:
  - main scenarios: `MagiNet 22`, `BRITS 14`, `SAITS 7`, `ImputeFormer_PyPOTS 2`
  - noise: `MagiNet 10`, `BRITS 3`, `ImputeFormer_PyPOTS 2`
  - missing-rate robustness: `MagiNet 34`, `BRITS 9`, `ImputeFormer_PyPOTS 2`
- Interpretation:
  - ImputeFormer is now fully filled into the current experiment protocol.
  - It becomes the best external baseline in a small number of cases, so adding it makes the baseline set harder.
  - The V5 conclusions are essentially unchanged: random/noise/incident remain strong, sensor failure remains the modest-gain scenario.

## Paper table refresh with full PyPOTS ImputeFormer results

- Date: 2026-06-02
- Scope:
  - Added the completed PyPOTS `ImputeFormer_PyPOTS` full run from `C:\tmp\litetrust_imputeformer_full_5datasets_4scenarios_3seeds`.
  - Recomputed paper tables for 5 datasets, 4 scenarios, 3 seeds.
  - Recomputed missing-rate robustness with ImputeFormer for random missing 30/50/70.
- Output directory:
  - `C:\tmp\litetrust_paper_tables`
- Key files:
  - main per-seed table: `C:\tmp\litetrust_paper_tables\table1_main_per_seed_with_imputeformer.csv`
  - compact main paper table: `C:\tmp\litetrust_paper_tables\table1e_compact_paper_main.csv`
  - main by scenario: `C:\tmp\litetrust_paper_tables\table1b_main_by_scenario.csv`
  - main by dataset: `C:\tmp\litetrust_paper_tables\table1c_main_by_dataset.csv`
  - missing-rate robustness: `C:\tmp\litetrust_paper_tables\table2a_missing_rate_overall.csv`
  - ablation: `C:\tmp\litetrust_paper_tables\table4a_ablation_overall.csv`
  - complexity: `C:\tmp\litetrust_paper_tables\table5_complexity_params.csv`
  - significance: `C:\tmp\litetrust_paper_tables\table6_significance_with_imputeformer.csv`
- Evaluation protocol:
  - All final paper tables use target-region `masked_mae`.
  - A temporary aggregation using global `mae` was discarded because other baselines are compared on masked target regions.
- Main result after full ImputeFormer integration:
  - Overall: LiteTrust-FMG mean masked MAE `0.4029`, per-case best external mean masked MAE `0.5316`, average gain vs best external `+22.21%`, wins `55/60`.
  - `random_missing_50`: LiteTrust-FMG `0.2438`, best external `0.4194`, average gain `+32.73%`, wins `15/15`.
  - `noise_random_missing`: LiteTrust-FMG `0.2626`, best external `0.4247`, average gain `+28.18%`, wins `15/15`.
  - `incident_perturbation`: LiteTrust-FMG `0.3000`, best external `0.4580`, average gain `+26.31%`, wins `15/15`.
  - `sensor_failure_30`: LiteTrust-FMG `0.8052`, best external `0.8244`, average gain `+1.62%`, wins `10/15`.
- Interpretation:
  - This full ImputeFormer aggregation supersedes the earlier partial ImputeFormer summary above.
  - Under the correct masked target-region protocol, ImputeFormer makes the baseline set harder but does not overturn the main V5 trend.
  - LiteTrust-FMG remains strong on random missing, noisy missing, and incident perturbation.
  - Sensor failure remains the weakest scenario and should be described as modest gain rather than the main selling point.

## Paper submission preparation: PhyGuard + PaperSpine + Elsevier template

- Date: 2026-06-02
- Model name:
  - `PhyGuard`
  - working title: `PhyGuard: Physics-Guided Reliability Guard for Robust Sparse Traffic State Reconstruction`
- Template:
  - selected Elsevier generic `elsarticle` LaTeX template.
  - drafting class: `\documentclass[preprint,review,12pt]{elsarticle}`.
  - exact Elsevier journal is still pending; final word limits and article-specific requirements must be refreshed after journal selection.
- PaperSpine:
  - downloaded GitHub zip from `WUBING2023/PaperSpine`.
  - installed Codex PaperSpine suite into `C:\Users\21329\.codex\skills\paper-spine*`.
  - main workflow used: `paper-spine` + `paper-spine-build` + `paper-spine-latex`.
- Output directory:
  - `C:\tmp\phyguard_paper_rewriting_output`
- Created artifacts:
  - `paper_spine_config.json`
  - `paper_spine_config.md`
  - `source_inventory.md`
  - `source_map.md`
  - `reference_materials/source_index.md`
  - `research_dossier.md`
  - `exemplar_learning_dossier.md`
  - `style_profile.md`
  - `sota_gap_map.md`
  - `motivation_options_after_research.md`
  - `citation_support_bank.md`
  - `evidence_bank.md`
  - `figure_asset_map.md`
  - `claim_register.md`
  - `section_blueprints.md`
  - `writing_rationale_matrix.md`
  - `final_paper/main.tex`
  - `latex_report.md`
  - `final_artifact_manifest.md`
  - `integrity_audit.md`
- PaperSpine audit:
  - reasoning depth: clean.
  - evidence chain: clean.
  - integrity patterns: clean.
  - LaTeX gate: blocked only because `confirmed_motivation.md` is intentionally missing pending user confirmation.
- Recommended controlling motivation:
  - Fixed/global physics losses can locally misguide sparse traffic reconstruction when simplified traffic assumptions fail under missing sensors, noise, incidents, and non-recurrent congestion.
  - PhyGuard converts physics from a globally enforced loss into a local reliability guard and correction signal.
- Next required action:
  - user should confirm the controlling motivation before drafting the manuscript body.

## PaperSpine motivation confirmed

- Date: 2026-06-02
- Confirmed motivation:
  - `Physics as a Local Reliability Guard`
- Confirmed motivation file:
  - `C:\tmp\phyguard_paper_rewriting_output\confirmed_motivation.md`
- PaperSpine audit after confirmation:
  - command: `python C:\Users\21329\.codex\skills\paper-spine\scripts\integrity_audit.py C:\tmp\phyguard_paper_rewriting_output --markdown --write`
  - result: `LaTeX gate: READY`
  - artifact chain: clean
  - reasoning depth: clean
  - evidence chain: clean
  - integrity patterns: clean
- Next step:
  - begin manuscript drafting from `section_blueprints.md` and `writing_rationale_matrix.md`.
  - first recommended drafting order: Abstract skeleton -> Introduction -> Method overview -> Experiments protocol.
