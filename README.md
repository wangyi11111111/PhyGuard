# PhyGuard

**PhyGuard** is a lightweight physics-guided reliability guard for robust sparse
traffic state reconstruction.

The central idea is simple: physics should not always be enforced with a fixed
global loss weight. In open traffic systems, simplified physical assumptions can
be locally unreliable under missing sensors, noisy observations, incidents, and
non-recurrent congestion. PhyGuard converts physics from a globally enforced
constraint into a **local reliability guard** and a **guarded correction signal**.

Working paper title:

> PhyGuard: A Physics-Reliability-Aware Guard Framework for Sparse and
> Disrupted Traffic State Reconstruction

## Method Overview

![PhyGuard mechanism](assets/phyguard_mechanism.png)

**Figure 1. PhyGuard overview.** PhyGuard first obtains an initial traffic-state
estimate from a spatiotemporal reconstruction core, then uses physics residuals,
temporal evidence, and failure-mode signals to estimate local physical
reliability. Reliable regions receive promoted physical correction, while risky
regions keep the final estimate close to the reconstruction core to avoid
harmful correction.

PhyGuard is designed as a small correction layer on top of a strong
spatiotemporal reconstruction core. The current research prototype uses a strong
graph-time reconstruction core and adds the following PhyGuard components:

- **Physics residual bank**: builds candidate physical correction signals from
  flow-density, graph smoothness, and temporal consistency residuals.
- **Temporal evidence bank**: extracts local trend, mask pattern, and neighbor
  agreement evidence.
- **Failure-mode scorer**: estimates whether the local region is likely to be
  reliable or risky for physical correction.
- **Region-wise guarded correction**: promotes physical correction in reliable
  regions and suppresses it when physics may harm the reconstruction.

The final correction follows the local form:

```text
x_hat = x0 + gamma(i,t) * Delta_phys(i,t)
```

where `x0` is the initial reconstruction, `Delta_phys(i,t)` is a local physical
correction candidate, and `gamma(i,t)` is controlled by local reliability.

## Key Results

Current paper evidence evaluates PhyGuard as a plug-in guard on top of four
strong reconstruction backbones: BRITS, SAITS, ImputeFormer, and MagiNet. The
main protocol uses five traffic datasets, three disruption scenarios, and three
random seeds. Values are target-region masked MAE.

The four backbones are selected to cover different reconstruction biases rather
than minor variants of one architecture. BRITS represents recurrent
bidirectional imputation through forward--backward temporal state propagation.
SAITS represents self-attention-based time-series imputation and emphasizes
long-range temporal dependency modeling. ImputeFormer represents
transformer-style masked spatiotemporal reconstruction with structured
missing-pattern modeling. MagiNet represents mask-aware graph repair for traffic
data, where spatial neighborhood structure and missingness are jointly encoded.
This diversity is used to test whether PhyGuard works as a backbone-agnostic
local reliability guard.

| Evidence | Result |
|---|---:|
| Average MAE reduction over all paired runs | 7.04% |
| Random missing 50% reduction | 8.98% |
| Incident perturbation reduction | 7.92% |
| Sensor failure 30% reduction | 4.23% |
| Additional trainable parameters | 5,954 |
| Additional forward time per batch | < 1 ms |

The main claim is not that PhyGuard replaces a strong backbone. PhyGuard is a
lightweight reliability guard that decides when a local backbone output should
be preserved, corrected, or corrected only within a bounded range.

Aggregated paper evidence is in `results/phyguard_paper_evidence/`. The visual
case used in the manuscript is in `results/phyguard_visual_case/`.
The current manuscript source is placed under `paper/`.

## Repository Structure

```text
configs/       small experiment configurations
data/          dataset loading, masking, corruption, normalization
losses/        metrics and training losses
models/        lightweight baselines and wrappers
physics/       traffic residual definitions and collocation utilities
scripts/       training, evaluation, PhyGuard, baselines, and paper experiments
tests/         smoke tests and unit tests
results/       selected evidence summaries and smoke-test artifacts
```

Primary paper reproduction scripts are:

```text
reproduce/run_phyguard_plugin_strong_backbones.py
reproduce/run_phyguard_plugin_ablation.py
reproduce/build_phyguard_paper_evidence.py
reproduce/create_phyguard_visual_case.py
```

## Installation

This repository targets a conservative Windows + RTX 4060 setup, but the code is
standard Python/PyTorch.

```bash
pip install -r requirements.txt
```

For PyPOTS baselines such as BRITS, SAITS, and ImputeFormer:

```bash
pip install pypots huggingface_hub tables
```

## Quick Start

Run a smoke test on synthetic toy data:

```bash
python reproduce/run_smoke.py
```

Run one PhyGuard quick evaluation:

```bash
python reproduce/run_phyguard_plugin_strong_backbones.py \
  --datasets PEMS08 \
  --scenarios random_missing_50 incident_perturbation \
  --seeds 1 \
  --backbones SAITS \
  --epochs 5 \
  --output-dir results/reproduce_quick
```

Run the component ablation:

```bash
python reproduce/run_phyguard_plugin_ablation.py \
  --dataset PEMS08 \
  --scenarios random_missing_50 sensor_failure_30 incident_perturbation \
  --seeds 1 \
  --backbone SAITS \
  --epochs 5 \
  --output-dir results/reproduce_ablation
```

The direct PhyGuard entry point is still available:

```bash
python scripts/run_maginet_physics_guard_quick.py \
  --dataset PEMS08 \
  --seed 1 \
  --epochs 20 \
  --guard-epochs 120 \
  --scenarios random_missing_50 incident_perturbation \
  --output-dir results/phyguard_pems08_seed1
```

## Data

The paper experiments use:

- `PEMS03`
- `PEMS04`
- `PEMS08`
- `PEMS-BAY`
- `METR-LA`
- `METR-LA`

`PEMS03`, `PEMS04`, and `PEMS08` are loaded from Zenodo record `7816008`.
`PEMS-BAY` is loaded through the Hugging Face dataset `MintBruce/SkyTraffic`.
`METR-LA` is loaded through the Hugging Face dataset `witgaw/METR-LA`.
Synthetic traffic-like data are used only for smoke tests and are not paper
evidence.

See `DATASETS.md` and `THIRD_PARTY_NOTICES.md` before redistributing data or
third-party baseline code.

## Evaluation Protocol

The main paper protocol evaluates target-region masked MAE under:

- `random_missing_50`
- `sensor_failure_30`
- `incident_perturbation`

Additional robustness experiments use random missing rates 30%, 50%, and 70%.

Formal reproduction scripts are in `reproduce/`.

## Reproducibility Notes

- Main paper evidence is aggregated under `results/phyguard_paper_evidence/`.
- The repository does not redistribute raw traffic datasets.
- The current implementation is a research prototype. Exact paper runs should
  be repeated on the release branch before camera-ready submission.

## Citation

If you use this repository, please cite the forthcoming paper:

```bibtex
@misc{phyguard2026,
  title  = {PhyGuard: Physics-Guided Reliability Guard for Robust Sparse Traffic State Reconstruction},
  author = {Anonymous},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

This repository is prepared for research release. The PhyGuard code can be
released under the MIT License after confirming the licenses of any third-party
backbone code and datasets used in final experiments.

