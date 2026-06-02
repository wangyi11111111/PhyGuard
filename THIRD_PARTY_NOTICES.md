# Third-Party Notices

This repository is prepared for research release. The PhyGuard code is released
under the repository license, but third-party baselines, datasets, and external
model implementations remain governed by their own licenses and terms.

## Python Packages

| Component | Role | License / Terms | Source |
|---|---|---|---|
| PyTorch | training and tensor backend | BSD-style license | https://github.com/pytorch/pytorch |
| PyPOTS | BRITS, SAITS, ImputeFormer baseline implementations | BSD license as reported by PyPI/GitHub metadata | https://github.com/WenjieDu/PyPOTS |
| scikit-learn | KNN imputation baseline | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| pandas / NumPy | data processing | BSD-style licenses | https://pandas.pydata.org/ / https://numpy.org/ |
| Hugging Face Hub | dataset access helper | Apache-2.0 | https://github.com/huggingface/huggingface_hub |

## Baselines and Related Methods

| Method | How it is used here | License status for this repo |
|---|---|---|
| BRITS | PyPOTS implementation baseline | governed by PyPOTS and the original paper/code license |
| SAITS | PyPOTS implementation baseline | governed by PyPOTS and the original paper/code license |
| ImputeFormer | PyPOTS baseline; official repository reports MIT license | cite the KDD 2024 paper and check the official repo before redistribution |
| GRIN / GRINLite | lightweight internal baseline inspired by graph recurrent imputation | this repo does not bundle official GRIN code; cite original GRIN if used for comparison |
| MagiNet | strong reconstruction core/baseline in experiments | verify official code license before bundling any external implementation |

## Datasets

| Dataset | Role | License / Access Notes |
|---|---|---|
| PEMS03 / PEMS04 / PEMS08 | traffic benchmark datasets | derived from Caltrans PeMS data and public benchmark releases; users should comply with the original PeMS/data-release terms |
| METR-LA | traffic benchmark dataset | commonly distributed through DCRNN/traffic forecasting benchmark mirrors; check the mirror license and original data source terms |
| PEMS-BAY | traffic benchmark dataset | commonly distributed through DCRNN/traffic forecasting benchmark mirrors; check the mirror license and original data source terms |

## Release Guidance

- Do not redistribute large raw datasets in this repository.
- Do not vendor official third-party model code unless its license permits
  redistribution and attribution is included.
- If a paper submission requires an artifact package, include links to original
  dataset/model sources and list exact package versions.
- Re-run final paper tables on the release branch before camera-ready
  submission.

