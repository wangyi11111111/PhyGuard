# Dataset Notes

PhyGuard uses the following dataset identifiers in the reproduction scripts:

- `PEMS03`
- `PEMS04`
- `PEMS08`
- `PEMS-BAY`
- `METR-LA`

The main paper experiments use the five datasets above. `PEMS08_debug` and
synthetic toy data are only for development and smoke tests.

## Sources Used in This Repository

| Dataset | Source used by the code | Files |
|---|---|---|
| PEMS03 | Zenodo record `7816008` | `PEMS03.npz`, `PEMS03.csv`, `PEMS03.txt` |
| PEMS04 | Zenodo record `7816008` | `PEMS04.npz`, `PEMS04.csv` |
| PEMS08 | Zenodo record `7816008` | `PEMS08.npz`, `PEMS08.csv` |
| PEMS-BAY | Hugging Face dataset `MintBruce/SkyTraffic` | `pems-bay.h5`, `pems/adj_mx_bay.pkl` |
| METR-LA | Hugging Face dataset `witgaw/METR-LA` | `train.parquet`, `val.parquet`, `test.parquet`, `sensor_graph/adj_mx.npy` |

The loader downloads these files into local caches through `huggingface_hub` or
the Zenodo file API. Raw datasets are not redistributed in this repository.

## Evaluation Scenarios

The main protocol uses target-region masked MAE under:

- `random_missing_50`
- `sensor_failure_30`
- `incident_perturbation`

Missing-rate robustness uses:

- `random_missing_30`
- `random_missing_50`
- `random_missing_70`

## Licensing and Access

Traffic benchmark datasets are not redistributed in this repository. The code
loads public mirrors where supported, but users remain responsible for checking
and complying with the original dataset terms. In particular:

- Caltrans PeMS data may require user registration and compliance with PeMS
  terms of use even when benchmark files are obtained from a public mirror.
- Zenodo record `7816008`, `MintBruce/SkyTraffic`, and `witgaw/METR-LA` are
  external sources. Check the license and terms of the exact mirror before
  redistribution.
- Synthetic toy data are provided only for smoke testing and cannot be used as
  paper evidence.
