# Dataset Notes

PhyGuard supports the following benchmark names in the reproduction scripts:

- `PEMS03`
- `PEMS04`
- `PEMS08`
- `PEMS-BAY`
- `METR-LA`
- `PEMS08_debug`

## Evaluation Scenarios

The main protocol uses target-region masked MAE under:

- `random_missing_50`
- `noise_random_missing`
- `incident_perturbation`
- `sensor_failure_30`

Missing-rate robustness uses:

- `random_missing_30`
- `random_missing_50`
- `random_missing_70`

## Licensing and Access

Traffic benchmark datasets are not redistributed in this repository. The code
loads public mirrors where supported, but users remain responsible for checking
and complying with the original dataset terms. In particular:

- Caltrans PeMS data may require user registration and compliance with PeMS
  terms of use.
- METR-LA and PEMS-BAY are commonly distributed through traffic forecasting
  benchmark mirrors; check the license of the specific mirror you use.
- Synthetic toy data are provided only for smoke testing and cannot be used as
  paper evidence.

