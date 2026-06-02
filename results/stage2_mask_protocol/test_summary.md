# Stage 2 Mask and Corruption Protocol Test Summary

## Command

`pytest tests/test_masks.py`

## Result

- Collected tests: 6
- Passed: 6
- Failed: 0

## Covered Cases

- `random_missing_mask(shape, missing_rate)`
- `sensor_failure_mask(shape, fail_rate)`
- `block_missing_mask(shape, adj, block_size)`
- `temporal_missing_mask(shape, missing_rate, duration)`
- `add_gaussian_noise(x, noise_std)`
- `incident_perturbation(x, adj, drop_ratio, duration, region_size)`

## Notes

- Tests use toy arrays only.
- No real traffic dataset was downloaded.
- Stage 2 passes the required minimum validation and is eligible to proceed to Stage 3 when requested.
- Regression check also passed: `pytest tests/test_masks.py tests/test_model_shapes.py tests/test_physics_residuals.py` collected 9 tests and passed 9.
