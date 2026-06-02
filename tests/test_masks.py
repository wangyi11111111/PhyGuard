import numpy as np

from data.corruptions import add_gaussian_noise, incident_perturbation
from data.masks import block_missing_mask, random_missing_mask, sensor_failure_mask, temporal_missing_mask


def test_random_missing_mask_shape_and_ratio():
    shape = (4, 12, 10, 3)
    mask = random_missing_mask(shape, 0.5, seed=1)
    assert mask.shape == shape
    missing_ratio = 1.0 - mask.mean()
    assert 0.4 <= missing_ratio <= 0.6


def test_sensor_failure_mask_shape_and_ratio():
    shape = (2, 8, 10, 3)
    mask = sensor_failure_mask(shape, 0.3, seed=2)
    assert mask.shape == shape
    missing_ratio = 1.0 - mask.mean()
    assert 0.25 <= missing_ratio <= 0.35
    node_keep = mask.mean(axis=(0, 1, 3))
    assert (node_keep == 0.0).sum() == 3
    assert (node_keep == 1.0).sum() == 7


def test_block_missing_mask_shape_and_ratio():
    shape = (2, 6, 8, 3)
    adj = np.eye(8, dtype=np.float32)
    adj[3, 2] = 0.9
    adj[3, 4] = 0.8
    mask = block_missing_mask(shape, adj, block_size=3, seed=5)
    assert mask.shape == shape
    missing_ratio = 1.0 - mask.mean()
    assert 0.34 <= missing_ratio <= 0.40
    node_keep = mask.mean(axis=(0, 1, 3))
    assert (node_keep == 0.0).sum() == 3


def test_temporal_missing_mask_shape_ratio_and_contiguous_block():
    shape = (2, 20, 5, 3)
    mask = temporal_missing_mask(shape, 0.3, duration=3, seed=3)
    assert mask.shape == shape
    missing_ratio = 1.0 - mask.mean()
    assert 0.25 <= missing_ratio <= 0.4
    zeros_per_time = (mask == 0.0).sum(axis=(0, 2, 3))
    missing_steps = zeros_per_time > 0
    assert missing_steps.sum() >= 6
    contiguous_run_exists = any(missing_steps[i : i + 3].all() for i in range(len(missing_steps) - 2))
    assert contiguous_run_exists


def test_add_gaussian_noise_changes_values_but_not_shape():
    x = np.ones((2, 10, 6, 3), dtype=np.float32)
    y = add_gaussian_noise(x, noise_std=0.1, seed=6)
    assert y.shape == x.shape
    assert not np.allclose(x, y)
    assert abs(float((y - x).mean())) < 0.05


def test_incident_perturbation_locality():
    shape = (2, 12, 8, 3)
    x = np.ones(shape, dtype=np.float32)
    adj = np.eye(shape[2], dtype=np.float32)
    for i in range(shape[2] - 1):
        adj[i, i + 1] = 0.8
        adj[i + 1, i] = 0.8

    y = incident_perturbation(x, adj, drop_ratio=0.5, duration=3, region_size=2, seed=4)
    assert y.shape == x.shape

    changed = np.abs(y - x) > 1e-6
    changed_by_time_node = changed.any(axis=(0, 3))
    changed_times = np.flatnonzero(changed_by_time_node.any(axis=1))
    changed_nodes = np.flatnonzero(changed_by_time_node.any(axis=0))

    assert len(changed_times) == 3
    assert np.all(np.diff(changed_times) == 1)
    assert len(changed_nodes) == 2
    assert changed[..., 0].any()
    assert changed[..., 2].any()
    assert not changed[..., 1].any()


def test_incident_perturbation_can_return_region_mask():
    shape = (2, 12, 8, 3)
    x = np.ones(shape, dtype=np.float32)
    adj = np.eye(shape[2], dtype=np.float32)
    y, incident_mask = incident_perturbation(
        x,
        adj,
        drop_ratio=0.5,
        duration=3,
        region_size=2,
        seed=4,
        return_mask=True,
    )
    assert y.shape == x.shape
    assert incident_mask.shape == x.shape
    assert incident_mask.sum() == 2 * 3 * 2 * 3
