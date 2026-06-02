import torch

from physics.traffic_residuals import fundamental_residual, fundamental_residual_from_prediction, graph_speed_residual
from data.normalization import StandardScaler


def test_fundamental_residual_shape_and_finite():
    flow = torch.randn(4, 12, 5, 1)
    speed = torch.randn(4, 12, 5, 1)
    occ = torch.randn(4, 12, 5, 1)
    residual = fundamental_residual(flow, speed, occ)
    assert residual.shape == flow.shape
    assert torch.isfinite(residual).all()


def test_fundamental_residual_supports_inverse_normalizer():
    scaler = StandardScaler(
        mean=torch.tensor([[[[40.0, 0.2, 30.0]]]]).numpy(),
        std=torch.tensor([[[[10.0, 0.1, 5.0]]]]).numpy(),
    )
    flow = torch.zeros(2, 4, 3, 1)
    speed = torch.zeros(2, 4, 3, 1)
    occ = torch.zeros(2, 4, 3, 1)
    residual = fundamental_residual(flow, speed, occ, normalizer=scaler)
    assert residual.shape == flow.shape
    assert torch.isfinite(residual).all()


def test_fundamental_residual_from_prediction_uses_pems_channel_order():
    pred = torch.zeros(2, 4, 3, 3)
    pred[..., 0] = 20.0
    pred[..., 1] = 0.5
    pred[..., 2] = 40.0
    residual = fundamental_residual_from_prediction(pred, channel_order="flow_occupancy_speed")
    assert residual.shape == (2, 4, 3, 1)
    assert torch.allclose(residual, torch.zeros_like(residual))


def test_fundamental_residual_uses_calibrated_fd_scale():
    physical = torch.zeros(2, 4, 3, 3).numpy().astype("float32")
    physical[..., 1] = 0.5
    physical[..., 2] = 40.0
    physical[..., 0] = 4.0 * physical[..., 1] * physical[..., 2]
    scaler = StandardScaler.fit(physical)
    normalized = torch.tensor(scaler.transform(physical))
    residual = fundamental_residual_from_prediction(normalized, normalizer=scaler)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_graph_speed_residual_shape_and_finite():
    speed = torch.randn(4, 12, 5, 1)
    adj = torch.eye(5)
    residual = graph_speed_residual(speed, adj)
    assert residual.shape == (4, 11, 5, 1)
    assert torch.isfinite(residual).all()
