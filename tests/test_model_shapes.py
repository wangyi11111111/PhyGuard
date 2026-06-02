import torch

from models.base_model import BaseTCNGraph
from models.grin_baseline import GRINLite
from models.litetrust_pinn import LiteTrustGRIN, LiteTrustGRINCorrection, LiteTrustGRINReliabilityRouter, LiteTrustGRINRiskRouter, LiteTrustPINN


def test_base_model_output_shape():
    model = BaseTCNGraph(input_dim=6, hidden_dim=16, output_dim=3, num_layers=2, dropout=0.1)
    x_obs = torch.randn(8, 24, 20, 3)
    mask = torch.ones(8, 24, 20, 3)
    adj = torch.eye(20)
    pred = model(x_obs, mask, adj)
    assert pred.shape == (8, 24, 20, 3)


def test_litetrust_model_output_and_trust_shape():
    model = LiteTrustPINN(input_dim=6, hidden_dim=16, output_dim=3, num_layers=2, dropout=0.1)
    x_obs = torch.randn(8, 24, 20, 3)
    mask = torch.ones(8, 24, 20, 3)
    adj = torch.eye(20)
    residual_abs = torch.rand(8, 24, 20, 1)
    output = model(x_obs, mask, adj, residual_abs=residual_abs)
    assert output["mu"].shape == (8, 24, 20, 3)
    assert output["h"].shape == (8, 24, 20, 16)
    assert output["trust"].shape == (8, 24, 20, 1)
    assert torch.all(output["trust"] >= 0.0)
    assert torch.all(output["trust"] <= 1.0)


def test_litetrust_uncertainty_shape_and_clamp():
    model = LiteTrustPINN(input_dim=6, hidden_dim=16, output_dim=3, num_layers=2, dropout=0.1, use_uncertainty=True)
    x_obs = torch.randn(8, 24, 20, 3)
    mask = torch.ones(8, 24, 20, 3)
    adj = torch.eye(20)
    output = model(x_obs, mask, adj)
    assert output["mu"].shape == (8, 24, 20, 3)
    assert output["log_var"].shape == (8, 24, 20, 3)
    assert torch.all(output["log_var"] >= -6.0)
    assert torch.all(output["log_var"] <= 3.0)


def test_litetrust_trust_accepts_extra_gate_feature():
    model = LiteTrustPINN(input_dim=6, hidden_dim=16, output_dim=3, num_layers=2, dropout=0.1)
    x_obs = torch.randn(8, 24, 20, 3)
    mask = torch.ones(8, 24, 20, 3)
    adj = torch.eye(20)
    residual_abs = torch.rand(8, 24, 20, 1)
    extra_feature = torch.rand(8, 24, 20, 1)
    output = model(x_obs, mask, adj, residual_abs=residual_abs, extra_feature=extra_feature)
    assert output["trust"].shape == (8, 24, 20, 1)
    assert torch.all(output["trust"] >= 0.0)
    assert torch.all(output["trust"] <= 1.0)


def test_litetrust_trust_accepts_multi_extra_gate_features():
    model = LiteTrustPINN(input_dim=6, hidden_dim=16, output_dim=3, num_layers=2, dropout=0.1, extra_feature_dim=4)
    x_obs = torch.randn(8, 24, 20, 3)
    mask = torch.ones(8, 24, 20, 3)
    adj = torch.eye(20)
    residual_abs = torch.rand(8, 24, 20, 1)
    extra_feature = torch.rand(8, 24, 20, 4)
    output = model(x_obs, mask, adj, residual_abs=residual_abs, extra_feature=extra_feature)
    assert output["trust"].shape == (8, 24, 20, 1)
    assert torch.all(output["trust"] >= 0.0)
    assert torch.all(output["trust"] <= 1.0)


def test_grin_lite_output_shape():
    model = GRINLite(input_dim=3, hidden_dim=16, output_dim=3, dropout=0.1)
    x_obs = torch.randn(4, 12, 10, 3)
    mask = torch.ones(4, 12, 10, 3)
    adj = torch.eye(10)
    pred = model(x_obs, mask, adj)
    assert pred.shape == (4, 12, 10, 3)


def test_litetrust_grin_output_shapes():
    model = LiteTrustGRIN(input_dim=3, hidden_dim=16, output_dim=3, dropout=0.1, extra_feature_dim=4)
    x_obs = torch.randn(4, 12, 10, 3)
    mask = torch.ones(4, 12, 10, 3)
    adj = torch.eye(10)
    output = model(x_obs, mask, adj)
    assert output["mu"].shape == (4, 12, 10, 3)
    assert output["h"].shape == (4, 12, 10, 16)
    assert output["log_var"].shape == (4, 12, 10, 3)
    trust = model.trust_from_residual(
        output["h"],
        torch.rand(4, 12, 10, 1),
        mask,
        log_var=output["log_var"].mean(dim=-1, keepdim=True),
        extra_feature=torch.rand(4, 12, 10, 4),
    )
    assert trust.shape == (4, 12, 10, 1)


def test_litetrust_grin_correction_output_shapes():
    model = LiteTrustGRINCorrection(input_dim=3, hidden_dim=16, output_dim=3, dropout=0.1, extra_feature_dim=4)
    x_obs = torch.randn(4, 12, 10, 3)
    mask = torch.ones(4, 12, 10, 3)
    adj = torch.eye(10)
    base = model(x_obs, mask, adj)
    assert base["mu"].shape == (4, 12, 10, 3)
    assert base["mu_data"].shape == (4, 12, 10, 3)
    out = model(
        x_obs,
        mask,
        adj,
        residual_abs=torch.rand(4, 12, 10, 1),
        log_var=base["log_var"].mean(dim=-1, keepdim=True),
        extra_feature=torch.rand(4, 12, 10, 4),
    )
    assert out["mu"].shape == (4, 12, 10, 3)
    assert out["delta_phys"].shape == (4, 12, 10, 3)
    assert out["graph_delta"].shape == (4, 12, 10, 3)
    assert out["expert_weights"].shape == (4, 12, 10, 3)
    assert out["data_weight"].shape == (4, 12, 10, 1)
    assert out["graph_weight"].shape == (4, 12, 10, 1)
    assert out["phys_weight"].shape == (4, 12, 10, 1)
    assert torch.allclose(out["expert_weights"].sum(dim=-1), torch.ones(4, 12, 10), atol=1e-6)
    assert out["trust"].shape == (4, 12, 10, 1)


def test_litetrust_grin_risk_router_output_shapes():
    model = LiteTrustGRINRiskRouter(input_dim=3, hidden_dim=16, output_dim=3, dropout=0.1, extra_feature_dim=6)
    x_obs = torch.randn(4, 12, 10, 3)
    mask = torch.ones(4, 12, 10, 3)
    adj = torch.eye(10)
    base = model(x_obs, mask, adj)
    out = model(
        x_obs,
        mask,
        adj,
        residual_abs=torch.rand(4, 12, 10, 1),
        log_var=base["log_var"].mean(dim=-1, keepdim=True),
        extra_feature=torch.rand(4, 12, 10, 6),
    )
    assert out["mu"].shape == (4, 12, 10, 3)
    assert out["x_graph"].shape == (4, 12, 10, 3)
    assert out["x_phys"].shape == (4, 12, 10, 3)
    assert out["risk_pred"].shape == (4, 12, 10, 3)
    assert out["expert_weights"].shape == (4, 12, 10, 3)
    assert torch.allclose(out["expert_weights"].sum(dim=-1), torch.ones(4, 12, 10), atol=1e-6)


def test_litetrust_grin_reliability_router_output_shapes():
    model = LiteTrustGRINReliabilityRouter(
        input_dim=3,
        hidden_dim=16,
        output_dim=3,
        dropout=0.1,
        extra_feature_dim=6,
        use_spatial_physics=True,
        use_directional_physics=True,
    )
    x_obs = torch.randn(4, 12, 10, 3)
    mask = torch.ones(4, 12, 10, 3)
    adj = torch.eye(10)
    base = model(x_obs, mask, adj)
    out = model(
        x_obs,
        mask,
        adj,
        residual_abs=torch.rand(4, 12, 10, 1),
        log_var=base["log_var"].mean(dim=-1, keepdim=True),
        extra_feature=torch.rand(4, 12, 10, 6),
    )
    assert out["mu"].shape == (4, 12, 10, 3)
    assert out["reliability_scores"].shape == (4, 12, 10, 3)
    assert out["projection_gamma"].shape == (4, 12, 10, 1)
    assert out["physics_validity"].shape == (4, 12, 10, 1)
    assert out["spatial_phys_delta"].shape == (4, 12, 10, 3)
    assert out["spatial_phys_gate"].shape == (4, 12, 10, 1)
    assert out["directional_phys_delta"].shape == (4, 12, 10, 3)
    assert out["directional_phys_gate"].shape == (4, 12, 10, 1)
    assert out["directional_shift"].shape == (4, 12, 10, 1)
    assert out["directional_conservation_residual"].shape == (4, 12, 10, 1)
    assert out["expert_weights"].shape == (4, 12, 10, 3)
    assert torch.allclose(out["expert_weights"].sum(dim=-1), torch.ones(4, 12, 10), atol=1e-6)
