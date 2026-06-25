import torch

from eval.eval_reentry_drift import bridge_gradient_liveness
from models.bridge import IdentityGatedBridge
from models.reentry_adapter import ReentryAffineAdapter, SpectralLowRankCorrection
from training.reentry_repair import apply_reentry_repair_controls


def test_identity_gated_bridge_preserves_hidden_state_at_init():
    bridge = IdentityGatedBridge(hidden_size=8)
    hidden = torch.randn(2, 5, 8)
    out = bridge(hidden)
    assert torch.equal(out, hidden)


def test_identity_gated_bridge_does_not_double_hidden_state():
    bridge = IdentityGatedBridge(hidden_size=4)
    hidden = torch.randn(1, 3, 4)
    out = bridge(hidden)
    assert torch.allclose(out, hidden)
    assert not torch.allclose(out, hidden + hidden)


def test_identity_gated_bridge_default_is_gradient_live():
    bridge = IdentityGatedBridge(hidden_size=4)
    sample = torch.randn(2, 3, 4)

    out = bridge_gradient_liveness(bridge, sample)

    assert float(bridge.bridge_gate.detach()) == 1.0
    assert out["gate_grad_abs"] == 0.0
    assert out["weight_grad_rms"] > 0.0
    assert out["bias_grad_rms"] > 0.0


def test_reentry_repair_controls_can_revive_dead_identity_bridge():
    bridge = IdentityGatedBridge(hidden_size=4, gate_init=0.0)
    wrapper = type("Wrapper", (), {"bridge": bridge})()
    sample = torch.randn(2, 3, 4)

    before = bridge_gradient_liveness(wrapper.bridge, sample)
    info = apply_reentry_repair_controls(
        wrapper,
        {"bridge_reset_identity": True, "bridge_gate_override": 1.0},
    )
    after = bridge_gradient_liveness(wrapper.bridge, sample)

    assert before["weight_grad_rms"] == 0.0
    assert info["applied"] is True
    assert info["bridge_gate_before"] == 0.0
    assert info["bridge_gate_after"] == 1.0
    assert info["bridge_identity_max_abs_diff_after"] == 0.0
    assert after["weight_grad_rms"] > 0.0
    assert after["bias_grad_rms"] > 0.0


def test_reentry_affine_adapter_preserves_hidden_state_at_init():
    adapter = ReentryAffineAdapter(hidden_size=8)
    hidden = torch.randn(2, 5, 8)
    out = adapter(hidden)

    assert torch.equal(out, hidden)


def test_reentry_affine_adapter_has_live_identity_gradients():
    adapter = ReentryAffineAdapter(hidden_size=4)
    hidden = torch.randn(2, 3, 4)

    output = adapter(hidden)
    loss = output.float().square().mean()
    loss.backward()

    assert adapter.scale.grad is not None
    assert adapter.bias.grad is not None
    assert adapter.scale.grad.float().square().mean().sqrt() > 0.0
    assert adapter.bias.grad.float().square().mean().sqrt() > 0.0


def test_spectral_low_rank_correction_matches_dense_spectral_norm():
    torch.manual_seed(0)
    correction = SpectralLowRankCorrection(hidden_size=6, rank=3, max_depth=4)
    with torch.no_grad():
        correction.U.normal_(mean=0.0, std=0.2)
        correction.V.normal_(mean=0.0, std=0.2)

    dense = correction.U @ correction.V.t()
    exact = torch.linalg.svdvals(dense.float()).max()
    estimated = correction.spectral_norm(num_iters=80, update=False)

    assert torch.allclose(estimated.float(), exact, rtol=1e-3, atol=1e-4)


def test_reentry_spectral_mode_is_near_identity_at_init_and_gradient_live():
    torch.manual_seed(0)
    adapter = ReentryAffineAdapter(hidden_size=5)
    hidden = torch.randn(2, 3, 5)

    out = adapter(hidden, loop_idx=2, mode="spectral")
    assert (out - hidden).abs().max() < 1e-2

    loss = out.float().square().mean()
    loss.backward()

    assert adapter.spectral_correction.U.grad is not None
    assert adapter.spectral_correction.V.grad is not None
    assert adapter.spectral_correction.theta.grad is not None
    assert adapter.spectral_correction.U.grad.float().square().mean().sqrt() > 0.0
    assert adapter.spectral_correction.V.grad.float().square().mean().sqrt() > 0.0
    assert adapter.spectral_correction.theta.grad.float().abs().max() > 0.0


def test_reentry_adapter_rejects_unknown_mode():
    adapter = ReentryAffineAdapter(hidden_size=4)
    hidden = torch.randn(1, 2, 4)

    try:
        adapter(hidden, mode="sideways")
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("Expected invalid re-entry adapter mode to raise")
