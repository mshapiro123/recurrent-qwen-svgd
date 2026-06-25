import torch

from eval.eval_reentry_drift import bridge_gradient_liveness
from models.bridge import IdentityGatedBridge
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
