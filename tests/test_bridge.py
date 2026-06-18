import torch

from models.bridge import IdentityGatedBridge


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
