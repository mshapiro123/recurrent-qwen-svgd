import torch

from models.latent_policy import LatentTrajectoryModule


def test_latent_trajectory_shapes_and_small_initial_delta():
    module = LatentTrajectoryModule(hidden_dim=16, latent_dim=4, latent_scale_init=0.01)
    hidden = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7, dtype=torch.long)
    out, stats = module(hidden, mask, sample=True)
    assert out.shape == hidden.shape
    assert stats.mu.shape == (3, 4)
    assert stats.logvar.shape == (3, 4)
    assert stats.kl.shape == (3,)
