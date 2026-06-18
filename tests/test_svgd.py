import torch

from models.svgd import svgd_particle_update


def _mean_pairwise_distance(flat_state, num_particles):
    pooled = flat_state.view(1, num_particles, 1, -1).squeeze(2)
    diff = pooled[:, :, None, :] - pooled[:, None, :, :]
    dist = diff.pow(2).sum(-1).sqrt()
    mask = ~torch.eye(num_particles, dtype=torch.bool)
    return dist[:, mask].mean()


def test_svgd_k1_recovers_standard_update_exactly():
    previous = torch.randn(2, 3, 4)
    standard = torch.randn(2, 3, 4)
    updated, stats = svgd_particle_update(
        previous,
        standard,
        attention_mask=None,
        num_particles=1,
        eps=1.0,
    )
    assert torch.equal(updated, standard)
    assert stats.repulsion_rms.item() == 0.0


def test_svgd_coincident_particles_do_not_separate_without_symmetry_breaking():
    previous = torch.ones(4, 1, 2)
    standard = previous.clone()
    updated, stats = svgd_particle_update(
        previous,
        standard,
        attention_mask=None,
        num_particles=4,
        eps=0.2,
    )
    assert torch.allclose(updated, previous)
    assert stats.repulsion_rms.item() == 0.0


def test_svgd_repulsion_increases_spread_without_drift():
    torch.manual_seed(0)
    state = 0.02 * torch.randn(4, 1, 2)
    initial_spread = _mean_pairwise_distance(state, 4)
    for _ in range(20):
        state, _ = svgd_particle_update(
            state,
            state,
            attention_mask=None,
            num_particles=4,
            eps=0.1,
        )
    final_spread = _mean_pairwise_distance(state, 4)
    assert final_spread > initial_spread * 2.0


def test_svgd_drift_moves_mean_toward_target_without_collapse():
    torch.manual_seed(1)
    state = 0.05 * torch.randn(4, 1, 2)
    target = torch.tensor([[[1.0, -0.5]]]).expand_as(state)
    initial_mean_distance = (state.mean(dim=0) - target[:1].mean(dim=0)).norm()
    for _ in range(30):
        standard = state + 0.15 * (target - state)
        state, _ = svgd_particle_update(
            state,
            standard,
            attention_mask=None,
            num_particles=4,
            eps=0.1,
            repulsion_scale=0.5,
        )
    final_mean_distance = (state.mean(dim=0) - target[:1].mean(dim=0)).norm()
    final_spread = _mean_pairwise_distance(state, 4)
    assert final_mean_distance < initial_mean_distance
    assert final_spread > 0.01
