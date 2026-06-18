import torch

from models.halting import centered_geometric_prior, expected_loop_count, pondernet_halting_probabilities


def test_pondernet_final_loop_absorbs_remaining_probability():
    halt_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    weights = pondernet_halting_probabilities(halt_probs)
    expected = torch.tensor([[0.25, 0.1875, 0.140625, 0.421875]])
    assert torch.allclose(weights, expected)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))


def test_expected_loop_count_shape():
    weights = torch.tensor([[[0.2, 0.3, 0.5]]])
    assert expected_loop_count(weights).shape == (1, 1)


def test_centered_geometric_prior_normalizes():
    prior = centered_geometric_prior(torch.tensor([2, 4]), max_loops=5)
    assert prior.shape == (2, 5)
    assert torch.allclose(prior.sum(dim=-1), torch.ones(2))
    assert prior[0].argmax().item() == 1
    assert prior[1].argmax().item() == 3
