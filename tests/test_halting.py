import torch

from models.halting import (
    SequenceHaltingPredictor,
    centered_geometric_prior,
    expected_loop_count,
    pondernet_halting_probabilities,
)


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


def test_loop_conditioned_halting_is_identity_at_initialization():
    predictor = SequenceHaltingPredictor(hidden_size=4, initial_halt_prob=0.25)
    pooled = torch.randn(3, 4)

    unconditioned = predictor(pooled)
    conditioned = predictor(pooled, loop_idx=2)

    assert torch.allclose(unconditioned, conditioned)


def test_loop_conditioned_halting_can_shift_specific_loop():
    predictor = SequenceHaltingPredictor(hidden_size=4, initial_halt_prob=0.25)
    pooled = torch.randn(3, 4)

    with torch.no_grad():
        predictor.loop_bias[2] = 1.0

    baseline = predictor(pooled)
    shifted = predictor(pooled, loop_idx=2)

    assert torch.all(shifted > baseline)
