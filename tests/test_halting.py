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


def test_target_loop_control_is_identity_at_initialization():
    predictor = SequenceHaltingPredictor(hidden_size=4, initial_halt_prob=0.25)
    pooled = torch.randn(3, 4)
    target_loops = torch.tensor([1, 2, 4])

    unconditioned = predictor(pooled, loop_idx=1)
    controlled = predictor(pooled, loop_idx=1, target_loop_counts=target_loops)

    assert torch.allclose(unconditioned, controlled)


def test_target_loop_control_bias_can_shift_specific_target_and_loop():
    predictor = SequenceHaltingPredictor(hidden_size=4, initial_halt_prob=0.25)
    pooled = torch.randn(3, 4)
    target_loops = torch.tensor([1, 3, 3])

    with torch.no_grad():
        predictor.target_loop_bias[2, 1] = 1.0

    baseline = predictor(pooled, loop_idx=1)
    shifted = predictor(pooled, loop_idx=1, target_loop_counts=target_loops).squeeze(-1)

    assert torch.allclose(shifted[:1], baseline.squeeze(-1)[:1])
    assert torch.all(shifted[1:] > baseline.squeeze(-1)[1:])


def test_soft_target_loop_control_bias_matches_weighted_bias():
    predictor = SequenceHaltingPredictor(hidden_size=4, initial_halt_prob=0.25)
    pooled = torch.randn(2, 4)
    probs = torch.tensor(
        [
            [1.0] + [0.0] * 15,
            [0.0, 0.25, 0.75] + [0.0] * 13,
        ]
    )

    with torch.no_grad():
        predictor.target_loop_bias[0, 1] = 1.0
        predictor.target_loop_bias[1, 1] = 2.0
        predictor.target_loop_bias[2, 1] = 4.0

    shifted = predictor(pooled, loop_idx=1, target_loop_probs=probs).squeeze(-1)
    baseline_logit = torch.logit(predictor(pooled, loop_idx=1).squeeze(-1))
    shifted_logit = torch.logit(shifted)

    expected_bias = torch.tensor([1.0, 3.5])
    assert torch.allclose(shifted_logit - baseline_logit, expected_bias, atol=1e-5)


def test_target_loop_router_logits_shape():
    predictor = SequenceHaltingPredictor(hidden_size=4, initial_halt_prob=0.25)
    pooled = torch.randn(3, 4)

    logits = predictor.target_loop_logits(pooled)

    assert logits.shape == (3, 16)
