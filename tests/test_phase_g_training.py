from __future__ import annotations

from types import SimpleNamespace

import torch

from training.phase_g_training import (
    PhaseGEMA,
    backward_phase_g_trajectories,
    first_active_loop_token_ids,
    posterior_target_embeddings,
)


class TinyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding


def test_phase_g_targets_use_first_active_symbol_and_detach_keeper() -> None:
    labels = torch.tensor(
        [
            [
                [-100, -100, 3, 4],
                [-100, 7, 8, 9],
            ]
        ]
    )
    base = TinyBase()

    assert first_active_loop_token_ids(labels).tolist() == [[3, 7]]
    targets = posterior_target_embeddings(base, labels)
    assert targets.shape == (1, 2, 4)
    assert targets.requires_grad is False


def test_phase_g_targets_reject_unsupervised_requested_loop() -> None:
    labels = torch.full((1, 2, 3), -100)
    try:
        first_active_loop_token_ids(labels)
    except ValueError as exc:
        assert "needs a gold symbol" in str(exc)
    else:
        raise AssertionError("Missing posterior targets must fail")


def test_phase_g_ema_updates_copies_and_restores() -> None:
    module = torch.nn.Linear(2, 1)
    ema = PhaseGEMA(module.named_parameters(), decay=0.5)
    original = {name: value.detach().clone() for name, value in module.named_parameters()}
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(2.0)
    ema.update(module.named_parameters())
    trained = {name: value.detach().clone() for name, value in module.named_parameters()}

    backup = ema.copy_to(module.named_parameters())
    for name, parameter in module.named_parameters():
        assert torch.allclose(parameter, original[name] + 1.0)
    PhaseGEMA.restore(module.named_parameters(), backup)
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, trained[name])


class SeededTrajectoryObjective(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        *,
        num_trajectories: int,
        phase_g_trajectory_seeds: list[int],
        **_: object,
    ) -> SimpleNamespace:
        assert num_trajectories == len(phase_g_trajectory_seeds)
        seeds = torch.tensor(
            phase_g_trajectory_seeds,
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        per_trajectory = (self.weight * seeds - 2.0).square()
        loss = per_trajectory.mean()
        return SimpleNamespace(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "seed_mean": seeds.mean(),
                "per_loop_label_active": torch.tensor(
                    float(num_trajectories),
                    device=self.weight.device,
                ),
            },
        )


def test_trajectory_microbatching_preserves_loss_gradient_and_additive_metrics() -> None:
    seeds = [11, 17, 23, 29]
    vectorized = SeededTrajectoryObjective()
    microbatched = SeededTrajectoryObjective()
    microbatched.load_state_dict(vectorized.state_dict())

    vectorized_result = backward_phase_g_trajectories(
        vectorized,
        forward_kwargs={},
        trajectory_seeds=seeds,
        microbatch_size=4,
    )
    microbatched_result = backward_phase_g_trajectories(
        microbatched,
        forward_kwargs={},
        trajectory_seeds=seeds,
        microbatch_size=1,
    )

    assert vectorized_result.loss == microbatched_result.loss
    assert torch.equal(vectorized.weight.grad, microbatched.weight.grad)
    assert vectorized_result.metrics["seed_mean"] == microbatched_result.metrics["seed_mean"]
    assert vectorized_result.metrics["per_loop_label_active"] == 4.0
    assert microbatched_result.metrics["per_loop_label_active"] == 4.0
    assert microbatched_result.microbatch_count == 4
