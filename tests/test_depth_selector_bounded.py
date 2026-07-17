from __future__ import annotations

import torch
import pytest

from models.halting import SequenceHaltingPredictor
from training.depth_selector_bounded import (
    assert_frozen_gradients_zero,
    configure_selector_only,
    evaluate_s1_gate,
    evaluate_s2_gate,
    frozen_parameter_hash,
    halting_weights_from_features,
    spearman_correlation,
    truncated_geometric_prior,
)


class TinySelectorWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.halt_predictor = SequenceHaltingPredictor(4, max_loop_embeddings=12)


def test_configure_selector_only_excludes_oracle_target_controls() -> None:
    wrapper = TinySelectorWrapper()
    trainable = configure_selector_only(wrapper)

    assert trainable == {
        "halt_predictor.loop_bias",
        "halt_predictor.loop_embedding.weight",
        "halt_predictor.proj.bias",
        "halt_predictor.proj.weight",
    }
    assert not wrapper.backbone.weight.requires_grad
    assert not wrapper.halt_predictor.target_loop_router.weight.requires_grad


def test_frozen_hash_ignores_selector_but_detects_backbone_change() -> None:
    wrapper = TinySelectorWrapper()
    configure_selector_only(wrapper)
    start = frozen_parameter_hash(wrapper)
    with torch.no_grad():
        wrapper.halt_predictor.proj.bias.add_(1.0)
    assert frozen_parameter_hash(wrapper) == start
    with torch.no_grad():
        wrapper.backbone.bias.add_(1.0)
    assert frozen_parameter_hash(wrapper) != start


def test_frozen_gradient_assertion_rejects_nonzero_frozen_gradient() -> None:
    wrapper = TinySelectorWrapper()
    configure_selector_only(wrapper)
    features = torch.randn(2, 3, 4)
    loss = -halting_weights_from_features(wrapper.halt_predictor, features)[:, 1].log().mean()
    loss.backward()
    assert_frozen_gradients_zero(wrapper)

    wrapper.backbone.weight.grad = torch.ones_like(wrapper.backbone.weight)
    try:
        assert_frozen_gradients_zero(wrapper)
    except RuntimeError as exc:
        assert "backbone.weight" in str(exc)
    else:
        raise AssertionError("Expected a nonzero frozen gradient to fail")


def test_truncated_geometric_prior_has_locked_mean() -> None:
    prior = truncated_geometric_prior(max_loops=12, target_mean=6.0)
    loops = torch.arange(1, 13, dtype=prior.dtype)
    assert torch.isclose(prior.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose((prior * loops).sum(), torch.tensor(6.0), atol=1e-5)


def test_spearman_handles_ties() -> None:
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_correlation([1, 1, 2, 2], [3, 3, 4, 4]) == pytest.approx(1.0)
    assert spearman_correlation([1, 1, 1], [2, 3, 4]) == 0.0


def _selector_rows(*, selected_offset: int = 0, miss_every: int = 0) -> list[dict]:
    rows = []
    for depth in range(1, 13):
        for index in range(64):
            selected = min(12, max(1, depth + selected_offset))
            selected_hit = not miss_every or index % miss_every != 0
            rows.append(
                {
                    "depth": depth,
                    "selected_loop": selected,
                    "forced_hit": True,
                    "selected_hit": selected_hit,
                }
            )
    return rows


def test_s1_gate_requires_every_depth_and_answer_nonregression() -> None:
    passed = evaluate_s1_gate(_selector_rows())
    assert passed["status"] == "pass"
    assert passed["all_depth_selection_gates_pass"]
    assert passed["all_answer_delta_gates_pass"]

    blocked = evaluate_s1_gate(_selector_rows(selected_offset=1))
    assert blocked["status"] == "blocked"
    assert not blocked["all_depth_selection_gates_pass"]


def test_s2_gate_bands_strong_partial_and_collapse() -> None:
    s1 = evaluate_s1_gate(_selector_rows())
    stable_trace = [
        {"step": step, "loss": 2.0 - min(step, 1000) / 1000.0, "kl": 0.5}
        for step in range(1, 2001)
    ]
    strong = evaluate_s2_gate(_selector_rows(), training_trace=stable_trace, s1_gate=s1)
    assert strong["status"] == "strong"
    assert strong["spearman_selected_vs_true"] == 1.0

    partial_rows = _selector_rows()
    for index, row in enumerate(partial_rows):
        row["selected_loop"] = 1 + ((int(row["depth"]) + index % 4) % 12)
    partial = evaluate_s2_gate(partial_rows, training_trace=stable_trace, s1_gate=s1)
    assert partial["status"] in {"partial", "collapse"}

    collapsed = evaluate_s2_gate(
        _selector_rows(selected_offset=1, miss_every=2),
        training_trace=stable_trace,
        s1_gate=s1,
    )
    assert collapsed["status"] == "collapse"
