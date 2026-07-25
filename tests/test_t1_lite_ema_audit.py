from __future__ import annotations

import torch

from eval.eval_t1_lite_ema_audit import (
    blend_states,
    fixed_screen_rows,
    parameter_group,
    scalar_ema_integrity,
    state_geometry,
    swap_group,
    validate_state_pair,
)


def states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw = {
        "base_model.model.embed_tokens.control_rows": torch.tensor([[1.0, 2.0]]),
        "bridge.bridge_gate": torch.tensor([1.0]),
        "base_model.model.layers.6.weight": torch.tensor([2.0, 0.0]),
    }
    ema = {
        "base_model.model.embed_tokens.control_rows": torch.tensor([[3.0, 4.0]]),
        "bridge.bridge_gate": torch.tensor([0.5]),
        "base_model.model.layers.6.weight": torch.tensor([0.0, 2.0]),
    }
    return raw, ema


def test_parameter_group_partition_is_exhaustive_for_t1_trainables() -> None:
    raw, _ = states()

    assert {parameter_group(name) for name in raw} == {
        "control_rows",
        "bridge",
        "recurrent_block",
    }


def test_state_pair_validation_rejects_unknown_and_nonfinite_tensors() -> None:
    raw, ema = states()
    assert validate_state_pair(raw, ema)["passed"] is True

    bad = dict(ema)
    bad["unexpected.weight"] = torch.tensor([float("nan")])
    receipt = validate_state_pair(raw, bad)
    assert receipt["passed"] is False
    assert receipt["ema_only"] == ["unexpected.weight"]


def test_blend_endpoints_and_midpoint_are_exact() -> None:
    raw, ema = states()

    left = blend_states(raw, ema, 0.0)
    middle = blend_states(raw, ema, 0.5)
    right = blend_states(raw, ema, 1.0)

    for name in raw:
        assert torch.equal(left[name], raw[name])
        assert torch.equal(right[name], ema[name])
        assert torch.equal(middle[name], (raw[name] + ema[name]) / 2.0)


def test_group_swap_changes_exactly_one_family() -> None:
    raw, ema = states()
    swapped = swap_group(raw, ema, "bridge")

    assert torch.equal(swapped["bridge.bridge_gate"], ema["bridge.bridge_gate"])
    assert torch.equal(
        swapped["base_model.model.layers.6.weight"],
        raw["base_model.model.layers.6.weight"],
    )
    assert torch.equal(
        swapped["base_model.model.embed_tokens.control_rows"],
        raw["base_model.model.embed_tokens.control_rows"],
    )


def test_geometry_reports_orthogonal_recurrent_block() -> None:
    raw, ema = states()
    geometry = state_geometry(raw, ema)

    assert geometry["recurrent_block"]["parameters"] == 2
    assert geometry["recurrent_block"]["cosine"] == 0.0
    assert geometry["bridge"]["difference_norm"] == 0.5


def test_fixed_screen_is_seeded_balanced_and_reproducible() -> None:
    rows = [
        {"id": f"d{depth}_{index}", "depth": depth}
        for depth in range(1, 4)
        for index in range(12)
    ]

    left = fixed_screen_rows(rows, seed=17, per_depth=4)
    right = fixed_screen_rows(rows, seed=17, per_depth=4)

    assert left == right
    assert len(left) == 12
    assert {depth: sum(row["depth"] == depth for row in left) for depth in range(1, 4)} == {
        1: 4,
        2: 4,
        3: 4,
    }


def test_device_ema_matches_exact_scalar_recurrence() -> None:
    receipt = scalar_ema_integrity()

    assert receipt["passed"] is True
    assert receipt["absolute_error"] <= 1e-7
