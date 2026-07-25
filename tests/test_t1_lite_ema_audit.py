from __future__ import annotations

import torch

from eval.eval_t1_lite_ema_audit import (
    blend_states,
    fixed_screen_rows,
    parameter_group,
    scalar_ema_integrity,
    stage_checkpoint_coverage,
    recurrent_layer_group,
    swap_recurrent_layer_group,
    state_geometry,
    swap_group,
    validate_state_pair,
)
from colab.run_stage5_t1_lite_ema_audit import checkpoint_file_usable


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


def test_stage_checkpoint_coverage_reports_partial_without_inference(tmp_path) -> None:
    (tmp_path / "t1_progress_step_500.pt").write_bytes(b"p" * 1024)
    (tmp_path / "t1_progress_step_2500.pt").write_bytes(b"")

    receipt = stage_checkpoint_coverage(tmp_path)

    assert receipt["required"] == 4
    assert receipt["available"] == 1
    assert receipt["available_names"] == ["t1_progress_step_500.pt"]
    assert receipt["complete"] is False
    assert "t1_progress_step_2500.pt" in receipt["missing_names"]
    assert "t1_progress_step_8500.pt" in receipt["missing_names"]


def test_checkpoint_file_usable_rejects_zero_byte_and_truncated_files(tmp_path) -> None:
    empty = tmp_path / "empty.pt"
    truncated = tmp_path / "truncated.pt"
    valid_sized = tmp_path / "valid-sized.pt"
    empty.write_bytes(b"")
    truncated.write_bytes(b"x" * 100)
    valid_sized.write_bytes(b"x" * 1024)

    assert checkpoint_file_usable(empty) is False
    assert checkpoint_file_usable(truncated) is False
    assert checkpoint_file_usable(valid_sized) is True


def test_recurrent_layer_groups_partition_all_twelve_looped_layers() -> None:
    assert recurrent_layer_group("base_model.model.layers.6.self_attn.q_proj.weight") == "early_6_9"
    assert recurrent_layer_group("base_model.model.layers.11.mlp.down_proj.weight") == "middle_10_13"
    assert recurrent_layer_group("base_model.model.layers.17.self_attn.o_proj.weight") == "late_14_17"
    assert recurrent_layer_group("bridge.bridge_gate") is None


def test_layer_group_swap_changes_only_requested_recurrent_layers() -> None:
    raw = {
        "base_model.model.layers.6.weight": torch.tensor([1.0]),
        "base_model.model.layers.10.weight": torch.tensor([2.0]),
        "base_model.model.layers.14.weight": torch.tensor([3.0]),
        "bridge.weight": torch.tensor([4.0]),
    }
    ema = {name: value + 10 for name, value in raw.items()}
    swapped = swap_recurrent_layer_group(raw, ema, "middle_10_13")
    assert torch.equal(swapped["base_model.model.layers.6.weight"], raw["base_model.model.layers.6.weight"])
    assert torch.equal(swapped["base_model.model.layers.10.weight"], ema["base_model.model.layers.10.weight"])
    assert torch.equal(swapped["base_model.model.layers.14.weight"], raw["base_model.model.layers.14.weight"])
    assert torch.equal(swapped["bridge.weight"], raw["bridge.weight"])
