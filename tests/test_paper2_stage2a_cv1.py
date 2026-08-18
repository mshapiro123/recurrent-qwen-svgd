from __future__ import annotations

import torch
from torch import nn

from analysis.analyze_paper2_stage2a_cv1_d5 import compare_conditions, holm_adjust
from eval.eval_paper2_phase3_p34_task_inference import P34TaskInferenceGraph
from eval.eval_paper2_stage2a_cv1 import (
    apply_value_condition,
    fixed_map_digest,
    summarize_rows,
    value_bank,
)
from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_stage2a_runtime import Stage2AMemorySystem


def _memory(arm: str) -> Stage2AMemorySystem:
    generator = torch.Generator().manual_seed(11)
    return Stage2AMemorySystem(
        arm=arm,
        memory_slots=32,
        memory_keys=torch.randn((32, 128), generator=generator),
        teacher_values=torch.randn((32, 128), generator=generator),
        seed=0,
    )


def _row_multiset(values: torch.Tensor) -> list[bytes]:
    return sorted(row.contiguous().numpy().tobytes() for row in values.float().cpu())


def test_crossed_value_conditions_preserve_fixed_map_and_distribution() -> None:
    for arm in ("t3a", "t3b"):
        memory = _memory(arm)
        original = value_bank(memory)
        fixed = fixed_map_digest(memory)

        shuffled = _memory(arm)
        shuffled.load_state_dict(memory.state_dict())
        audit = apply_value_condition(
            shuffled, "shuffled", shuffle_seed=20260818, random_seed=20260819
        )
        assert fixed_map_digest(shuffled) == fixed
        assert _row_multiset(value_bank(shuffled)) == _row_multiset(original)
        assert audit["maximum_mean_error"] < 1e-6
        assert audit["maximum_std_error"] < 1e-6

        random_first = _memory(arm)
        random_first.load_state_dict(memory.state_dict())
        apply_value_condition(
            random_first, "random", shuffle_seed=20260818, random_seed=20260819
        )
        random_second = _memory(arm)
        random_second.load_state_dict(memory.state_dict())
        random_audit = apply_value_condition(
            random_second, "random", shuffle_seed=20260818, random_seed=20260819
        )
        torch.testing.assert_close(value_bank(random_first), value_bank(random_second))
        assert random_audit["maximum_mean_error"] < 1e-5
        assert random_audit["maximum_std_error"] < 1e-5
        assert fixed_map_digest(random_second) == fixed


def test_nonunit_stage2a_scale_requires_diagnostic_authorization() -> None:
    embedding = nn.Embedding(43, 16)
    sidecar = Phase3StudentModules(
        tied_embedding=embedding,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        control_dim=4,
        draft_rank=4,
        stage2a_memory_dim=None,
    )
    memory = _memory("t3a")
    try:
        P34TaskInferenceGraph(
            base_model=nn.Linear(1, 1),
            sidecar=sidecar,
            stage2a_memory_system=memory,
            stage2a_geometry={},
            stage2a_value_scale=0.5,
        )
    except ValueError as error:
        assert "score-only authorization" in str(error)
    else:
        raise AssertionError("nonunit value scale was accepted without authorization")

    graph = P34TaskInferenceGraph(
        base_model=nn.Linear(1, 1),
        sidecar=sidecar,
        stage2a_memory_system=memory,
        stage2a_geometry={},
        stage2a_value_scale=0.5,
        stage2a_diagnostic_value_scale_authorized=True,
    )
    assert graph.stage2a_value_scale == 0.5


def test_both_comparator_summary_and_cross_condition_pairing() -> None:
    rows = [
        {"item_id": "a", "base_correct": False, "augmented_correct": True},
        {"item_id": "b", "base_correct": True, "augmented_correct": False},
        {"item_id": "c", "base_correct": False, "augmented_correct": True},
    ]
    initialization = {
        "a": {"augmented_correct": True},
        "b": {"augmented_correct": True},
        "c": {"augmented_correct": False},
    }
    summary = summarize_rows(rows, initialization)
    assert summary["base"]["delta_rows"] == 1
    assert summary["initialization"]["delta_rows"] == 0
    assert summary["initialization"]["fixes"] == 1
    assert summary["initialization"]["regressions"] == 1

    left = {row["item_id"]: row for row in rows}
    right = {
        "a": {"augmented_correct": False},
        "b": {"augmented_correct": False},
        "c": {"augmented_correct": True},
    }
    contrast = compare_conditions(left, right)
    assert contrast["left_only_correct"] == 1
    assert contrast["right_only_correct"] == 0
    assert contrast["net_rows_left_minus_right"] == 1
    assert holm_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
