from __future__ import annotations

import ast
from pathlib import Path

import torch
import pytest

from eval.cache_speculative_depth_d0_teachers import (
    completed_cache_receipt,
    validate_teacher_drafter_tokenizer_alignment,
)
from colab.run_stage5_paper2_d0_teacher_cache import resolve_checkpoint_source
from eval.eval_speculative_depth_d0 import first_stop, simulate_windows, spearman
from training.speculative_depth_d0_postlock import (
    D0_LOCK_COMMIT,
    build_training_schedule,
    cache_plan,
    calibration_verdict,
    fit_depth_mapping,
    fit_isotonic,
    predict_isotonic,
    rejection_run_lengths,
    score_teacher_signals,
    validate_cache_summary,
)


def test_postlock_cache_plan_preserves_single_pass_teacher_contract() -> None:
    plan = cache_plan()
    assert D0_LOCK_COMMIT == "90cbc48c9aa749cb2e53dfef35bb2af9a24d9ae3"
    assert plan["teacher_7b"]["partitions"] == [
        "label_train",
        "calibration",
        "evaluation",
        "in_era_contrast",
    ]
    assert plan["teacher_14b"]["partitions"] == ["calibration"]
    assert plan["teacher_7b"]["full_logit_scope"] == "registered_natural_training_positions"
    assert plan["teacher_14b"]["full_logit_scope"] == "none"


def test_training_schedule_is_exactly_70_30_and_deterministic() -> None:
    first = build_training_schedule(total_steps=4000, natural_positions=10_000, seed=0)
    second = build_training_schedule(total_steps=4000, natural_positions=10_000, seed=0)
    assert first == second
    assert sum(row["kind"] == "natural" for row in first) == 2800
    assert sum(row["kind"] == "rehearsal" for row in first) == 1200
    assert all(0 <= int(row["position_index"]) < 10_000 for row in first if row["kind"] == "natural")


def test_rejection_run_lengths_are_local_to_each_row() -> None:
    assert rejection_run_lengths([False, True, True, False, True]) == [0, 2, 2, 0, 1]
    assert rejection_run_lengths([True, True]) == [2, 2]


def test_teacher_signal_scorer_masks_control_rows_and_reports_exact_values() -> None:
    teacher = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    drafter = torch.tensor([[1.0, 2.0, 0.0, 99.0], [0.0, 1.0, 3.0, 99.0]])
    target = torch.tensor([1, 2])
    result = score_teacher_signals(teacher, drafter, target)
    assert result["teacher_greedy_token_id"].tolist() == [0, 1]
    assert result["drafter_greedy_token_id"].tolist() == [1, 2]
    assert result["accepted"].tolist() == [False, False]
    assert result["drafter_token_rank_under_teacher"].tolist() == [2, 2]
    assert torch.isfinite(result["teacher_entropy"]).all()
    assert torch.isfinite(result["teacher_to_plain_drafter_kl"]).all()


def test_teacher_signal_scorer_ignores_scale_specific_padded_model_rows() -> None:
    teacher = torch.tensor([[1.0, 2.0, 3.0, 100.0, 90.0]])
    drafter = torch.tensor([[3.0, 2.0, 1.0, 200.0]])
    target = torch.tensor([2])

    result = score_teacher_signals(
        teacher, drafter, target, shared_vocab_size=3
    )

    assert result["teacher_greedy_token_id"].tolist() == [2]
    assert result["drafter_greedy_token_id"].tolist() == [0]
    assert result["accepted"].tolist() == [False]


def test_teacher_signal_scorer_rejects_targets_outside_shared_tokenizer_space() -> None:
    with pytest.raises(ValueError, match="target token IDs exceed"):
        score_teacher_signals(
            torch.zeros(1, 5),
            torch.zeros(1, 4),
            torch.tensor([3]),
            shared_vocab_size=3,
        )


def test_calibration_verdict_uses_locked_two_bin_gain_rule() -> None:
    graded = calibration_verdict(
        {
            "q1": [0.4, 0.48, 0.50, 0.50, 0.50, 0.50],
            "q2": [0.3, 0.31, 0.31, 0.31, 0.31, 0.31],
            "q3": [0.2, 0.25, 0.29, 0.30, 0.30, 0.30],
            "q4": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )
    assert graded["branch"] == "graded_floor_curve"
    assert graded["targets"] == {"q1": 3, "q2": 1, "q3": 3, "q4": 1}
    flat = calibration_verdict({name: [0.2] * 6 for name in ("q1", "q2", "q3", "q4")})
    assert flat["branch"] == "flat_floor_dynamic_targets"


def test_cache_summary_requires_every_locked_partition_and_no_teacher_reload() -> None:
    summary = {
        "status": "complete",
        "lock_commit": D0_LOCK_COMMIT,
        "teacher_reloaded_after_completed_cache": False,
        "tokenizer_alignment": {
            teacher: {
                "status": "exact_pre_resize_vocabulary_match",
                "logit_space": "shared_pre_resize_tokenizer_vocabulary",
                "vocabulary_size": 151665,
            }
            for teacher in cache_plan()
        },
        "caches": {
            "teacher_7b": {name: {"status": "complete"} for name in cache_plan()["teacher_7b"]["partitions"]},
            "teacher_14b": {"calibration": {"status": "complete"}},
        },
    }
    validate_cache_summary(summary)
    summary["caches"]["teacher_7b"].pop("evaluation")
    try:
        validate_cache_summary(summary)
    except AssertionError as error:
        assert "evaluation" in str(error)
    else:
        raise AssertionError("missing cache partition was accepted")


def test_first_stop_and_exhaustion_are_exact() -> None:
    selected, exhausted = first_stop(
        torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]), 4
    )
    assert selected.tolist() == [2, 4, 1]
    assert exhausted.tolist() == [False, True, False]


def test_teacher_forced_speculative_window_simulation_advances_after_correction() -> None:
    records = [
        {
            "row_index": 0,
            "plain_token_id": value,
            "adaptive_token_id": value,
            "teacher_token_id": target,
            "selected_loop": 2,
        }
        for value, target in [(1, 1), (2, 9), (3, 3), (4, 4)]
    ]
    result = simulate_windows(records, 4, adaptive=True)
    assert result["windows"] == 2
    assert result["draft_tokens_proposed"] == 6
    assert result["draft_tokens_accepted"] == 3
    assert result["loop_cost"] == 12


def test_spearman_handles_ties_without_external_statistics_dependency() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_completed_teacher_cache_can_resume_without_loading_teacher(tmp_path) -> None:
    path = tmp_path / "teacher_7b" / "evaluation" / "rows_000000_000001.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "kind": "paper2_d0_teacher_cache_shard",
            "lock_commit": D0_LOCK_COMMIT,
            "logit_space": "shared_pre_resize_tokenizer_vocabulary",
            "shared_vocab_size": 3,
            "teacher": "teacher_7b",
            "partition": "evaluation",
            "row_start": 0,
            "row_stop": 1,
            "rows": [
                {
                    "full_logit_local_positions": torch.tensor([2, 4]),
                }
            ],
        },
        path,
    )
    receipt = completed_cache_receipt(
        cache_root=tmp_path, teacher="teacher_7b", partition="evaluation", expected_rows=1
    )
    assert receipt is not None
    assert receipt["full_logit_rows"] == 1
    assert receipt["full_logit_positions"] == 2
    assert receipt["shared_vocab_size"] == 3


def test_teacher_cache_checkpoint_resolver_accepts_only_sha_identical_fallback(tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    wrong = tmp_path / "wrong.pt"
    right = tmp_path / "stage_states" / "raw.pt"
    wrong.write_bytes(b"wrong")
    right.parent.mkdir(parents=True)
    right.write_bytes(b"locked checkpoint")

    import hashlib

    expected = hashlib.sha256(b"locked checkpoint").hexdigest()
    resolved, diagnostics = resolve_checkpoint_source(
        [missing, wrong, right], expected_sha256=expected
    )

    assert resolved == right
    assert [row["status"] for row in diagnostics] == ["missing", "sha_mismatch", "matched"]


def test_teacher_cache_checkpoint_resolver_fails_with_candidate_diagnostics(tmp_path) -> None:
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"wrong")

    import pytest

    with pytest.raises(FileNotFoundError, match="No SHA-identical locked D0 drafter checkpoint"):
        resolve_checkpoint_source([tmp_path / "missing.pt", wrong], expected_sha256="0" * 64)


def test_teacher_alignment_uses_pre_resize_vocab_and_audits_frozen_ids() -> None:
    class _Tokenizer:
        def get_vocab(self):
            return {"a": 0, "b": 1, "c": 2}

    receipt = validate_teacher_drafter_tokenizer_alignment(
        teacher_tokenizer=_Tokenizer(),
        drafter_original_vocab={"a": 0, "b": 1, "c": 2},
        rows_by_partition={"label_train": [{"input_ids": [0, 2, 1]}]},
    )

    assert receipt["status"] == "exact_pre_resize_vocabulary_match"
    assert receipt["vocabulary_size"] == 3
    assert receipt["token_ids_checked"] == 3


def test_teacher_alignment_rejects_out_of_vocabulary_frozen_ids() -> None:
    class _Tokenizer:
        def get_vocab(self):
            return {"a": 0, "b": 1}

    with pytest.raises(RuntimeError, match="outside the shared teacher/drafter vocabulary"):
        validate_teacher_drafter_tokenizer_alignment(
            teacher_tokenizer=_Tokenizer(),
            drafter_original_vocab={"a": 0, "b": 1},
            rows_by_partition={"evaluation": [{"input_ids": [0, 2]}]},
        )


def test_all_d0_load_drafter_callers_unpack_pre_resize_vocabulary() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "eval/cache_speculative_depth_d0_teachers.py",
        root / "eval/eval_speculative_depth_d0.py",
        root / "eval/eval_speculative_depth_d0_floor.py",
        root / "eval/eval_speculative_depth_d0_arc_allocation.py",
        root / "eval/eval_speculative_depth_d0_t1_retention.py",
    ]
    observed = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            function = node.value.func
            if not isinstance(function, ast.Name) or function.id != "load_drafter":
                continue
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
            assert len(node.targets[0].elts) == 4, f"stale load_drafter unpack in {path}"
            observed += 1
    assert observed == 6


def test_isotonic_mapping_is_monotone_and_depth_capped() -> None:
    model = fit_isotonic([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 2.0, 4.0])
    predictions = [predict_isotonic(model, value) for value in [0.0, 1.0, 2.0, 3.0]]
    assert predictions == sorted(predictions)
    assert all(1 <= value <= 4 for value in predictions)


def test_depth_mapping_records_primary_and_all_locked_comparators() -> None:
    examples = [
        {
            "kl": index / 10,
            "run_length": 1 + index % 4,
            "required_depth": 1 + min(3, index // 4),
            "rank": 1 + index,
            "teacher_entropy": index / 20,
            "negative_drafter_logprob_under_teacher": index / 8,
        }
        for index in range(20)
    ]
    receipt = fit_depth_mapping(examples)
    assert receipt["primary_fit"]["kind"] == "monotone_isotonic"
    assert set(receipt["heldout_scores"]) == {
        "isotonic",
        "linear_run_length",
        "linear_log_kl",
        "saturating",
    }
    assert receipt["run_length_tail_excluded_above"] == 8
