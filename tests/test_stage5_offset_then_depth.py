from __future__ import annotations

import importlib
from pathlib import Path


def _row(delta: int, paired: int = 256) -> dict:
    return {
        "paired_examples": paired,
        "base_correct": 100,
        "recurrent_correct": 100 + delta,
        "correct_delta_recurrent_vs_base": delta,
        "wins": max(delta, 0),
        "losses": max(-delta, 0),
        "ties": paired - abs(delta),
        "sign_test_p_value": 1.0,
    }


def _payload(delta: int = 0, paired: int = 256) -> dict:
    return {
        "status": "completed",
        "failures": [],
        "paired_comparisons": {
            "arc_easy": {
                "content_question_only": {"mean": _row(delta, paired)},
                "cyclic_label_aggregated": {"permutation_mean": _row(delta, paired)},
            },
            "arc_challenge": {
                "content_question_only": {"mean": _row(delta, paired)},
                "cyclic_label_aggregated": {"permutation_mean": _row(delta, paired)},
            },
        },
    }


def module():
    return importlib.import_module("colab.run_stage5_arc_mix_offset_then_depth")


def test_offset_assessment_passes_four_nonnegative_readouts() -> None:
    assessed = module().assess_offset_confirmation(_payload(delta=0))

    assert assessed["status"] == "offset_confirmed_debiased_flat"
    assert assessed["passed"] is True
    assert assessed["content_replicated"] is False
    assert assessed["debiased_positive"] is False
    assert len(assessed["evidence"]) == 4
    assert all(row["passed"] for row in assessed["evidence"])


def test_offset_assessment_blocks_negative_content_readout() -> None:
    payload = _payload(delta=0)
    payload["paired_comparisons"]["arc_challenge"]["content_question_only"]["mean"] = _row(-1)

    assessed = module().assess_offset_confirmation(payload)

    assert assessed["status"] == "offset_regressed"
    assert assessed["passed"] is False
    failed = [row for row in assessed["evidence"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["benchmark"] == "arc_challenge"
    assert failed[0]["score_target"] == "content_question_only"


def test_offset_assessment_requires_coverage() -> None:
    assessed = module().assess_offset_confirmation(_payload(delta=1, paired=128))

    assert assessed["status"] == "offset_regressed"
    assert assessed["passed"] is False
    assert all(row["paired_examples"] == 128 for row in assessed["evidence"])
    assert any(
        row["benchmark"] == "arc_easy" and row["paired_examples"] < row["required_examples"]
        for row in assessed["evidence"]
    )


def test_offset_assessment_accepts_custom_post_depth_min_examples() -> None:
    assessed = module().assess_offset_confirmation(_payload(delta=0, paired=128), min_examples=128)

    assert assessed["status"] == "offset_confirmed_debiased_flat"
    assert assessed["passed"] is True


def test_offset_assessment_accepts_flat_debiased_small_arc_challenge_slice() -> None:
    payload = _payload(delta=0)
    payload["paired_comparisons"]["arc_easy"]["content_question_only"]["mean"] = _row(10, paired=256)
    payload["paired_comparisons"]["arc_easy"]["cyclic_label_aggregated"]["permutation_mean"] = _row(0, paired=256)
    payload["paired_comparisons"]["arc_challenge"]["content_question_only"]["mean"] = _row(1, paired=43)
    payload["paired_comparisons"]["arc_challenge"]["cyclic_label_aggregated"]["permutation_mean"] = _row(0, paired=43)

    assessed = module().assess_offset_confirmation(payload)

    assert assessed["status"] == "offset_confirmed_debiased_flat"
    assert assessed["passed"] is True
    assert assessed["content_replicated"] is True
    assert assessed["debiased_positive"] is False
    assert assessed["min_examples_by_benchmark"]["arc_challenge"] == 32
    assert all(row["passed"] for row in assessed["evidence"])


def test_offset_assessment_blocks_material_debiased_regression() -> None:
    payload = _payload(delta=0)
    payload["paired_comparisons"]["arc_easy"]["cyclic_label_aggregated"]["permutation_mean"] = _row(-1, paired=256)

    assessed = module().assess_offset_confirmation(payload)

    assert assessed["status"] == "offset_regressed"
    assert assessed["passed"] is False
    failed = [row for row in assessed["evidence"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["score_target"] == "cyclic_label_aggregated"


def test_offset_assessment_can_explicitly_tolerate_small_debiased_negative() -> None:
    payload = _payload(delta=0)
    payload["paired_comparisons"]["arc_easy"]["content_question_only"]["mean"] = _row(10, paired=256)
    payload["paired_comparisons"]["arc_easy"]["cyclic_label_aggregated"]["permutation_mean"] = _row(-1, paired=256)

    assessed = module().assess_offset_confirmation(payload, debiased_allowed_negative_delta=1)

    assert assessed["status"] == "offset_confirmed_debiased_tolerated_negative"
    assert assessed["passed"] is True
    assert assessed["content_replicated"] is True
    assert assessed["debiased_positive"] is False


def test_offset_assessment_blocks_completed_with_failures() -> None:
    payload = _payload(delta=1)
    payload["status"] = "completed_with_failures"
    payload["failures"] = [{"benchmark": "arc_easy"}]

    assessed = module().assess_offset_confirmation(payload)

    assert assessed["status"] == "offset_incomplete"
    assert assessed["passed"] is False


def test_offset_summary_override_reuses_existing_summary(monkeypatch, tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("STAGE5_ARC_MIX_CHAIN_OFFSET_SUMMARY", str(summary))

    imported = importlib.reload(module())

    assert imported.run_offset_confirmation() == summary
    monkeypatch.delenv("STAGE5_ARC_MIX_CHAIN_OFFSET_SUMMARY", raising=False)
    importlib.reload(imported)
