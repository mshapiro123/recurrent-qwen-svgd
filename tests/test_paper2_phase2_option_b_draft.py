from __future__ import annotations

import json
from pathlib import Path

from training.paper2_phase2_stage0a import STAGE0A_CONFIG
from training.speculative_depth_d0_spec import (
    FINEWEB_DATASET,
    FINEWEB_DUMP,
    FINEWEB_REVISION,
    STACK_DATASET,
    STACK_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "training/paper2_phase2_option_b_preregistration.draft.json"
PROTOCOL = ROOT / "docs/PAPER2_PHASE2_OPTION_B_EXPLORATION_PROTOCOL_DRAFT_20260806.md"


def test_option_b_draft_is_staged_and_cannot_launch() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert registration["status"] == "draft_training_prohibited"
    assert registration["pre_splice_authorized"] is False
    assert registration["post_splice_authorized"] is False
    assert registration["training_authorized"] is False
    assert registration["fixed_constants"]["target_splice_step"] == 4000
    assert registration["fixed_constants"]["stable_learning_rate_through_step"] == 18000
    assert registration["teacher_pass"]["new_training_anchor_target"] == 140_000
    assert registration["teacher_pass"]["new_training_anchor_minimum"] == 100_000
    assert registration["teacher_pass"]["hash_only_amendment_required_before_splice"] is True
    assert "teacher_14b_state_coverage_policy" in registration["lock_blockers"]
    assert registration["analysis"]["not_a_general_unique_data_scaling_law"] is True


def test_option_b_units_and_teacher_revisions_are_explicit() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    units = registration["population_units_banked"]
    assert units["stage0a_anchors"] == 50_000
    assert units["stage0a_horizons_per_anchor"] == 4
    assert units["stage0a_horizon_samples"] == 200_000
    assert units["a2_training_anchors"] == 41_969
    assert units["a2_evaluation_anchors"] == 8_031
    models = registration["teacher_pass"]["models"]
    for key in ("student_0p5b", "teacher_7b", "teacher_14b", "teacher_32b"):
        assert models[key]["model"] == STAGE0A_CONFIG["models"][key]["model"]
        assert models[key]["revision"] == STAGE0A_CONFIG["models"][key]["revision"]
    sources = registration["teacher_pass"]["sources"]
    assert sources["general"] == {
        "dataset": FINEWEB_DATASET,
        "revision": FINEWEB_REVISION,
        "dump": FINEWEB_DUMP,
    }
    assert sources["code"]["dataset"] == STACK_DATASET
    assert sources["code"]["revision"] == STACK_REVISION


def test_option_b_protocol_requires_lock_and_hash_only_splice_amendment() -> None:
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "draft, not locked, training prohibited" in protocol
    assert "No Option B training or teacher-pass launcher may exist" in protocol
    assert "hash-only amendment" in protocol
    assert "single splice identifies a general unique-data scaling law" in protocol
    assert not (ROOT / "training/run_paper2_phase2_option_b.py").exists()
    assert not (ROOT / "colab/run_stage5_paper2_phase2_option_b.py").exists()


def test_option_b_resource_note_separates_runtime_classes() -> None:
    note = (
        ROOT / "docs/PAPER2_PHASE2_OPTION_B_TEACHER_PASS_RESOURCE_NOTE_DRAFT_20260806.md"
    ).read_text(encoding="utf-8")
    assert "A100 80GB class" in note
    assert "One A100 cannot run the teacher pass and Segment 1 concurrently" in note
    assert "100,000-anchor floor" in note
    assert "25-percent reserve" in note
    assert "does not authorize" in note
