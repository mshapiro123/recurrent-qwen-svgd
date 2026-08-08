from __future__ import annotations

import json
import hashlib
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
REGISTRATION = ROOT / "training/paper2_phase2_option_b_preregistration.json"
PROTOCOL = ROOT / "docs/PAPER2_PHASE2_OPTION_B_EXPLORATION_PROTOCOL_LOCKED_20260806.md"
RULES = ROOT / "training/paper2_phase2_option_b_rule_inventory.json"


def test_option_b_hash_amendment_authorizes_the_locked_training_protocol() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert registration["status"] == (
        "locked_post_transport_and_endpoint_errata_training_authorized"
    )
    assert registration["locked_before_teacher_pass"] is True
    assert registration["teacher_pass_authorized"] is True
    assert registration["lock_blockers"] == []
    assert registration["pre_splice_authorized"] is True
    assert registration["post_splice_authorized"] is True
    assert registration["training_authorized"] is True
    assert registration["fixed_constants"]["target_splice_step"] == 4000
    assert registration["fixed_constants"]["stable_learning_rate_through_step"] == 18000
    assert registration["teacher_pass"]["new_training_anchor_target"] == 140_000
    assert registration["teacher_pass"]["new_training_anchor_minimum"] == 100_000
    assert registration["teacher_pass"]["hash_only_amendment_required_before_splice"] is True
    assert registration["teacher_pass"]["teacher_14b_state_coverage_policy"] == "all_admitted_anchors"
    assert registration["teacher_pass"]["per_anchor_label_tier_admission_required"] is True
    assert registration["analysis"]["not_a_general_unique_data_scaling_law"] is True
    erratum = registration["endpoint_reserialization_erratum"]
    assert erratum["locked_before_option_b_optimizer_updates"] is True
    assert erratum["option_b_optimizer_updates_before_erratum"] == 0
    assert erratum["executed_optimizer_updates_per_arm"] == 2_000
    assert erratum["executed_schedule_sha256"] == (
        "a2718f46a22ff47a91f14fac2bb1fb38719fa29c4edb9663cab5143f139524c6"
    )
    transport = registration["teacher_cache_summary_transport_erratum"]
    assert transport["locked_before_option_b_optimizer_updates"] is True
    assert transport["option_b_optimizer_updates_before_erratum"] == 0
    assert transport["original_windows_crlf_sha256"] == registration[
        "post_generation_hash_amendment"
    ]["teacher_cache_summary_sha256"]
    assert transport["newline_count"] == 142


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


def test_option_b_protocol_and_post_generation_amendment_are_both_immutable() -> None:
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "locked before teacher pass" in protocol
    assert "teacher/cache generation authorized" in protocol
    assert "Option B training remains prohibited" in protocol
    assert "hash-only amendment" in protocol
    assert "single splice identifies a general unique-data scaling law" in protocol
    assert (ROOT / "training/run_paper2_phase2_option_b.py").exists()
    assert (ROOT / "colab/run_stage5_paper2_phase2_option_b.py").exists()
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    amendment = registration["post_generation_hash_amendment"]
    path = ROOT / amendment["amendment_document"]
    assert path.stat().st_size == amendment["amendment_document_bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == amendment["amendment_document_sha256"]
    endpoint_erratum = registration["governing_documents"][
        "endpoint_reserialization_erratum"
    ]
    endpoint_path = ROOT / endpoint_erratum["path"]
    assert endpoint_path.stat().st_size == endpoint_erratum["bytes"]
    assert hashlib.sha256(endpoint_path.read_bytes()).hexdigest() == endpoint_erratum["sha256"]
    transport = registration["governing_documents"]["teacher_summary_transport_erratum"]
    transport_path = ROOT / transport["path"]
    transport_bytes = transport_path.read_bytes().replace(b"\r\n", b"\n")
    assert len(transport_bytes) == transport["git_lf_bytes"]
    assert hashlib.sha256(transport_bytes).hexdigest() == transport["git_lf_sha256"]


def test_option_b_endpoint_byte_and_semantic_hashes_are_locked() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    assert registration["source_checkpoints"] == {
        "seed_0_full_a2": "5ebc1ec1f2299344b24fb055799c5e35a8236982a4840f2013418fd7513a6373",
        "seed_0_draft_only_control": "69f0b3970dd1de174d728ce062ceba242a55d9ae9c670e4c5dd0d27ad9249b1a",
        "seed_1_full_a2": "5960ef967f3834db0c83eef26a2d9c896e43cc4f07f6a6d6047700dbcf5d4e76",
        "seed_1_draft_only_control": "691e102c0dd258f543e55aee291ac9d05675a9a8e8e8b170f47015e4782d1760",
    }
    digests = registration["source_checkpoint_semantic_digests"]
    assert digests["algorithm"] == "sorted_name_dtype_shape_tensor_bytes_sha256"
    for name in registration["source_checkpoints"]:
        assert len(digests[name]) == 64


def test_option_b_localization_and_existing_population_hashes_are_banked() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    unresolved = registration["post_generation_hash_amendment"]
    assert unresolved["localization_mask_rule_result"] == "no_structural_group_qualified"
    assert unresolved["structural_mask"] is None
    for key in (
        "localization_receipt_sha256",
        "localization_markdown_receipt_sha256",
        "existing_training_manifest_sha256",
        "existing_document_partition_sha256",
        "evaluation_exclusion_sha256",
        "fixed_old_train_subset_sha256",
        "teacher_pass_resource_note_sha256",
    ):
        assert len(unresolved[key]) == 64
    assert unresolved["new_training_anchor_count"] == 140_000
    assert unresolved["new_horizon_sample_count"] == 560_000
    assert unresolved["teacher_14b_state_sample_count"] == 560_000
    assert unresolved["recorded_splice_step"] == 4_000
    for key in (
        "teacher_cache_summary_sha256",
        "new_data_sha256",
        "new_training_manifest_sha256",
        "new_document_partition_sha256",
        "excluded_document_partition_sha256",
        "new_position_key_sha256",
        "new_lattice_summary_sha256",
        "full_logit_audit_sample_keys_sha256",
        "anchor_admission_ledger_sha256",
        "fixed_new_train_subset_sha256",
        "exclusion_lineage_closure_sha256",
    ):
        assert len(unresolved[key]) == 64


def test_option_b_resource_note_separates_runtime_classes() -> None:
    note = (
        ROOT / "docs/PAPER2_PHASE2_OPTION_B_TEACHER_PASS_RESOURCE_NOTE_LOCKED_20260806.md"
    ).read_text(encoding="utf-8")
    assert "A100 80GB class" in note
    assert "One A100 cannot run the teacher pass and Segment 1 concurrently" in note
    assert "100,000-anchor floor" in note
    assert "25-percent reserve" in note
    assert "authorize only the teacher/cache pass" in note


def test_option_b_rule_inventory_is_complete_and_only_named_cliffs_stop() -> None:
    inventory = json.loads(RULES.read_text(encoding="utf-8"))
    assert inventory["status"] == "locked_before_teacher_pass"
    assert len(inventory["rules"]) == 18
    required = {
        "name",
        "threshold",
        "estimator",
        "reference",
        "cadence",
        "disposition",
        "named_cliff",
    }
    for rule in inventory["rules"]:
        assert set(rule) == required
        if rule["disposition"] in {"stop", "stop_before_updates", "refuse_splice"}:
            assert rule["named_cliff"]
        else:
            assert rule["named_cliff"] is None


def test_option_b_locked_artifact_hashes_match() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for artifact in registration["lock_artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.exists()
        locked_bytes = path.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(locked_bytes).hexdigest() == artifact["sha256"]
