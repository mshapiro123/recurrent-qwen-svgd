from __future__ import annotations

import pytest

from training.speculative_depth_d0_spec import (
    D0ExecutionPolicy,
    build_only_contract,
    calibrated_depth_targets,
    d0_draft,
    depth_recoverable_fraction,
    dynamic_depth_target,
    mask_teacher_added_token_probabilities,
    locked_d0_from_manifest,
    prelock_contract,
    unresolved_paths,
    validate_locked_d0,
)
from training.speculative_depth_d0_corpus import (
    SourceDocument,
    assert_document_disjoint,
    choose_stratum_mix,
    collect_partition_rows,
    collect_probe_rows,
    partition_for_document,
    select_pilot_rows,
    token_quotas,
)

import torch
from eval.eval_speculative_depth_d0_density import score_density


def test_d0_draft_is_fail_closed_and_has_no_launcher() -> None:
    spec = d0_draft()
    assert spec["status"] == "draft_not_locked"
    assert spec["training_authorized"] is False
    assert spec["launch_target_exists"] is False
    assert spec["substrate_family"] == "Qwen"
    assert spec["dependency"]["t1_lite_verdict_required"] is True
    assert spec["dependency"]["automatic_launch_from_t1"] is False
    assert len(unresolved_paths(spec)) >= 10


def test_d0_draft_cannot_validate_as_locked() -> None:
    with pytest.raises(AssertionError, match="not locked"):
        validate_locked_d0(d0_draft())


def test_d0_build_only_contract_forbids_labeling_and_training() -> None:
    contract = build_only_contract()
    assert contract["status"] == "build_only_no_labeling_no_training"
    assert contract["labeling_gpu_authorized"] is False
    assert contract["training_authorized"] is False
    D0ExecutionPolicy().assert_allowed(labeling=False, training=False)
    with pytest.raises(RuntimeError, match="labeling"):
        D0ExecutionPolicy().assert_allowed(labeling=True, training=False)
    with pytest.raises(RuntimeError, match="training"):
        D0ExecutionPolicy().assert_allowed(labeling=False, training=True)


def test_d0_prelock_contract_authorizes_density_only() -> None:
    contract = prelock_contract()
    assert contract["density_probe_authorized"] is True
    assert contract["labeling_gpu_authorized"] is False
    assert contract["training_authorized"] is False
    D0ExecutionPolicy(density_probe_authorized=True).assert_allowed(
        density_probe=True, labeling=False, training=False
    )
    with pytest.raises(RuntimeError, match="labeling"):
        D0ExecutionPolicy(density_probe_authorized=True).assert_allowed(
            density_probe=True, labeling=True, training=False
        )


def test_d0_dynamic_target_uses_first_teacher_match_and_caps_at_four() -> None:
    assert dynamic_depth_target([False, True, True, True], max_depth=4) == 2
    assert dynamic_depth_target([False, False, False, False], max_depth=4) == 4
    assert dynamic_depth_target([True], max_depth=4) == 1


def test_d0_graded_mapping_selects_smallest_depth_near_depth4_plateau() -> None:
    curves = {
        "q1": [0.40, 0.48, 0.50, 0.50],
        "q2": [0.30, 0.31, 0.31, 0.31],
        "q3": [0.20, 0.25, 0.29, 0.30],
        "q4": [0.10, 0.10, 0.10, 0.10],
    }
    result = calibrated_depth_targets(curves)
    assert result["branch"] == "graded_floor_curve"
    assert result["targets"] == {"q1": 3, "q2": 1, "q3": 3, "q4": 1}


def test_d0_teacher_added_token_mask_renormalizes() -> None:
    probabilities = torch.tensor([0.2, 0.3, 0.1, 0.4])
    masked = mask_teacher_added_token_probabilities(probabilities, added_token_ids=[2, 3])
    assert torch.allclose(masked, torch.tensor([0.4, 0.6, 0.0, 0.0]))
    assert float(masked.sum()) == pytest.approx(1.0)


def test_d0_depth_recoverable_fraction_is_incremental_match_rate() -> None:
    receipt = depth_recoverable_fraction(loop1_matches=20, self_halted_matches=35, rejected_positions=100)
    assert receipt["depth_recoverable_fraction"] == pytest.approx(0.15)


class _Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


def _documents(count: int = 1000) -> list[SourceDocument]:
    return [
        SourceDocument(
            document_id=f"doc-{index}",
            text=" ".join(f"t{value}" for value in range(64)),
            metadata={"license": "fixture"},
        )
        for index in range(count)
    ]


def test_d0_mix_rule_preserves_domain_floor() -> None:
    assert choose_stratum_mix({"general": 100.0, "code": 80.0})["mix"] == {
        "general": 0.5,
        "code": 0.5,
    }
    assert choose_stratum_mix({"general": 100.0, "code": 20.0})["mix"] == {
        "general": 0.6,
        "code": 0.4,
    }


def test_d0_token_quotas_are_exact() -> None:
    quotas = token_quotas(2_000_000, {"general": 0.6, "code": 0.4})
    assert sum(sum(parts.values()) for parts in quotas.values()) == 2_000_000
    assert quotas["general"] == {
        "label_train": 960_000,
        "calibration": 120_000,
        "evaluation": 120_000,
    }


def test_d0_probe_and_partitions_are_deterministic_and_document_disjoint() -> None:
    tokenizer = _Tokenizer()
    probe, excluded = collect_probe_rows(
        _documents(), tokenizer, stratum="general", token_budget=128
    )
    assert sum(row["token_count"] for row in probe) == 128
    quotas = {"label_train": 256, "calibration": 128, "evaluation": 128}
    partitions = collect_partition_rows(
        _documents(),
        tokenizer,
        stratum="general",
        quotas=quotas,
        excluded_document_ids=excluded,
    )
    assert {name: sum(row["token_count"] for row in rows) for name, rows in partitions.items()} == quotas
    assert assert_document_disjoint(partitions)["document_disjoint"] is True
    assert partition_for_document("stable") == partition_for_document("stable")
    assert len(select_pilot_rows(partitions["label_train"], count=4)) == 4


def test_d0_locked_payload_requires_all_frozen_artifacts() -> None:
    artifacts = {
        name: {"sha256": "a" * 64, "tokens": 1}
        for name in (
            "label_train",
            "calibration",
            "evaluation",
            "density_general",
            "density_code",
            "in_era_contrast",
            "pilot_256",
            "general_label_train",
            "general_calibration",
            "general_evaluation",
            "code_label_train",
            "code_calibration",
            "code_evaluation",
        )
    }
    payload = locked_d0_from_manifest(
        {"document_disjoint": True, "artifacts": artifacts}
    )
    validate_locked_d0(payload)
    assert payload["calibration"]["forced_measurement_depths"] == [1, 2, 3, 4, 5, 6]
    assert payload["calibration"]["training_target_cap"] == 4


def test_d0_density_counts_next_token_disagreements() -> None:
    rows = [{"input_ids": [1, 2, 3, 4], "token_count": 4}]
    receipt = score_density(rows, [[1, 2, 3]], [[1, 9, 3]])
    assert receipt["prediction_positions"] == 3
    assert receipt["disagreements"] == 1
    assert receipt["rejection_density_per_1000_tokens"] == pytest.approx(1000 / 3)
