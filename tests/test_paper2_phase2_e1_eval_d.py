from __future__ import annotations

import json
from pathlib import Path

import torch

from training.paper2_phase2_e1_confirmation import REQUIRED_CACHE_FIELDS
from training.paper2_phase2_e1_eval_d import (
    ANCHORS_PER_STRATUM,
    SELECTION_RULE,
    build_freeze_receipt,
    build_score_blind_config,
    dev_mixture_weights,
)
from training.paper2_phase2_option_b import load_locked_registration


ROOT = Path(__file__).resolve().parents[1]


def test_score_blind_config_copies_locked_teacher_stack(tmp_path: Path) -> None:
    data = tmp_path / "eval_d.jsonl"
    data.write_text('{"document_id":"d","input_ids":[1,2,3,4,5]}\n', encoding="utf-8")
    registration = load_locked_registration()
    config = build_score_blind_config(registration=registration, data_path=data)
    teacher = registration["teacher_pass"]
    assert config["anchor_count"] == 8_000
    assert config["anchors_per_stratum"] == ANCHORS_PER_STRATUM
    assert config["boundary_sample_count"] == 32_000
    assert config["seed"] == 20260808
    assert config["selection_rule"] == SELECTION_RULE
    assert config["models"] == teacher["models"]
    assert config["cascade"]["query_32b_on_7b_14b_argmax_disagreement"]
    assert config["score_blind"] is True
    assert config["endpoint_checkpoints_loaded"] is False
    assert config["model_quality_scores_computed"] is False


def test_dev_mixture_uses_immutable_document_partition() -> None:
    samples = []
    for anchor in range(1_000):
        for horizon in range(1, 5):
            samples.append(
                {
                    "anchor_index": anchor,
                    "document_id": f"doc-{anchor}",
                    "stratum": "general" if anchor < 500 else "code",
                    "horizon": horizon,
                }
            )
    result = dev_mixture_weights(samples)
    assert result["anchor_count"] > 0
    assert set(result["weights"]) == {"general", "code"}
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-12


def test_freeze_receipt_is_score_blind_and_evaluator_compatible(tmp_path: Path) -> None:
    private_cache = tmp_path / "cache.pt"
    private_cache.write_bytes(b"transport")
    cache = {field: object() for field in REQUIRED_CACHE_FIELDS}
    cache.update(
        {
            "kind": "paper2_phase2_matched_alpha_cache_v1",
            "documents": [f"document-{index // 4}" for index in range(8_000)],
            "strata": ["general"] * 4_000 + ["code"] * 4_000,
            "positions": torch.arange(8_000),
        }
    )
    receipt = build_freeze_receipt(
        cache=cache,
        private_cache_path=private_cache,
        data_sha256="a" * 64,
        document_count=2_000,
        canonicalizer_sha256="b" * 64,
        sample_manifest_sha256="c" * 64,
        position_key_sha256_value="e" * 64,
        admission_ledger_sha256="d" * 64,
        cascade_count=5_200,
        dev_mixture={
            "source": "immutable",
            "anchor_count": 8_031,
            "counts": {"general": 4_100, "code": 3_931},
            "weights": {"general": 4_100 / 8_031, "code": 3_931 / 8_031},
        },
        model_revisions={"teacher_14b": "revision"},
        cross_partition_document_overlap=[],
    )
    assert receipt["status"] == "complete_frozen_unscored"
    assert receipt["scores_exposed"] is False
    assert receipt["read_once_scoring_spent"] is False
    for key in (
        "model_quality_scores_computed",
        "eal_computed",
        "retention_computed",
        "acceptance_computed",
        "student_teacher_quality_aggregates_emitted",
    ):
        assert receipt[key] is False
    assert set(receipt["option_b_cache"]["fields"]) == REQUIRED_CACHE_FIELDS
    assert receipt["option_b_cache"]["anchor_count"] == 8_000


def test_score_blind_lattice_omits_quality_aggregation() -> None:
    source = (ROOT / "eval/cache_paper2_phase2_stage0a.py").read_text(encoding="utf-8")
    assert "if not score_blind:" in source
    assert '"student_teacher_quality_aggregates_emitted": not score_blind' in source
    assert '"endpoint_models_loaded": False' in source
    assert '"eal_computed": False' in source
    assert '"retention_computed": False' in source
    assert '"acceptance_computed": False' in source


def test_e1_cache_runner_cannot_load_endpoints_or_train() -> None:
    runner = (
        ROOT / "colab/run_stage5_paper2_phase2_e1_eval_d_freeze.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "seed_0_full_a2",
        "seed_1_full_a2",
        "draft_only_control/resume.pt",
        "torch.optim",
        "optimizer.step",
        "relative_mean_eal_gain",
    )
    assert not any(marker in runner for marker in forbidden)
    assert "--score_blind" in runner
    assert "eval.finalize_paper2_phase2_e1_eval_d_cache" in runner
    assert "read_once_scoring_spent" in runner


def test_e1_eval_d_target_is_wired_and_guarded() -> None:
    cell = (
        ROOT / "colab/STAGE5_PAPER2_PHASE2_E1_EVAL_D_FREEZE_CELL.py"
    ).read_text(encoding="utf-8")
    for path in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py",
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md",
    ):
        bootstrap = path.read_text(encoding="utf-8")
        assert "paper2_phase2_e1_eval_d_freeze" in bootstrap
        assert "STAGE5_PAPER2_PHASE2_E1_EVAL_D_FREEZE_CELL.py" in bootstrap
        assert "read-once scoring remains unspent and readiness only authorizes lock" in bootstrap
    assert "paper2_phase2_e1_eval_d_freeze_v1" in cell
    assert "memory >= 70000" in cell
    assert "no EAL no retention no acceptance no optimizer no training" in cell


def test_machine_preregistration_carries_ratified_population() -> None:
    payload = json.loads(
        (
            ROOT / "training/paper2_phase2_e1_confirmation_preregistration.draft.json"
        ).read_text(encoding="utf-8")
    )
    population = payload["evaluation"]["population"]
    assert population["anchor_count"] == 8_000
    assert population["anchors_per_stratum"] == ANCHORS_PER_STRATUM
    assert population["selection_seed"] == 20260808
    assert population["selection_rule"] == SELECTION_RULE
    assert payload["primary"]["population_weighting"] == "balanced_general_0p5_code_0p5"
