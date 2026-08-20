from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

import eval.eval_paper2_stage2b_autopsy as autopsy_eval

from training.paper2_stage2b_autopsy import (
    discrete_mutual_information,
    decision_mapping,
    margin_correlation_receipt,
    normalized_gram_eigengap,
    spherical_kmeans,
    stable_dev2_subsample,
    validate_autopsy_lock,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "training/paper2_stage2b_autopsy_lock.json"


def test_autopsy_signed_lock_is_score_only_and_sealed() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_autopsy_lock(lock, require_signature=True)
    assert lock["status"] == "SIGNED"
    assert lock["mark_signed"] is True
    assert lock["locked_before_model_contact"] is True
    assert lock["authority"]["signature_record_drive_id"] == (
        "1OSaglrQTMNkf_hWDLudeMIXYnnNLdrwK"
    )
    assert lock["authority"]["signature_record_sha256"] == (
        "bbdd5c05d08e6e6e9fc2c4d2a3d128b657f7b4b479c185c18b089b756aee481b"
    )
    assert lock["optimizer_steps_allowed"] == 0
    assert lock["training_authorized"] is False
    assert lock["sealed_partitions"]["remain_sealed"] is True

    unsigned = copy.deepcopy(lock)
    unsigned["status"] = "DRAFT_UNEXECUTABLE"
    unsigned["mark_signed"] = False
    with pytest.raises(RuntimeError, match="unsigned"):
        validate_autopsy_lock(unsigned, require_signature=True)


def test_autopsy_lock_rejects_training_or_seal_contact() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    training = copy.deepcopy(lock)
    training["optimizer_steps_allowed"] = 1
    with pytest.raises(RuntimeError, match="score-only"):
        validate_autopsy_lock(training, require_signature=False)
    unsealed = copy.deepcopy(lock)
    unsealed["sealed_partitions"]["confirm_scored"] = True
    with pytest.raises(RuntimeError, match="sealed-partition"):
        validate_autopsy_lock(unsealed, require_signature=False)


def test_dev2_subsample_is_deterministic_and_stratified() -> None:
    rows = [
        {"item_id": f"{battery}-{index}", "battery": battery}
        for battery, count in (("gsm8k", 80), ("mbpp", 10), ("mmlu", 8), ("tier1", 2))
        for index in range(count)
    ]
    first = stable_dev2_subsample(rows, size=25)
    second = stable_dev2_subsample(list(reversed(rows)), size=25)
    assert first == second
    counts = Counter(row["battery"] for row in first)
    assert len(first) == 25
    assert set(counts) == {"gsm8k", "mbpp", "mmlu", "tier1"}


def test_margin_correlation_reports_pearson_and_spearman() -> None:
    rows = [
        {"per_loop_mean_teacher_token_margin": [value, 0.0, 0.0, 2.0 * value]}
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    receipt = margin_correlation_receipt(rows)
    assert receipt["pearson"] == pytest.approx(1.0)
    assert receipt["spearman"] == pytest.approx(1.0)


def test_decision_mapping_composes_hypotheses() -> None:
    assert decision_mapping({"h_b_magnitude": True, "h_a_attractor": True}) == [
        "radius_control_successor",
        "task_preservation_anchor_required",
    ]


def test_arm6_geometry_primitives_detect_separated_directions() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn((16, 8), generator=generator) * 0.01
    right = torch.randn((16, 8), generator=generator) * 0.01
    left[:, 0] += 1.0
    right[:, 0] -= 1.0
    values = torch.cat([left, right])
    labels, silhouette = spherical_kmeans(
        values, clusters=2, restarts=4, iterations=20, seed=11
    )
    assert silhouette > 0.9
    gap = normalized_gram_eigengap(values, max_rank=4)
    assert gap["maximum"] > 0.0
    association = discrete_mutual_information(
        labels.tolist(), ["left"] * 16 + ["right"] * 16
    )
    assert association["normalized_by_battery_entropy"] == pytest.approx(1.0)


def test_autopsy_runner_contains_no_optimizer_or_sealed_partition_path() -> None:
    evaluator = (ROOT / "eval/eval_paper2_stage2b_autopsy.py").read_text(encoding="utf-8")
    orchestrator = (ROOT / "colab/run_stage5_paper2_stage2b_autopsy.py").read_text(
        encoding="utf-8"
    )
    assert "torch.optim" not in evaluator
    assert "optimizer.step" not in evaluator
    assert "stage5_paper2_phase3_confirm" not in orchestrator.lower()
    assert "stage5_paper2_eval_e" not in orchestrator.lower()
    assert 'f"receipts/seed_{seed}/summary.json"' in orchestrator
    assert "validate_autopsy_lock(lock, require_signature=True)" in orchestrator
    assert '"optimizer_steps": 0' in evaluator
    assert '"optimizer_steps": 0' in orchestrator


def test_dev1_condition_resumes_from_atomic_generation_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows_by_id = [
        ("arc_easy", "arc_easy"),
        ("arc_challenge", "arc_challenge"),
        ("mmlu", "mmlu"),
        *((f"gsm8k-{index}", "gsm8k") for index in range(8)),
        ("mbpp", "mbpp"),
        ("tier1", "tier1"),
    ]
    panel = [{"item_id": item_id, "battery": battery} for item_id, battery in rows_by_id]
    comparators = {
        item_id: {"item_id": item_id, "correct": False, "augmented_correct": False}
        for item_id, _battery in rows_by_id
    }

    class FakeGraph:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def scored(rows: list[dict[str, str]]) -> list[dict[str, object]]:
        return [
            {
                "item_id": row["item_id"],
                "battery": row["battery"],
                "augmented_correct": True,
                "prediction": "ok",
            }
            for row in rows
        ]

    calls: list[list[str]] = []

    def interrupted_generation(
        _graph: object,
        _tokenizer: object,
        rows: list[dict[str, str]],
        *,
        batch_size: int,
        emit_batch: object,
    ) -> list[dict[str, object]]:
        del batch_size
        calls.append([row["item_id"] for row in rows])
        emit_batch(scored(rows[:8]))
        raise RuntimeError("simulated backend loss")

    monkeypatch.setattr(autopsy_eval, "Stage2BTaskInferenceGraph", FakeGraph)
    monkeypatch.setattr(
        autopsy_eval,
        "score_mcq",
        lambda _graph, _tokenizer, rows, *, batch_size: scored(rows),
    )
    monkeypatch.setattr(autopsy_eval, "score_generation", interrupted_generation)
    with pytest.raises(RuntimeError, match="backend loss"):
        autopsy_eval._score_dev1_condition(
            wrapper=object(),
            tokenizer=object(),
            panel=panel,
            base_rows=comparators,
            initialization_rows=comparators,
            seed=0,
            gamma=0.05,
            mode="standard",
            condition="resume_test",
            private_dir=tmp_path,
            mcq_batch_size=8,
            generation_batch_size=2,
        )
    partial = tmp_path / "dev1__resume_test.partial.jsonl"
    assert partial.is_file()
    assert len(autopsy_eval.read_jsonl(partial)) == 11

    def completed_generation(
        _graph: object,
        _tokenizer: object,
        rows: list[dict[str, str]],
        *,
        batch_size: int,
        emit_batch: object,
    ) -> list[dict[str, object]]:
        del batch_size
        calls.append([row["item_id"] for row in rows])
        result = scored(rows)
        emit_batch(result)
        return result

    monkeypatch.setattr(autopsy_eval, "score_generation", completed_generation)
    rows, summary = autopsy_eval._score_dev1_condition(
        wrapper=object(),
        tokenizer=object(),
        panel=panel,
        base_rows=comparators,
        initialization_rows=comparators,
        seed=0,
        gamma=0.05,
        mode="standard",
        condition="resume_test",
        private_dir=tmp_path,
        mcq_batch_size=8,
        generation_batch_size=2,
    )
    expected_generation = [f"gsm8k-{index}" for index in range(8)] + ["mbpp", "tier1"]
    assert calls == [expected_generation, ["mbpp", "tier1"]]
    assert [row["item_id"] for row in rows] == [item_id for item_id, _battery in rows_by_id]
    assert summary["rows"] == len(rows_by_id)
    assert not partial.exists()
    assert (tmp_path / "dev1__resume_test.jsonl").is_file()
