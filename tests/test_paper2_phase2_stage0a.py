from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from eval.cache_paper2_phase2_stage0a import (
    _score_candidates,
    apply_authoritative_topk,
    completed_union_score_shard,
)
from colab.run_stage5_paper2_phase2_stage0a import select_local_scratch
from training.paper2_phase2_stage0a import (
    STAGE0A_CONFIG,
    build_sparse_union,
    coarse_lattice_metrics,
    post_block_hidden_state_indices,
    select_stage0a_samples,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(row_id: str, stratum: str, length: int = 20) -> dict:
    return {
        "row_id": row_id,
        "document_id": f"doc-{row_id}",
        "stratum": stratum,
        "input_ids": list(range(length)),
    }


def test_stage0a_config_locks_models_geometry_and_dev_only_scope() -> None:
    assert STAGE0A_CONFIG["data_sha256"] == (
        "05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d"
    )
    assert STAGE0A_CONFIG["anchor_count"] == 50_000
    assert STAGE0A_CONFIG["boundary_sample_count"] == 200_000
    assert STAGE0A_CONFIG["horizons"] == [1, 2, 3, 4]
    assert STAGE0A_CONFIG["top_k"] == 128
    assert STAGE0A_CONFIG["selected_layer_ordinals_one_based"] == [16, 32, 44]
    assert STAGE0A_CONFIG["teacher_state_model"]["model"] == (
        "Qwen/Qwen2.5-14B-Instruct"
    )
    assert STAGE0A_CONFIG["models"]["teacher_32b"]["revision"] == (
        "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    )
    assert STAGE0A_CONFIG["training_started"] is False
    assert STAGE0A_CONFIG["frozen_evaluation_partitions_touched"] == []


def test_stage0a_sample_selection_is_deterministic_stratified_and_in_bounds() -> None:
    rows = [
        _row("g0", "general"),
        _row("g1", "general"),
        _row("c0", "code"),
        _row("c1", "code"),
    ]
    first = select_stage0a_samples(
        rows,
        anchors_per_stratum={"general": 4, "code": 4},
        horizons=(1, 2, 3, 4),
        seed=17,
    )
    second = select_stage0a_samples(
        rows,
        anchors_per_stratum={"general": 4, "code": 4},
        horizons=(1, 2, 3, 4),
        seed=17,
    )
    assert first == second
    assert len(first["anchors"]) == 8
    assert len(first["samples"]) == 32
    assert first["counts_by_stratum"] == {"code": 16, "general": 16}
    assert len({sample["sample_key"] for sample in first["samples"]}) == 32
    assert len(
        {
            (sample["row_index"], sample["prediction_position"])
            for sample in first["samples"]
        }
    ) == 32
    for sample in first["samples"]:
        row = rows[sample["row_index"]]
        assert sample["prediction_position"] + 1 == sample["state_position"]
        assert sample["state_position"] < len(row["input_ids"])
        assert 1 <= sample["horizon"] <= 4


def test_stage0a_selection_refuses_missing_document_ids_or_insufficient_samples() -> None:
    with pytest.raises(ValueError, match="document_id"):
        select_stage0a_samples(
            [_row("a", "general"), {**_row("b", "code"), "document_id": ""}],
            anchors_per_stratum={"general": 1, "code": 1},
            horizons=(1, 2, 3, 4),
            seed=1,
        )
    with pytest.raises(ValueError, match="insufficient eligible anchors"):
        select_stage0a_samples(
            [_row("a", "general", length=5), _row("b", "code", length=5)],
            anchors_per_stratum={"general": 2, "code": 2},
            horizons=(1, 2, 3, 4),
            seed=1,
        )


def test_post_block_layer_ordinals_map_to_hidden_state_tuple_indices() -> None:
    assert post_block_hidden_state_indices(
        num_hidden_layers=48, ordinals_one_based=(16, 32, 44)
    ) == (16, 32, 44)
    with pytest.raises(ValueError, match="outside"):
        post_block_hidden_state_indices(
            num_hidden_layers=48, ordinals_one_based=(16, 49)
        )


def test_sparse_union_is_stable_and_capped_by_model_topk_union() -> None:
    union, mask = build_sparse_union(
        [
            torch.tensor([[4, 1, 9], [8, 2, 3]]),
            torch.tensor([[9, 2, 5], [3, 7, 8]]),
        ]
    )
    assert union.tolist() == [[1, 2, 4, 5, 9], [2, 3, 7, 8, -1]]
    assert mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, True, False],
    ]


def test_coarse_lattice_metrics_separates_agreement_gap_and_teachability() -> None:
    # Candidate ids are [0, 1], followed by one exact tail bucket.
    student = torch.log(torch.tensor([0.60, 0.20, 0.20]))
    teacher_7b = torch.log(torch.tensor([0.10, 0.70, 0.20]))
    teacher_14b = torch.log(torch.tensor([0.10, 0.70, 0.20]))
    result = coarse_lattice_metrics(
        student_log_probs=student,
        teacher_log_probs=[teacher_7b, teacher_14b],
        student_topk_mask=torch.tensor([True, False, False]),
    )
    assert result["teacher_count"] == 2
    assert result["normalized_teacher_agreement"] == pytest.approx(1.0)
    assert result["student_gap_coarse_kl"] > 0
    assert result["teachability_student_topk"] == pytest.approx(0.10)
    assert result["teacher_tail_mass"] == pytest.approx(0.20)


def test_stage0a_launcher_and_runner_enforce_resume_and_no_training_contracts() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(
        encoding="utf-8"
    )
    cell = (ROOT / "colab/STAGE5_PAPER2_PHASE2_STAGE0A_CELL.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "colab/run_stage5_paper2_phase2_stage0a.py").read_text(
        encoding="utf-8"
    )
    evaluator = (ROOT / "eval/cache_paper2_phase2_stage0a.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "paper2_phase2_stage0a",
        "STAGE5_PAPER2_PHASE2_STAGE0A_CELL.py",
        "paper2_phase2_stage0a_v1",
        "DEV-C only sparse lattice and teacher states no optimizer no training",
    ):
        assert marker in bootstrap
    assert "minimum_vram_mib=70000" in cell
    assert "A100-SXM4-80GB" in cell
    assert "private/stage0a" in runner
    assert "stage0a_status.json" in runner
    assert "training_started" in runner and "optimizer_steps" in runner
    assert "torch.optim" not in runner
    assert "training/train_" not in runner
    assert "completed_model_shard" in evaluator
    assert "stage0a_union_resume_complete" in evaluator
    assert "atomic_torch_save" in evaluator
    assert "teacher_forward_passes" in evaluator
    assert "frozen_evaluation_partitions_touched" in evaluator


def test_stage0a_machine_config_matches_confirmed_v1d_constants() -> None:
    path = ROOT / "training/paper2_phase2_dc2_constants.json"
    raw = path.read_bytes()
    constants = json.loads(raw)
    assert constants["status"] == "confirmed_by_v1d"
    assert STAGE0A_CONFIG["dc2_constants_sha256"] == hashlib.sha256(raw).hexdigest()


def test_union_rescoring_reproduces_exact_cached_candidate_probabilities() -> None:
    torch.manual_seed(7)
    head = torch.randn(11, 5, dtype=torch.float32)
    hidden = torch.randn(2, 5, dtype=torch.float32)
    logits = hidden @ head.T
    log_partition = torch.logsumexp(logits, dim=-1)
    candidate_ids = torch.tensor([[0, 3, 8, -1], [1, 2, 7, 10]])
    candidate_mask = candidate_ids >= 0
    scores, tail = _score_candidates(
        hidden=hidden,
        head=head,
        candidate_ids=candidate_ids,
        candidate_mask=candidate_mask,
        log_partition=log_partition,
        device="cpu",
    )
    expected = torch.full(candidate_ids.shape, float("-inf"))
    for row in range(candidate_ids.shape[0]):
        ids = candidate_ids[row][candidate_mask[row]]
        expected[row, candidate_mask[row]] = torch.log_softmax(logits[row], dim=0)[ids]
    assert torch.allclose(
        scores[candidate_mask].float(), expected[candidate_mask], atol=0.02, rtol=0
    )
    total_mass = torch.where(candidate_mask, scores.float().exp(), 0).sum(dim=-1)
    assert torch.allclose(total_mass + tail.exp(), torch.ones(2), atol=0.01, rtol=0)


def test_completed_union_score_shard_validates_its_union_source(tmp_path: Path) -> None:
    path = tmp_path / "score.pt"
    torch.save(
        {
            "kind": "paper2_phase2_stage0a_union_score_shard",
            "score_schema_version": 2,
            "model_key": "teacher_7b",
            "row_start": 0,
            "row_stop": 8,
            "union_sha256": "abc",
        },
        path,
    )
    assert completed_union_score_shard(
        path,
        model_key="teacher_7b",
        row_start=0,
        row_stop=8,
        union_sha256="abc",
    )
    assert not completed_union_score_shard(
        path,
        model_key="teacher_7b",
        row_start=0,
        row_stop=8,
        union_sha256="different",
    )


def test_authoritative_topk_replaces_kernel_drift_and_recomputes_tail() -> None:
    candidate = torch.log(torch.tensor([0.50, 0.25, 0.10, 0.05]))
    union_ids = torch.tensor([1, 3, 5, 8])
    union_mask = torch.tensor([True, True, True, True])
    cached_ids = torch.tensor([1, 5])
    cached = torch.log(torch.tensor([0.48, 0.12]))
    corrected, tail, diagnostics = apply_authoritative_topk(
        candidate_log_probs=candidate,
        union_ids=union_ids,
        union_mask=union_mask,
        cached_topk_ids=cached_ids,
        cached_topk_log_probs=cached,
    )
    assert torch.equal(corrected[[0, 2]], cached)
    assert diagnostics["log_probability_max_abs_error"] > 0
    assert diagnostics["probability_max_abs_error"] == pytest.approx(0.02)
    assert corrected.exp().sum() + tail.exp() == pytest.approx(1.0)


def test_stage0a_prefers_large_named_local_scratch(tmp_path: Path) -> None:
    root = tmp_path / "content"
    scratch = tmp_path / "local-scratch"
    root.mkdir()
    scratch.mkdir()
    gib = 1024**3
    listing = "\n".join(
        [
            "Filesystem Mounted-on 1B-blocks Avail",
            f"overlay {root} {235 * gib} {70 * gib}",
            f"/dev/nvme1n1 {scratch} {368 * gib} {360 * gib}",
        ]
    )
    assert select_local_scratch(listing) == scratch
