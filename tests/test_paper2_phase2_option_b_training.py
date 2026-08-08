from __future__ import annotations

import json
from pathlib import Path

import torch

from training.run_paper2_phase2_option_b import (
    EXPECTED_STATUS,
    canonical_json_sha256,
    learning_rate_at_step,
    load_training_lock,
    merge_caches,
    normalized_lf_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _cache(rows: int, width: int) -> dict:
    return {
        "documents": [f"doc-{index}" for index in range(rows)],
        "strata": ["general"] * rows,
        "positions": torch.arange(rows),
        "student_hidden": torch.zeros(rows, 4, 2, dtype=torch.bfloat16),
        "target_centered_raw": torch.zeros(rows, 8, 2, dtype=torch.bfloat16),
        "candidate_ids": torch.zeros(rows, 4, width, dtype=torch.int32),
        "candidate_mask": torch.ones(rows, 4, width, dtype=torch.bool),
        "base_log_probs": torch.zeros(rows, 4, width, dtype=torch.bfloat16),
        "base_tail": torch.zeros(rows, 4, dtype=torch.bfloat16),
        "teacher_log_probs": torch.zeros(rows, 4, width, dtype=torch.bfloat16),
        "teacher_tail": torch.zeros(rows, 4, dtype=torch.bfloat16),
        "teacher_topk_ids": torch.zeros(rows, 4, 128, dtype=torch.int32),
        "teacher_topk_log_probs": torch.zeros(rows, 4, 128, dtype=torch.bfloat16),
        "whiten_basis": torch.eye(2),
        "whiten_eigenvalues": torch.ones(2),
        "decoder_weight_alpha_0p5": torch.eye(2),
        "decoder_bias": torch.zeros(2),
        "source": {"rows": rows},
    }


def test_post_generation_lock_is_complete_and_rule_inventory_rides() -> None:
    registration, inventory = load_training_lock()
    assert registration["status"] == EXPECTED_STATUS
    assert registration["training_authorized"] is True
    assert registration["post_generation_hash_amendment"]["recorded_splice_step"] == 4_000
    assert len(inventory["rules"]) == 18


def test_teacher_summary_integrity_is_transport_stable_and_semantic(tmp_path: Path) -> None:
    registration = json.loads(
        (ROOT / "training/paper2_phase2_option_b_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    transport = registration["teacher_cache_summary_transport_erratum"]
    source = ROOT / "outputs/stage5/stage5_paper2_phase2_option_b_teacher_cache_20260806/summary.json"
    payload = source.read_bytes().replace(b"\r\n", b"\n")
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_path.write_bytes(payload)
    crlf_path.write_bytes(payload.replace(b"\n", b"\r\n"))
    assert normalized_lf_sha256(lf_path) == transport["git_lf_sha256"]
    assert normalized_lf_sha256(crlf_path) == transport["git_lf_sha256"]
    parsed = json.loads(lf_path.read_text(encoding="utf-8"))
    assert canonical_json_sha256(parsed) == transport["canonical_json_sha256"]
    parsed["selected_anchor_count"] += 1
    assert canonical_json_sha256(parsed) != transport["canonical_json_sha256"]


def test_learning_rate_matches_locked_warmup_plateau_and_cooldown() -> None:
    constants = json.loads(
        (ROOT / "training/paper2_phase2_option_b_preregistration.json").read_text(
            encoding="utf-8"
        )
    )["fixed_constants"]
    assert learning_rate_at_step(0, constants) == 0.0
    assert learning_rate_at_step(100, constants) == 1.5e-4
    assert learning_rate_at_step(200, constants) == 3e-4
    assert learning_rate_at_step(18_000, constants) == 3e-4
    assert abs(learning_rate_at_step(20_000, constants) - 3e-5) < 1e-12


def test_cache_merge_preserves_order_and_pads_only_sparse_candidate_axis() -> None:
    old = _cache(2, 3)
    new = _cache(3, 5)
    new["documents"] = [f"new-{index}" for index in range(3)]
    merged = merge_caches(old, new)
    assert merged["documents"] == ["doc-0", "doc-1", "new-0", "new-1", "new-2"]
    assert merged["candidate_ids"].shape == (5, 4, 5)
    assert merged["candidate_mask"][:2, :, 3:].sum() == 0
    assert torch.isneginf(merged["base_log_probs"][:2, :, 3:]).all()


def test_launcher_is_wired_with_resume_and_scientific_boundaries() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (ROOT / "colab/STAGE5_PAPER2_PHASE2_OPTION_B_CELL.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "colab/run_stage5_paper2_phase2_option_b.py").read_text(
        encoding="utf-8"
    )
    training = (ROOT / "training/run_paper2_phase2_option_b.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "paper2_phase2_option_b_v3",
        "four A2 endpoint arms fresh AdamW state exact step 4000 splice",
        "fixed evaluation excluded from both training populations",
        "teacher summary normalized Git-LF plus canonical JSON integrity",
    ):
        assert marker in bootstrap or marker in cell
    assert "checkpoint_step_" in training
    assert "expanded_train_indices" in training
    assert "fixed_evaluation" in training
    assert "DRIVE_A2" in runner
    assert "private/option_b" in runner
    assert "trainable_state_digest" in runner
    assert "source_checkpoint_semantic_digests" in training
    assert "endpoint_state_digest" in training
    assert "a2_noop_resume_preserved" in (
        ROOT / "training/run_paper2_phase2_a2.py"
    ).read_text(encoding="utf-8")
