from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.audit_paper2_phase2_e1_sparse_support import (
    MODEL_KEYS,
    QC_KIND,
    audit_sparse_support,
)
from training.paper2_phase2_e1_confirmation import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_sparse_support_qc_replaces_infinite_errors_with_explicit_counts(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    filename = "rows_000000_000001.pt"
    (private / "union").mkdir(parents=True)
    union = {
        "sample_indices": torch.tensor([0]),
        "union_ids": torch.tensor([[1, 2, 3]]),
        "union_mask": torch.tensor([[True, True, True]]),
    }
    torch.save(union, private / "union" / filename)

    union_scores = {}
    for model_key in MODEL_KEYS:
        root = private / "union_scores" / model_key
        root.mkdir(parents=True)
        score_path = root / filename
        torch.save(
            {
                "sample_indices": torch.tensor([0]),
                "audit_sample_indices": torch.tensor([0]),
                "candidate_log_probs": torch.tensor(
                    [[-1.0, float("-inf"), -2.0]], dtype=torch.float32
                ),
                "tail_log_probs": torch.tensor([-0.5], dtype=torch.float32),
                "full_log_probs_bfloat16": torch.log_softmax(
                    torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]), dim=-1
                ).to(torch.bfloat16),
            },
            score_path,
        )
        union_scores[model_key] = {
            "shards": [{"path": str(score_path), "sha256": sha256_file(score_path)}]
        }

    source_path = tmp_path / "source.json"
    source = {
        "status": "complete_frozen_unscored",
        "score_blind": True,
        "endpoint_models_loaded": False,
        "training_started": False,
        "acceptance_computed": False,
        "eal_computed": False,
        "retention_computed": False,
        "student_teacher_quality_aggregates_emitted": False,
        "optimizer_steps": 0,
        "manifest": {"full_logit_audit_samples": 1},
        "lattice": {"shards": [{"path": str(tmp_path / filename)}]},
        "union_scores": union_scores,
    }
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "qc.json"
    result = audit_sparse_support(
        source_summary=source_path,
        private_dir=private,
        output_summary=output,
    )

    assert result["kind"] == QC_KIND
    assert result["all_emitted_metrics_finite"] is True
    assert result["read_once_scoring_spent"] is False
    for model in result["models"].values():
        assert model["counts"]["support_mismatch_entries"] == 1
        assert model["counts"]["support_mismatch_fraction_all_entries"] == 1 / 3
        for metric in model["metrics"].values():
            assert metric["nan_count"] == 0
            assert metric["positive_infinity_count"] == 0
            assert metric["negative_infinity_count"] == 0
    rendered = output.read_text(encoding="utf-8")
    assert "Infinity" not in rendered
    assert "NaN" not in rendered


def test_sparse_qc_target_is_cpu_only_and_bootstrap_wired() -> None:
    cell = (ROOT / "colab/STAGE5_PAPER2_PHASE2_E1_SPARSE_QC_CELL.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "colab/run_stage5_paper2_phase2_e1_sparse_qc.py").read_text(
        encoding="utf-8"
    )
    assert "paper2_phase2_e1_sparse_qc_v1" in cell
    assert "no EAL no retention no acceptance no optimizer no training" in cell
    assert "eval.audit_paper2_phase2_e1_sparse_support" in runner
    assert "e1_readiness_v2.json" in runner
    for path in (
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py",
        ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.md",
    ):
        bootstrap = path.read_text(encoding="utf-8")
        assert "paper2_phase2_e1_sparse_qc" in bootstrap
        assert "STAGE5_PAPER2_PHASE2_E1_SPARSE_QC_CELL.py" in bootstrap
