from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eval.cache_paper2_phase2_stage0a import _materialize_weight

from training.paper2_phase2_option_b import (
    build_anchor_admission_rows,
    build_cache_config,
    choose_full_anchor_count,
    load_locked_registration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_locked_option_b_config_preserves_teacher_and_training_boundaries(tmp_path: Path) -> None:
    registration = load_locked_registration()
    data = tmp_path / "rows.jsonl"
    data.write_text('{"document_id":"new","input_ids":[1,2,3,4,5]}\n', encoding="utf-8")
    config = build_cache_config(
        registration=registration,
        data_path=data,
        anchor_count=100_000,
        run_id="test_full",
    )
    assert config["anchor_count"] == 100_000
    assert config["boundary_sample_count"] == 400_000
    assert config["teacher_14b_state_coverage_policy"] == "all_admitted_anchors"
    assert config["training_started"] is False
    assert config["optimizer_steps"] == 0


def test_storage_preflight_chooses_target_then_floor_and_never_lower() -> None:
    target = choose_full_anchor_count(
        target=140_000,
        floor=100_000,
        pilot_anchors=500,
        pilot_total_bytes=2_000_000,
        pilot_fixed_bytes=1_000_000,
        scratch_free_bytes=1_000_000_000,
        drive_free_bytes=1_000_000_000,
    )
    assert target["selected_anchor_count"] == 140_000
    floor = choose_full_anchor_count(
        target=140_000,
        floor=100_000,
        pilot_anchors=500,
        pilot_total_bytes=2_000_000,
        pilot_fixed_bytes=1_000_000,
        scratch_free_bytes=300_000_000,
        drive_free_bytes=300_000_000,
    )
    assert floor["selected_anchor_count"] == 100_000
    with pytest.raises(RuntimeError, match="100,000-anchor floor"):
        choose_full_anchor_count(
            target=140_000,
            floor=100_000,
            pilot_anchors=500,
            pilot_total_bytes=2_000_000,
            pilot_fixed_bytes=1_000_000,
            scratch_free_bytes=10,
            drive_free_bytes=10,
        )


def test_admission_ledger_separates_states_from_label_cascade() -> None:
    samples = [
        {
            "anchor_index": 0,
            "sample_index": horizon - 1,
            "document_id": "doc-a",
            "stratum": "general",
            "horizon": horizon,
        }
        for horizon in range(1, 5)
    ]
    rows = build_anchor_admission_rows(samples, {1, 3})
    assert len(rows) == 1
    assert rows[0]["teacher_14b_states_by_horizon"] == {
        "1": True,
        "2": True,
        "3": True,
        "4": True,
    }
    assert rows[0]["label_tier_admission"]["teacher_7b_by_horizon"] == {
        "1": True,
        "2": True,
        "3": True,
        "4": True,
    }
    assert rows[0]["label_tier_admission"]["teacher_14b_by_horizon"] == {
        "1": True,
        "2": True,
        "3": True,
        "4": True,
    }
    assert rows[0]["label_tier_admission"]["teacher_32b_by_horizon"] == {
        "1": False,
        "2": True,
        "3": False,
        "4": True,
    }


def test_launcher_is_teacher_only_and_bootstrap_target_is_wired() -> None:
    runner = (ROOT / "colab/run_stage5_paper2_phase2_option_b_teacher_cache.py").read_text(
        encoding="utf-8"
    )
    cell = (ROOT / "colab/STAGE5_PAPER2_PHASE2_OPTION_B_TEACHER_CACHE_CELL.py").read_text(
        encoding="utf-8"
    )
    evaluator = (ROOT / "eval/cache_paper2_phase2_stage0a.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    for text in (runner, cell):
        assert "no optimizer no training" in text
        assert "all-admitted-anchor 14B states" in text
    assert "--config_json" in evaluator
    assert "ACTIVE_CONFIG" in evaluator
    assert "paper2_phase2_option_b_teacher_cache" in bootstrap
    assert "STAGE5_PAPER2_PHASE2_OPTION_B_TEACHER_CACHE_CELL.py" in bootstrap
    assert "torch.optim" not in runner
    assert "STAGE5_PHASE2_OPTION_B_OFFLOAD_32B" in runner
    assert "a100_40gb_32b_accelerate_offload" in runner
    assert "memory >= 38000" in cell
    assert "paper2_phase2_option_b_teacher_cache_v4" in cell
    assert "pinned bf16 32B Accelerate offload on CUDA" in cell
    assert 'MIN_SCRATCH_TOTAL_GIB"] = "200"' in cell
    assert 'MIN_SCRATCH_FREE_GIB"] = "150"' in cell
    assert 'OPTION_B_PREFLIGHT_ONLY"] = "1"' in cell
    assert "complete_preflight_full_cache_not_started" in runner
    assert '"full_cache_started": False' in runner
    assert not (ROOT / "colab/run_stage5_paper2_phase2_option_b.py").exists()


def test_materialize_weight_keeps_resident_tensor_exact() -> None:
    module = torch.nn.Linear(3, 2, bias=False)
    assert _materialize_weight(module).data_ptr() == module.weight.data_ptr()


def test_materialize_weight_reads_accelerate_offload_map() -> None:
    module = torch.nn.Linear(3, 2, bias=False, device="meta")
    expected = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    module._hf_hook = SimpleNamespace(weights_map={"weight": expected})
    assert torch.equal(_materialize_weight(module), expected)


def test_40gb_mode_is_an_execution_only_amendment() -> None:
    amendment = (
        ROOT / "docs/PAPER2_PHASE2_OPTION_B_A10040_OFFLOAD_AMENDMENT_20260806.md"
    ).read_text(encoding="utf-8")
    for marker in (
        "model IDs, revisions, bf16 dtype",
        "no quantization, optimizer, training, or threshold change",
        "complete `hf_device_map`",
    ):
        assert marker in amendment
