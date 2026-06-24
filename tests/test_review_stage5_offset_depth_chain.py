from __future__ import annotations

import json
from pathlib import Path

from colab import review_stage5_offset_depth_chain as module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def chain_payload(**overrides) -> dict:
    payload = {
        "run_id": "chain",
        "kind": "stage5_arc_mix_offset_then_depth_chain",
        "status": "depth_completed",
        "source_summary": "outputs/stage5/source/summary.json",
        "offset_assessment": {"passed": True},
        "depth_launched": True,
        "depth_returncode": 0,
        "depth_summary": "outputs/stage5/depth/summary.json",
        "post_depth_debiased_summary": "outputs/stage5/post/summary.json",
        "post_depth_debiased_assessment": {"passed": True},
    }
    payload.update(overrides)
    return payload


def test_review_stops_when_offset_not_confirmed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    summary = write_json(
        tmp_path / "outputs" / "stage5" / "chain" / "summary.json",
        chain_payload(offset_assessment={"passed": False}, depth_launched=False),
    )

    review = module.classify(summary, module.read_json(summary))

    assert review["action"] == "stop_offset_not_confirmed"
    assert review["dense_control_ready"] is False


def test_review_flags_depth_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    summary = write_json(
        tmp_path / "outputs" / "stage5" / "chain" / "summary.json",
        chain_payload(depth_returncode=1, depth_summary=None, post_depth_debiased_summary=None),
    )

    review = module.classify(summary, module.read_json(summary))

    assert review["action"] == "inspect_depth_failure"
    assert review["depth_returncode"] == 1


def test_review_requests_post_depth_gate_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    summary = write_json(
        tmp_path / "outputs" / "stage5" / "chain" / "summary.json",
        chain_payload(post_depth_debiased_summary=None, post_depth_debiased_assessment=None),
    )
    write_json(
        tmp_path / "outputs" / "stage5" / "depth" / "summary.json",
        {"kind": "stage5_balanced_arc_mix_gate", "data": {"mixed_train_jsonl": "data/mix.jsonl"}},
    )

    review = module.classify(summary, module.read_json(summary))

    assert review["action"] == "run_post_depth_debiased_gate"
    assert review["mixed_train_jsonl"] == "data/mix.jsonl"


def test_review_blocks_dense_control_when_positive_sft_source_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    summary = write_json(tmp_path / "outputs" / "stage5" / "chain" / "summary.json", chain_payload())
    write_json(
        tmp_path / "outputs" / "stage5" / "depth" / "summary.json",
        {"kind": "stage5_balanced_arc_mix_gate", "data": {"mixed_train_jsonl": "data/mix.jsonl"}},
    )

    review = module.classify(summary, module.read_json(summary))

    assert review["action"] == "dense_control_blocked_missing_positive_sft_source"
    assert review["mixed_train_jsonl"] == "data/mix.jsonl"
    assert review["positive_sft_locator"] == {}
    assert review["dense_control_env"] == {}


def test_review_emits_dense_control_env_when_source_and_extra_rows_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = write_json(
        tmp_path / "outputs" / "stage5" / "source" / "summary.json",
        {
            "kind": "stage5_capability_ladder_trace_collection",
            "dataset": {"source_positive_sft": "data/curriculum/source/positive_sft.jsonl"},
        },
    )
    summary = write_json(
        tmp_path / "outputs" / "stage5" / "chain" / "summary.json",
        chain_payload(source_summary=module.path_for_cli(source)),
    )
    write_json(
        tmp_path / "outputs" / "stage5" / "depth" / "summary.json",
        {"kind": "stage5_balanced_arc_mix_gate", "data": {"mixed_train_jsonl": "data/mix.jsonl"}},
    )

    review = module.classify(summary, module.read_json(summary))

    assert review["action"] == "run_dense_mcq_trace_sft_control"
    assert review["dense_control_ready"] is True
    assert review["dense_control_env"] == {
        "STAGE5_CURRENT_A100_TARGET": "dense_mcq_trace_sft_control",
        "STAGE5_DENSE_MCQ_SOURCE_SUMMARY": "outputs/stage5/source/summary.json",
        "STAGE5_DENSE_MCQ_RECURRENT_BENCHMARK_SUMMARY": "outputs/stage5/post/summary.json",
        "STAGE5_DENSE_MCQ_EXTRA_TRAIN_JSONL": "data/mix.jsonl",
        "STAGE5_DENSE_MCQ_RUN_ID": "stage5_dense_control_after_chain",
    }


def test_latest_summary_selects_newest_chain_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    old = write_json(tmp_path / "outputs" / "stage5" / "old" / "summary.json", chain_payload(run_id="old"))
    new = write_json(tmp_path / "outputs" / "stage5" / "new" / "summary.json", chain_payload(run_id="new"))
    old.touch()
    new.touch()

    assert module.latest_summary() == new


def test_write_review_updates_current_pointer(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = module.write_review(
        {
            "kind": "stage5_arc_mix_offset_depth_review",
            "reviewed_summary": "outputs/stage5/chain/summary.json",
            "reviewed_status": "depth_completed",
            "action": "run_dense_mcq_trace_sft_control",
            "next_step": "Run dense control.",
            "offset_passed": True,
            "depth_launched": True,
            "depth_returncode": 0,
            "depth_summary_visible": True,
            "post_depth_debiased_passed": True,
            "mixed_train_jsonl": "data/mix.jsonl",
            "positive_sft_locator": {"source_summary": "outputs/stage5/source/summary.json"},
            "dense_control_ready": True,
            "dense_control_env": {"STAGE5_CURRENT_A100_TARGET": "dense_mcq_trace_sft_control"},
        },
        run_id="review",
    )

    assert output == tmp_path / "outputs" / "stage5" / "review" / "summary.json"
    assert (tmp_path / "outputs" / "stage5" / "review" / "summary.md").exists()
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip() == (
        "outputs/stage5/review/summary.json"
    )
