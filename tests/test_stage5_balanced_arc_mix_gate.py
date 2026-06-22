from __future__ import annotations

from pathlib import Path

import pytest

from colab.run_stage5_balanced_arc_mix_gate import (
    arm_config,
    build_summary,
    checkpoint_run_id,
    is_safe_output_artifact,
    paired_mcq_diagnostics,
    selected_checkpoint,
)


def _arm(
    name: str,
    *,
    start: int,
    base: int,
    best: int,
    margin_delta: float = 0.0,
    prediction_shift: int = 0,
    calibration_ok: bool = True,
) -> dict:
    return {
        "arm": name,
        "base_arc": {"mean": {"correct": base, "total": 128, "accuracy": base / 128}},
        "phase1_start": {
            "checkpoint": "outputs/stage5/source/phase1/phase1_step_150.pt",
            "val": {},
            "arc": {"mean": {"correct": start, "total": 128, "accuracy": start / 128}},
        },
        "checkpoints": [],
        "best_checkpoint": {
            "checkpoint": f"outputs/stage5/child/{name}/phase1/phase1_step_50.pt",
            "arc": {"mean": {"correct": best, "total": 128, "accuracy": best / 128}},
            "comparison_to_base": {
                "helped": max(best - base, 0),
                "hurt": max(base - best, 0),
                "tied": 128 - abs(best - base),
                "prediction_changes": prediction_shift,
                "mean_margin_delta": margin_delta,
                "max_abs_prediction_count_delta": prediction_shift,
                "calibration_ok": calibration_ok,
            },
        },
    }


def test_checkpoint_run_id_from_stage5_path() -> None:
    assert (
        checkpoint_run_id("outputs/stage5/stage5_run/phase1/phase1_step_150.pt")
        == "stage5_run"
    )


def test_arm_config_defaults_to_arc_mix_no_distill() -> None:
    cfg = arm_config("arc_mix_nodistill_lr3e6")

    assert cfg.distill_enabled == "0"
    assert cfg.learning_rate == "3e-6"
    assert cfg.steps == "150"


def test_arm_config_exposes_competence_preserving_distill_arm() -> None:
    cfg = arm_config("arc_mix_response_w01_lr2e6")

    assert cfg.distill_enabled == "1"
    assert cfg.distill_weight == "0.10"
    assert cfg.learning_rate == "2e-6"
    assert cfg.steps == "150"


def test_build_mixed_train_uses_separate_arc_repeats(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_balanced_arc_mix_gate as module

    opus = tmp_path / "opus.jsonl"
    opus_val = tmp_path / "opus_val.jsonl"
    arc_challenge = tmp_path / "arc_challenge.jsonl"
    arc_easy = tmp_path / "arc_easy.jsonl"
    mixed = tmp_path / "mixed.jsonl"
    module.write_jsonl(opus, [{"id": "opus"}])
    module.write_jsonl(opus_val, [{"id": "val"}])
    module.write_jsonl(arc_challenge, [{"id": "challenge"}])
    module.write_jsonl(arc_easy, [{"id": "easy"}])

    monkeypatch.setattr(module, "prepare_opus", lambda: None)
    monkeypatch.setattr(module, "prepare_arc_sft", lambda config, output: None)
    monkeypatch.setattr(module, "OPUS_TRAIN_JSONL", opus)
    monkeypatch.setattr(module, "OPUS_VAL_JSONL", opus_val)
    monkeypatch.setattr(module, "ARC_CHALLENGE_TRAIN_JSONL", arc_challenge)
    monkeypatch.setattr(module, "ARC_EASY_TRAIN_JSONL", arc_easy)
    monkeypatch.setattr(module, "MIXED_TRAIN_JSONL", mixed)
    monkeypatch.setattr(module, "ARC_REPEAT", 2)
    monkeypatch.setattr(module, "ARC_CHALLENGE_REPEAT", 1)
    monkeypatch.setattr(module, "ARC_EASY_REPEAT", 3)

    summary = module.build_mixed_train()
    rows = module.read_jsonl(mixed)

    assert summary["arc_repeat"] == 2
    assert summary["arc_challenge_repeat"] == 1
    assert summary["arc_easy_repeat"] == 3
    assert summary["mixed_rows"] == 5
    assert summary["arc_prompt_style"] == "with_options"
    assert summary["arc_score_target"] == "label"
    assert sum(1 for row in rows if row["id"] == "challenge") == 1
    assert sum(1 for row in rows if row["id"] == "easy") == 3


def test_selected_checkpoint_accepts_top_level_checkpoint(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_balanced_arc_mix_gate as module

    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert selected_checkpoint({"checkpoint": "outputs/stage5/source/phase1.pt"}) == (
        tmp_path / "outputs" / "stage5" / "source" / "phase1.pt"
    )


def test_selected_checkpoint_reads_source_summary_with_windows_path(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_balanced_arc_mix_gate as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "outputs" / "stage5" / "suite" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        '{"checkpoint": "outputs/stage5/source/phase1/phase1_step_150.pt"}',
        encoding="utf-8",
    )

    assert selected_checkpoint({"source_summary": "outputs\\stage5\\suite\\summary.json"}) == (
        tmp_path / "outputs" / "stage5" / "source" / "phase1" / "phase1_step_150.pt"
    )


def test_build_summary_passes_when_arc_mix_lifts_proxy(tmp_path) -> None:
    payload = build_summary(
        source_summary=tmp_path / "balanced.json",
        source_payload={"status": "needs_competence_recovery"},
        resume_checkpoint=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
        data_summary={"mixed_rows": 10},
        arms=[_arm("arc_mix", start=71, base=72, best=73)],
    )

    assert payload["status"] == "proxy_lift"
    assert payload["passed"] is True
    assert payload["decision"] == "run_full_balanced_assessment"
    assert payload["blocked_reason"] is None
    assert payload["best_arm"]["arm"] == "arc_mix"
    assert "answer-calibration drift" in payload["objective_rationale"]["failure_mode"]
    assert "response-only" in payload["objective_rationale"]["proxy_hypothesis"]
    assert "label-only completions" in payload["objective_rationale"]["response_distillation_reason"]


def test_build_summary_fails_without_proxy_lift(tmp_path) -> None:
    payload = build_summary(
        source_summary=tmp_path / "balanced.json",
        source_payload={"status": "needs_competence_recovery"},
        resume_checkpoint=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
        data_summary={"mixed_rows": 10},
        arms=[_arm("arc_mix", start=71, base=72, best=70)],
    )

    assert payload["status"] == "no_proxy_lift"
    assert payload["passed"] is False
    assert payload["decision"] == "stop_and_revise_objective"
    assert payload["blocked_reason"]


def test_build_summary_blocks_lift_with_calibration_warning(tmp_path) -> None:
    payload = build_summary(
        source_summary=tmp_path / "balanced.json",
        source_payload={"status": "needs_competence_recovery"},
        resume_checkpoint=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
        data_summary={"mixed_rows": 10},
        arms=[
            _arm(
                "arc_mix",
                start=71,
                base=72,
                best=73,
                margin_delta=-1.0,
                prediction_shift=32,
                calibration_ok=False,
            )
        ],
    )

    assert payload["status"] == "proxy_lift_calibration_warning"
    assert payload["passed"] is False
    assert payload["decision"] == "stop_for_calibration_repair"
    assert "calibration" in payload["blocked_reason"]
    assert "Do not run the full paid assessment" in payload["next_step"]


def test_paired_mcq_diagnostics_reports_margin_and_prediction_shift(tmp_path) -> None:
    import colab.run_stage5_balanced_arc_mix_gate as module

    reference = tmp_path / "reference.jsonl"
    variant = tmp_path / "variant.jsonl"
    module.write_jsonl(
        reference,
        [
            {
                "id": "a",
                "aggregate": "mean",
                "prediction": "A",
                "answer": "A",
                "hit": True,
                "scores": {"A": 0.9, "B": 0.1},
            },
            {
                "id": "b",
                "aggregate": "mean",
                "prediction": "B",
                "answer": "A",
                "hit": False,
                "scores": {"A": 0.4, "B": 0.6},
            },
        ],
    )
    module.write_jsonl(
        variant,
        [
            {
                "id": "a",
                "aggregate": "mean",
                "prediction": "B",
                "answer": "A",
                "hit": False,
                "scores": {"A": 0.3, "B": 0.7},
            },
            {
                "id": "b",
                "aggregate": "mean",
                "prediction": "A",
                "answer": "A",
                "hit": True,
                "scores": {"A": 0.8, "B": 0.2},
            },
        ],
    )

    stats = paired_mcq_diagnostics(variant, reference)

    assert stats["helped"] == 1
    assert stats["hurt"] == 1
    assert stats["prediction_changes"] == 2
    assert stats["mean_margin_delta"] == pytest.approx(-0.2)
    assert stats["prediction_count_delta"] == {"A": 0, "B": 0}


def test_arc_mix_commit_artifacts_exclude_checkpoints(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_balanced_arc_mix_gate as module

    run_dir = tmp_path / "outputs" / "stage5" / "run"
    phase_dir = run_dir / "arc_mix_response_w01_lr2e6" / "phase1"
    phase_dir.mkdir(parents=True)
    safe_summary = run_dir / "summary.json"
    safe_report = run_dir / "summary.md"
    safe_log = run_dir / "train.log"
    checkpoint = phase_dir / "phase1_step_50.pt"
    weights = phase_dir / "adapter.safetensors"
    unknown = run_dir / "scratch.npy"
    safe_summary.write_text("{}", encoding="utf-8")
    safe_report.write_text("# report\n", encoding="utf-8")
    safe_log.write_text("log\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    weights.write_bytes(b"weights")
    unknown.write_bytes(b"array")

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)

    assert is_safe_output_artifact(safe_summary)
    assert not is_safe_output_artifact(checkpoint)
    assert not is_safe_output_artifact(weights)
    assert not is_safe_output_artifact(unknown)
    assert module.committable_run_files() == [
        "outputs/stage5/run/summary.json",
        "outputs/stage5/run/summary.md",
        "outputs/stage5/run/train.log",
    ]
