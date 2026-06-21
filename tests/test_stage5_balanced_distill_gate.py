from __future__ import annotations

from pathlib import Path

from colab.run_stage5_balanced_distill_gate import (
    ArmConfig,
    arm_config,
    build_gate_summary,
    checkpoint_run_id,
    child_env,
)


def test_checkpoint_run_id_from_stage5_path() -> None:
    assert (
        checkpoint_run_id("outputs/stage5/run_abc/phase1/phase1_step_150.pt")
        == "run_abc"
    )


def test_arm_config_exposes_response_distill_preset() -> None:
    cfg = arm_config("response_w005_lr3e6")

    assert cfg.distill_enabled == "1"
    assert cfg.distill_weight == "0.05"
    assert cfg.distill_on == "response"
    assert cfg.learning_rate == "3e-6"


def test_child_env_resumes_selected_checkpoint_and_disables_child_push(monkeypatch) -> None:
    monkeypatch.setenv("EXISTING", "1")
    cfg = ArmConfig(
        name="toy",
        learning_rate="1e-6",
        beta="0.12",
        steps="10",
        save_every="5",
        distill_enabled="1",
        distill_weight="0.05",
        distill_temperature="2.0",
        distill_on="response",
    )

    env = child_env(
        cfg,
        child_run_id="child",
        resume_from=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
    )

    assert env["EXISTING"] == "1"
    assert env["STAGE5_RUN_ID"] == "child"
    assert env["STAGE5_RESUME_FROM"].replace("\\", "/").endswith(
        "outputs/stage5/source/phase1/phase1_step_150.pt"
    )
    assert env["STAGE5_PUSH"] == "0"


def test_build_gate_summary_passes_on_proxy_lift(tmp_path) -> None:
    summary = build_gate_summary(
        source_summary=tmp_path / "source.json",
        source_payload={"status": "needs_competence_recovery"},
        resume_checkpoint=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
        arm_summaries=[
            {
                "arm": "response",
                "checkpoint": "outputs/stage5/child/phase1/phase1_step_50.pt",
                "start_arc": {"correct": 71, "total": 128, "accuracy": 71 / 128},
                "base_arc": {"correct": 72, "total": 128, "accuracy": 72 / 128},
                "best_arc": {"correct": 73, "total": 128, "accuracy": 73 / 128},
                "lift_vs_start": 2,
                "gap_vs_base": 1,
            }
        ],
    )

    assert summary["status"] == "proxy_lift"
    assert summary["passed"] is True
    assert "full ARC-Easy and ARC-Challenge" in summary["next_step"]


def test_build_gate_summary_fails_without_proxy_lift(tmp_path) -> None:
    summary = build_gate_summary(
        source_summary=tmp_path / "source.json",
        source_payload={"status": "needs_competence_recovery"},
        resume_checkpoint=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
        arm_summaries=[
            {
                "arm": "response",
                "checkpoint": "outputs/stage5/child/phase1/phase1_step_50.pt",
                "start_arc": {"correct": 71, "total": 128, "accuracy": 71 / 128},
                "base_arc": {"correct": 72, "total": 128, "accuracy": 72 / 128},
                "best_arc": {"correct": 71, "total": 128, "accuracy": 71 / 128},
                "lift_vs_start": 0,
                "gap_vs_base": -1,
            }
        ],
    )

    assert summary["status"] == "no_proxy_lift"
    assert summary["passed"] is False
