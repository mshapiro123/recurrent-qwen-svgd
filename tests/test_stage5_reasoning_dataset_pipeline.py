from __future__ import annotations

import subprocess

import colab.run_stage5_reasoning_dataset_pipeline as module


def test_pipeline_audit_env_sets_safe_defaults(monkeypatch) -> None:
    monkeypatch.setattr(module, "AUDIT_RUN_ID", "audit")
    monkeypatch.setattr(module, "PUSH_RESULTS", False)

    env = module.audit_env()

    assert env["STAGE5_DATASET_AUDIT_RUN_ID"] == "audit"
    assert "opus47_sft" in env["STAGE5_DATASET_AUDIT_KEYS"]
    assert "fable5_pi_agent" in env["STAGE5_DATASET_AUDIT_KEYS"]
    assert env["STAGE5_DATASET_AUDIT_LIMIT"] == "1000"
    assert env["STAGE5_DATASET_AUDIT_PUSH"] == "0"


def test_pipeline_defaults_to_planning_next_action() -> None:
    assert module.EXECUTE_NEXT is False


def test_pipeline_next_action_env_points_to_audit_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "AUDIT_RUN_ID", "audit")
    monkeypatch.setattr(module, "NEXT_ACTION_RUN_ID", "next")
    monkeypatch.setattr(module, "EXECUTE_NEXT", True)
    monkeypatch.setattr(module, "MAX_ACTIONS", 1)

    env = module.next_action_env()

    assert env["STAGE5_ARC_AGI_NEXT_ACTION_RUN_ID"] == "next"
    assert env["STAGE5_ARC_AGI_NEXT_ACTION_SOURCE_SUMMARY"] == "outputs/stage5/audit/summary.json"
    assert env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] == "1"
    assert env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "1"


def test_pipeline_summary_records_planned_action_without_execute(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_ID", "pipe")
    monkeypatch.setattr(module, "AUDIT_RUN_ID", "audit")
    monkeypatch.setattr(module, "NEXT_ACTION_RUN_ID", "next")
    monkeypatch.setattr(module, "EXECUTE_NEXT", False)

    payload = module.build_summary(
        audit_payload={"status": "ok", "next_step": "Prepare filtered small-train mix."},
        next_payload={
            "steps": [
                {
                    "selected_action": {
                        "name": "Run audited modified-Opus recurrent fine-tune",
                    }
                }
            ]
        },
    )

    assert payload["status"] == "next_action_planned"
    assert payload["audit_summary"] == "outputs/stage5/audit/summary.json"
    assert payload["next_action_summary"] == "outputs/stage5/next/summary.json"
    assert payload["next_executed"] is False
    assert payload["next_step"] == "Run audited modified-Opus recurrent fine-tune"


def test_pipeline_summary_reports_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_ID", "pipe")
    monkeypatch.setattr(module, "AUDIT_RUN_ID", "audit")
    monkeypatch.setattr(module, "NEXT_ACTION_RUN_ID", "next")

    payload = module.build_summary(audit_payload=None, next_payload=None, error="boom")

    assert payload["status"] == "pipeline_failed"
    assert payload["error"] == "boom"
    assert "Inspect pipeline logs" in payload["next_step"]


def test_pipeline_commit_results_adds_only_safe_text_artifacts(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "pipe"
    audit_dir = tmp_path / "outputs" / "stage5" / "audit"
    next_dir = tmp_path / "outputs" / "stage5" / "next"
    for directory in (run_dir, audit_dir, next_dir):
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("{}", encoding="utf-8")
        (directory / "checkpoint.pt").write_bytes(b"checkpoint")

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], *, env=None, check: bool = True, log_name: str | None = None):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "AUDIT_RUN_ID", "audit")
    monkeypatch.setattr(module, "NEXT_ACTION_RUN_ID", "next")
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results()

    add_commands = [cmd for cmd in commands if cmd[:3] == ["git", "add", "-f"]]
    assert add_commands
    added = " ".join(" ".join(cmd) for cmd in add_commands)
    assert "summary.json" in added
    assert "checkpoint.pt" not in added
