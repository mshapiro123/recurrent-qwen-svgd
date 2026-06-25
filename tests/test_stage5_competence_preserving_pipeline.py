from __future__ import annotations

import subprocess


def test_child_env_defaults_to_distilled_easy_weighted_arc_mix(monkeypatch) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    monkeypatch.delenv("STAGE5_ARC_MIX_ARMS", raising=False)
    monkeypatch.delenv("STAGE5_ARC_MIX_ARC_EASY_REPEAT", raising=False)
    monkeypatch.delenv("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", raising=False)

    env = module.child_env()

    assert (
        env["STAGE5_ARC_MIX_SOURCE_SUMMARY"].replace("\\", "/")
        == "outputs/stage5/stage5_debiased_benchmark_assessment_20260625_121302/summary.json"
    )
    assert env["STAGE5_ARC_MIX_ARMS"] == "arc_mix_response_w01_lr2e6,arc_mix_response_w02_lr2e6"
    assert env["STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT"] == "2"
    assert env["STAGE5_ARC_MIX_ARC_EASY_REPEAT"] == "4"
    assert env["STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY"].replace("\\", "/").endswith("_arc_mix/summary.json")


def test_arc_mix_passed_accepts_lift_or_base_match() -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    assert module.arc_mix_passed({"status": "proxy_lift"}) is True
    assert module.arc_mix_passed({"status": "proxy_matches_base"}) is True
    assert module.arc_mix_passed({"status": "proxy_lift_calibration_warning"}) is False
    assert module.arc_mix_passed({"status": "proxy_matches_base_calibration_warning"}) is False
    assert module.arc_mix_passed({"status": "no_proxy_lift"}) is False
    assert module.arc_mix_passed(None) is False


def test_preflight_checkpoint_restore_uses_selected_checkpoint(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    checkpoint = tmp_path / "outputs" / "stage5" / "run" / "phase1" / "phase1_step_75.pt"
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PREFLIGHT_CHECKPOINT_RESTORE", True)
    monkeypatch.setattr(module, "selected_checkpoint", lambda payload: checkpoint)
    monkeypatch.setattr(module, "checkpoint_run_id", lambda path: "run")
    monkeypatch.setattr(
        module,
        "restore_checkpoint_if_needed",
        lambda path, *, run_id: calls.append((str(path), run_id)),
    )

    assert module.preflight_checkpoint_restore({"checkpoint": "unused"}) == checkpoint
    assert calls == [(str(checkpoint), "run")]


def test_preflight_checkpoint_restore_can_be_disabled(monkeypatch) -> None:
    import pytest
    import colab.run_stage5_competence_preserving_pipeline as module

    monkeypatch.setattr(module, "PREFLIGHT_CHECKPOINT_RESTORE", False)
    monkeypatch.setattr(module, "selected_checkpoint", lambda payload: pytest.fail("should not select checkpoint"))

    assert module.preflight_checkpoint_restore({}) is None


def test_build_summary_waits_for_full_assessment_after_arc_mix_passes() -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    payload = module.build_summary(
        source_payload={"status": "needs_competence_recovery"},
        arc_payload={"status": "proxy_lift"},
        full_payload=None,
    )

    assert payload["status"] == "full_assessment_missing"
    assert payload["arc_mix_status"] == "proxy_lift"


def test_build_summary_reports_full_assessment_status() -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    payload = module.build_summary(
        source_payload={"status": "needs_competence_recovery"},
        arc_payload={"status": "proxy_lift"},
        full_payload={"status": "balanced_nonnegative", "next_step": "ship it"},
    )

    assert payload["status"] == "full_assessment_balanced_nonnegative"
    assert payload["next_step"] == "ship it"


def test_competence_pipeline_write_report_updates_current_source_summary(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    run_dir = tmp_path / "outputs" / "stage5" / "competence"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "RUN_ID", "competence")

    module.write_report(
        {
            "status": "full_assessment_balanced_nonnegative",
            "source_summary": "outputs/stage5/source/summary.json",
            "arc_mix_summary": "outputs/stage5/competence_arc_mix/summary.json",
            "full_assessment_summary": "outputs/stage5/competence_full/summary.json",
            "next_step": "benchmark",
            "child_log_tail": "important child traceback tail",
        }
    )

    assert (run_dir / "summary.json").exists()
    assert "important child traceback tail" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/competence/summary.json\n"


def test_competence_pipeline_commit_stages_current_source_pointer(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    run_dir = tmp_path / "outputs" / "stage5" / "competence"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/competence/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        if [str(item) for item in cmd] == ["git", "diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(cmd, 1, "", None)
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results()

    add_commands = [cmd for cmd in commands if cmd[:2] == ["git", "add"]]
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/competence" in staged
    assert "config/stage5_current_source_summary.txt" in staged
    commit_commands = [cmd for cmd in commands if cmd[:2] == ["git", "commit"]]
    assert commit_commands
    assert "[skip ci]" in " ".join(commit_commands[0])


def test_competence_pipeline_commit_retries_failed_push_with_autostash_rebase(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    run_dir = tmp_path / "outputs" / "stage5" / "competence"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/competence/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []
    push_calls = 0

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        nonlocal push_calls
        command = [str(item) for item in cmd]
        commands.append(command)
        if command == ["git", "diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(cmd, 1, "", None)
        if command == ["git", "push", "origin", "main"]:
            push_calls += 1
            return subprocess.CompletedProcess(cmd, 1 if push_calls == 1 else 0, "", None)
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results()

    assert commands.count(["git", "push", "origin", "main"]) == 2
    assert ["git", "pull", "--rebase", "--autostash", "origin", "main"] in commands


def test_failure_summary_diagnoses_checkpoint_restore_drive_mount_failure(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    run_dir = tmp_path / "outputs" / "stage5" / "competence"
    run_dir.mkdir(parents=True)
    (run_dir / "arc_mix.log").write_text(
        "\n".join(
            [
                "Drive mount skipped/failed: 'NoneType' object has no attribute 'kernel'",
                "FileNotFoundError: Missing recovered checkpoint /content/recurrent-qwen-svgd/outputs/stage5/run/phase1/phase1_step_75.pt.",
                "Could not restore it from Drive.",
                "Drive visibility:",
                "- /content/drive: exists=False is_dir=False",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)

    payload = module.failure_summary(
        stage="arc_mix",
        error="command failed: /usr/bin/python3 colab/run_stage5_balanced_arc_mix_gate.py",
        source_payload={"status": "needs_review"},
    )

    assert payload["failure_diagnosis"] == "checkpoint_restore_or_drive_mount_failed"
    assert "Mount Google Drive" in payload["next_step"]
    assert "Missing recovered checkpoint" in payload["child_log_tail"]


def test_failure_summary_diagnoses_parent_checkpoint_restore_failure(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "outputs" / "stage5" / "competence")

    payload = module.failure_summary(
        stage="checkpoint_restore",
        error=(
            "Missing recovered checkpoint /content/recurrent-qwen-svgd/outputs/stage5/run/phase1/phase1_step_75.pt. "
            "Could not restore it from Drive."
        ),
        source_payload={"status": "needs_review"},
    )

    assert payload["failed_stage"] == "checkpoint_restore"
    assert payload["failure_diagnosis"] == "checkpoint_restore_or_drive_mount_failed"
    assert "Mount Google Drive" in payload["next_step"]
