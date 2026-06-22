from __future__ import annotations


def test_selected_checkpoint_from_distill_autopilot(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovery_full_assessment as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "kind": "stage5_balanced_recovery_autopilot",
        "status": "distill_gate_passed",
        "distill": {
            "status": "proxy_lift",
            "best_arm": {
                "checkpoint": "outputs/stage5/distill_child/phase1/phase1_step_50.pt",
            },
        },
    }

    gate, checkpoint, gate_payload = module.selected_checkpoint(payload)

    assert gate == "distill"
    assert checkpoint == tmp_path / "outputs" / "stage5" / "distill_child" / "phase1" / "phase1_step_50.pt"
    assert gate_payload["status"] == "proxy_lift"


def test_selected_checkpoint_from_arc_mix_autopilot(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovery_full_assessment as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "kind": "stage5_balanced_recovery_autopilot",
        "status": "arc_mix_gate_passed",
        "arc_mix": {
            "status": "proxy_matches_base",
            "best_arm": {
                "best_checkpoint": {
                    "checkpoint": "outputs/stage5/arc_mix_child/phase1/phase1_step_100.pt",
                },
            },
        },
    }

    gate, checkpoint, _ = module.selected_checkpoint(payload)

    assert gate == "arc_mix"
    assert checkpoint == tmp_path / "outputs" / "stage5" / "arc_mix_child" / "phase1" / "phase1_step_100.pt"


def test_selected_checkpoint_rejects_failed_autopilot() -> None:
    import pytest
    import colab.run_stage5_recovery_full_assessment as module

    with pytest.raises(ValueError, match="did not pass"):
        module.selected_checkpoint(
            {
                "kind": "stage5_balanced_recovery_autopilot",
                "status": "no_recovery_gate_lift",
            }
        )


def test_infer_stage5_run_id() -> None:
    import colab.run_stage5_recovery_full_assessment as module

    assert module.infer_stage5_run_id("outputs/stage5/run/phase1/phase1_step_50.pt") == "run"


def test_run_full_benchmark_reuses_existing_summary(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovery_full_assessment as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_ID", "full_assess")
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(module, "run", fake_run)
    summary = tmp_path / "outputs" / "stage5" / "full_assess_balanced_full" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")

    assert module.run_full_benchmark(tmp_path / "checkpoint.pt") == summary
    assert called is False


def test_update_current_source_summary_writes_relative_summary_pointer(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovery_full_assessment as module

    summary = tmp_path / "outputs" / "stage5" / "full_assess" / "summary.json"
    summary.parent.mkdir(parents=True)

    monkeypatch.setattr(module, "ROOT", tmp_path)

    written = module.update_current_source_summary(summary)

    assert written == tmp_path / "config" / "stage5_current_source_summary.txt"
    assert written.read_text(encoding="utf-8") == "outputs/stage5/full_assess/summary.json\n"


def test_write_report_advances_current_source_summary(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovery_full_assessment as module

    run_dir = tmp_path / "outputs" / "stage5" / "full_assess"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "RUN_ID", "full_assess")

    module.write_report(
        {
            "status": "balanced_passed",
            "passed": True,
            "source_summary": "outputs/stage5/source/summary.json",
            "selected_gate": "arc_mix",
            "selected_checkpoint": "outputs/stage5/source/phase1/phase1_step_50.pt",
            "benchmark_summary": "outputs/stage5/full_assess_balanced_full/summary.json",
            "balanced_assessment_summary": "outputs/stage5/full_assess/balanced_assessment/summary.json",
            "next_step": "continue",
        }
    )

    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/full_assess/summary.json\n"


def test_commit_results_stages_current_source_pointer(tmp_path, monkeypatch) -> None:
    import subprocess

    import colab.run_stage5_recovery_full_assessment as module

    run_dir = tmp_path / "outputs" / "stage5" / "full_assess"
    benchmark_dir = tmp_path / "outputs" / "stage5" / "full_assess_balanced_full"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    benchmark_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (benchmark_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/full_assess/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results([run_dir, benchmark_dir])

    add_commands = [cmd for cmd in commands if cmd[:2] == ["git", "add"]]
    assert add_commands
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/full_assess" in staged
    assert "outputs/stage5/full_assess_balanced_full" in staged
    assert "config/stage5_current_source_summary.txt" in staged


def test_main_writes_failure_summary_when_benchmark_child_fails(tmp_path, monkeypatch) -> None:
    import json
    import subprocess
    import sys

    import colab.run_stage5_recovery_full_assessment as module

    root = tmp_path
    run_dir = root / "outputs" / "stage5" / "full_assess"
    source = root / "outputs" / "stage5" / "repair" / "summary.json"
    checkpoint = root / "outputs" / "stage5" / "repair" / "phase1" / "phase1_step_150.pt"
    source.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    source.write_text(
        json.dumps(
            {
                "kind": "stage5_routing_repair",
                "status": "repair_proxy_lift",
                "passed": True,
                "best_checkpoint": {"checkpoint": checkpoint.relative_to(root).as_posix()},
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        assert cmd == [sys.executable, "colab/run_stage5_benchmark_suite.py"]
        return subprocess.CompletedProcess(cmd, 7, "benchmark died\n", None)

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "RUN_ID", "full_assess")
    monkeypatch.setattr(module, "SOURCE_SUMMARY", source)
    monkeypatch.setattr(module, "PUSH_RESULTS", False)
    monkeypatch.setattr(module, "restore_checkpoint_if_needed", lambda checkpoint, *, run_id: None)
    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "backup_to_drive", lambda paths: None)

    assert module.main() == 7

    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "stage5_recovery_full_assessment"
    assert payload["status"] == "benchmark_child_failed"
    assert payload["passed"] is False
    assert payload["child_returncode"] == 7
    assert payload["child_summary_path"] == "outputs/stage5/full_assess_balanced_full/summary.json"
    assert "benchmark died" in payload["child_stdout_tail"]
    assert payload["best_checkpoint"] is None
    assert payload["balanced_assessment"] is None
    assert (root / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/full_assess/summary.json\n"


def test_main_writes_failure_summary_when_balanced_assessment_child_fails(tmp_path, monkeypatch) -> None:
    import json
    import subprocess
    import sys

    import colab.run_stage5_recovery_full_assessment as module

    root = tmp_path
    run_dir = root / "outputs" / "stage5" / "full_assess"
    source = root / "outputs" / "stage5" / "repair" / "summary.json"
    checkpoint = root / "outputs" / "stage5" / "repair" / "phase1" / "phase1_step_150.pt"
    benchmark_summary = root / "outputs" / "stage5" / "full_assess_balanced_full" / "summary.json"
    source.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    source.write_text(
        json.dumps(
            {
                "kind": "stage5_routing_repair",
                "status": "repair_proxy_lift",
                "passed": True,
                "best_checkpoint": {"checkpoint": checkpoint.relative_to(root).as_posix()},
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        calls.append([str(item) for item in cmd])
        if cmd == [sys.executable, "colab/run_stage5_benchmark_suite.py"]:
            benchmark_summary.parent.mkdir(parents=True)
            benchmark_summary.write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "benchmark ok\n", None)
        assert cmd[:2] == [sys.executable, "colab/assess_stage5_balanced_mcq.py"]
        return subprocess.CompletedProcess(cmd, 8, "assessment died\n", None)

    backed_up: list[list[str]] = []

    def fake_backup(paths):
        backed_up.append([module.path_for_cli(path) for path in paths])

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "RUN_ID", "full_assess")
    monkeypatch.setattr(module, "SOURCE_SUMMARY", source)
    monkeypatch.setattr(module, "PUSH_RESULTS", False)
    monkeypatch.setattr(module, "restore_checkpoint_if_needed", lambda checkpoint, *, run_id: None)
    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "backup_to_drive", fake_backup)

    assert module.main() == 8

    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "balanced_assessment_child_failed"
    assert payload["child_returncode"] == 8
    assert payload["benchmark_summary"] == "outputs/stage5/full_assess_balanced_full/summary.json"
    assert payload["child_summary_path"] == "outputs/stage5/full_assess/balanced_assessment/summary.json"
    assert "assessment died" in payload["child_stdout_tail"]
    assert backed_up == [["outputs/stage5/full_assess", "outputs/stage5/full_assess_balanced_full"]]
    assert calls[0] == [sys.executable, "colab/run_stage5_benchmark_suite.py"]
