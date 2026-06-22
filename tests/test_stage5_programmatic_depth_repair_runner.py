from __future__ import annotations

import subprocess


def test_programmatic_depth_write_report_updates_current_source_summary(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_programmatic_depth_repair as module

    run_dir = tmp_path / "outputs" / "stage5" / "programmatic"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "RUN_ID", "programmatic")

    module.write_report(
        {
            "status": "complete",
            "resume_checkpoint": "outputs/stage5/source/phase1/phase1_step_150.pt",
            "train_report": {"exported_examples": 10},
            "val_report": {"exported_examples": 2},
            "best_checkpoint": "outputs/stage5/programmatic/phase1/phase1_step_100.pt",
            "start_eval": {"mean_expected_loops": 2.0},
            "best_eval": {"mean_expected_loops": 2.5},
            "next_step": "benchmark",
        }
    )

    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/programmatic/summary.json\n"


def test_programmatic_depth_commit_stages_current_source_pointer(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_programmatic_depth_repair as module

    run_dir = tmp_path / "outputs" / "stage5" / "programmatic"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/programmatic/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results()

    add_commands = [cmd for cmd in commands if cmd[:2] == ["git", "add"]]
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/programmatic/summary.json" in staged
    assert "config/stage5_current_source_summary.txt" in staged
