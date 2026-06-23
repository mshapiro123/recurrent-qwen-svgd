from pathlib import Path
import subprocess


def test_source_benchmark_summary_follows_assessment_source(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_surface_alignment_repair as module

    benchmark = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"kind":"stage5_benchmark_suite"}', encoding="utf-8")
    source = tmp_path / "outputs" / "stage5" / "assess" / "summary.json"
    source.parent.mkdir(parents=True)

    monkeypatch.setattr(module, "ROOT", tmp_path)

    resolved = module.source_benchmark_summary(
        {"kind": "stage5_broader_benchmark_suite", "source_summary": "outputs/stage5/bench/summary.json"},
        source,
    )

    assert resolved == benchmark


def test_train_config_targets_direct_surface_and_resume_checkpoint(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_surface_alignment_repair as module

    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(module, "MAX_STEPS", 12)
    monkeypatch.setattr(module, "SAVE_EVERY", 6)
    monkeypatch.setattr(module, "LEARNING_RATE", "1e-6")
    monkeypatch.setattr(module, "DISTILL_WEIGHT", "0.05")

    cfg = module.train_config(
        checkpoint=Path("outputs/stage5/source/phase1_step_75.pt"),
        train_jsonl=Path("data/train.jsonl"),
    )

    assert cfg["resume_from"] == "outputs/stage5/source/phase1_step_75.pt"
    assert cfg["max_steps"] == 12
    assert cfg["save_every"] == 6
    assert cfg["learning_rate"] == 1e-6
    assert cfg["use_target_loop_control"] is True
    assert cfg["halt_target_nll_weight"] == 0.05
    assert cfg["distillation"]["enabled"] is True
    assert cfg["distillation"]["on"] == "response"


def test_final_checkpoint_uses_surface_align_output_dir(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_surface_alignment_repair as module

    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "surface")
    monkeypatch.setattr(module, "MAX_STEPS", 50)

    assert module.final_checkpoint() == tmp_path / "surface" / "phase1_surface_align" / "phase1_step_50.pt"


def test_assess_surface_repair_invokes_before_after_assessor(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_surface_alignment_repair as module

    commands: list[list[str]] = []
    output_json = tmp_path / "run" / "surface_repair_assessment.json"
    output_json.parent.mkdir(parents=True)
    output_json.write_text('{"status":"surface_repair_passed","passed":true}', encoding="utf-8")

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(module, "run", fake_run)

    result = module.assess_surface_repair(
        tmp_path / "outputs" / "stage5" / "source" / "summary.json",
        tmp_path / "outputs" / "stage5" / "repaired" / "summary.json",
    )

    assert result["status"] == "surface_repair_passed"
    assert commands[0][:2] == ["python", "colab/assess_stage5_surface_repair.py"] or commands[0][
        1
    ] == "colab/assess_stage5_surface_repair.py"
    assert "--source_benchmark_summary" in commands[0]
    assert "--repaired_benchmark_summary" in commands[0]
