from pathlib import Path
import subprocess
import json


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
        source_order_diagnosis=tmp_path / "outputs" / "stage5" / "source" / "arc_easy_order_sensitivity_diagnosis.json",
        repaired_order_diagnosis=tmp_path / "outputs" / "stage5" / "repaired" / "arc_easy_order_sensitivity_diagnosis.json",
    )

    assert result["status"] == "surface_repair_passed"
    assert commands[0][:2] == ["python", "colab/assess_stage5_surface_repair.py"] or commands[0][
        1
    ] == "colab/assess_stage5_surface_repair.py"
    assert "--source_benchmark_summary" in commands[0]
    assert "--repaired_benchmark_summary" in commands[0]
    assert "--source_order_diagnosis" in commands[0]
    assert "--repaired_order_diagnosis" in commands[0]


def test_ensure_order_sensitivity_diagnosis_invokes_analyzer(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_surface_alignment_repair as module

    benchmark = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    benchmark.parent.mkdir(parents=True)
    for name in [
        "arc_easy_base_content_question_only.jsonl",
        "arc_easy_recurrent_content_question_only.jsonl",
        "arc_easy_recurrent_cyclic_label_aggregated.jsonl",
        "arc_easy_base_cyclic_label_aggregated.jsonl",
    ]:
        (benchmark.parent / name).write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, env=None, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        (benchmark.parent / "arc_easy_order_sensitivity_diagnosis.json").write_text(
            '{"summary":{"recommendation":"prioritize_conditional_invariance_repair"}}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", fake_run)

    diagnosis = module.ensure_order_sensitivity_diagnosis(benchmark)

    assert diagnosis == benchmark.parent / "arc_easy_order_sensitivity_diagnosis.json"
    assert commands[0][1] == "eval/analyze_mcq_order_sensitivity.py"
    assert "--candidate_cyclic" in commands[0]


def test_repair_objective_uses_conditional_invariance_when_order_diagnostic_requests_it() -> None:
    import colab.run_stage5_surface_alignment_repair as module

    objective = module.repair_objective(
        {"summary": {"recommendation": "prioritize_conditional_invariance_repair"}},
        {"summary": {"recommendation": "prioritize_content_cyclic_surface_alignment"}},
    )

    assert objective == "conditional_invariance"


def test_repair_objective_defaults_to_surface_alignment() -> None:
    import colab.run_stage5_surface_alignment_repair as module

    objective = module.repair_objective(
        {"summary": {"recommendation": "diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation"}},
        {"summary": {"recommendation": "prioritize_content_cyclic_surface_alignment"}},
    )

    assert objective == "content_cyclic_surface_alignment"


def test_surface_alignment_drive_backup_skips_when_drive_not_mounted(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_surface_alignment_repair as module

    monkeypatch.setattr(module, "DRIVE_BACKUP_ENABLED", True)
    monkeypatch.setattr(module, "DRIVE_MYDRIVE", tmp_path / "missing_drive")

    info = module.backup_to_drive()

    assert info["status"] == "skipped_drive_not_mounted"


def test_surface_alignment_drive_backup_copies_run_and_private_data(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_surface_alignment_repair as module

    run_dir = tmp_path / "outputs" / "stage5" / "run"
    private_dir = tmp_path / "data" / "stage5_surface_alignment" / "run"
    run_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text('{"ok": true}', encoding="utf-8")
    (private_dir / "score_alignment_train.jsonl").write_text("{}\n", encoding="utf-8")
    drive = tmp_path / "drive" / "MyDrive"
    drive.mkdir(parents=True)

    monkeypatch.setattr(module, "DRIVE_BACKUP_ENABLED", True)
    monkeypatch.setattr(module, "DRIVE_MYDRIVE", drive)
    monkeypatch.setattr(module, "DRIVE_BACKUP_ROOT", drive / "recurrent-qwen-svgd-artifacts")
    monkeypatch.setattr(module, "RUN_ID", "run")
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", private_dir)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", Path("outputs/stage5/source/summary.json"))
    monkeypatch.setattr(module, "ROOT", tmp_path)

    info = module.backup_to_drive()

    dest = Path(info["dest_root"])
    assert info["status"] == "backed_up"
    assert (dest / "run_dir" / "summary.json").exists()
    assert (dest / "private_data" / "score_alignment_train.jsonl").exists()
    assert json.loads((dest / "backup_manifest.json").read_text(encoding="utf-8"))["run_id"] == "run"
