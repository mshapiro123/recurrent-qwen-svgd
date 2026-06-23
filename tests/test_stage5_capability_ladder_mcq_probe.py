from __future__ import annotations

import json

import colab.run_stage5_capability_ladder_mcq_probe as module


def test_parse_model_specs_requires_key_value_pairs() -> None:
    specs = module.parse_model_specs(
        "qwen_0_5b=Qwen/Qwen2.5-0.5B-Instruct,qwen_1_5b=Qwen/Qwen2.5-1.5B-Instruct"
    )

    assert [spec.key for spec in specs] == ["qwen_0_5b", "qwen_1_5b"]
    assert specs[1].model_name == "Qwen/Qwen2.5-1.5B-Instruct"


def test_score_config_prefers_cheap_content_only_default() -> None:
    config = module.score_config("content_question_only")

    assert config.public_name == "content_question_only"
    assert config.prompt_style == "question_only"
    assert config.score_target == "option_text"
    assert config.cyclic is False


def test_score_config_supports_cyclic_label_aggregation() -> None:
    config = module.score_config("cyclic_label_aggregated")

    assert config.public_name == "cyclic_label_aggregated"
    assert config.prompt_style == "with_options"
    assert config.score_target == "label"
    assert config.cyclic is True


def test_probe_status_distinguishes_gate_ready_and_sparse() -> None:
    gate_ready = module.probe_status(
        {"counts": {"typed_records": 2, "mode_counts": {"direct": 1, "deep_narrow": 1}}},
        {"go": True},
    )
    needs_review = module.probe_status(
        {"counts": {"typed_records": 2, "mode_counts": {"direct": 1, "deep_narrow": 1}}},
        {"go": False},
    )
    sparse = module.probe_status(
        {"counts": {"typed_records": 3, "mode_counts": {"direct": 3}}},
        {"go": False},
    )

    assert gate_ready == "capability_ladder_probe_gate_ready"
    assert needs_review == "capability_ladder_probe_needs_review"
    assert sparse == "capability_ladder_probe_sparse"


def test_generic_model_ladder_controls_required_keys_and_max_loop(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "MODEL_LADDER",
        "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4",
    )

    assert module.required_model_keys() == ["qwen_0_5b", "qwen_1_5b", "qwen_3b", "qwen_7b"]
    assert module.max_target_loop() == 4


def test_write_probe_summary_records_depth_probe_caveat(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "probe"
    work_dir = tmp_path / "data" / "curriculum" / "probe"
    run_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    tasks = tmp_path / "tasks.jsonl"
    scored = tmp_path / "scored.jsonl"
    curriculum = work_dir / "summary.json"
    gate = run_dir / "curriculum_sft_gate.json"
    score = run_dir / "qwen_0_5b_content_question_only.jsonl"
    tasks.write_text("", encoding="utf-8")
    scored.write_text("", encoding="utf-8")
    score.write_text(
        json.dumps({"id": "x", "prediction": "A", "answer": "A", "hit": True}) + "\n",
        encoding="utf-8",
    )
    curriculum.write_text(
        json.dumps(
            {
                "counts": {
                    "typed_records": 2,
                    "positive_sft_rows": 2,
                    "mode_counts": {"direct": 1, "deep_narrow": 1},
                    "target_loop_counts": {"1": 1, "2": 1},
                }
            }
        ),
        encoding="utf-8",
    )
    gate.write_text(json.dumps({"go": True, "status": "go"}), encoding="utf-8")

    monkeypatch.setattr(module, "RUN_ID", "probe")
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "WORK_DIR", work_dir)
    monkeypatch.setattr(module, "MODEL_SPECS", "qwen_0_5b=Qwen/Qwen2.5-0.5B-Instruct")

    summary = module.write_probe_summary(
        tasks_jsonl=tasks,
        score_paths={"qwen_0_5b": score},
        scored_jsonl=scored,
        curriculum_summary_path=curriculum,
        gate_path=gate,
        drive_backup={"enabled": False},
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))

    assert payload["kind"] == "stage5_capability_ladder_mcq_probe"
    assert payload["status"] == "capability_ladder_probe_gate_ready"
    assert "answer-only MCQ predictions" in payload["caveat"]
    assert payload["score_summaries"]["qwen_0_5b"]["correct"] == 1


def test_commit_paths_include_private_scored_rows_and_curriculum(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "probe"
    private_data_dir = tmp_path / "data" / "stage5_capability_ladder" / "probe"
    work_dir = tmp_path / "data" / "curriculum" / "probe"
    config_dir = tmp_path / "config"
    summary = run_dir / "summary.json"
    for path in (run_dir, private_data_dir, work_dir, config_dir):
        path.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    (private_data_dir / "scored_capability_rows.jsonl").write_text("{}", encoding="utf-8")
    (work_dir / "positive_sft.jsonl").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", private_data_dir)
    monkeypatch.setattr(module, "WORK_DIR", work_dir)
    monkeypatch.setattr(module, "current_source_summary_file", lambda: config_dir / "stage5_current_source_summary.txt")
    monkeypatch.setattr(module, "path_for_cli", lambda path: str(path).replace("\\", "/"))

    paths = module.commit_paths(summary)

    assert str(run_dir).replace("\\", "/") in paths
    assert str(private_data_dir).replace("\\", "/") in paths
    assert str(work_dir).replace("\\", "/") in paths
    assert str(config_dir / "stage5_current_source_summary.txt").replace("\\", "/") in paths


def test_safe_commit_uses_skip_ci(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "probe"
    private_data_dir = tmp_path / "data" / "stage5_capability_ladder" / "probe"
    work_dir = tmp_path / "data" / "curriculum" / "probe"
    config_dir = tmp_path / "config"
    summary = run_dir / "summary.json"
    for path in (run_dir, private_data_dir, work_dir, config_dir):
        path.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        if [str(item) for item in cmd] == ["git", "diff", "--cached", "--quiet"]:
            return module.subprocess.CompletedProcess(cmd, 1, "", None)
        return module.subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", private_data_dir)
    monkeypatch.setattr(module, "WORK_DIR", work_dir)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "current_source_summary_file", lambda: config_dir / "stage5_current_source_summary.txt")
    monkeypatch.setattr(module, "path_for_cli", lambda path: str(path).replace("\\", "/"))

    module.safe_commit(summary)

    assert ["git", "commit", "-m", "Record capability-ladder MCQ probe [skip ci]"] in commands
