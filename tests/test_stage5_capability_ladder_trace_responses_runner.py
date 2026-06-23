from __future__ import annotations

import json

import colab.run_stage5_capability_ladder_trace_responses as runner


def test_provider_config_requires_explicit_provider_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BACKEND", "openai_compatible")
    monkeypatch.setattr(runner, "RUN_PROVIDER", False)

    ready, reason = runner.provider_config_ready()

    assert ready is False
    assert "RUN_PROVIDER=1" in reason


def test_provider_config_accepts_local_hf_without_provider_spend(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BACKEND", "hf_local")
    monkeypatch.setattr(runner, "RUN_PROVIDER", False)
    monkeypatch.setattr(runner, "RUN_LOCAL_HF", True)
    monkeypatch.setattr(runner, "HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    ready, reason = runner.provider_config_ready()

    assert ready is True
    assert reason == "ready"


def test_provider_config_requires_local_hf_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BACKEND", "hf_local")
    monkeypatch.setattr(runner, "RUN_LOCAL_HF", False)
    monkeypatch.setattr(runner, "HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    ready, reason = runner.provider_config_ready()

    assert ready is False
    assert "RUN_LOCAL_HF=1" in reason


def test_provider_config_requires_local_hf_model_name(monkeypatch) -> None:
    monkeypatch.setattr(runner, "BACKEND", "hf_local")
    monkeypatch.setattr(runner, "RUN_LOCAL_HF", True)
    monkeypatch.setattr(runner, "HF_MODEL_NAME", "")

    ready, reason = runner.provider_config_ready()

    assert ready is False
    assert "HF_MODEL_NAME" in reason


def test_inline_model_map_is_written_to_run_dir(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "trace_responses"
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(
        runner,
        "MODEL_MAP_JSON_INLINE",
        '{"opus-strong":"provider/model-a","glm-strong":"provider/model-b"}',
    )

    path = runner.inline_model_map_path()

    assert path == run_dir / "model_map.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "opus-strong": "provider/model-a",
        "glm-strong": "provider/model-b",
    }


def test_run_responses_passes_local_hf_flags(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "trace_responses"
    run_dir.mkdir(parents=True)
    jobs = root / "outputs" / "stage5" / "trace_jobs" / "jobs.jsonl"
    jobs.parent.mkdir(parents=True)
    jobs.write_text("", encoding="utf-8")
    recorded: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        recorded.append(cmd)
        report = run_dir / "trace_response_report.json"
        report.write_text(
            json.dumps(
                {
                    "backend": "hf_local",
                    "jobs": 2,
                    "selected": 2,
                    "written": 2,
                    "skipped": 0,
                    "errors": 0,
                    "timeouts": 0,
                }
            ),
            encoding="utf-8",
        )
        class Proc:
            returncode = 0
            stdout = ""
        return Proc()

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "BACKEND", "hf_local")
    monkeypatch.setattr(runner, "HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setattr(runner, "HF_DTYPE", "bfloat16")
    monkeypatch.setattr(runner, "HF_DEVICE", "cuda")
    monkeypatch.setattr(runner, "HF_TOP_P", "0.8")
    monkeypatch.setattr(runner, "HF_ATTN_IMPLEMENTATION", "eager")
    monkeypatch.setattr(runner, "HF_TRUST_REMOTE_CODE", True)
    monkeypatch.setattr(runner, "ALLOW_STUDENT_LINEAGE", True)
    monkeypatch.setattr(runner, "LIMIT", "3")
    monkeypatch.setattr(runner, "run", fake_run)

    _responses, _report_path, report = runner.run_responses(jobs)

    cmd = recorded[0]
    assert report["backend"] == "hf_local"
    assert "--hf_model_name" in cmd
    assert cmd[cmd.index("--hf_model_name") + 1] == "Qwen/Qwen2.5-7B-Instruct"
    assert "--hf_dtype" in cmd
    assert "--hf_attn_implementation" in cmd
    assert "--hf_trust_remote_code" in cmd
    assert "--allow_student_lineage" in cmd
    assert "--limit" in cmd


def test_write_summary_records_safe_response_artifacts(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "trace_responses"
    run_dir.mkdir(parents=True)
    source = root / "outputs" / "stage5" / "trace_jobs" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"kind": "stage5_capability_ladder_trace_jobs"}), encoding="utf-8")
    jobs = run_dir / "capability_ladder_trace_jobs.jsonl"
    jobs_report = run_dir / "capability_ladder_trace_jobs_report.json"
    responses = run_dir / "trace_responses.jsonl"
    response_report = run_dir / "trace_response_report.json"
    for path in (jobs, jobs_report, responses):
        path.write_text("", encoding="utf-8")
    response_payload = {
        "backend": "openai_compatible",
        "jobs": 3,
        "selected": 3,
        "written": 3,
        "skipped": 0,
        "errors": 0,
        "timeouts": 0,
        "output_jsonl": str(responses),
    }
    response_report.write_text(json.dumps(response_payload), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "RUN_ID", "trace_responses")
    monkeypatch.setattr(runner, "BACKEND", "openai_compatible")
    monkeypatch.setattr(runner, "RUN_PROVIDER", True)

    summary = runner.write_summary(
        source_summary=source,
        jobs_jsonl=jobs,
        jobs_report=jobs_report,
        restore_report={"jobs": {"restored": False}},
        responses_jsonl=responses,
        response_report_json=response_report,
        response_report=response_payload,
        drive_backup={"enabled": False},
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["kind"] == "stage5_capability_ladder_trace_responses"
    assert payload["status"] == "responses_ready"
    assert payload["response_report"]["written"] == 3
    assert payload["artifacts"]["responses_jsonl"] == "outputs/stage5/trace_responses/trace_responses.jsonl"
    assert "trace_collect_cpu" in payload["next_action"]


def test_safe_commit_can_include_response_jsonl(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "trace_responses"
    run_dir.mkdir(parents=True)
    summary = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    report = run_dir / "trace_response_report.json"
    responses = run_dir / "trace_responses.jsonl"
    for path in (summary, summary_md, report, responses):
        path.write_text("{}", encoding="utf-8")
    recorded: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        recorded.append(cmd)
        class Proc:
            returncode = 0
            stdout = ""
        return Proc()

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "PUSH_RESULTS", True)
    monkeypatch.setattr(runner, "COMMIT_RESPONSES", True)
    monkeypatch.setattr(runner, "run", fake_run)

    runner.safe_commit(summary, update_pointer=True, include_responses=True)

    add_commands = [cmd for cmd in recorded if cmd[:3] == ["git", "add", "-f"]]
    assert any(cmd[-1] == "outputs/stage5/trace_responses/trace_responses.jsonl" for cmd in add_commands)
