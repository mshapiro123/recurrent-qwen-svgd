from __future__ import annotations

import json

import pytest

import colab.run_stage5_capability_ladder_trace_collect as runner


def test_resolve_source_summary_falls_back_from_stale_pointer_to_latest_trace_response(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    stale = root / "outputs" / "stage5" / "old_mcq" / "summary.json"
    response_summary = root / "outputs" / "stage5" / "trace_responses" / "summary.json"
    trace_jobs_summary = root / "outputs" / "stage5" / "trace_jobs" / "summary.json"
    for path in (stale, response_summary, trace_jobs_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(
        json.dumps({"kind": "stage5_mcq_debias_diagnostic", "status": "complete"}),
        encoding="utf-8",
    )
    trace_jobs_summary.write_text(
        json.dumps(
            {
                "run_id": "trace_jobs",
                "kind": "stage5_capability_ladder_trace_jobs",
                "status": "ready",
                "artifacts": {
                    "jobs_jsonl": "outputs/stage5/trace_jobs/trace_jobs.jsonl",
                    "report_json": "outputs/stage5/trace_jobs/report.json",
                },
            }
        ),
        encoding="utf-8",
    )
    response_summary.write_text(
        json.dumps(
            {
                "run_id": "trace_responses",
                "kind": "stage5_capability_ladder_trace_responses",
                "status": "responses_ready",
                "source_summary": "outputs/stage5/trace_jobs/summary.json",
                "artifacts": {
                    "responses_jsonl": "outputs/stage5/trace_responses/trace_responses.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "config").mkdir(parents=True)
    (root / "config" / "stage5_current_source_summary.txt").write_text(
        "outputs/stage5/old_mcq/summary.json\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "SOURCE_SUMMARY", "")
    monkeypatch.setattr(runner, "drive_search_roots", lambda: [])

    assert runner.resolve_source_summary() == response_summary


def test_resolve_source_summary_rejects_explicit_unrelated_summary(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    source = root / "outputs" / "stage5" / "old_mcq" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"kind": "stage5_mcq_debias_diagnostic"}), encoding="utf-8")

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "SOURCE_SUMMARY", "outputs/stage5/old_mcq/summary.json")
    monkeypatch.setattr(runner, "drive_search_roots", lambda: [])

    with pytest.raises(ValueError, match="unsupported source summary kind"):
        runner.resolve_source_summary()


def test_trace_response_summary_restores_missing_trace_job_summary_from_drive(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    drive = tmp_path / "drive"
    response_summary = root / "outputs" / "stage5" / "trace_responses" / "summary.json"
    drive_jobs_summary = drive / "stage5_capability_ladder_trace_jobs" / "trace_jobs" / "summary.json"
    response_summary.parent.mkdir(parents=True)
    drive_jobs_summary.parent.mkdir(parents=True)
    response_summary.write_text(
        json.dumps(
            {
                "run_id": "trace_responses",
                "kind": "stage5_capability_ladder_trace_responses",
                "status": "responses_ready",
                "source_summary": "outputs/stage5/trace_jobs/summary.json",
                "artifacts": {
                    "responses_jsonl": "outputs/stage5/trace_responses/trace_responses.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    drive_jobs_summary.write_text(
        json.dumps(
            {
                "run_id": "trace_jobs",
                "kind": "stage5_capability_ladder_trace_jobs",
                "status": "ready",
                "artifacts": {
                    "jobs_jsonl": "outputs/stage5/trace_jobs/trace_jobs.jsonl",
                    "report_json": "outputs/stage5/trace_jobs/report.json",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "drive_search_roots", lambda: [drive])

    restored = runner.trace_jobs_summary_for_collection(response_summary)

    assert restored == root / "outputs" / "stage5" / "trace_jobs" / "summary.json"
    assert json.loads(restored.read_text(encoding="utf-8"))["run_id"] == "trace_jobs"


def test_response_summary_resolves_trace_jobs_and_responses(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    trace_jobs_summary = root / "outputs" / "stage5" / "trace_jobs" / "summary.json"
    response_summary = root / "outputs" / "stage5" / "trace_responses" / "summary.json"
    responses = root / "outputs" / "stage5" / "trace_responses" / "trace_responses.jsonl"
    trace_jobs_summary.parent.mkdir(parents=True)
    response_summary.parent.mkdir(parents=True)
    responses.write_text("", encoding="utf-8")
    trace_jobs_summary.write_text(
        json.dumps(
            {
                "kind": "stage5_capability_ladder_trace_jobs",
                "artifacts": {
                    "jobs_jsonl": "outputs/stage5/trace_jobs/trace_jobs.jsonl",
                    "report_json": "outputs/stage5/trace_jobs/report.json",
                },
            }
        ),
        encoding="utf-8",
    )
    response_summary.write_text(
        json.dumps(
            {
                "kind": "stage5_capability_ladder_trace_responses",
                "source_summary": "outputs/stage5/trace_jobs/summary.json",
                "artifacts": {
                    "responses_jsonl": "outputs/stage5/trace_responses/trace_responses.jsonl",
                    "jobs_jsonl": "outputs/stage5/trace_jobs/trace_jobs.jsonl",
                    "jobs_report": "outputs/stage5/trace_jobs/report.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", root)

    assert runner.trace_jobs_summary_for_collection(response_summary) == trace_jobs_summary
    assert runner.response_path_from_response_summary(response_summary) == responses


def test_main_collects_directly_from_response_summary(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "collect"
    work_dir = root / "data" / "curriculum" / "collect"
    trace_jobs_dir = root / "outputs" / "stage5" / "trace_jobs"
    trace_responses_dir = root / "outputs" / "stage5" / "trace_responses"
    trace_jobs_summary = trace_jobs_dir / "summary.json"
    response_summary = trace_responses_dir / "summary.json"
    jobs_jsonl = trace_jobs_dir / "trace_jobs.jsonl"
    report_json = trace_jobs_dir / "report.json"
    scored_jsonl = root / "data" / "scored.jsonl"
    responses_jsonl = trace_responses_dir / "trace_responses.jsonl"
    for path in (trace_jobs_dir, trace_responses_dir, scored_jsonl.parent):
        path.mkdir(parents=True, exist_ok=True)
    jobs_jsonl.write_text("", encoding="utf-8")
    scored_jsonl.write_text("", encoding="utf-8")
    responses_jsonl.write_text("", encoding="utf-8")
    report_json.write_text(
        json.dumps({"source": {"scored_jsonl": "data/scored.jsonl"}}),
        encoding="utf-8",
    )
    trace_jobs_summary.write_text(
        json.dumps(
            {
                "run_id": "trace_jobs",
                "kind": "stage5_capability_ladder_trace_jobs",
                "artifacts": {
                    "jobs_jsonl": "outputs/stage5/trace_jobs/trace_jobs.jsonl",
                    "report_json": "outputs/stage5/trace_jobs/report.json",
                },
            }
        ),
        encoding="utf-8",
    )
    response_summary.write_text(
        json.dumps(
            {
                "run_id": "trace_responses",
                "kind": "stage5_capability_ladder_trace_responses",
                "status": "responses_ready",
                "source_summary": "outputs/stage5/trace_jobs/summary.json",
                "artifacts": {
                    "responses_jsonl": "outputs/stage5/trace_responses/trace_responses.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "config").mkdir(parents=True)
    (root / "config" / "stage5_current_source_summary.txt").write_text(
        "outputs/stage5/trace_responses/summary.json\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "RUN_ID", "collect")
    monkeypatch.setattr(runner, "WORK_DIR", work_dir)
    monkeypatch.setattr(runner, "SOURCE_SUMMARY", "")
    monkeypatch.setattr(runner, "RESPONSES_JSONL", "")
    monkeypatch.setattr(runner, "BACKUP_DRIVE", False)
    monkeypatch.setattr(runner, "PUSH_RESULTS", False)
    monkeypatch.setattr(runner, "REFUSE_GPU_RUNTIME", False)

    def fake_collect(scored, jobs, responses):
        assert scored == scored_jsonl
        assert jobs == jobs_jsonl
        assert responses == responses_jsonl
        run_dir.mkdir(parents=True, exist_ok=True)
        traced = run_dir / "scored_rows_with_traces.jsonl"
        collection_report = run_dir / "trace_collection_report.json"
        traced.write_text("", encoding="utf-8")
        collection_report.write_text(json.dumps({"accepted_rows": 2}), encoding="utf-8")
        return traced, collection_report, {
            "accepted_rows": 2,
            "status_counts": {"accepted": 2},
            "target_loop_counts": {"1": 1, "2": 1},
            "tier_counts": {"base_preservation": 1, "deep_narrow": 1},
        }

    def fake_build(traced, collection_source_summary):
        assert traced == run_dir / "scored_rows_with_traces.jsonl"
        assert collection_source_summary == trace_jobs_summary
        work_dir.mkdir(parents=True, exist_ok=True)
        summary = work_dir / "summary.json"
        summary.write_text(
            json.dumps({"counts": {"positive_sft_rows": 2, "mode_counts": {"direct": 1, "deep_narrow": 1}}}),
            encoding="utf-8",
        )
        return summary, "qwen_0_5b:1,qwen_1_5b:2"

    def fake_gate(curriculum_summary, *, model_ladder=""):
        assert curriculum_summary == work_dir / "summary.json"
        assert model_ladder == "qwen_0_5b:1,qwen_1_5b:2"
        gate = run_dir / "curriculum_sft_gate.json"
        gate.write_text(json.dumps({"kind": "curriculum_sft_gate", "go": True}), encoding="utf-8")
        return gate

    monkeypatch.setattr(runner, "collect_traces", fake_collect)
    monkeypatch.setattr(runner, "build_curriculum", fake_build)
    monkeypatch.setattr(runner, "gate_curriculum", fake_gate)
    monkeypatch.setattr(runner, "safe_commit", lambda summary: runner.update_current_source_summary(summary))

    assert runner.main() == 0

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "trace_curriculum_gate_ready"
    assert summary["source_summary"] == "outputs/stage5/trace_jobs/summary.json"
    assert summary["response_summary"] == "outputs/stage5/trace_responses/summary.json"
    assert summary["artifacts"]["responses_jsonl"] == "outputs/stage5/trace_responses/trace_responses.jsonl"
    assert (root / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip() == (
        "outputs/stage5/collect/summary.json"
    )


def test_write_summary_records_gate_and_artifacts(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "collect"
    work_dir = root / "data" / "curriculum" / "collect"
    run_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    source = root / "outputs" / "stage5" / "trace_jobs" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"kind": "stage5_capability_ladder_trace_jobs"}), encoding="utf-8")
    scored = root / "data" / "scored.jsonl"
    jobs = run_dir / "jobs.jsonl"
    responses = run_dir / "responses.jsonl"
    traced = run_dir / "scored_rows_with_traces.jsonl"
    collection_report = run_dir / "trace_collection_report.json"
    curriculum = work_dir / "summary.json"
    gate = run_dir / "curriculum_sft_gate.json"
    for path in (scored, jobs, responses, traced, collection_report):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    curriculum.write_text(
        json.dumps({"counts": {"positive_sft_rows": 3, "mode_counts": {"direct": 1, "deep_narrow": 2}}}),
        encoding="utf-8",
    )
    gate.write_text(json.dumps({"kind": "curriculum_sft_gate", "go": True}), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "RUN_ID", "collect")
    monkeypatch.setattr(runner, "WORK_DIR", work_dir)

    summary = runner.write_summary(
        source_summary=source,
        scored_jsonl=scored,
        jobs_jsonl=jobs,
        responses_jsonl=responses,
        traced_jsonl=traced,
        collection_report=collection_report,
        collection_payload={
            "accepted_rows": 3,
            "status_counts": {"accepted": 3},
            "target_loop_counts": {"1": 1, "2": 2},
            "tier_counts": {"base_preservation": 1},
        },
        curriculum_summary=curriculum,
        gate_json=gate,
        model_ladder="qwen_0_5b:1,qwen_1_5b:2",
        restore_report={"jobs": {"restored": False}},
        drive_backup={"enabled": False},
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["kind"] == "stage5_capability_ladder_trace_collection"
    assert payload["status"] == "trace_curriculum_gate_ready"
    assert payload["collection"]["accepted_rows"] == 3
    assert payload["curriculum"]["counts"]["positive_sft_rows"] == 3
    assert payload["curriculum"]["model_ladder"] == "qwen_0_5b:1,qwen_1_5b:2"
    assert payload["gate"]["go"] is True
    assert payload["artifacts"]["traced_scored_rows"] == "outputs/stage5/collect/scored_rows_with_traces.jsonl"


def test_gate_curriculum_allows_answer_line_verified_trace_shards(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "collect"
    work_dir = root / "data" / "curriculum" / "collect"
    summary = work_dir / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("{}", encoding="utf-8")
    captured: list[list[str]] = []

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "WORK_DIR", work_dir)
    monkeypatch.setattr(runner, "MIN_POSITIVE_ROWS", "16")
    monkeypatch.setattr(runner, "MIN_MODE_ROWS", "")

    def fake_run(cmd, *, check=True, log_name=None):
        captured.append(cmd)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "curriculum_sft_gate.json").write_text('{"go": true}', encoding="utf-8")

    monkeypatch.setattr(runner, "run", fake_run)

    assert runner.gate_curriculum(summary, model_ladder="qwen_0_5b:1,qwen_1_5b:2") == (
        run_dir / "curriculum_sft_gate.json"
    )

    cmd = captured[0]
    assert "--allow_answer_line_verification" in cmd
    assert "--min_positive_rows" in cmd
    assert cmd[cmd.index("--min_positive_rows") + 1] == "16"


def test_model_ladder_from_collection_source_prefers_trace_job_report(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    trace_jobs_dir = root / "outputs" / "stage5" / "trace_jobs"
    trace_jobs_dir.mkdir(parents=True)
    summary = trace_jobs_dir / "summary.json"
    report = trace_jobs_dir / "report.json"
    summary.write_text(
        json.dumps(
            {
                "kind": "stage5_capability_ladder_trace_jobs",
                "artifacts": {"report_json": "outputs/stage5/trace_jobs/report.json"},
            }
        ),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "model_ladder": [
                    {"key": "qwen_0_5b", "target_loop_count": 1},
                    {"key": "qwen_1_5b", "target_loop_count": 2},
                    {"key": "qwen_3b", "target_loop_count": 3},
                    {"key": "qwen_7b", "target_loop_count": 4},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "MODEL_LADDER", "")

    assert runner.model_ladder_from_collection_source(summary) == (
        "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3,qwen_7b:4"
    )
    assert runner.max_loop_for_ladder("qwen_0_5b:1,qwen_7b:4") == 4
