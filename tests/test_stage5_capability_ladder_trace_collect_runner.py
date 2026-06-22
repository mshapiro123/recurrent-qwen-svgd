from __future__ import annotations

import json

import colab.run_stage5_capability_ladder_trace_collect as runner


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
        restore_report={"jobs": {"restored": False}},
        drive_backup={"enabled": False},
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["kind"] == "stage5_capability_ladder_trace_collection"
    assert payload["status"] == "trace_curriculum_gate_ready"
    assert payload["collection"]["accepted_rows"] == 3
    assert payload["curriculum"]["counts"]["positive_sft_rows"] == 3
    assert payload["gate"]["go"] is True
    assert payload["artifacts"]["traced_scored_rows"] == "outputs/stage5/collect/scored_rows_with_traces.jsonl"
