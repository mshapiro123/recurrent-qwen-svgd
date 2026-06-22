from __future__ import annotations

import json

import colab.run_stage5_capability_ladder_trace_jobs as runner


def test_write_summary_records_trace_job_artifacts(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    run_dir = root / "outputs" / "stage5" / "trace_jobs"
    run_dir.mkdir(parents=True)
    source = root / "outputs" / "stage5" / "probe" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"kind": "stage5_capability_ladder_mcq_probe"}), encoding="utf-8")
    jobs = run_dir / "capability_ladder_trace_jobs.jsonl"
    report = run_dir / "capability_ladder_trace_jobs_report.json"
    jobs.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN_DIR", run_dir)
    monkeypatch.setattr(runner, "RUN_ID", "trace_jobs")

    summary = runner.write_summary(
        source_summary=source,
        restore_report={"restored": False},
        jobs_jsonl=jobs,
        report_json=report,
        trace_report={
            "jobs": 6,
            "selected_rows": 3,
            "tier_counts": {"base_preservation": 1},
            "by_target_loop": {"1": 1, "2": 2},
            "by_model": {"opus-strong": 3, "glm-strong": 3},
        },
        drive_backup={"enabled": False},
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["kind"] == "stage5_capability_ladder_trace_jobs"
    assert payload["status"] == "ready"
    assert payload["trace_jobs"]["jobs"] == 6
    assert payload["artifacts"]["jobs_jsonl"] == "outputs/stage5/trace_jobs/capability_ladder_trace_jobs.jsonl"
    assert "run_curriculum_job_responses.py" in payload["next_action"]
