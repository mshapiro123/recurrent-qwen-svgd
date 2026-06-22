from __future__ import annotations

import json
import sys

from training.run_curriculum_job_responses import main, read_jsonl, run_jobs


def job(job_id: str = "job-1") -> dict:
    return {
        "job_id": job_id,
        "stage": "ground_truth_solve",
        "role": "solver",
        "model": "opus-test",
        "prompt": "What is 2+2?",
    }


def write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_dry_run_writes_standard_response_rows(tmp_path) -> None:
    output = tmp_path / "responses.jsonl"

    report = run_jobs([job("job-1"), job("job-2")], output_jsonl=output, backend="dry_run")

    rows = read_jsonl(output)
    assert report["written"] == 2
    assert rows[0]["job_id"] == "job-1"
    assert rows[0]["backend"] == "dry_run"
    assert rows[0]["status"] == "ok"
    assert "response_text" in rows[0]


def test_resume_skips_existing_job_ids(tmp_path) -> None:
    output = tmp_path / "responses.jsonl"
    write_jsonl(output, [{"job_id": "job-1", "response_text": "already done"}])

    report = run_jobs([job("job-1"), job("job-2")], output_jsonl=output, backend="dry_run", resume=True)

    rows = read_jsonl(output)
    assert report["skipped"] == 1
    assert report["written"] == 1
    assert [row["job_id"] for row in rows] == ["job-1", "job-2"]


def test_command_backend_passes_job_json_on_stdin(tmp_path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import json, sys\n"
        "job = json.loads(sys.stdin.read())\n"
        "print('ANSWER: ' + job['job_id'])\n",
        encoding="utf-8",
    )
    output = tmp_path / "responses.jsonl"

    report = run_jobs(
        [job("job-abc")],
        output_jsonl=output,
        backend="command",
        command=f"{sys.executable} {helper}",
    )

    rows = read_jsonl(output)
    assert report["written"] == 1
    assert rows[0]["backend"] == "command"
    assert rows[0]["status"] == "ok"
    assert rows[0]["response_text"] == "ANSWER: job-abc"


def test_command_backend_records_errors_without_raising(tmp_path) -> None:
    helper = tmp_path / "bad_helper.py"
    helper.write_text("import sys\nprint('bad')\nsys.exit(3)\n", encoding="utf-8")
    output = tmp_path / "responses.jsonl"

    report = run_jobs(
        [job("job-bad")],
        output_jsonl=output,
        backend="command",
        command=f"{sys.executable} {helper}",
    )

    rows = read_jsonl(output)
    assert report["errors"] == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["returncode"] == 3


def test_cli_runs_dry_run_with_report(tmp_path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    output = tmp_path / "responses.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(jobs, [job("job-1")])

    assert main(
        [
            "--jobs_jsonl",
            str(jobs),
            "--output_jsonl",
            str(output),
            "--report_json",
            str(report),
            "--backend",
            "dry_run",
        ]
    ) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["written"] == 1
    assert read_jsonl(output)[0]["job_id"] == "job-1"


def test_read_jsonl_accepts_utf8_bom(tmp_path) -> None:
    jobs = tmp_path / "bom_jobs.jsonl"
    jobs.write_text(json.dumps(job("job-bom")) + "\n", encoding="utf-8-sig")

    assert read_jsonl(jobs)[0]["job_id"] == "job-bom"
