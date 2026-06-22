from __future__ import annotations

import json
import sys

import training.run_curriculum_job_responses as runner
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


def test_openai_compatible_backend_posts_chat_completion_request(tmp_path, monkeypatch) -> None:
    output = tmp_path / "responses.jsonl"
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ANSWER: 4"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["headers"] = dict(request.header_items())
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    report = run_jobs(
        [{**job("job-openai"), "model": "logical-opus", "expects_json": True}],
        output_jsonl=output,
        backend="openai_compatible",
        api_key="test-key",
        base_url="https://example.test/v1",
        model_map={"logical-opus": "real-model"},
        json_mode=True,
        max_tokens=321,
        temperature=0.0,
        system_prompt="Return concise answers.",
    )

    rows = read_jsonl(output)
    assert report["written"] == 1
    assert rows[0]["backend"] == "openai_compatible"
    assert rows[0]["status"] == "ok"
    assert rows[0]["resolved_model"] == "real-model"
    assert rows[0]["response_text"] == "ANSWER: 4"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["payload"]["model"] == "real-model"
    assert seen["payload"]["max_tokens"] == 321
    assert seen["payload"]["temperature"] == 0.0
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "Return concise answers."}


def test_openai_compatible_backend_requires_api_key(tmp_path) -> None:
    output = tmp_path / "responses.jsonl"

    try:
        run_jobs(
            [job("job-openai")],
            output_jsonl=output,
            backend="openai_compatible",
            api_key_env="MISSING_TEST_API_KEY",
        )
    except ValueError as exc:
        assert "API key missing" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected missing API key error")


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
