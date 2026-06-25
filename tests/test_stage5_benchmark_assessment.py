from __future__ import annotations

import json
import subprocess

from colab.assess_stage5_benchmark_suite import assess_benchmark_suite, main


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _suite(
    *,
    arc_delta: int = 0,
    arc_easy_delta: int | None = None,
    gpqa_delta: int = 0,
    arc_n: int = 128,
    arc_easy_n: int = 128,
    gpqa_n: int = 16,
    checkpoint: str = "outputs/stage5/run/phase1.pt",
):
    paired = {
        "arc_challenge": {
            "label": {
                "mean": {
                    "paired_examples": arc_n,
                    "base_correct": 64,
                    "recurrent_correct": 64 + arc_delta,
                    "correct_delta_recurrent_vs_base": arc_delta,
                    "wins": max(arc_delta, 0),
                    "losses": max(-arc_delta, 0),
                    "ties": arc_n - abs(arc_delta),
                    "sign_test_p_value": 1.0,
                }
            }
        },
        "gpqa_lite": {
            "label": {
                "mean": {
                    "paired_examples": gpqa_n,
                    "base_correct": 4,
                    "recurrent_correct": 4 + gpqa_delta,
                    "correct_delta_recurrent_vs_base": gpqa_delta,
                    "wins": max(gpqa_delta, 0),
                    "losses": max(-gpqa_delta, 0),
                    "ties": gpqa_n - abs(gpqa_delta),
                    "sign_test_p_value": 1.0,
                }
            }
        },
    }
    benchmarks = ["arc_challenge", "gpqa_lite"]
    if arc_easy_delta is not None:
        benchmarks.insert(0, "arc_easy")
        paired["arc_easy"] = {
            "label": {
                "mean": {
                    "paired_examples": arc_easy_n,
                    "base_correct": 80,
                    "recurrent_correct": 80 + arc_easy_delta,
                    "correct_delta_recurrent_vs_base": arc_easy_delta,
                    "wins": max(arc_easy_delta, 0),
                    "losses": max(-arc_easy_delta, 0),
                    "ties": arc_easy_n - abs(arc_easy_delta),
                    "sign_test_p_value": 1.0,
                }
            }
        }
    return {
        "run_id": "suite",
        "kind": "stage5_benchmark_suite",
        "status": "completed",
        "checkpoint": checkpoint,
        "benchmarks": benchmarks,
        "failures": [],
        "paired_comparisons": paired,
    }


def test_benchmark_assessment_passes_nonnegative_paired_evidence(tmp_path) -> None:
    source = tmp_path / "suite" / "summary.json"
    payload = _suite(arc_delta=1, gpqa_delta=0)

    assessed = assess_benchmark_suite(summary_json=source, payload=payload)

    assert assessed["status"] == "passed"
    assert assessed["passed"] is True
    assert assessed["checkpoint"] == "outputs/stage5/run/phase1.pt"
    assert [row["passed"] for row in assessed["criteria"]] == [True, True, True]


def test_benchmark_assessment_preserves_after_confirmation_dense_control(tmp_path) -> None:
    source = tmp_path / "suite" / "summary.json"
    payload = _suite(arc_delta=1, gpqa_delta=0)
    payload["after_confirmation_dense_control"] = {
        "run_suffix": "dense_after_confirm",
        "extra_train_jsonl": "data/repair/surface_alignment_train.jsonl",
    }

    assessed = assess_benchmark_suite(summary_json=source, payload=payload)

    assert assessed["status"] == "passed"
    assert assessed["after_confirmation_dense_control"] == {
        "run_suffix": "dense_after_confirm",
        "extra_train_jsonl": "data/repair/surface_alignment_train.jsonl",
    }


def test_benchmark_assessment_routes_negative_delta_to_recovery(tmp_path) -> None:
    source = tmp_path / "suite" / "summary.json"
    payload = _suite(arc_delta=-2, gpqa_delta=0)

    assessed = assess_benchmark_suite(summary_json=source, payload=payload)

    assert assessed["status"] == "needs_recurrent_recovery"
    assert assessed["passed"] is False
    assert assessed["criteria"][2]["passed"] is False


def test_benchmark_assessment_requires_paired_coverage(tmp_path) -> None:
    source = tmp_path / "suite" / "summary.json"
    payload = _suite(arc_delta=0, gpqa_delta=0, arc_n=20)

    assessed = assess_benchmark_suite(summary_json=source, payload=payload)

    assert assessed["status"] == "needs_benchmark_confirmation"
    assert assessed["criteria"][1]["passed"] is False


def test_benchmark_assessment_requires_real_arc_easy_coverage_when_present(tmp_path) -> None:
    source = tmp_path / "suite" / "summary.json"
    payload = _suite(arc_delta=0, arc_easy_delta=0, gpqa_delta=0, arc_easy_n=1)

    assessed = assess_benchmark_suite(summary_json=source, payload=payload)

    assert assessed["status"] == "needs_benchmark_confirmation"
    arc_easy = next(row for row in assessed["benchmarks"] if row["benchmark"] == "arc_easy")
    assert arc_easy["required_examples"] == 128
    assert arc_easy["paired_examples"] == 1


def test_benchmark_assessment_caps_arc_challenge_at_validation_size(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_benchmark_suite as module

    source = tmp_path / "suite" / "summary.json"
    payload = _suite(arc_delta=0, gpqa_delta=0, arc_n=299)
    monkeypatch.setattr(module, "MIN_ARC_EXAMPLES", 512)
    monkeypatch.setattr(module, "ARC_CHALLENGE_VALIDATION_EXAMPLES", 299)

    assessed = assess_benchmark_suite(summary_json=source, payload=payload)

    assert assessed["status"] == "passed"
    assert assessed["criteria"][1]["passed"] is True
    assert assessed["benchmarks"][0]["required_examples"] == 299


def test_benchmark_assessment_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    source = tmp_path / "suite" / "summary.json"
    output_json = tmp_path / "assessment.json"
    output_md = tmp_path / "assessment.md"
    _write(source, _suite(arc_delta=1, gpqa_delta=1))
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_benchmark_suite.py",
            "--summary_json",
            str(source),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
    )

    assert main() == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "passed"
    assert "Stage 5 Broader Benchmark Assessment" in output_md.read_text(encoding="utf-8")


def test_benchmark_assessment_updates_current_source_summary(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_benchmark_suite as module

    source = tmp_path / "suite" / "summary.json"
    output_json = tmp_path / "outputs" / "stage5" / "assessment" / "summary.json"
    output_md = output_json.with_suffix(".md")
    _write(source, _suite(arc_delta=1, gpqa_delta=1))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PUSH_RESULTS", False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "assess_stage5_benchmark_suite.py",
            "--summary_json",
            str(source),
        ],
    )
    monkeypatch.setattr(module, "RUN_ID", "assessment")

    assert main() == 0
    assert output_json.exists()
    assert output_md.exists()
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/assessment/summary.json\n"


def test_benchmark_assessment_commit_stages_current_source_pointer(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_benchmark_suite as module

    run_dir = tmp_path / "outputs" / "stage5" / "assessment"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/assessment/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True):
        commands.append([str(item) for item in cmd])
        if [str(item) for item in cmd] == ["git", "diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(cmd, 1, "", None)
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_ID", "assessment")
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results(run_dir)

    add_commands = [cmd for cmd in commands if cmd[:2] == ["git", "add"]]
    assert add_commands
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/assessment" in staged
    assert "config/stage5_current_source_summary.txt" in staged
    assert ["git", "commit", "-m", "Record Stage 5 benchmark assessment assessment [skip ci]"] in commands
