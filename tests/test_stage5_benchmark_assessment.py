from __future__ import annotations

import json

from colab.assess_stage5_benchmark_suite import assess_benchmark_suite, main


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _suite(
    *,
    arc_delta: int = 0,
    gpqa_delta: int = 0,
    arc_n: int = 128,
    gpqa_n: int = 16,
    checkpoint: str = "outputs/stage5/run/phase1.pt",
):
    return {
        "run_id": "suite",
        "kind": "stage5_benchmark_suite",
        "status": "completed",
        "checkpoint": checkpoint,
        "benchmarks": ["arc_challenge", "gpqa_lite"],
        "failures": [],
        "paired_comparisons": {
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
        },
    }


def test_benchmark_assessment_passes_nonnegative_paired_evidence(tmp_path) -> None:
    source = tmp_path / "suite" / "summary.json"
    payload = _suite(arc_delta=1, gpqa_delta=0)

    assessed = assess_benchmark_suite(summary_json=source, payload=payload)

    assert assessed["status"] == "passed"
    assert assessed["passed"] is True
    assert assessed["checkpoint"] == "outputs/stage5/run/phase1.pt"
    assert [row["passed"] for row in assessed["criteria"]] == [True, True, True]


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
