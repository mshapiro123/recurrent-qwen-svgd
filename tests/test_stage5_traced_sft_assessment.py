from __future__ import annotations

import json

from colab.assess_stage5_traced_sft import assess


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sft_summary(*, rows: int = 32, direct_loops: float = 1.08, deep_loops: float = 1.95):
    return {
        "run_id": "sft",
        "kind": "stage5_curriculum_sft",
        "dataset": {"rows": rows},
        "phase1_val_by_mode": {
            "direct": {"mean_expected_loops": direct_loops},
            "deep_narrow": {"mean_expected_loops": deep_loops},
        },
    }


def _pair(*, n: int, base: int, recurrent: int):
    delta = recurrent - base
    return {
        "paired_examples": n,
        "base_correct": base,
        "recurrent_correct": recurrent,
        "correct_delta_recurrent_vs_base": delta,
        "wins": max(delta, 0),
        "losses": max(-delta, 0),
        "ties": n - abs(delta),
        "sign_test_p_value": 1.0,
    }


def _benchmark(*, sft_summary: str, content_delta: int = -1, cyclic_delta: int = 1, n: int = 96):
    return {
        "run_id": "bench",
        "kind": "stage5_benchmark_suite",
        "status": "completed",
        "source_summary": sft_summary,
        "checkpoint": "outputs/stage5/sft/phase1/phase1_step_150.pt",
        "benchmarks": ["arc_challenge"],
        "failures": [],
        "paired_comparisons": {
            "arc_challenge": {
                "content_question_only": {
                    "mean": _pair(n=n, base=34, recurrent=34 + content_delta),
                },
                "cyclic_label_aggregated": {
                    "permutation_mean": _pair(n=n, base=51, recurrent=51 + cyclic_delta),
                },
            }
        },
    }


def test_traced_sft_assessment_routes_small_positive_pilot_to_scale_more(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_traced_sft as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    sft_path = tmp_path / "outputs/stage5/sft/summary.json"
    bench_path = tmp_path / "outputs/stage5/bench/summary.json"
    _write(sft_path, _sft_summary(rows=32))
    payload = _benchmark(sft_summary="outputs/stage5/sft/summary.json")

    result = assess(benchmark_summary_path=bench_path, benchmark_summary=payload)

    assert result["status"] == "scale_trace_curriculum"
    assert result["trace_rows"] == 32
    assert result["depth"]["passed"] is True


def test_traced_sft_assessment_blocks_negative_cyclic_delta(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_traced_sft as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    sft_path = tmp_path / "outputs/stage5/sft/summary.json"
    bench_path = tmp_path / "outputs/stage5/bench/summary.json"
    _write(sft_path, _sft_summary(rows=64))
    payload = _benchmark(sft_summary="outputs/stage5/sft/summary.json", content_delta=0, cyclic_delta=-1)

    result = assess(benchmark_summary_path=bench_path, benchmark_summary=payload)

    assert result["status"] == "needs_calibration_repair"
    assert result["criteria"][3]["passed"] is False


def test_traced_sft_assessment_blocks_weak_depth_gradient(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_traced_sft as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    sft_path = tmp_path / "outputs/stage5/sft/summary.json"
    bench_path = tmp_path / "outputs/stage5/bench/summary.json"
    _write(sft_path, _sft_summary(rows=64, direct_loops=1.7, deep_loops=1.9))
    payload = _benchmark(sft_summary="outputs/stage5/sft/summary.json", content_delta=0, cyclic_delta=1)

    result = assess(benchmark_summary_path=bench_path, benchmark_summary=payload)

    assert result["status"] == "needs_depth_routing_repair"
    assert result["depth"]["passed"] is False


def test_traced_sft_assessment_allows_phase2_after_rows_and_benchmark_coverage(tmp_path, monkeypatch) -> None:
    import colab.assess_stage5_traced_sft as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    sft_path = tmp_path / "outputs/stage5/sft/summary.json"
    bench_path = tmp_path / "outputs/stage5/bench/summary.json"
    _write(sft_path, _sft_summary(rows=128))
    payload = _benchmark(sft_summary="outputs/stage5/sft/summary.json", content_delta=0, cyclic_delta=1, n=128)

    result = assess(benchmark_summary_path=bench_path, benchmark_summary=payload)

    assert result["status"] == "ready_for_phase2_probe"
    assert result["passed"] is True
