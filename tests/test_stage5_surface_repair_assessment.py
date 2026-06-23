from __future__ import annotations

import json

from colab import assess_stage5_surface_repair as module


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def rows(hits: list[bool], *, aggregate: str) -> list[dict[str, object]]:
    return [
        {
            "id": f"item_{idx}",
            "aggregate": aggregate,
            "hit": hit,
            "prediction": "A" if hit else "B",
            "answer": "A",
        }
        for idx, hit in enumerate(hits)
    ]


def make_benchmark_pair(tmp_path, *, repaired_easy_content=None, repaired_challenge_content=None):
    source = tmp_path / "outputs" / "stage5" / "source_benchmark"
    repaired = tmp_path / "outputs" / "stage5" / "repaired_benchmark"
    specs = {
        ("arc_easy", "content_question_only", "mean"): {
            "source": [True, False, False, False],
            "repaired": repaired_easy_content or [True, True, True, False],
            "source_base_delta": -2,
            "repaired_base_delta": 0,
        },
        ("arc_easy", "cyclic_label_aggregated", "permutation_mean"): {
            "source": [True, True, True, False],
            "repaired": [True, True, True, False],
            "source_base_delta": 1,
            "repaired_base_delta": 1,
        },
        ("arc_challenge", "content_question_only", "mean"): {
            "source": [True, False, False, False],
            "repaired": repaired_challenge_content or [True, False, False, False],
            "source_base_delta": 1,
            "repaired_base_delta": 1,
        },
        ("arc_challenge", "cyclic_label_aggregated", "permutation_mean"): {
            "source": [True, True, False, False],
            "repaired": [True, True, False, False],
            "source_base_delta": 0,
            "repaired_base_delta": 0,
        },
    }

    paired_source: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    paired_repaired: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for (benchmark, score_target, aggregate), spec in specs.items():
        write_jsonl(
            source / f"{benchmark}_recurrent_{score_target}.jsonl",
            rows(spec["source"], aggregate=aggregate),
        )
        write_jsonl(
            repaired / f"{benchmark}_recurrent_{score_target}.jsonl",
            rows(spec["repaired"], aggregate=aggregate),
        )
        paired_source.setdefault(benchmark, {}).setdefault(score_target, {})[aggregate] = {
            "paired_examples": 4,
            "correct_delta_recurrent_vs_base": spec["source_base_delta"],
        }
        paired_repaired.setdefault(benchmark, {}).setdefault(score_target, {})[aggregate] = {
            "paired_examples": 4,
            "correct_delta_recurrent_vs_base": spec["repaired_base_delta"],
        }

    source_summary = source / "summary.json"
    repaired_summary = repaired / "summary.json"
    write_json(
        source_summary,
        {
            "kind": "stage5_benchmark_suite",
            "run_id": "source",
            "paired_comparisons": paired_source,
        },
    )
    write_json(
        repaired_summary,
        {
            "kind": "stage5_benchmark_suite",
            "run_id": "repaired",
            "paired_comparisons": paired_repaired,
        },
    )
    return source_summary, repaired_summary


def test_surface_repair_assessment_passes_when_easy_lifts_and_hard_preserved(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source, repaired = make_benchmark_pair(tmp_path)

    payload = module.assess_surface_repair(source_benchmark_summary=source, repaired_benchmark_summary=repaired)

    assert payload["status"] == "surface_repair_passed"
    assert payload["passed"] is True
    easy = payload["comparisons"]["arc_easy_content"]["source_recurrent_vs_repaired_recurrent"]
    assert easy["correct_delta_candidate_vs_reference"] == 2
    assert payload["comparisons"]["arc_easy_content"]["repaired_recurrent_vs_base"]["correct_delta_recurrent_vs_base"] == 0


def test_surface_repair_assessment_flags_tradeoff(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source, repaired = make_benchmark_pair(
        tmp_path,
        repaired_challenge_content=[False, False, False, False],
    )

    payload = module.assess_surface_repair(source_benchmark_summary=source, repaired_benchmark_summary=repaired)

    assert payload["status"] == "surface_repair_tradeoff"
    assert payload["passed"] is False
    assert payload["comparisons"]["arc_challenge_content"]["source_recurrent_vs_repaired_recurrent"][
        "correct_delta_candidate_vs_reference"
    ] == -1


def test_surface_repair_assessment_flags_no_easy_content_lift(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source, repaired = make_benchmark_pair(
        tmp_path,
        repaired_easy_content=[True, False, False, False],
    )

    payload = module.assess_surface_repair(source_benchmark_summary=source, repaired_benchmark_summary=repaired)

    assert payload["status"] == "surface_repair_no_easy_content_lift"
    assert payload["passed"] is False


def test_surface_repair_assessment_writes_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source, repaired = make_benchmark_pair(tmp_path)
    output_json = tmp_path / "out" / "summary.json"
    output_md = tmp_path / "out" / "summary.md"

    payload = module.assess_surface_repair(source_benchmark_summary=source, repaired_benchmark_summary=repaired)
    module.write_report(payload, output_json=output_json, output_md=output_md)

    assert json.loads(output_json.read_text(encoding="utf-8"))["kind"] == "stage5_surface_repair_assessment"
    assert "arc_easy_content" in output_md.read_text(encoding="utf-8")


def test_order_sensitivity_repair_assessment_detects_reduction(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source_order.json"
    repaired = tmp_path / "repaired_order.json"
    write_json(
        source,
        {
            "summary": {
                "candidate_order_sensitive_rows": 10,
                "candidate_order_sensitive_fraction": 0.5,
                "content_losses": 6,
                "content_losses_order_sensitive": 4,
                "content_losses_order_sensitive_fraction": 0.67,
                "content_losses_rescued_by_cyclic": 3,
                "order_sensitivity_loss_rate_lift": 0.3,
            }
        },
    )
    write_json(
        repaired,
        {
            "summary": {
                "candidate_order_sensitive_rows": 6,
                "candidate_order_sensitive_fraction": 0.3,
                "content_losses": 5,
                "content_losses_order_sensitive": 2,
                "content_losses_order_sensitive_fraction": 0.4,
                "content_losses_rescued_by_cyclic": 2,
                "order_sensitivity_loss_rate_lift": 0.1,
            }
        },
    )

    payload = module.assess_order_sensitivity_repair(
        source_order_diagnosis=source,
        repaired_order_diagnosis=repaired,
    )

    assert payload["status"] == "order_sensitivity_reduced"
    assert payload["improved"] is True
    assert payload["deltas_repaired_minus_source"]["content_losses_order_sensitive"] == -2


def test_surface_repair_assessment_can_embed_order_sensitivity_repair(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source_benchmark, repaired_benchmark = make_benchmark_pair(tmp_path)
    source_order = tmp_path / "source_order.json"
    repaired_order = tmp_path / "repaired_order.json"
    write_json(source_order, {"summary": {"candidate_order_sensitive_rows": 3, "content_losses": 2, "content_losses_order_sensitive": 2}})
    write_json(repaired_order, {"summary": {"candidate_order_sensitive_rows": 1, "content_losses": 2, "content_losses_order_sensitive": 1}})

    payload = module.assess_surface_repair(
        source_benchmark_summary=source_benchmark,
        repaired_benchmark_summary=repaired_benchmark,
        source_order_diagnosis=source_order,
        repaired_order_diagnosis=repaired_order,
    )

    assert payload["order_sensitivity_repair"]["status"] == "order_sensitivity_reduced"
