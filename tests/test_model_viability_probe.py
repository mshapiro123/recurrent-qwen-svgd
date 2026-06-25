import json
from pathlib import Path

from colab.run_stage5_model_viability_queue import (
    ModelProbeSpec,
    assess_child,
    parse_model_queue,
    should_skip_for_vram,
)
from colab.run_stage5_model_viability_probe import parse_int_csv, summarize_pair


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def row(row_id: str, hit: bool, aggregate: str = "mean", expected_loops=None) -> dict:
    payload = {
        "id": row_id,
        "hit": hit,
        "aggregate": aggregate,
    }
    if expected_loops is not None:
        payload["loop_diagnostics"] = {"mean_expected_loops": expected_loops}
    return payload


def test_parse_int_csv_requires_at_least_one_loop():
    assert parse_int_csv("1,2, 3") == [1, 2, 3]

    try:
        parse_int_csv(" ")
    except ValueError as exc:
        assert "Expected at least one loop" in str(exc)
    else:
        raise AssertionError("parse_int_csv should reject an empty loop list")


def test_summarize_pair_reports_paired_delta_and_loop_telemetry(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    recurrent = tmp_path / "recurrent.jsonl"
    write_jsonl(
        base,
        [
            row("a", True),
            row("b", False),
            row("c", True),
        ],
    )
    write_jsonl(
        recurrent,
        [
            row("a", True, expected_loops=1.0),
            row("b", True, expected_loops=2.0),
            row("c", False, expected_loops=3.0),
        ],
    )

    summary = summarize_pair(base, recurrent)["mean"]

    assert summary["paired_examples"] == 3
    assert summary["base_correct"] == 2
    assert summary["recurrent_correct"] == 2
    assert summary["delta"] == 0
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1
    assert summary["mean_expected_loops"] == 2.0


def test_parse_model_queue_supports_generic_qwen_sizes():
    specs = parse_model_queue(
        "qwen_3b=Qwen/Qwen2.5-3B-Instruct|auto|1,2|22|32|32|float32;"
        "qwen_7b=Qwen/Qwen2.5-7B-Instruct|auto|1,2,3|39|24|24|float32"
    )

    assert [spec.label for spec in specs] == ["qwen_3b", "qwen_7b"]
    assert specs[0].model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert specs[0].min_vram_gb == 22
    assert specs[1].loops == "1,2,3"
    assert specs[1].arc_easy_limit == "24"


def test_model_queue_skip_logic_is_vram_aware():
    spec = ModelProbeSpec(label="qwen_7b", model_name="Qwen/Qwen2.5-7B-Instruct", min_vram_gb=39)

    assert should_skip_for_vram(spec, available_vram_gb=22.5)
    assert should_skip_for_vram(spec, available_vram_gb=40.0) is None
    assert should_skip_for_vram(spec, available_vram_gb=22.5, allow_insufficient_vram=True) is None


def test_assess_child_promotes_identity_passed_loop1_preserving_scale():
    summary = {
        "identity": {"passed": True},
        "comparisons": {
            "arc_easy": {
                "label": {
                    "1": {"mean": {"delta": -1}},
                    "2": {"mean": {"delta": 0}},
                }
            },
            "arc_challenge": {
                "label": {
                    "1": {"mean": {"delta": 1}},
                    "2": {"mean": {"delta": -1}},
                }
            },
        },
    }

    assessment = assess_child(summary, returncode=0)

    assert assessment["status"] == "viable_for_training_probe"
    assert assessment["loop1_min_delta"] == -1
    assert assessment["loop2_min_delta"] == -1
    assert assessment["promote_to_training_probe"] is True


def test_assess_child_rejects_large_loop1_regression():
    summary = {
        "identity": {"passed": True},
        "comparisons": {
            "arc_easy": {"label": {"1": {"mean": {"delta": -2}}}},
        },
    }

    assessment = assess_child(summary, returncode=0)

    assert assessment["status"] == "loop1_regression_too_large"
    assert assessment["promote_to_training_probe"] is False


def test_model_viability_outputs_are_information_only_scale_probes():
    probe = (ROOT / "colab/run_stage5_model_viability_probe.py").read_text(encoding="utf-8")
    queue = (ROOT / "colab/run_stage5_model_viability_queue.py").read_text(encoding="utf-8")

    assert '"program_phase": "standing_scale_probe"' in probe
    assert '"program_phase": "standing_scale_probe"' in queue
    assert "Information-only scale probe" in probe
    assert "information only; this does not unlock Stage 4 recovery" in queue
