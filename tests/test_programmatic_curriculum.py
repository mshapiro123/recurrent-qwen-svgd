from __future__ import annotations

import json
import re

from training.generate_programmatic_curriculum import (
    generate_records,
    main,
    make_arithmetic_record,
)
from training.prepare_curriculum_jsonl import convert_curriculum_records, validate_curriculum_record


def answer_from_trace(text: str) -> str:
    match = re.search(r"ANSWER:\s*(-?\d+)", text)
    assert match, text
    return match.group(1)


def test_make_arithmetic_record_is_verified_and_depth_typed() -> None:
    import random

    record = make_arithmetic_record(
        rng=random.Random(3),
        index=0,
        mode="deep_narrow",
        step_range=(6, 6),
        max_abs_value=500,
        max_target_loops=4,
    )

    assert validate_curriculum_record(record) == []
    assert record["mode"] == "deep_narrow"
    assert record["depth"]["min_steps"] == 6
    assert record["target_loop_count"] == 3
    assert record["width_signature"]["width"] == 1
    assert record["answer"]["value"] == answer_from_trace(record["traces"][0]["text"])
    assert record["traces"][0]["role"] == "positive_depth"
    assert record["traces"][0]["answer_match"] == {
        "matched": True,
        "source": "constructed_python_eval",
        "parsed_answer": record["answer"]["value"],
        "parsed_answer_normalized": record["answer"]["value"],
        "verified_answer_normalized": record["answer"]["value"],
    }


def test_direct_records_get_shallow_loop_targets() -> None:
    import random

    record = make_arithmetic_record(
        rng=random.Random(4),
        index=0,
        mode="direct",
        step_range=(1, 1),
        max_abs_value=500,
        max_target_loops=4,
    )

    assert record["target_loop_count"] == 1
    assert record["traces"][0]["role"] == "positive_direct"
    assert record["traces"][0]["answer_match"]["matched"] is True


def test_generate_records_mixes_direct_and_deep_narrow_and_converts_to_sft() -> None:
    records = generate_records(
        num_direct=3,
        num_deep_narrow=4,
        direct_steps=(1, 2),
        deep_steps=(5, 7),
        seed=9,
        max_abs_value=500,
        max_target_loops=4,
    )

    assert len(records) == 7
    assert {record["mode"] for record in records} == {"direct", "deep_narrow"}
    assert all(validate_curriculum_record(record) == [] for record in records)

    examples, report = convert_curriculum_records(records)
    assert len(examples) == 7
    assert report["exported_examples"] == 7
    assert set(report["exported_role_counts"]) == {"positive_depth", "positive_direct"}
    assert {example["routing_type"] for example in examples} == {"direct", "deep_narrow"}


def test_programmatic_curriculum_cli_writes_jsonl_and_report(tmp_path) -> None:
    output_jsonl = tmp_path / "typed.jsonl"
    report_json = tmp_path / "report.json"

    assert main(
        [
            "--output_jsonl",
            str(output_jsonl),
            "--report_json",
            str(report_json),
            "--num_direct",
            "2",
            "--num_deep_narrow",
            "2",
            "--seed",
            "11",
        ]
    ) == 0

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert report["records"] == 4
    assert report["by_mode"] == {"deep_narrow": 2, "direct": 2}


def test_programmatic_depth_runner_finds_nested_arc_mix_checkpoint() -> None:
    from colab.run_stage5_programmatic_depth_repair import checkpoint_from_payload

    assert checkpoint_from_payload(
        {
            "kind": "stage5_routing_repair",
            "arc_mix": {
                "best_arm": {
                    "best_checkpoint": {
                        "checkpoint": "outputs/stage5/child/phase1/phase1_step_150.pt",
                    }
                }
            },
        }
    ) == "outputs/stage5/child/phase1/phase1_step_150.pt"
