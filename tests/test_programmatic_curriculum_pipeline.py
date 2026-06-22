from __future__ import annotations

import json

from colab.check_stage5_a100_go_no_go import apply_checkpoint_guard, classify_action
from colab.plan_stage5_next_run import plan_next_actions
from training.check_curriculum_sft_gate import build_gate_payload, parse_args
from training.run_programmatic_curriculum_pipeline import main


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_programmatic_curriculum_pipeline_writes_gate_ready_work_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_CURRICULUM_RESUME_FROM", raising=False)
    monkeypatch.delenv("STAGE5_CURRICULUM_WORK_DIR", raising=False)
    monkeypatch.delenv("STAGE5_CURRICULUM_SUMMARY_JSON", raising=False)

    work_dir = tmp_path / "programmatic_direct_deep"

    assert main(
        [
            "--work_dir",
            str(work_dir),
            "--num_direct",
            "3",
            "--num_deep_narrow",
            "4",
            "--seed",
            "21",
        ]
    ) == 0

    summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "programmatic_curriculum_pipeline"
    assert summary["status"] == "complete"
    assert summary["counts"]["typed_records"] == 7
    assert summary["counts"]["positive_sft_rows"] == 7

    typed_rows = read_jsonl(work_dir / "typed_records.jsonl")
    sft_rows = read_jsonl(work_dir / "positive_sft.jsonl")
    assert {row["mode"] for row in typed_rows} == {"direct", "deep_narrow"}
    assert {row["curriculum_mode"] for row in sft_rows} == {"direct", "deep_narrow"}
    assert all(row["trace_role"].startswith("positive_") for row in sft_rows)
    assert all(row["answer_match"]["matched"] is True for row in sft_rows)
    assert all(row["answer_match"]["source"] == "constructed_python_eval" for row in sft_rows)
    assert all(row["source_model"] == "programmatic_generator" for row in sft_rows)

    gate = build_gate_payload(
        parse_args(
            [
                "--summary_json",
                str(work_dir / "summary.json"),
                "--min_positive_rows",
                "7",
                "--min_mode_rows",
                "direct=3,deep_narrow=4",
                "--max_loop_target",
                "4",
            ]
        )
    )
    assert gate["go"] is True
    assert gate["issues"] == []
    assert gate["checks"]["positive_sft"]["mode_requirements"] == {
        "deep_narrow": {"required": 4, "observed": 4, "passed": True},
        "direct": {"required": 3, "observed": 3, "passed": True},
    }

    gate_path = work_dir / "curriculum_sft_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    actions = plan_next_actions(gate, source_summary=gate_path)
    decision = classify_action(actions[0], source_payload=gate)
    guarded, preflight = apply_checkpoint_guard(decision, source_payload=gate)

    assert actions[0]["name"] == "Run generated curriculum recurrent SFT"
    assert "python colab/run_stage5_curriculum_sft.py" in actions[0]["command"]
    assert "STAGE5_CURRICULUM_MIN_MODE_ROWS=deep_narrow=4,direct=3" in actions[0]["command"]
    assert decision["status"] == "go_curriculum_sft"
    assert guarded["go"] is True
    assert guarded["status"] == "go_curriculum_sft"
    assert preflight["available"] is True
    assert preflight["input_preflight"]["local_available"] is True


def test_programmatic_curriculum_pipeline_gate_blocks_wrong_mode_requirement(tmp_path) -> None:
    work_dir = tmp_path / "programmatic_direct_deep"
    assert main(
        [
            "--work_dir",
            str(work_dir),
            "--num_direct",
            "2",
            "--num_deep_narrow",
            "2",
        ]
    ) == 0

    gate = build_gate_payload(
        parse_args(
            [
                "--summary_json",
                str(work_dir / "summary.json"),
                "--min_positive_rows",
                "4",
                "--min_mode_rows",
                "wide=1",
            ]
        )
    )

    assert gate["go"] is False
    assert any(issue["code"] == "too_few_mode_rows" for issue in gate["issues"])
