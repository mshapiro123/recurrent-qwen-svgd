from __future__ import annotations

import json

import colab.build_stage5_arc_agi_reproduced_baseline_registry as builder
import colab.validate_arc_agi_baseline_registry as validator


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metadata() -> dict:
    return {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "arc_version": "1",
        "arc_split": "evaluation",
        "params_b": 0.5,
    }


def test_reproduced_registry_builder_handles_direct_summary_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    summary_path = tmp_path / "outputs" / "stage5" / "run" / "summary.json"
    _write(
        summary_path,
        {
            "run_id": "stage5_arc_agi_sft",
            "metadata": _metadata(),
            "base": {
                "selected_exact": 7,
                "best_of_k_exact": 8,
                "first_exact": 6,
                "examples_with_targets": 10,
            },
        },
    )

    registry = builder.build_registry(
        summary_paths=[summary_path],
        labels=["base"],
        metric="selected_accuracy",
        benchmark="ARC-AGI public evaluation",
        min_params_b=0.3,
        max_params_b=1.0,
        name_prefix="reproduced",
        reproduction_command="python colab/run_stage5_arc_agi_sft.py",
        git_commit="abc1234",
        accessed_date="2026-06-21",
    )
    validation = validator.validate_registry_payload(registry, source_path=tmp_path / "config" / "baselines.json")

    assert validation["status"] == "passed"
    row = registry["baselines"][0]
    assert row["name"] == "reproduced-qwen-qwen2-5-0-5b-instruct-base"
    assert row["accuracy"] == 0.7
    assert row["source_artifact"] == "outputs/stage5/run/summary.json"


def test_reproduced_registry_builder_handles_nested_dense_control_blocks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    summary_path = tmp_path / "outputs" / "stage5" / "dense" / "summary.json"
    _write(
        summary_path,
        {
            "run_id": "dense_sft_control",
            "kind": "dense_sft_control",
            "metadata": _metadata(),
            "dense_tuned": {
                "summary": {
                    "selected_exact": 9,
                    "best_of_k_exact": 9,
                    "first_exact": 9,
                    "examples_with_targets": 10,
                }
            },
        },
    )

    registry = builder.build_registry(
        summary_paths=[summary_path],
        labels=["dense_tuned"],
        metric="selected_accuracy",
        benchmark="ARC-AGI public evaluation",
        min_params_b=0.3,
        max_params_b=1.0,
        name_prefix="reproduced",
        reproduction_command=None,
        git_commit="abc1234",
        accessed_date="2026-06-21",
    )
    row = registry["baselines"][0]

    assert row["accuracy"] == 0.9
    assert row["reproduction_command"] == "python colab/run_stage5_arc_agi_dense_sft.py"
    assert validator.validate_registry_payload(registry, source_path=tmp_path / "config" / "baselines.json")[
        "passed"
    ]


def test_reproduced_registry_builder_cli_writes_validator_passing_registry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "current_git_commit", lambda: "abc1234")
    summary_path = tmp_path / "outputs" / "stage5" / "run" / "summary.json"
    output_json = tmp_path / "config" / "arc_agi_same_size_baselines.json"
    validation_json = tmp_path / "outputs" / "stage5" / "registry" / "summary.json"
    _write(
        summary_path,
        {
            "run_id": "stage5_arc_agi_sft",
            "metadata": _metadata(),
            "base": {
                "selected_exact": 7,
                "best_of_k_exact": 8,
                "first_exact": 6,
                "examples_with_targets": 10,
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_stage5_arc_agi_reproduced_baseline_registry.py",
            "--summary_json",
            str(summary_path),
            "--labels",
            "base",
            "--output_json",
            str(output_json),
            "--validation_json",
            str(validation_json),
        ],
    )

    assert builder.main() == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    validation = validator.validate_registry_payload(payload, source_path=output_json)

    assert validation["passed"] is True
    assert payload["baselines"][0]["git_commit"] == "abc1234"
    validation_payload = json.loads(validation_json.read_text(encoding="utf-8"))
    assert validation_payload["passed"] is True
    assert validation_payload["criteria"][0]["passed"] is True


def test_reproduced_registry_builder_rejects_missing_arc_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    summary_path = tmp_path / "outputs" / "stage5" / "run" / "summary.json"
    _write(
        summary_path,
        {
            "run_id": "bad",
            "metadata": {"params_b": 0.5},
            "base": {"selected_exact": 1, "examples_with_targets": 10},
        },
    )

    try:
        builder.build_registry(
            summary_paths=[summary_path],
            labels=["base"],
            metric="selected_accuracy",
            benchmark="ARC-AGI public evaluation",
            min_params_b=0.3,
            max_params_b=1.0,
            name_prefix="reproduced",
            reproduction_command="python colab/run_stage5_arc_agi_sft.py",
            git_commit="abc1234",
            accessed_date="2026-06-21",
        )
    except ValueError as exc:
        assert "metadata.arc_version" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected missing ARC metadata to fail")
