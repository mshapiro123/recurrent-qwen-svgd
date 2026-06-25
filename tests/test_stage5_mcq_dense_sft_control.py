from __future__ import annotations

import json
import subprocess

import yaml

from colab import run_stage5_mcq_dense_sft_control as module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_source_positive_sft_path_prefers_dataset_source(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "dataset": {
            "source_positive_sft": "data/curriculum/run/positive_sft.jsonl",
        },
        "gate": {
            "artifacts": {
                "positive_sft": "/wrong/positive_sft.jsonl",
            }
        },
    }

    assert module.source_positive_sft_path(payload) == tmp_path / "data" / "curriculum" / "run" / "positive_sft.jsonl"


def test_source_positive_sft_path_falls_back_to_gate_artifact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "gate": {
            "artifacts": {
                "positive_sft": "data/curriculum/trace/positive_sft.jsonl",
            }
        },
    }

    assert module.source_positive_sft_path(payload) == tmp_path / "data" / "curriculum" / "trace" / "positive_sft.jsonl"


def test_source_positive_sft_path_follows_benchmark_source_chain(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    train_summary = tmp_path / "outputs" / "stage5" / "train" / "summary.json"
    benchmark_summary = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    train_summary.parent.mkdir(parents=True)
    benchmark_summary.parent.mkdir(parents=True)
    train_summary.write_text(
        json.dumps({"dataset": {"source_positive_sft": "data/curriculum/train/positive_sft.jsonl"}}),
        encoding="utf-8",
    )
    benchmark_summary.write_text(
        json.dumps({"kind": "stage5_benchmark_suite", "source_summary": module.path_for_cli(train_summary)}),
        encoding="utf-8",
    )

    assert module.source_positive_sft_path(json.loads(benchmark_summary.read_text(encoding="utf-8"))) == (
        tmp_path / "data" / "curriculum" / "train" / "positive_sft.jsonl"
    )


def test_source_summary_path_defaults_to_current_pointer(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "")
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    summary = tmp_path / "outputs" / "stage5" / "current" / "summary.json"
    pointer.parent.mkdir(parents=True)
    summary.parent.mkdir(parents=True)
    pointer.write_text("outputs/stage5/current/summary.json\n", encoding="utf-8")
    summary.write_text(json.dumps({"kind": "stage5_reentry_repair_smoke"}), encoding="utf-8")

    assert module.source_summary_path() == summary


def test_recurrent_benchmark_summary_resolves_current_assessment_source(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "")
    monkeypatch.setattr(module, "RECURRENT_BENCHMARK_SUMMARY", "")
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    assessment = tmp_path / "outputs" / "stage5" / "assess" / "summary.json"
    benchmark = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    pointer.parent.mkdir(parents=True)
    assessment.parent.mkdir(parents=True)
    benchmark.parent.mkdir(parents=True)
    pointer.write_text("outputs/stage5/assess/summary.json\n", encoding="utf-8")
    assessment.write_text(
        json.dumps(
            {
                "gate": "stage5_broader_benchmark_suite",
                "source_summary": "outputs/stage5/bench/summary.json",
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"kind": "stage5_benchmark_suite"}), encoding="utf-8")

    assert module.recurrent_benchmark_summary_path() == benchmark


def test_recurrent_benchmark_summary_resolves_explicit_assessment_source(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RECURRENT_BENCHMARK_SUMMARY", "outputs/stage5/assess/summary.json")
    assessment = tmp_path / "outputs" / "stage5" / "assess" / "summary.json"
    benchmark = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    assessment.parent.mkdir(parents=True)
    benchmark.parent.mkdir(parents=True)
    assessment.write_text(
        json.dumps(
            {
                "kind": "stage5_broader_benchmark_assessment",
                "source_summary": "outputs/stage5/bench/summary.json",
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(json.dumps({"kind": "stage5_benchmark_suite"}), encoding="utf-8")

    assert module.recurrent_benchmark_summary_path() == benchmark


def test_resolve_curriculum_source_follows_benchmark_to_training_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    train_summary = tmp_path / "outputs" / "stage5" / "train" / "summary.json"
    benchmark_summary = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    train_summary.parent.mkdir(parents=True)
    benchmark_summary.parent.mkdir(parents=True)
    train_summary.write_text(
        json.dumps(
            {
                "kind": "stage5_reentry_recovery_training",
                "dataset": {"source_positive_sft": "data/curriculum/train/positive_sft.jsonl"},
                "config": {"depth_hint_style": "natural", "max_steps": 37, "learning_rate": 5e-5},
            }
        ),
        encoding="utf-8",
    )
    benchmark_summary.write_text(
        json.dumps(
            {
                "kind": "stage5_benchmark_suite",
                "source_summary": module.path_for_cli(train_summary),
                "config": {"depth_hint_style": "none", "max_steps": 999, "learning_rate": 9e-4},
            }
        ),
        encoding="utf-8",
    )

    resolved_path, resolved_payload = module.resolve_curriculum_source(
        json.loads(benchmark_summary.read_text(encoding="utf-8")),
        source_path=benchmark_summary,
    )

    assert resolved_path == train_summary
    assert resolved_payload["kind"] == "stage5_reentry_recovery_training"
    assert resolved_payload["config"]["depth_hint_style"] == "natural"
    assert resolved_payload["config"]["max_steps"] == 37
    assert resolved_payload["config"]["learning_rate"] == 5e-5


def test_resolve_curriculum_source_accepts_stage4_curriculum_sft_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(module, "ROOT", tmp_path)
    train_summary = tmp_path / "outputs" / "stage5" / "stage4" / "summary.json"
    benchmark_summary = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    train_summary.parent.mkdir(parents=True)
    benchmark_summary.parent.mkdir(parents=True)
    train_summary.write_text(
        json.dumps(
            {
                "kind": "stage5_curriculum_sft",
                "phase1_checkpoint": "outputs/stage5/stage4/phase1/phase1_step_75.pt",
                "dataset": {"source_positive_sft": "data/curriculum/stage4/positive_sft.jsonl"},
                "config": {"depth_hint_style": "natural", "max_steps": 75, "learning_rate": 5e-6},
            }
        ),
        encoding="utf-8",
    )
    benchmark_summary.write_text(
        json.dumps({"kind": "stage5_benchmark_suite", "source_summary": module.path_for_cli(train_summary)}),
        encoding="utf-8",
    )

    resolved_path, resolved_payload = module.resolve_curriculum_source(
        json.loads(benchmark_summary.read_text(encoding="utf-8")),
        source_path=benchmark_summary,
    )

    assert resolved_path == train_summary
    assert resolved_payload["kind"] == "stage5_curriculum_sft"
    assert module.source_positive_sft_path(resolved_payload) == (
        tmp_path / "data" / "curriculum" / "stage4" / "positive_sft.jsonl"
    )
    assert resolved_payload["config"]["max_steps"] == 75
    assert resolved_payload["config"]["learning_rate"] == 5e-6


def test_paired_dense_vs_base_reports_wins_losses_and_ties(tmp_path) -> None:
    base = tmp_path / "base.jsonl"
    dense = tmp_path / "dense.jsonl"
    write_jsonl(
        base,
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": False},
            {"id": "c", "aggregate": "mean", "hit": True},
            {"id": "d", "aggregate": "mean", "hit": False},
        ],
    )
    write_jsonl(
        dense,
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": True},
            {"id": "c", "aggregate": "mean", "hit": False},
            {"id": "d", "aggregate": "mean", "hit": False},
        ],
    )

    paired = module.paired_dense_vs_base(base, dense)["mean"]

    assert paired["paired_examples"] == 4
    assert paired["base_correct"] == 2
    assert paired["dense_correct"] == 2
    assert paired["correct_delta_dense_vs_base"] == 0
    assert paired["wins"] == 1
    assert paired["losses"] == 1
    assert paired["ties"] == 2


def test_prepare_train_val_appends_extra_repair_rows_to_train_only(monkeypatch, tmp_path) -> None:
    source = tmp_path / "data" / "trace" / "positive_sft.jsonl"
    extra = tmp_path / "data" / "repair" / "surface_alignment_train.jsonl"
    write_jsonl(
        source,
        [
            {"id": "source_0", "prompt": "p0", "completion": "c0", "curriculum_mode": "direct", "target_loop_count": 1},
            {"id": "source_1", "prompt": "p1", "completion": "c1", "curriculum_mode": "deep_narrow", "target_loop_count": 3},
            {"id": "source_2", "prompt": "p2", "completion": "c2", "curriculum_mode": "direct", "target_loop_count": 1},
            {"id": "source_3", "prompt": "p3", "completion": "c3", "curriculum_mode": "deep_narrow", "target_loop_count": 3},
        ],
    )
    write_jsonl(
        extra,
        [
            {
                "id": "repair_0",
                "prompt": "rp",
                "completion": "rc",
                "curriculum_mode": "surface_alignment",
                "target_loop_count": 1,
            }
        ],
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", tmp_path / "private")
    monkeypatch.setattr(module, "EXTRA_TRAIN_JSONL_ENV", str(extra))
    monkeypatch.setattr(module, "VAL_FRACTION", 0.25)
    monkeypatch.setattr(module, "VAL_MIN_ROWS", 1)

    train_jsonl, val_jsonl, dataset = module.prepare_train_val(
        {"dataset": {"source_positive_sft": str(source)}}
    )

    train_rows = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(line) for line in val_jsonl.read_text(encoding="utf-8").splitlines()]
    assert dataset["source_rows"] == 4
    assert dataset["extra_train_rows"] == 1
    assert dataset["rows"] == 5
    assert dataset["extra_train_jsonls"] == [
        {
            "path": "data/repair/surface_alignment_train.jsonl",
            "rows": 1,
            "mode_counts": {"surface_alignment": 1},
            "target_loop_counts": {"1": 1},
        }
    ]
    assert dataset["source_mode_counts"] == {"direct": 2, "deep_narrow": 2}
    assert dataset["source_target_loop_counts"] == {"1": 2, "3": 2}
    assert dataset["extra_train_mode_counts"] == {"surface_alignment": 1}
    assert dataset["extra_train_target_loop_counts"] == {"1": 1}
    assert dataset["train_target_loop_counts"] == {"1": 3, "3": 1}
    assert dataset["val_target_loop_counts"] == {"3": 1}
    assert any(row["id"] == "repair_0" for row in train_rows)
    assert all(row["id"] != "repair_0" for row in val_rows)


def test_prepare_train_val_uses_resolved_curriculum_depth_hints(monkeypatch, tmp_path) -> None:
    source = tmp_path / "data" / "trace" / "positive_sft.jsonl"
    write_jsonl(
        source,
        [
            {"id": "source_0", "prompt": "p0", "completion": "c0", "curriculum_mode": "direct", "target_loop_count": 1},
            {"id": "source_1", "prompt": "p1", "completion": "c1", "curriculum_mode": "deep_narrow", "target_loop_count": 3},
        ],
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", tmp_path / "private")
    monkeypatch.setattr(module, "EXTRA_TRAIN_JSONL_ENV", "")
    monkeypatch.setattr(module, "VAL_FRACTION", 0.5)
    monkeypatch.setattr(module, "VAL_MIN_ROWS", 1)
    monkeypatch.delenv("STAGE5_DENSE_MCQ_DEPTH_HINT_STYLE", raising=False)

    train_jsonl, val_jsonl, dataset = module.prepare_train_val(
        {
            "dataset": {"source_positive_sft": str(source)},
            "config": {"depth_hint_style": "natural"},
        }
    )

    all_rows = [
        json.loads(line)
        for path in (train_jsonl, val_jsonl)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert dataset["depth_hint_style"] == "natural"
    assert {row["depth_hint_style"] for row in all_rows} == {"natural"}
    assert all(str(row["prompt"]).startswith("Depth hint:") for row in all_rows)


def test_train_dense_lora_inherits_resolved_curriculum_training_defaults(monkeypatch, tmp_path) -> None:
    captured_config = {}

    def fake_run(cmd, *, check=True, log_name=None):
        cfg_path = tmp_path / cmd[cmd.index("--config") + 1]
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        captured_config.update(cfg)
        checkpoint = tmp_path / cfg["output_dir"] / f"dense_lora_step_{cfg['max_steps']}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "outputs" / "stage5" / "dense")
    monkeypatch.setattr(module, "TRAIN_STEPS_ENV", "")
    monkeypatch.setattr(module, "SAVE_EVERY_ENV", "")
    monkeypatch.setattr(module, "LEARNING_RATE_ENV", "")
    monkeypatch.setattr(module, "run", fake_run)

    checkpoint = module.train_dense_lora(
        tmp_path / "train.jsonl",
        {"config": {"max_steps": 37, "learning_rate": 5e-5}},
    )

    assert checkpoint.exists()
    assert captured_config["max_steps"] == 37
    assert captured_config["save_every"] == 37
    assert captured_config["learning_rate"] == 5e-5


def test_write_summary_updates_current_pointer(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "dense"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    payload = {
        "run_id": "dense",
        "source_summary": "outputs/source/summary.json",
        "dataset": {"rows": 3},
        "dense_checkpoint": "outputs/dense/dense_lora.pt",
        "config": {"dense_lora_layer_range": "6,18"},
        "paired_comparisons": {
            "arc_easy": {
                "content_question_only": {
                    "mean": {
                        "dense_correct": 2,
                        "base_correct": 1,
                        "paired_examples": 3,
                        "correct_delta_dense_vs_base": 1,
                        "wins": 1,
                        "losses": 0,
                        "ties": 2,
                        "sign_test_p_value": 1.0,
                    }
                }
            }
        },
    }

    module.write_summary(payload)

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip() == (
        "outputs/stage5/dense/summary.json"
    )


def test_write_summary_points_front_of_queue_to_recipe_assessment(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "dense"
    assessment = run_dir / "mcq_recipe_control_assessment.json"
    assessment.parent.mkdir(parents=True)
    assessment.write_text(json.dumps({"kind": "stage5_mcq_recipe_control_assessment"}), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    payload = {
        "run_id": "dense",
        "source_summary": "outputs/source/summary.json",
        "dataset": {"rows": 3},
        "dense_checkpoint": "outputs/dense/dense_lora.pt",
        "config": {"dense_lora_layer_range": "6,18"},
        "paired_comparisons": {},
        "recipe_control_assessment": {
            "ran": True,
            "summary_json": "outputs/stage5/dense/mcq_recipe_control_assessment.json",
        },
    }

    module.write_summary(payload)

    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(encoding="utf-8").strip() == (
        "outputs/stage5/dense/mcq_recipe_control_assessment.json"
    )


def test_commit_results_excludes_checkpoint_unless_explicitly_enabled(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "dense"
    checkpoint = run_dir / "dense_lora" / "dense_lora_step_10.pt"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    checkpoint.parent.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    pointer.write_text("outputs/stage5/dense/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "COMMIT_CHECKPOINT", False)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results(checkpoint)

    add_commands = [cmd for cmd in commands if cmd[:3] == ["git", "add", "-f"]]
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/dense/summary.json" in staged
    assert "outputs/stage5/dense/summary.md" in staged
    assert "config/stage5_current_source_summary.txt" in staged
    assert "outputs/stage5/dense/dense_lora/dense_lora_step_10.pt" not in staged
