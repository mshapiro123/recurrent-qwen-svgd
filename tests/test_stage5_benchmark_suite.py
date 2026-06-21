from __future__ import annotations

import json

from colab.run_stage5_benchmark_suite import (
    EvalJob,
    BenchmarkSpec,
    benchmark_specs,
    build_summary,
    checkpoint_candidates_from_payload,
    compare_arm_summaries,
    paired_arm_summaries,
    parse_csv,
    parse_optional_limit,
    resolve_checkpoint,
    summarize_rows,
    two_sided_sign_p_value,
)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_parse_csv_trims_empty_items() -> None:
    assert parse_csv(" arc_challenge, gpqa_lite ,,") == ["arc_challenge", "gpqa_lite"]


def test_parse_optional_limit_accepts_full_aliases() -> None:
    assert parse_optional_limit("256") == 256
    assert parse_optional_limit("full") is None
    assert parse_optional_limit("all") is None
    assert parse_optional_limit("0") is None


def test_summarize_rows_groups_by_aggregate() -> None:
    rows = [
        {"aggregate": "mean", "hit": True},
        {"aggregate": "mean", "hit": False},
        {"aggregate": "vote", "hit": True},
    ]

    summary = summarize_rows(rows)

    assert summary["mean"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert summary["vote"] == {"correct": 1, "total": 1, "accuracy": 1.0}


def test_compare_arm_summaries_reports_recurrent_delta() -> None:
    comparison = compare_arm_summaries(
        {"mean": {"correct": 3, "total": 5, "accuracy": 0.6}},
        {"mean": {"correct": 4, "total": 5, "accuracy": 0.8}},
    )

    assert comparison["mean"]["correct_delta_recurrent_vs_base"] == 1
    assert round(comparison["mean"]["accuracy_delta_recurrent_vs_base"], 4) == 0.2


def test_paired_arm_summaries_report_wins_losses_and_sign_test() -> None:
    paired = paired_arm_summaries(
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": False},
            {"id": "c", "aggregate": "mean", "hit": True},
            {"id": "base_only", "aggregate": "mean", "hit": True},
        ],
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": True},
            {"id": "c", "aggregate": "mean", "hit": False},
            {"id": "recurrent_only", "aggregate": "mean", "hit": True},
        ],
    )

    assert paired["mean"]["paired_examples"] == 3
    assert paired["mean"]["base_correct"] == 2
    assert paired["mean"]["recurrent_correct"] == 2
    assert paired["mean"]["wins"] == 1
    assert paired["mean"]["losses"] == 1
    assert paired["mean"]["ties"] == 1
    assert paired["mean"]["sign_test_p_value"] == 1.0


def test_sign_test_returns_none_without_disagreements() -> None:
    assert two_sided_sign_p_value(0, 0) is None


def test_checkpoint_candidates_include_hf_export_checkpoint(tmp_path) -> None:
    source = tmp_path / "outputs" / "hf_exports" / "export" / "summary.json"
    payload = {
        "checkpoint": "missing/original.pt",
        "export_dir": str(source.parent),
    }

    candidates = checkpoint_candidates_from_payload(source, payload)

    assert source.parent / "recurrent_adapter_checkpoint.pt" in candidates


def test_resolve_checkpoint_prefers_existing_export_adapter(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    source = tmp_path / "outputs" / "hf_exports" / "export" / "summary.json"
    checkpoint = source.parent / "recurrent_adapter_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "checkpoint": str(tmp_path / "missing.pt"),
        "export_dir": str(source.parent),
    }
    monkeypatch.setattr(module, "EXPLICIT_CHECKPOINT", "")

    assert resolve_checkpoint(source, payload) == checkpoint


def test_build_summary_compares_base_and_recurrent_rows(tmp_path) -> None:
    base_jsonl = tmp_path / "arc_base_label.jsonl"
    recurrent_jsonl = tmp_path / "arc_recurrent_label.jsonl"
    _write_jsonl(
        base_jsonl,
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": False},
        ],
    )
    _write_jsonl(
        recurrent_jsonl,
        [
            {"id": "a", "aggregate": "mean", "hit": True},
            {"id": "b", "aggregate": "mean", "hit": True},
        ],
    )

    payload = build_summary(
        source_summary=None,
        checkpoint=tmp_path / "checkpoint.pt",
        specs=[BenchmarkSpec("arc_challenge", tmp_path / "arc.jsonl", [])],
        jobs=[
            EvalJob("arc_challenge", "base", "label", base_jsonl, []),
            EvalJob("arc_challenge", "recurrent", "label", recurrent_jsonl, []),
        ],
        failures=[],
        elapsed_seconds=12.5,
    )

    delta = payload["comparisons"]["arc_challenge"]["label"]["mean"]
    paired = payload["paired_comparisons"]["arc_challenge"]["label"]["mean"]
    assert payload["status"] == "completed"
    assert delta["correct_delta_recurrent_vs_base"] == 1
    assert delta["base"]["correct"] == 1
    assert delta["recurrent"]["correct"] == 2
    assert paired["paired_examples"] == 2
    assert paired["wins"] == 1
    assert paired["losses"] == 0


def test_eval_jobs_passes_phase2_svgd_flags(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    checkpoint = tmp_path / "phase2.pt"
    data = tmp_path / "arc.jsonl"
    monkeypatch.setattr(module, "RECURRENT_MODE", "phase2")
    monkeypatch.setattr(module, "RECURRENT_NUM_TRAJECTORIES", 4)
    monkeypatch.setattr(module, "RECURRENT_SAMPLE_LATENTS", True)
    monkeypatch.setattr(module, "RECURRENT_PARTICLE_UPDATE_MODE", "svgd")
    monkeypatch.setattr(module, "RECURRENT_PARTICLE_INIT_NOISE", "0.05")
    monkeypatch.setattr(module, "RECURRENT_SVGD_REPULSION_SCALE", "2.0")
    monkeypatch.setattr(module, "RECURRENT_SVGD_REPULSION_MAX_NORM", "none")
    monkeypatch.setattr(module, "RECURRENT_SVGD_KERNEL_PROJECTION_DIM", "8")
    monkeypatch.setattr(module, "RECURRENT_SVGD_KERNEL_PROJECTION_PATH", "outputs/calibration/proj.pt")
    monkeypatch.setattr(module, "RECURRENT_SVGD_KERNEL_GEOMETRY", "euclidean")

    jobs = module.eval_jobs([BenchmarkSpec("arc_challenge", data, [])], checkpoint=checkpoint)
    recurrent_cmd = next(job.cmd for job in jobs if job.arm == "recurrent")

    assert "--mode" in recurrent_cmd
    assert recurrent_cmd[recurrent_cmd.index("--mode") + 1] == "phase2"
    assert "--sample_latents" in recurrent_cmd
    assert recurrent_cmd[recurrent_cmd.index("--num_trajectories") + 1] == "4"
    assert recurrent_cmd[recurrent_cmd.index("--particle_update_mode") + 1] == "svgd"
    assert recurrent_cmd[recurrent_cmd.index("--particle_init_noise") + 1] == "0.05"
    assert recurrent_cmd[recurrent_cmd.index("--svgd_repulsion_scale") + 1] == "2.0"
    assert recurrent_cmd[recurrent_cmd.index("--svgd_repulsion_max_norm") + 1] == "none"
    assert recurrent_cmd[recurrent_cmd.index("--svgd_kernel_projection_dim") + 1] == "8"
    assert recurrent_cmd[recurrent_cmd.index("--svgd_kernel_projection_path") + 1] == "outputs/calibration/proj.pt"


def test_benchmark_specs_supports_arc_easy(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "ARC_EASY_LIMIT", None)

    spec = benchmark_specs(["arc_easy"])[0]

    assert spec.name == "arc_easy"
    assert spec.data_jsonl == tmp_path / "arc_easy_validation_full.jsonl"
    assert "ARC-Easy" in spec.prepare_cmd
    assert "--limit" not in spec.prepare_cmd


def test_benchmark_specs_supports_limited_arc_easy(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "ARC_EASY_LIMIT", 64)

    spec = benchmark_specs(["arc_easy"])[0]

    assert spec.data_jsonl == tmp_path / "arc_easy_validation_64.jsonl"
    assert spec.prepare_cmd[spec.prepare_cmd.index("--limit") + 1] == "64"
