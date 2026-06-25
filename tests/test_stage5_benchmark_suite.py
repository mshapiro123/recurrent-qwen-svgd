from __future__ import annotations

import json
import subprocess

import pytest

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
    suite_profile_defaults,
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


def test_checkpoint_candidates_include_balanced_arc_mix_best_arm(tmp_path) -> None:
    import colab.run_stage5_benchmark_suite as module

    source = tmp_path / "outputs" / "stage5" / "depth" / "summary.json"
    checkpoint = tmp_path / "outputs" / "stage5" / "depth" / "arm" / "phase1" / "phase1_step_150.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "best_arm": {
            "best_checkpoint": {
                "checkpoint": str(checkpoint),
            }
        }
    }

    monkeypatch = None
    candidates = module.checkpoint_candidates_from_payload(source, payload)

    assert checkpoint in candidates


def test_checkpoint_candidates_include_direct_preservation_best_checkpoint(tmp_path) -> None:
    import colab.run_stage5_benchmark_suite as module

    source = tmp_path / "outputs" / "stage5" / "direct_preserve" / "summary.json"
    checkpoint = tmp_path / "outputs" / "stage5" / "direct_preserve" / "phase1" / "phase1_step_75.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "kind": "stage5_direct_preservation_probe",
        "best_checkpoint": {
            "checkpoint": str(checkpoint),
        },
    }

    candidates = module.checkpoint_candidates_from_payload(source, payload)

    assert checkpoint in candidates


def test_checkpoint_candidates_include_benchmark_suite_checkpoint(tmp_path) -> None:
    import colab.run_stage5_benchmark_suite as module

    source = tmp_path / "outputs" / "stage5" / "bench" / "summary.json"
    checkpoint = tmp_path / "outputs" / "stage5" / "surface" / "phase1_surface_align" / "phase1_step_50.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "kind": "stage5_benchmark_suite",
        "checkpoint": str(checkpoint),
    }

    candidates = module.checkpoint_candidates_from_payload(source, payload)

    assert checkpoint in candidates
    assert module.checkpoint_bearing_source_summary(source, payload) == source


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


def test_resolve_checkpoint_restores_missing_stage5_checkpoint_from_drive(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    source = tmp_path / "outputs" / "stage5" / "run_a" / "summary.json"
    checkpoint = tmp_path / "outputs" / "stage5" / "run_a" / "phase1" / "phase1_step_150.pt"
    drive_checkpoint = tmp_path / "drive" / "run_a" / "run_dir" / "phase1" / "phase1_step_150.pt"
    drive_checkpoint.parent.mkdir(parents=True)
    drive_checkpoint.write_bytes(b"checkpoint")
    payload = {"phase1_checkpoint": str(checkpoint)}

    monkeypatch.setattr(module, "EXPLICIT_CHECKPOINT", "")
    monkeypatch.setattr(module, "mount_drive_if_possible", lambda: None)
    monkeypatch.setattr(module, "candidate_drive_checkpoints", lambda run_id, filename: [drive_checkpoint])

    restored = module.resolve_checkpoint(source, payload)

    assert restored == checkpoint
    assert checkpoint.read_bytes() == b"checkpoint"


def test_resolve_checkpoint_missing_error_includes_drive_diagnostics(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    source = tmp_path / "outputs" / "stage5" / "run_a" / "summary.json"
    checkpoint = tmp_path / "outputs" / "stage5" / "run_a" / "phase1" / "phase1_step_150.pt"
    payload = {"phase1_checkpoint": str(checkpoint)}

    monkeypatch.setattr(module, "EXPLICIT_CHECKPOINT", "")
    monkeypatch.setattr(module, "mount_drive_if_possible", lambda: None)
    monkeypatch.setattr(module, "candidate_drive_checkpoints", lambda run_id, filename: [])
    monkeypatch.setattr(module, "drive_diagnostics", lambda: "Drive visibility: mocked")

    with pytest.raises(FileNotFoundError, match="Drive visibility: mocked"):
        module.resolve_checkpoint(source, payload)


def test_checkpoint_bearing_source_summary_follows_mcq_policy_chain(tmp_path) -> None:
    import colab.run_stage5_benchmark_suite as module

    checkpoint = tmp_path / "outputs" / "stage5" / "train" / "phase1" / "phase1_step_150.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    train_summary = tmp_path / "outputs" / "stage5" / "train" / "summary.json"
    pair_summary = tmp_path / "outputs" / "stage5" / "pair" / "summary.json"
    policy_summary = tmp_path / "outputs" / "stage5" / "policy" / "summary.json"
    debias_summary = tmp_path / "outputs" / "stage5" / "debias" / "summary.json"
    _write_jsonl(train_summary, [{"kind": "stage5_curriculum_sft", "phase1_checkpoint": str(checkpoint)}])
    _write_jsonl(
        debias_summary,
        [
            {
                "kind": "stage5_mcq_debias_diagnostic",
                "nested_source_summary": str(train_summary),
            }
        ],
    )
    _write_jsonl(
        pair_summary,
        [
            {
                "kind": "stage5_mcq_debias_pair_assessment",
                "source_summaries": {"arc_challenge": str(debias_summary)},
            }
        ],
    )
    _write_jsonl(
        policy_summary,
        [
            {
                "kind": "stage5_mcq_scoring_policy",
                "status": "debiased_mcq_policy_active",
                "source_summary": str(pair_summary),
            }
        ],
    )

    policy_payload = json.loads(policy_summary.read_text(encoding="utf-8").splitlines()[0])

    assert module.checkpoint_bearing_source_summary(policy_summary, policy_payload) == train_summary


def test_checkpoint_bearing_source_summary_unwraps_benchmark_assessment_even_with_checkpoint(tmp_path) -> None:
    import colab.run_stage5_benchmark_suite as module

    checkpoint = tmp_path / "outputs" / "stage5" / "train" / "phase1" / "phase1_step_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    benchmark_summary = tmp_path / "outputs" / "stage5" / "benchmark" / "summary.json"
    assessment_summary = tmp_path / "outputs" / "stage5" / "assessment" / "summary.json"
    _write_jsonl(
        benchmark_summary,
        [
            {
                "kind": "stage5_benchmark_suite",
                "checkpoint": str(checkpoint),
            }
        ],
    )
    _write_jsonl(
        assessment_summary,
        [
            {
                "kind": "stage5_benchmark_assessment",
                "gate": "stage5_broader_benchmark_suite",
                "source_summary": str(benchmark_summary),
                "checkpoint": str(checkpoint),
            }
        ],
    )

    assessment_payload = json.loads(assessment_summary.read_text(encoding="utf-8").splitlines()[0])

    assert module.checkpoint_bearing_source_summary(assessment_summary, assessment_payload) == benchmark_summary


def test_checkpoint_bearing_source_summary_unwraps_forced_depth_diagnostic(tmp_path) -> None:
    import colab.run_stage5_benchmark_suite as module

    checkpoint = tmp_path / "outputs" / "stage5" / "train" / "phase1" / "phase1_step_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    benchmark_summary = tmp_path / "outputs" / "stage5" / "benchmark" / "summary.json"
    forced_summary = tmp_path / "outputs" / "stage5" / "forced_depth" / "summary.json"
    _write_jsonl(
        benchmark_summary,
        [
            {
                "kind": "stage5_benchmark_suite",
                "checkpoint": str(checkpoint),
            }
        ],
    )
    _write_jsonl(
        forced_summary,
        [
            {
                "kind": "stage5_forced_depth_diagnostic",
                "source_summary": str(benchmark_summary),
            }
        ],
    )

    forced_payload = json.loads(forced_summary.read_text(encoding="utf-8").splitlines()[0])

    assert module.checkpoint_bearing_source_summary(forced_summary, forced_payload) == benchmark_summary


def test_build_summary_compares_base_and_recurrent_rows(tmp_path) -> None:
    base_jsonl = tmp_path / "arc_base_label.jsonl"
    recurrent_jsonl = tmp_path / "arc_recurrent_label.jsonl"
    data_jsonl = tmp_path / "arc.jsonl"
    _write_jsonl(
        data_jsonl,
        [
            {"id": "a", "question": "Which option is already known?", "choices": {"A": "known", "B": "unknown"}, "answer": "A"},
            {"id": "b", "question": "What is 2 + 2?", "choices": {"A": "4", "B": "5"}, "answer": "A"},
        ],
    )
    _write_jsonl(
        base_jsonl,
        [
            {"id": "a", "aggregate": "mean", "answer": "A", "prediction": "A", "hit": True, "scores": {"A": 3.0, "B": 1.0}},
            {"id": "b", "aggregate": "mean", "answer": "A", "prediction": "B", "hit": False, "scores": {"A": 0.0, "B": 1.0}},
        ],
    )
    _write_jsonl(
        recurrent_jsonl,
        [
            {
                "id": "a",
                "aggregate": "mean",
                "answer": "A",
                "prediction": "A",
                "hit": True,
                "scores": {"A": 2.5, "B": 1.0},
                "loop_diagnostics": {"mean_expected_loops": 1.1, "answer_expected_loops": 1.0},
            },
            {
                "id": "b",
                "aggregate": "mean",
                "answer": "A",
                "prediction": "A",
                "hit": True,
                "scores": {"A": 2.0, "B": 1.5},
                "loop_diagnostics": {"mean_expected_loops": 3.5, "answer_expected_loops": 3.7},
            },
        ],
    )

    payload = build_summary(
        source_summary=None,
        checkpoint=tmp_path / "checkpoint.pt",
        specs=[BenchmarkSpec("arc_challenge", data_jsonl, [])],
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
    routing = payload["routing_diagnostics"]["arc_challenge"]["label"]
    assert "loss_examples" not in routing
    assert routing["routing_buckets"]["base_confident_direct_proxy"]["n"] == 1
    assert routing["routing_buckets"]["base_confident_direct_proxy"]["delta"] == 0
    assert routing["routing_buckets"]["deep_numeric_proxy"]["n"] == 1
    assert routing["routing_buckets"]["deep_numeric_proxy"]["delta"] == 1
    assert routing["routing_buckets"]["deep_numeric_proxy"]["mean_candidate_expected_loops"] == 3.5
    loop_summary = payload["loop_bucket_diagnostics"]["arc_challenge"]["label"]["routing_buckets"]
    assert loop_summary["deep_numeric_proxy"]["mean_candidate_expected_loops"] == 3.5
    assert loop_summary["base_confident_direct_proxy"]["mean_candidate_expected_loops"] == 1.1


def test_build_summary_reports_hard_content_signal(tmp_path) -> None:
    base_jsonl = tmp_path / "arc_base_content_question_only.jsonl"
    recurrent_jsonl = tmp_path / "arc_recurrent_content_question_only.jsonl"
    data_jsonl = tmp_path / "arc.jsonl"
    _write_jsonl(
        data_jsonl,
        [
            {"id": "a", "question": "Why does this happen?", "choices": {"A": "x", "B": "y"}, "answer": "A"},
            {"id": "b", "question": "What is 3 * 4?", "choices": {"A": "12", "B": "7"}, "answer": "A"},
        ],
    )
    _write_jsonl(
        base_jsonl,
        [
            {"id": "a", "aggregate": "mean", "answer": "A", "prediction": "B", "hit": False, "scores": {"A": 1.0, "B": 2.0}},
            {"id": "b", "aggregate": "mean", "answer": "A", "prediction": "A", "hit": True, "scores": {"A": 2.0, "B": 1.0}},
        ],
    )
    _write_jsonl(
        recurrent_jsonl,
        [
            {
                "id": "a",
                "aggregate": "mean",
                "answer": "A",
                "prediction": "A",
                "hit": True,
                "scores": {"A": 2.0, "B": 1.0},
                "loop_diagnostics": {"mean_expected_loops": 2.4},
            },
            {
                "id": "b",
                "aggregate": "mean",
                "answer": "A",
                "prediction": "A",
                "hit": True,
                "scores": {"A": 2.0, "B": 1.0},
                "loop_diagnostics": {"mean_expected_loops": 2.8},
            },
        ],
    )

    payload = build_summary(
        source_summary=None,
        checkpoint=tmp_path / "checkpoint.pt",
        specs=[BenchmarkSpec("arc_challenge", data_jsonl, [])],
        jobs=[
            EvalJob("arc_challenge", "base", "content_question_only", base_jsonl, []),
            EvalJob("arc_challenge", "recurrent", "content_question_only", recurrent_jsonl, []),
        ],
        failures=[],
        elapsed_seconds=1.0,
    )

    signal = payload["hard_content_signal"]["arc_challenge"]
    assert signal["score_target"] == "content_question_only"
    assert signal["aggregate"] == "mean"
    assert signal["correct_delta_recurrent_vs_base"] == 1
    assert signal["routing_buckets"]["conceptual_reasoning_proxy"]["delta"] == 1
    assert signal["routing_buckets"]["conceptual_reasoning_proxy"]["mean_candidate_expected_loops"] == 2.4


def test_build_summary_records_after_confirmation_dense_control(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    base_jsonl = tmp_path / "arc_base_label.jsonl"
    recurrent_jsonl = tmp_path / "arc_recurrent_label.jsonl"
    data_jsonl = tmp_path / "arc.jsonl"
    _write_jsonl(data_jsonl, [{"id": "a", "question": "Q", "choices": {"A": "x"}, "answer": "A"}])
    _write_jsonl(base_jsonl, [{"id": "a", "aggregate": "mean", "answer": "A", "prediction": "A", "hit": True}])
    _write_jsonl(recurrent_jsonl, [{"id": "a", "aggregate": "mean", "answer": "A", "prediction": "A", "hit": True}])
    monkeypatch.setattr(module, "AFTER_CONFIRM_DENSE_RUN_SUFFIX", "dense_after_confirm")
    monkeypatch.setattr(module, "AFTER_CONFIRM_DENSE_EXTRA_TRAIN_JSONL", "data/repair/train.jsonl")

    payload = module.build_summary(
        source_summary=None,
        checkpoint=tmp_path / "checkpoint.pt",
        specs=[module.BenchmarkSpec("arc_challenge", data_jsonl, [])],
        jobs=[
            module.EvalJob("arc_challenge", "base", "label", base_jsonl, []),
            module.EvalJob("arc_challenge", "recurrent", "label", recurrent_jsonl, []),
        ],
        failures=[],
        elapsed_seconds=1.0,
    )

    assert payload["after_confirmation_dense_control"] == {
        "run_suffix": "dense_after_confirm",
        "extra_train_jsonl": "data/repair/train.jsonl",
        "reason": "run dense same-curriculum control after this recurrent confirmation passes",
    }


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
    assert "--include_loop_diagnostics" in recurrent_cmd


def test_eval_jobs_include_loop_diagnostics_only_for_recurrent(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    monkeypatch.setattr(module, "RECURRENT_MODE", "phase1")
    monkeypatch.setattr(module, "RECURRENT_NUM_TRAJECTORIES", 1)
    monkeypatch.setattr(module, "INCLUDE_LOOP_DIAGNOSTICS", True)

    jobs = module.eval_jobs([BenchmarkSpec("arc_challenge", tmp_path / "arc.jsonl", [])], checkpoint=tmp_path / "phase1.pt")
    base_cmd = next(job.cmd for job in jobs if job.arm == "base")
    recurrent_cmd = next(job.cmd for job in jobs if job.arm == "recurrent")

    assert "--include_loop_diagnostics" not in base_cmd
    assert "--include_loop_diagnostics" in recurrent_cmd


def test_eval_jobs_passes_forced_loop_count_to_recurrent_only(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    monkeypatch.setattr(module, "RECURRENT_MODE", "phase1")
    monkeypatch.setattr(module, "RECURRENT_NUM_TRAJECTORIES", 1)
    monkeypatch.setattr(module, "RECURRENT_FORCED_LOOP_COUNT", "3")

    jobs = module.eval_jobs([BenchmarkSpec("arc_challenge", tmp_path / "arc.jsonl", [])], checkpoint=tmp_path / "phase1.pt")
    base_cmd = next(job.cmd for job in jobs if job.arm == "base")
    recurrent_cmd = next(job.cmd for job in jobs if job.arm == "recurrent")

    assert "--forced_loop_count" not in base_cmd
    assert recurrent_cmd[recurrent_cmd.index("--forced_loop_count") + 1] == "3"


def test_eval_jobs_support_content_question_only_target(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    monkeypatch.setattr(module, "SCORE_TARGETS", "content_question_only")
    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "run")

    jobs = module.eval_jobs([BenchmarkSpec("arc_challenge", tmp_path / "arc.jsonl", [])], checkpoint=tmp_path / "phase1.pt")
    base_job = next(job for job in jobs if job.arm == "base")

    assert base_job.score_target == "content_question_only"
    assert base_job.cmd[base_job.cmd.index("--prompt_style") + 1] == "question_only"
    assert base_job.cmd[base_job.cmd.index("--score_target") + 1] == "option_text"


def test_eval_jobs_support_cyclic_label_aggregation_target(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    data = tmp_path / "arc.jsonl"
    _write_jsonl(
        data,
        [
            {
                "id": "item",
                "question": "Which number is even?",
                "choices": {"A": "three", "B": "four"},
                "answer": "B",
            }
        ],
    )
    monkeypatch.setattr(module, "SCORE_TARGETS", "cyclic_label_aggregated")
    monkeypatch.setattr(module, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", tmp_path / "private")

    jobs = module.eval_jobs([BenchmarkSpec("arc_challenge", data, [])], checkpoint=tmp_path / "phase1.pt")
    base_job = next(job for job in jobs if job.arm == "base")

    assert base_job.score_target == "cyclic_label_aggregated"
    assert base_job.output_jsonl.name == "arc_challenge_base_cyclic_label_aggregated.jsonl"
    assert base_job.eval_output_jsonl.name == "arc_challenge_base_cyclic_label_raw.jsonl"
    assert base_job.permutation_jsonl is not None
    assert base_job.permutation_jsonl.exists()
    assert base_job.cmd[base_job.cmd.index("--data_jsonl") + 1].endswith("_cyclic_permuted.jsonl")
    assert base_job.cmd[base_job.cmd.index("--prompt_style") + 1] == "with_options"
    assert base_job.cmd[base_job.cmd.index("--score_target") + 1] == "label"


def test_aggregate_cyclic_label_output_writes_public_rows(tmp_path) -> None:
    import colab.run_stage5_benchmark_suite as module
    from eval.mcq_debias import cyclic_permutation_rows

    original = [
        {
            "id": "item",
            "question": "Which number is even?",
            "choices": {"A": "three", "B": "four"},
            "answer": "B",
        }
    ]
    permutation_jsonl = tmp_path / "permuted.jsonl"
    raw_jsonl = tmp_path / "raw.jsonl"
    public_jsonl = tmp_path / "public.jsonl"
    _write_jsonl(permutation_jsonl, cyclic_permutation_rows(original))
    _write_jsonl(
        raw_jsonl,
        [
            {
                "id": "item::perm0",
                "aggregate": "mean",
                "prediction": "B",
                "answer": "B",
                "hit": True,
                "scores": {"A": 0.0, "B": 2.0},
            },
            {
                "id": "item::perm1",
                "aggregate": "mean",
                "prediction": "A",
                "answer": "A",
                "hit": True,
                "scores": {"A": 2.0, "B": 0.0},
            },
        ],
    )
    job = module.EvalJob(
        "arc_challenge",
        "base",
        "cyclic_label_aggregated",
        public_jsonl,
        [],
        eval_output_jsonl=raw_jsonl,
        permutation_jsonl=permutation_jsonl,
    )

    module.aggregate_cyclic_label_output(job)

    rows = [json.loads(line) for line in public_jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "id": "item",
            "aggregate": "permutation_mean",
            "prediction": "B",
            "answer": "B",
            "hit": True,
            "scores": {"A": 0.0, "B": 2.0},
            "num_permutations": 2,
            "permutation_prediction_counts": {"B": 2},
        }
    ]


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


def test_benchmark_specs_supports_open_hard_arc_challenge_fallback(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    monkeypatch.setattr(module, "PRIVATE_DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "OPEN_HARD_ARC_CHALLENGE_LIMIT", 256)
    monkeypatch.setattr(module, "OPEN_HARD_ARC_CHALLENGE_OFFSET", 0)
    monkeypatch.setattr(module, "OPEN_HARD_ARC_CHALLENGE_SPLIT", "test")

    spec = benchmark_specs(["open_hard_arc_challenge"])[0]

    assert spec.name == "open_hard_arc_challenge"
    assert spec.data_jsonl == tmp_path / "open_hard_arc_challenge_test_256.jsonl"
    assert "ARC-Challenge" in spec.prepare_cmd
    assert spec.prepare_cmd[spec.prepare_cmd.index("--split") + 1] == "test"
    assert spec.prepare_cmd[spec.prepare_cmd.index("--limit") + 1] == "256"


def test_suite_profile_depth_signal_confirmation_defaults_to_hard_content_fallback() -> None:
    defaults = suite_profile_defaults("depth_signal_confirmation")

    assert defaults["benchmarks"] == "arc_easy,arc_challenge,open_hard_arc_challenge"
    assert defaults["score_targets"] == "content_question_only,cyclic_label_aggregated"
    assert defaults["arc_challenge_limit"] == "256"
    assert defaults["open_hard_arc_challenge_limit"] == "256"


def test_benchmark_suite_updates_current_source_summary(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    run_dir = tmp_path / "outputs" / "stage5" / "bench"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "RUN_ID", "bench")

    module.write_report(
        {
            "status": "completed",
            "source_summary": "outputs/stage5/source/summary.json",
            "checkpoint": "outputs/stage5/source/phase1/phase1_step_150.pt",
            "benchmarks": ["arc_challenge"],
            "recurrent_mode": "phase1",
            "recurrent_num_trajectories": 1,
            "elapsed_seconds": 1.0,
            "comparisons": {},
            "paired_comparisons": {},
            "routing_diagnostics": {},
            "failures": [],
        }
    )

    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "config" / "stage5_current_source_summary.txt").read_text(
        encoding="utf-8"
    ) == "outputs/stage5/bench/summary.json\n"


def test_benchmark_suite_commit_stages_current_source_pointer(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    run_dir = tmp_path / "outputs" / "stage5" / "bench"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/bench/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results()

    add_commands = [cmd for cmd in commands if cmd[:2] == ["git", "add"]]
    assert add_commands
    staged = {item for cmd in add_commands for item in cmd[3:]}
    assert "outputs/stage5/bench" in staged
    assert "config/stage5_current_source_summary.txt" in staged


def test_benchmark_suite_commit_retries_failed_push_with_autostash_rebase(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_benchmark_suite as module

    run_dir = tmp_path / "outputs" / "stage5" / "bench"
    pointer = tmp_path / "config" / "stage5_current_source_summary.txt"
    run_dir.mkdir(parents=True)
    pointer.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    pointer.write_text("outputs/stage5/bench/summary.json\n", encoding="utf-8")
    commands: list[list[str]] = []
    push_calls = 0

    def fake_run(cmd, *, check=True, log_name=None):
        nonlocal push_calls
        command = [str(item) for item in cmd]
        commands.append(command)
        if command == ["git", "diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(cmd, 1, "", None)
        if command == ["git", "push", "origin", "main"]:
            push_calls += 1
            return subprocess.CompletedProcess(cmd, 1 if push_calls == 1 else 0, "", None)
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "PUSH_RESULTS", True)
    monkeypatch.setattr(module, "run", fake_run)

    module.commit_results()

    assert commands.count(["git", "push", "origin", "main"]) == 2
    assert ["git", "pull", "--rebase", "--autostash", "origin", "main"] in commands
