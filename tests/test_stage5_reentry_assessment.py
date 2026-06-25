import json
import subprocess
import sys

from colab.assess_stage5_reentry import assess


def drift_summary(*, gate=0.0, weight_grad=0.0, bias_grad=0.0, overlap=0.37, loop8=1.037):
    return {
        "kind": "reentry_drift_diagnostic",
        "run_id": "stage5_reentry_drift_test",
        "aggregate": {
            "mean_exit_over_entry_rms": 1.002,
            "entry_exit_subspace": {
                "overlap": overlap,
                "aligned_dims_cos_ge_0p8": 1,
            },
            "loop_summary": {
                "8": {
                    "input_over_entry_rms": 1.03,
                    "output_over_entry_rms": loop8,
                }
            },
        },
        "bridge": {
            "bridge_gate": gate,
            "sample_bridge_delta_rms": 0.0,
        },
        "bridge_gradient_liveness": {
            "weight_grad_rms": weight_grad,
            "bias_grad_rms": bias_grad,
        },
    }


def norm_summary(*, none_hits=20, entry_hits=21, none_best=5, entry_best=5):
    def candidate(mode, hits, best):
        return {
            "by_mode": {
                mode: {
                    "task_groups": 10,
                    "best_hits": best,
                    "candidate_hits": hits,
                    "total_candidates": 40,
                    "mean_unique": 2.0,
                }
            }
        }

    return {
        "kind": "stage5_reentry_norm_eval_only",
        "run_id": "stage5_reentry_norm_test",
        "drift": {
            "none": {
                "loop_summary": {"8": {"output_over_entry_rms": 1.04}},
                "entry_exit_subspace": {"overlap": 0.37},
            },
            "entry_rms": {
                "loop_summary": {"8": {"output_over_entry_rms": 1.00}},
                "entry_exit_subspace": {"overlap": 0.39},
            },
        },
        "effective_pathways": {
            "none": {"mean_effective_pathways": {"2": 1.1}},
            "entry_rms": {"mean_effective_pathways": {"2": 1.2}},
        },
        "candidate_conversion": {
            "none": candidate("none", none_hits, none_best),
            "entry_rms": candidate("entry_rms", entry_hits, entry_best),
        },
    }


def repair_summary(
    *,
    post_delta=0.01,
    weight_grad=1e-4,
    bias_grad=1e-4,
    use_reentry_adapter=True,
    adapter_delta=0.01,
    adapter_scale_grad=1e-4,
    adapter_bias_grad=1e-4,
    source_best=4,
    trained_best=4,
    source_candidates=4,
    trained_candidates=4,
    source_groups=4,
    trained_groups=4,
):
    return {
        "kind": "stage5_reentry_repair_smoke",
        "run_id": "stage5_reentry_repair_smoke_test",
        "config": {
            "use_reentry_adapter": use_reentry_adapter,
        },
        "pre_bridge": {
            "bridge_gate": 0.0,
            "bridge_delta_rms": 0.0,
            "weight_grad_rms": 0.0,
            "bias_grad_rms": 0.0,
        },
        "post_bridge": {
            "bridge_gate": 1.0,
            "bridge_delta_rms": post_delta,
            "weight_grad_rms": weight_grad,
            "bias_grad_rms": bias_grad,
        },
        "pre_reentry_adapter": {
            "scale_identity_max_abs_diff": 0.0,
            "bias_max_abs": 0.0,
            "sample_adapter_delta_rms": 0.0,
        },
        "post_reentry_adapter": {
            "scale_identity_max_abs_diff": adapter_delta,
            "bias_max_abs": 0.0,
            "sample_adapter_delta_rms": adapter_delta,
        },
        "post_reentry_adapter_liveness": {
            "scale_grad_rms": adapter_scale_grad,
            "bias_grad_rms": adapter_bias_grad,
        },
        "loop1_preservation": {
            "source": {
                "task_groups": source_groups,
                "best_hits": source_best,
                "candidate_hits": source_candidates,
                "total_candidates": 4,
            },
            "trained": {
                "task_groups": trained_groups,
                "best_hits": trained_best,
                "candidate_hits": trained_candidates,
                "total_candidates": 4,
            },
            "best_hits_delta_trained_minus_source": trained_best - source_best,
            "candidate_hits_delta_trained_minus_source": trained_candidates - source_candidates,
        },
    }


def test_drift_assessment_flags_dead_bridge() -> None:
    out = assess(drift_summary())

    assert out["status"] == "bridge_dead"
    assert out["recommendation"] == "run_reentry_norm_then_repair_smoke"
    assert out["metrics"]["dead_bridge"] is True


def test_drift_assessment_flags_subspace_mismatch_without_dead_bridge() -> None:
    out = assess(drift_summary(gate=1.0, weight_grad=1e-4, bias_grad=1e-4, overlap=0.3))

    assert out["status"] == "subspace_mismatch"
    assert out["recommendation"] == "run_reentry_norm_diagnostic"


def test_norm_assessment_allows_repair_smoke_when_no_major_candidate_regression() -> None:
    out = assess(norm_summary(none_hits=20, entry_hits=21, none_best=5, entry_best=5))

    assert out["status"] == "entry_rms_safe_for_smoke"
    assert out["recommendation"] == "run_reentry_repair_smoke"


def test_norm_assessment_blocks_on_major_candidate_regression() -> None:
    out = assess(norm_summary(none_hits=20, entry_hits=10, none_best=5, entry_best=2))

    assert out["status"] == "entry_rms_eval_regression"
    assert out["recommendation"] == "review_before_trainable_repair"


def test_repair_smoke_assessment_passes_when_bridge_live_and_moved() -> None:
    out = assess(repair_summary())

    assert out["status"] == "bridge_repair_smoke_passed"
    assert out["recommendation"] == "run_bounded_recovery_training_with_reentry_repair"
    assert out["metrics"]["loop1_preservation_available"] is True
    assert out["metrics"]["adapter_live"] is True
    assert out["metrics"]["adapter_moved"] is True


def test_repair_smoke_assessment_detects_live_but_not_moved() -> None:
    out = assess(repair_summary(post_delta=0.0, weight_grad=1e-4, bias_grad=1e-4))

    assert out["status"] == "bridge_live_but_no_observed_movement"
    assert out["recommendation"] == "extend_reentry_repair_smoke_or_increase_bridge_lr"


def test_repair_smoke_assessment_blocks_when_enabled_adapter_not_live() -> None:
    out = assess(repair_summary(adapter_scale_grad=0.0, adapter_bias_grad=0.0))

    assert out["status"] == "reentry_adapter_not_gradient_live"
    assert out["recommendation"] == "fix_reentry_adapter_before_recovery_training"
    assert out["metrics"]["adapter_live"] is False


def test_repair_smoke_assessment_blocks_when_enabled_adapter_not_moved() -> None:
    out = assess(repair_summary(adapter_delta=0.0))

    assert out["status"] == "reentry_adapter_live_but_not_moved"
    assert out["recommendation"] == "extend_reentry_repair_smoke_or_increase_adapter_lr"
    assert out["metrics"]["adapter_moved"] is False


def test_repair_smoke_assessment_keeps_legacy_bridge_only_pass() -> None:
    out = assess(repair_summary(use_reentry_adapter=False, adapter_delta=0.0, adapter_scale_grad=0.0, adapter_bias_grad=0.0))

    assert out["status"] == "bridge_repair_smoke_passed"
    assert out["recommendation"] == "run_bounded_recovery_training_with_reentry_repair"


def test_repair_smoke_assessment_blocks_on_loop1_regression() -> None:
    out = assess(repair_summary(source_best=4, trained_best=3, source_candidates=4, trained_candidates=3))

    assert out["status"] == "bridge_live_but_loop1_regressed"
    assert out["recommendation"] == "review_or_reduce_repair_lr_before_recovery_training"
    assert out["metrics"]["loop1_regressed"] is True


def test_repair_smoke_assessment_blocks_when_loop1_preservation_missing() -> None:
    out = assess(repair_summary(source_groups=0, trained_groups=4))

    assert out["status"] == "loop1_preservation_missing_or_mismatched"
    assert out["recommendation"] == "fix_loop1_preservation_eval_before_recovery_training"
    assert out["metrics"]["loop1_preservation_available"] is False


def test_repair_smoke_assessment_blocks_when_loop1_task_groups_mismatch() -> None:
    out = assess(repair_summary(source_groups=4, trained_groups=3))

    assert out["status"] == "loop1_preservation_missing_or_mismatched"
    assert out["recommendation"] == "fix_loop1_preservation_eval_before_recovery_training"
    assert out["metrics"]["source_loop1_task_groups"] == 4
    assert out["metrics"]["trained_loop1_task_groups"] == 3


def test_reentry_assessment_cli_writes_outputs(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    output_json = tmp_path / "assessment.json"
    output_md = tmp_path / "assessment.md"
    summary.write_text(json.dumps(drift_summary()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "colab/assess_stage5_reentry.py",
            "--summary_json",
            str(summary),
            "--output_json",
            str(output_json),
            "--output_md",
            str(output_md),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["kind"] == "stage5_reentry_assessment"
    assert payload["status"] == "bridge_dead"
    assert "run_reentry_norm_then_repair_smoke" in result.stdout
    assert "Recommendation" in output_md.read_text(encoding="utf-8")
