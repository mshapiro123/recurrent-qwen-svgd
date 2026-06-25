"""Assess Stage 5 recurrent re-entry diagnostics and recommend the next step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
        return default
    return out


def finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def candidate_stats(summary: dict[str, Any], mode: str) -> dict[str, float]:
    payload = nested(summary, "candidate_conversion", mode, "by_mode", mode, default={})
    if not isinstance(payload, dict):
        return {"best_hits": 0.0, "candidate_hits": 0.0, "total_candidates": 0.0, "mean_unique": 0.0}
    return {
        "best_hits": finite_float(payload.get("best_hits")),
        "candidate_hits": finite_float(payload.get("candidate_hits")),
        "total_candidates": finite_float(payload.get("total_candidates")),
        "mean_unique": finite_float(payload.get("mean_unique")),
    }


def drift_metrics(summary: dict[str, Any], mode: str | None = None) -> dict[str, float]:
    root = nested(summary, "drift", mode, default={}) if mode else nested(summary, "aggregate", default={})
    if not isinstance(root, dict):
        root = {}
    loop8 = nested(root, "loop_summary", "8", default={})
    subspace = nested(root, "entry_exit_subspace", default={})
    return {
        "exit_over_entry_rms": finite_float(root.get("mean_exit_over_entry_rms"), 1.0),
        "loop8_input_over_entry_rms": finite_float(nested(loop8, "input_over_entry_rms"), 1.0),
        "loop8_output_over_entry_rms": finite_float(nested(loop8, "output_over_entry_rms"), 1.0),
        "subspace_overlap": finite_float(nested(subspace, "overlap"), 0.0),
        "aligned_dims_ge_0p8": finite_float(nested(subspace, "aligned_dims_cos_ge_0p8"), 0.0),
    }


def effective_metrics(summary: dict[str, Any], mode: str) -> dict[str, float]:
    root = nested(summary, "effective_pathways", mode, default={})
    if not isinstance(root, dict):
        root = {}
    return {
        "initial_distance": finite_float(root.get("mean_initial_pairwise_distance")),
        "final_distance": finite_float(root.get("mean_final_pairwise_distance")),
        "spread_ratio": finite_float(root.get("mean_spread_ratio_final_over_initial")),
        "q2_pathways": finite_float(nested(root, "mean_effective_pathways", "2")),
        "unique_next_token_argmax": finite_float(root.get("mean_unique_next_token_argmax")),
    }


def assess_drift(summary: dict[str, Any]) -> dict[str, Any]:
    drift = drift_metrics(summary)
    bridge = summary.get("bridge") if isinstance(summary.get("bridge"), dict) else {}
    live = summary.get("bridge_gradient_liveness") if isinstance(summary.get("bridge_gradient_liveness"), dict) else {}
    gate = finite_float(bridge.get("bridge_gate"))
    bridge_delta = finite_float(bridge.get("sample_bridge_delta_rms"))
    weight_grad = finite_float(live.get("weight_grad_rms"))
    bias_grad = finite_float(live.get("bias_grad_rms"))
    dead_bridge = abs(gate) < 1e-8 and bridge_delta < 1e-8 and weight_grad < 1e-12 and bias_grad < 1e-12
    norm_drift = abs(drift["loop8_output_over_entry_rms"] - 1.0)
    subspace_mismatch = drift["subspace_overlap"] < 0.6

    if dead_bridge:
        recommendation = "run_reentry_norm_then_repair_smoke"
        status = "bridge_dead"
        reason = "Bridge is identity with gate zero and no bridge gradients; repair smoke is justified after the eval-only norm check."
    elif subspace_mismatch:
        recommendation = "run_reentry_norm_diagnostic"
        status = "subspace_mismatch"
        reason = "Bridge is not fully dead, but entry/exit subspace overlap is low; compare eval-only re-entry normalization."
    elif norm_drift > 0.08:
        recommendation = "run_reentry_norm_diagnostic"
        status = "norm_drift"
        reason = "Loop norm drift is large enough to test eval-only re-entry normalization before training."
    else:
        recommendation = "run_reentry_repair_smoke"
        status = "repair_smoke_reasonable"
        reason = "Drift is bounded, but a trainable bridge/re-entry smoke can test whether recurrence can move usefully."

    return {
        "stage": "drift",
        "status": status,
        "recommendation": recommendation,
        "reason": reason,
        "metrics": {
            **drift,
            "bridge_gate": gate,
            "bridge_delta_rms": bridge_delta,
            "bridge_weight_grad_rms": weight_grad,
            "bridge_bias_grad_rms": bias_grad,
            "loop8_norm_drift_abs": norm_drift,
            "dead_bridge": dead_bridge,
        },
    }


def assess_norm(summary: dict[str, Any]) -> dict[str, Any]:
    none_drift = drift_metrics(summary, "none")
    entry_drift = drift_metrics(summary, "entry_rms")
    none_candidates = candidate_stats(summary, "none")
    entry_candidates = candidate_stats(summary, "entry_rms")
    none_effective = effective_metrics(summary, "none")
    entry_effective = effective_metrics(summary, "entry_rms")

    loop8_delta = entry_drift["loop8_output_over_entry_rms"] - none_drift["loop8_output_over_entry_rms"]
    candidate_delta = entry_candidates["candidate_hits"] - none_candidates["candidate_hits"]
    best_delta = entry_candidates["best_hits"] - none_candidates["best_hits"]
    pathway_delta = entry_effective["q2_pathways"] - none_effective["q2_pathways"]
    unique_delta = entry_candidates["mean_unique"] - none_candidates["mean_unique"]
    major_candidate_regression = best_delta <= -2 or candidate_delta <= -6

    if major_candidate_regression:
        status = "entry_rms_eval_regression"
        recommendation = "review_before_trainable_repair"
        reason = "Eval-only re-entry normalization appears to reduce candidate conversion materially."
    else:
        status = "entry_rms_safe_for_smoke"
        recommendation = "run_reentry_repair_smoke"
        reason = "Eval-only re-entry normalization does not show a large candidate regression; proceed to tiny trainable repair smoke."

    return {
        "stage": "norm",
        "status": status,
        "recommendation": recommendation,
        "reason": reason,
        "metrics": {
            "loop8_output_over_entry_delta_entry_minus_none": loop8_delta,
            "candidate_hits_delta_entry_minus_none": candidate_delta,
            "best_hits_delta_entry_minus_none": best_delta,
            "q2_pathways_delta_entry_minus_none": pathway_delta,
            "mean_unique_delta_entry_minus_none": unique_delta,
            "none_candidate_hits": none_candidates["candidate_hits"],
            "entry_candidate_hits": entry_candidates["candidate_hits"],
            "none_best_hits": none_candidates["best_hits"],
            "entry_best_hits": entry_candidates["best_hits"],
        },
    }


def assess_repair_smoke(summary: dict[str, Any]) -> dict[str, Any]:
    pre = summary.get("pre_bridge") if isinstance(summary.get("pre_bridge"), dict) else {}
    post = summary.get("post_bridge") if isinstance(summary.get("post_bridge"), dict) else {}
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    use_reentry_adapter = bool(config.get("use_reentry_adapter", False))
    halt_target_nll_weight = finite_float(config.get("halt_target_nll_weight"))
    pre_delta = finite_float(pre.get("bridge_delta_rms"))
    post_delta = finite_float(post.get("bridge_delta_rms"))
    post_identity_diff = finite_float(post.get("proj_identity_max_abs_diff"))
    post_bias_max = finite_float(post.get("proj_bias_max_abs"))
    post_weight_grad = finite_float(post.get("weight_grad_rms"))
    post_bias_grad = finite_float(post.get("bias_grad_rms"))
    post_gate = finite_float(post.get("bridge_gate"))
    bridge_live = post_weight_grad > 0.0 and post_bias_grad > 0.0
    bridge_projection_moved = post_identity_diff > 1e-6 or post_bias_max > 1e-6
    bridge_output_moved = abs(post_delta - pre_delta) > 1e-6 or post_delta > 1e-6
    bridge_moved = bridge_projection_moved or bridge_output_moved

    post_adapter = summary.get("post_reentry_adapter") if isinstance(summary.get("post_reentry_adapter"), dict) else {}
    post_adapter_live = (
        summary.get("post_reentry_adapter_liveness")
        if isinstance(summary.get("post_reentry_adapter_liveness"), dict)
        else {}
    )
    adapter_delta = finite_float(post_adapter.get("sample_adapter_delta_rms"))
    adapter_scale_diff = finite_float(post_adapter.get("scale_identity_max_abs_diff"))
    adapter_bias_max = finite_float(post_adapter.get("bias_max_abs"))
    adapter_scale_grad = finite_float(post_adapter_live.get("scale_grad_rms"))
    adapter_bias_grad = finite_float(post_adapter_live.get("bias_grad_rms"))
    adapter_live = (not use_reentry_adapter) or (adapter_scale_grad > 0.0 and adapter_bias_grad > 0.0)
    adapter_moved = (not use_reentry_adapter) or (
        adapter_delta > 1e-6 or adapter_scale_diff > 1e-6 or adapter_bias_max > 1e-6
    )

    train_log_metrics = summary.get("train_log_metrics") if isinstance(summary.get("train_log_metrics"), dict) else {}
    train_last_metrics = (
        train_log_metrics.get("last_metrics") if isinstance(train_log_metrics.get("last_metrics"), dict) else {}
    )
    train_metrics_available = bool(train_log_metrics.get("available")) and bool(train_last_metrics)
    train_loss = finite_float_or_none(train_last_metrics.get("loss"))
    train_expected_ce = finite_float_or_none(train_last_metrics.get("expected_ce"))
    train_mean_expected_loops = finite_float_or_none(train_last_metrics.get("mean_expected_loops"))
    train_target_loop_abs_error = finite_float_or_none(train_last_metrics.get("target_loop_abs_error"))
    train_halting_target_nll = finite_float_or_none(train_last_metrics.get("halting_target_nll"))
    depth_supervision_metrics_present = (
        halt_target_nll_weight <= 0.0
        or (train_target_loop_abs_error is not None and train_halting_target_nll is not None)
    )

    preservation = summary.get("loop1_preservation") if isinstance(summary.get("loop1_preservation"), dict) else {}
    source_preservation = preservation.get("source") if isinstance(preservation.get("source"), dict) else {}
    trained_preservation = preservation.get("trained") if isinstance(preservation.get("trained"), dict) else {}
    source_task_groups = finite_float(source_preservation.get("task_groups"))
    trained_task_groups = finite_float(trained_preservation.get("task_groups"))
    source_best_hits = finite_float(source_preservation.get("best_hits"))
    trained_best_hits = finite_float(trained_preservation.get("best_hits"))
    source_candidate_hits = finite_float(source_preservation.get("candidate_hits"))
    trained_candidate_hits = finite_float(trained_preservation.get("candidate_hits"))
    best_delta = finite_float(preservation.get("best_hits_delta_trained_minus_source"))
    candidate_delta = finite_float(preservation.get("candidate_hits_delta_trained_minus_source"))
    preservation_available = bool(
        source_preservation
        and trained_preservation
        and source_task_groups > 0
        and trained_task_groups > 0
        and source_task_groups == trained_task_groups
    )
    preservation_source_has_signal = preservation_available and source_best_hits > 0 and source_candidate_hits > 0
    loop1_regressed = preservation_available and (best_delta < 0 or candidate_delta <= -2)

    if not train_metrics_available:
        status = "repair_smoke_train_metrics_missing"
        recommendation = "fix_repair_smoke_training_log_before_recovery_training"
        reason = "The repair smoke did not publish final training metrics, so Stage 4 cannot judge whether the tiny repair optimized cleanly."
    elif train_loss is None:
        status = "repair_smoke_train_metrics_nonfinite"
        recommendation = "fix_repair_smoke_training_before_recovery_training"
        reason = "The repair smoke final training loss is missing or nonfinite."
    elif not depth_supervision_metrics_present:
        status = "repair_smoke_depth_metrics_missing"
        recommendation = "fix_repair_smoke_depth_supervision_before_recovery_training"
        reason = "The repair smoke requested supervised halting-depth loss but did not publish the expected depth metrics."
    elif not preservation_available:
        status = "loop1_preservation_missing_or_mismatched"
        recommendation = "fix_loop1_preservation_eval_before_recovery_training"
        reason = "The repair smoke did not produce comparable loop-1 preservation evidence."
    elif not preservation_source_has_signal:
        status = "loop1_preservation_source_has_no_signal"
        recommendation = "fix_loop1_preservation_eval_before_recovery_training"
        reason = "The loop-1 preservation source scored no correct examples, so non-regression would be uninformative."
    elif loop1_regressed:
        status = "bridge_live_but_loop1_regressed" if bridge_live else "bridge_still_dead_and_loop1_regressed"
        recommendation = "review_or_reduce_repair_lr_before_recovery_training"
        reason = "The repair smoke harmed deterministic loop-1 preservation; review before recovery training."
    elif use_reentry_adapter and not adapter_live:
        status = "reentry_adapter_not_gradient_live"
        recommendation = "fix_reentry_adapter_before_recovery_training"
        reason = "The repair smoke enabled the re-entry adapter, but adapter gradients were not live."
    elif use_reentry_adapter and not adapter_moved:
        status = "reentry_adapter_live_but_not_moved"
        recommendation = "extend_reentry_repair_smoke_or_increase_adapter_lr"
        reason = "The repair smoke enabled the re-entry adapter, but no adapter movement was observed."
    elif bridge_live and bridge_moved:
        status = "bridge_repair_smoke_passed"
        recommendation = "run_bounded_recovery_training_with_reentry_repair"
        reason = "Bridge and re-entry repair path are gradient-live and changed during the smoke run."
    elif bridge_live:
        status = "bridge_live_but_no_observed_movement"
        recommendation = "extend_reentry_repair_smoke_or_increase_bridge_lr"
        reason = "Bridge gradients are live, but the summary did not show measurable bridge-path movement."
    else:
        status = "bridge_still_dead"
        recommendation = "fix_reentry_repair_controls_before_more_training"
        reason = "Bridge gradients are still zero after the repair smoke."

    return {
        "stage": "repair_smoke",
        "status": status,
        "recommendation": recommendation,
        "reason": reason,
        "metrics": {
            "post_bridge_gate": post_gate,
            "pre_bridge_delta_rms": pre_delta,
            "post_bridge_delta_rms": post_delta,
            "post_bridge_proj_identity_max_abs_diff": post_identity_diff,
            "post_bridge_proj_bias_max_abs": post_bias_max,
            "bridge_projection_moved": bridge_projection_moved,
            "bridge_output_moved": bridge_output_moved,
            "post_weight_grad_rms": post_weight_grad,
            "post_bias_grad_rms": post_bias_grad,
            "bridge_live": bridge_live,
            "bridge_moved": bridge_moved,
            "use_reentry_adapter": use_reentry_adapter,
            "adapter_delta_rms": adapter_delta,
            "adapter_scale_identity_max_abs_diff": adapter_scale_diff,
            "adapter_bias_max_abs": adapter_bias_max,
            "adapter_scale_grad_rms": adapter_scale_grad,
            "adapter_bias_grad_rms": adapter_bias_grad,
            "adapter_live": adapter_live,
            "adapter_moved": adapter_moved,
            "train_metrics_available": train_metrics_available,
            "train_last_step": train_log_metrics.get("last_step"),
            "train_loss": train_loss,
            "train_expected_ce": train_expected_ce,
            "train_mean_expected_loops": train_mean_expected_loops,
            "train_target_loop_abs_error": train_target_loop_abs_error,
            "train_halting_target_nll": train_halting_target_nll,
            "depth_supervision_metrics_present": depth_supervision_metrics_present,
            "loop1_preservation_available": preservation_available,
            "source_loop1_task_groups": source_task_groups,
            "trained_loop1_task_groups": trained_task_groups,
            "source_loop1_best_hits": source_best_hits,
            "trained_loop1_best_hits": trained_best_hits,
            "loop1_best_hits_delta": best_delta,
            "source_loop1_candidate_hits": source_candidate_hits,
            "trained_loop1_candidate_hits": trained_candidate_hits,
            "loop1_candidate_hits_delta": candidate_delta,
            "loop1_source_has_correct_signal": preservation_source_has_signal,
            "loop1_regressed": loop1_regressed,
        },
    }


def assess(summary: dict[str, Any]) -> dict[str, Any]:
    kind = summary.get("kind")
    if kind == "reentry_drift_diagnostic":
        assessment = assess_drift(summary)
    elif kind == "stage5_reentry_norm_eval_only":
        assessment = assess_norm(summary)
    elif kind == "stage5_reentry_repair_smoke":
        assessment = assess_repair_smoke(summary)
    else:
        raise ValueError(f"Unsupported re-entry summary kind: {kind!r}")
    return {
        "kind": "stage5_reentry_assessment",
        "source_kind": kind,
        "source_run_id": summary.get("run_id"),
        **assessment,
    }


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    lines = [
        f"# Stage 5 Re-entry Assessment - {payload.get('source_run_id', '')}",
        "",
        f"- Source kind: `{payload.get('source_kind')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Recommendation: `{payload.get('recommendation')}`",
        f"- Reason: {payload.get('reason')}",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    if payload.get("stage") == "repair_smoke":
        lines.extend(
            [
                "",
                "## Loop-1 Preservation Gate",
                f"- Source has correct signal: `{metrics.get('loop1_source_has_correct_signal')}`",
                f"- Source best hits: `{metrics.get('source_loop1_best_hits')}`",
                f"- Trained best hits: `{metrics.get('trained_loop1_best_hits')}`",
                f"- Best-hit delta: `{metrics.get('loop1_best_hits_delta')}`",
                f"- Candidate-hit delta: `{metrics.get('loop1_candidate_hits_delta')}`",
            ]
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    args = parser.parse_args()

    payload = assess(read_json(args.summary_json))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(payload, args.output_md)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
