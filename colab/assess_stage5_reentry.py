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
    pre_delta = finite_float(pre.get("bridge_delta_rms"))
    post_delta = finite_float(post.get("bridge_delta_rms"))
    post_weight_grad = finite_float(post.get("weight_grad_rms"))
    post_bias_grad = finite_float(post.get("bias_grad_rms"))
    post_gate = finite_float(post.get("bridge_gate"))
    bridge_live = post_weight_grad > 0.0 and post_bias_grad > 0.0
    bridge_moved = abs(post_delta - pre_delta) > 1e-6 or post_delta > 1e-6

    if bridge_live and bridge_moved:
        status = "bridge_repair_smoke_passed"
        recommendation = "run_bounded_recovery_training_with_reentry_repair"
        reason = "Bridge is gradient-live and changed the re-entry path during the smoke run."
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
            "post_weight_grad_rms": post_weight_grad,
            "post_bias_grad_rms": post_bias_grad,
            "bridge_live": bridge_live,
            "bridge_moved": bridge_moved,
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
