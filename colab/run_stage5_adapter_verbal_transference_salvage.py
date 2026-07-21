"""Evaluation-only salvage for the guardrail-truncated E3b experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab import run_stage5_adapter_verbal_transference as e3b
from training.adapter_verbal_transference import (
    TRANSFER_THRESHOLD,
    classify_regression,
    first_threshold_crossing,
    guardrail_near_miss_context,
)


MATCHED_STEPS = (0, 1000, 2000, 3000)
ARM_T_STEPS = (0, 1000, 2000, 3000, 4000, 5000, 6000)
HARD_STOP_DELTA = -0.03


def read_hits(path: Path) -> dict[str, bool]:
    rows = e3b.read_jsonl(path)
    hits = {str(row["id"]): bool(row["hit"]) for row in rows}
    if len(hits) != len(rows):
        raise RuntimeError(f"Duplicate guardrail ids in {path}")
    return hits


def single_arm_transfer(arm: str, step: int) -> dict[str, Any]:
    families: dict[str, Any] = {}
    pooled: list[bool] = []
    for family in ("relay", "pointer"):
        hits = e3b.aligned_hits(arm, step, family)
        values = list(hits.values())
        families[family] = {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
        pooled.extend(values)
    return {
        "families": families,
        "pooled": {
            "correct": sum(pooled),
            "total": len(pooled),
            "accuracy": sum(pooled) / len(pooled),
        },
    }


def assert_truncated_training_receipt(summary: dict[str, Any]) -> None:
    arm_t = summary.get("training", {}).get("arm_t", {})
    arm_s = summary.get("training", {}).get("arm_s", {})
    arm_s_training = arm_s.get("summary", {})
    if summary.get("status") != "arm_s_training_blocked":
        raise RuntimeError(f"Expected blocked E3b receipt, found {summary.get('status')!r}")
    if arm_t.get("status") != "finished" or arm_t.get("summary", {}).get("final_step") != 6000:
        raise RuntimeError("Arm T 6,000-step receipt is incomplete")
    if arm_s.get("status") != "blocked":
        raise RuntimeError("Arm S is not recorded as blocked")
    if arm_s_training.get("status") != "hard_stopped_canary" or arm_s_training.get("final_step") != 3000:
        raise RuntimeError("Arm S did not stop at the registered step-3,000 canary boundary")
    if not arm_t.get("summary", {}).get("pretrained_base_hash_unchanged"):
        raise RuntimeError("Arm T frozen-base hash receipt failed")
    if not arm_s_training.get("pretrained_base_hash_unchanged"):
        raise RuntimeError("Arm S frozen-base hash receipt failed")


def write_progress(summary: dict[str, Any], *, status: str, message: str) -> None:
    summary["status"] = status
    summary["salvage_note"] = message
    e3b.write_json(e3b.RUN_DIR / "summary.json", summary)
    e3b.publish_run(e3b.RUN_DIR, message=message + " [skip ci]")


def main() -> int:
    e3b.assert_frozen_data()
    summary_path = e3b.RUN_DIR / "summary.json"
    summary = e3b.read_json(summary_path)
    assert_truncated_training_receipt(summary)

    arm_t_init, arm_t_restore = e3b.restore_arm_e_checkpoint(e3b.RUN_DIR / "init" / "arm_t_e1.pt")
    arm_s_init, arm_s_identity = e3b.fresh_arm_s_checkpoint()
    initial = {"arm_t": arm_t_init, "arm_s": arm_s_init}

    checkpoint_manifest: dict[str, dict[str, Any]] = {"arm_t": {}, "arm_s": {}}
    arm_t_transfer: dict[str, Any] = {}
    arm_t_synthetic: dict[str, dict[str, float]] = {}
    arm_t_tier1: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    arm_s_tier1: dict[str, Any] = {}

    for step in ARM_T_STEPS:
        checkpoint = e3b.checkpoint_for("arm_t", step, initial["arm_t"])
        checkpoint_manifest["arm_t"][str(step)] = {
            "path": e3b.path_for_cli(checkpoint),
            "sha256": e3b.sha256_file(checkpoint),
        }
        for family in ("relay", "pointer"):
            e3b.final_symbol_eval(arm="arm_t", step=step, family=family, checkpoint=checkpoint)
        synthetic = e3b.diagonal_eval(
            label=f"arm_t_step_{step}_synthetic",
            checkpoint=checkpoint,
            data_jsonl=e3b.SYNTHETIC_DATA,
            max_depth=8,
            value_prefix="letter:",
        )
        tier1 = e3b.diagonal_eval(
            label=f"arm_t_step_{step}_tier1",
            checkpoint=checkpoint,
            data_jsonl=e3b.TIER1_DATA,
            max_depth=1,
            value_prefix="name:",
        )
        arm_t_transfer[str(step)] = single_arm_transfer("arm_t", step)
        arm_t_synthetic[str(step)] = {
            key: float(value) for key, value in synthetic["active_diagonal"].items()
        }
        arm_t_tier1[str(step)] = {
            "correct": tier1["correct"],
            "total": tier1["rows"],
            "accuracy": tier1["accuracy"],
        }
        summary["salvage"] = {
            "mode": "evaluation_only_after_preregistered_guardrail_stop",
            "matched_steps": list(MATCHED_STEPS),
            "arm_t_steps": list(ARM_T_STEPS),
            "checkpoint_manifest": checkpoint_manifest,
            "arm_t_transfer_by_step": arm_t_transfer,
            "arm_t_synthetic_by_step": arm_t_synthetic,
            "arm_t_tier1_by_step": arm_t_tier1,
            "arm_t_restore": arm_t_restore,
            "arm_s_identity": arm_s_identity,
        }
        write_progress(
            summary,
            status=f"salvage_arm_t_evaluated_step_{step}",
            message=f"Record E3b salvage Arm T eval step {step} {e3b.RUN_ID}",
        )

    for step in MATCHED_STEPS:
        checkpoint = e3b.checkpoint_for("arm_s", step, initial["arm_s"])
        checkpoint_manifest["arm_s"][str(step)] = {
            "path": e3b.path_for_cli(checkpoint),
            "sha256": e3b.sha256_file(checkpoint),
        }
        for family in ("relay", "pointer"):
            e3b.final_symbol_eval(arm="arm_s", step=step, family=family, checkpoint=checkpoint)
        tier1 = e3b.diagonal_eval(
            label=f"arm_s_step_{step}_tier1_salvage",
            checkpoint=checkpoint,
            data_jsonl=e3b.TIER1_DATA,
            max_depth=1,
            value_prefix="name:",
        )
        arm_s_tier1[str(step)] = {
            "correct": tier1["correct"],
            "total": tier1["rows"],
            "accuracy": tier1["accuracy"],
        }
        comparisons[str(step)] = e3b.comparison_at(step)
        summary["salvage"].update(
            {
                "checkpoint_manifest": checkpoint_manifest,
                "arm_s_tier1_by_step": arm_s_tier1,
                "matched_comparisons": comparisons,
            }
        )
        write_progress(
            summary,
            status=f"salvage_matched_evaluated_step_{step}",
            message=f"Record E3b salvage matched eval step {step} {e3b.RUN_ID}",
        )

    baseline_hits = read_hits(
        e3b.RUN_DIR / "guardrails" / "arm_s_step_0_tier1_salvage" / "rows.jsonl"
    )
    stopped_hits = read_hits(
        e3b.RUN_DIR / "guardrails" / "arm_s_step_3000_tier1_salvage" / "rows.jsonl"
    )
    if baseline_hits.keys() != stopped_hits.keys():
        raise RuntimeError("Arm S canary rows changed between step 0 and step 3,000")
    row_ids = sorted(baseline_hits)
    near_miss = guardrail_near_miss_context(
        baseline_hits=[baseline_hits[row_id] for row_id in row_ids],
        observed_hits=[stopped_hits[row_id] for row_id in row_ids],
        hard_stop_delta=HARD_STOP_DELTA,
    )

    matched_t_curve = {
        step: float(comparison["pooled"]["arm_t"]["accuracy"])
        for step, comparison in comparisons.items()
    }
    matched_s_curve = {
        step: float(comparison["pooled"]["arm_s"]["accuracy"])
        for step, comparison in comparisons.items()
    }
    arm_t_full_curve = {
        step: float(result["pooled"]["accuracy"]) for step, result in arm_t_transfer.items()
    }
    regression = classify_regression(arm_t_synthetic)
    decision = {
        "status": "truncated_at_preregistered_guardrail",
        "planned_endpoint_step": 6000,
        "planned_endpoint_available": False,
        "last_matched_step": 3000,
        "truncated_transference": comparisons["3000"]["pooled"],
        "matched_dose_to_0p71": {
            "arm_t": first_threshold_crossing(matched_t_curve, threshold=TRANSFER_THRESHOLD),
            "arm_s": first_threshold_crossing(matched_s_curve, threshold=TRANSFER_THRESHOLD),
        },
        "arm_t_full_dose_to_0p71": first_threshold_crossing(
            arm_t_full_curve, threshold=TRANSFER_THRESHOLD
        ),
        "arm_t_step_6000_descriptive": arm_t_transfer["6000"],
        "arm_t_synthetic_regression": regression,
        "arm_s_guardrail_near_miss": near_miss,
        "claim_boundary": (
            "The registered 6,000-step positive/null transference endpoint is unavailable. "
            "Matched results through step 3,000 and Arm T-only later results are descriptive."
        ),
        "prospective_guardrail_policy": "docs/SMALL_SAMPLE_HARD_STOP_POLICY_20260721.md",
    }
    summary["status"] = "finished_truncated_guardrail_block"
    summary["salvage"].update(
        {
            "checkpoint_manifest": checkpoint_manifest,
            "arm_t_transfer_by_step": arm_t_transfer,
            "arm_t_synthetic_by_step": arm_t_synthetic,
            "arm_t_tier1_by_step": arm_t_tier1,
            "arm_s_tier1_by_step": arm_s_tier1,
            "matched_comparisons": comparisons,
            "decision": decision,
        }
    )
    summary["decision"] = decision
    e3b.write_json(summary_path, summary)
    e3b.write_json(e3b.RUN_DIR / "salvage_summary.json", summary["salvage"])
    (e3b.RUN_DIR / "salvage_summary.md").write_text(
        "\n".join(
            [
                f"# E3b Guardrail-Truncated Evaluation - {e3b.RUN_ID}",
                "",
                "- Training stop: Arm S step 3,000 at the locked Tier-1 boundary.",
                f"- Last matched comparison: `{decision['truncated_transference']}`",
                f"- Arm T step 6,000 descriptive: `{decision['arm_t_step_6000_descriptive']}`",
                f"- Arm T synthetic regression: `{regression}`",
                f"- Canary boundary context: `{near_miss}`",
                "- Registered step-6,000 Arm T/S endpoint: unavailable.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    e3b.publish_run(
        e3b.RUN_DIR,
        message=f"Record completed truncated E3b salvage {e3b.RUN_ID} [skip ci]",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
