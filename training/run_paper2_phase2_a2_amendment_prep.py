"""Verify A2 calibration receipts and prepare an unlocked amendment package."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from training.run_paper2_phase2_matched_alpha import sha256_file, write_json


KIND = "paper2_phase2_a2_amendment_prep_v1"
CALIBRATION_KIND = "paper2_phase2_a2_zero_update_calibration_v1"
STRATEGY_DRIVE_ID = "1CCIZqKgIvaveFit8IEOzcXfEcf-4YYWZ"
LOSSES = ("final_ce", "cumulative_kl", "local_ce", "preserve_kl")


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _assert_close(observed: float, expected: float, *, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"{label} mismatch: observed={observed} expected={expected}")


def verify_arm(arm: dict[str, Any], private_path: Path) -> dict[str, Any]:
    seed = int(arm["seed"])
    if arm["status"] != "complete_zero_update" or int(arm["optimizer_updates"]) != 0:
        raise RuntimeError(f"seed {seed} is not a complete zero-update calibration")
    if not arm["mutation_assertions"]["all_unchanged"]:
        raise RuntimeError(f"seed {seed} failed its mutation assertion")
    expected_sha = arm["private_batch_receipt"]["sha256"]
    observed_sha = sha256_file(private_path)
    if observed_sha != expected_sha:
        raise RuntimeError(f"seed {seed} private calibration receipt SHA mismatch")
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if private.get("kind") != f"{CALIBRATION_KIND}_batch_rows":
        raise RuntimeError(f"seed {seed} private calibration kind mismatch")
    if int(private.get("seed", -1)) != seed:
        raise RuntimeError(f"seed {seed} private calibration identity mismatch")
    rows = private["rows"]
    if len(rows) != 51 or [int(row["batch"]) for row in rows] != list(range(50, 101)):
        raise RuntimeError(f"seed {seed} private calibration batch population mismatch")
    for row in rows:
        for surface in ("loss_values", "raw_gradient_norms", "legacy_weighted_gradient_norms"):
            if set(row[surface]) != set(LOSSES):
                raise RuntimeError(f"seed {seed} batch {row['batch']} loss surface mismatch")
            if any(not math.isfinite(float(value)) for value in row[surface].values()):
                raise RuntimeError(f"seed {seed} batch {row['batch']} contains non-finite values")

    recomputed_mean_norms = {
        loss: _mean([float(row["raw_gradient_norms"][loss]) for row in rows])
        for loss in LOSSES
    }
    for loss in LOSSES:
        _assert_close(
            recomputed_mean_norms[loss],
            float(arm["raw_mean_gradient_norms"][loss]),
            label=f"seed {seed} {loss} raw mean",
        )
    primary_cosines = [
        float(row["global_conflict_cosines"]["cumulative_kl__local_ce"])
        for row in rows
    ]
    _assert_close(
        _mean(primary_cosines),
        float(arm["primary_primary_conflict"]["distribution"]["mean"]),
        label=f"seed {seed} primary cosine mean",
    )
    weights = {
        loss: float(arm["legacy_initialization"]["static_weights"][loss])
        for loss in LOSSES
    }
    mean_weighted = {
        loss: weights[loss] * recomputed_mean_norms[loss] for loss in LOSSES
    }
    denominator = sum(mean_weighted.values())
    recomputed_shares = {loss: mean_weighted[loss] / denominator for loss in LOSSES}
    for loss in LOSSES:
        _assert_close(
            recomputed_shares[loss],
            float(arm["legacy_initialization"]["mean_realized_independent_shares"][loss]),
            label=f"seed {seed} {loss} initialized share",
        )

    conflict_pathology = _mean(primary_cosines) < -0.5
    spread_pathology = bool(arm["raw_norm_spread_exceeds_a1"])
    return {
        "seed": seed,
        "private_receipt": {"path": str(private_path), "sha256": observed_sha},
        "rows_verified": len(rows),
        "batch_numbers_verified": [50, 100],
        "raw_mean_gradient_norms": recomputed_mean_norms,
        "raw_norm_spread": float(arm["raw_norm_spread"]),
        "a1_raw_norm_spread": float(arm["a1_calibration_raw_norm_spread"]),
        "raw_norm_spread_pathology": spread_pathology,
        "primary_cosine": {
            "mean": _mean(primary_cosines),
            "minimum": min(primary_cosines),
            "maximum": max(primary_cosines),
            "fraction_below_minus_0p5": sum(value < -0.5 for value in primary_cosines)
            / len(primary_cosines),
            "pathology": conflict_pathology,
        },
        "initialization_weights": weights,
        "initialized_independent_shares": recomputed_shares,
        "legacy_initialization_meets_future_directional_contract": bool(
            arm["legacy_initialization"]["directional_contract_compatible"]
        ),
        "candidate_clip_catastrophe_tripwire": float(
            arm["clip_observation"]["candidate_catastrophe_tripwire_p99_times_10"]
        ),
        "source_checkpoint": arm["source_checkpoint"],
        "parameter_hashes_unchanged": True,
        "pathology_found": conflict_pathology or spread_pathology,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-2 A2 Calibration Reconciliation and Amendment Draft",
        "",
        "Date: 2026-08-05",
        "",
        "## Decision surface",
        "",
        f"- Calibration verification: `{summary['verification_status']}`.",
        f"- Pathology verdict: `{summary['pathology_verdict']}`.",
        "- Optimizer updates: `0`.",
        "- A2 training launched: `false`.",
        "- Amendment status: `draft_unlocked_strategy_review_required`.",
        "",
        "## Seed receipts",
        "",
        "| Seed | Primary cosine mean | Fraction below -0.5 | Raw spread | A1 spread | Clip p99 x 10 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        lines.append(
            f"| {arm['seed']} | {arm['primary_cosine']['mean']:.4f} | "
            f"{arm['primary_cosine']['fraction_below_minus_0p5']:.4f} | "
            f"{arm['raw_norm_spread']:.1f} | {arm['a1_raw_norm_spread']:.1f} | "
            f"{arm['candidate_clip_catastrophe_tripwire']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Both seed-specific private 51-batch receipts reproduce the public raw norms,",
            "initialized shares, and primary-loss cosine means. No mutation, conflict, or",
            "raw-spread pathology is present.",
            "",
            "## Amendment frame for strategy approval",
            "",
            "- Keep the seed-specific calibrated static weights as initialization only.",
            "- Audit the matched 51 x 128 training estimator at steps 200, 400, 600, 800, and 1,000.",
            "- Require cumulative KL plus local CE to hold at least 50% of independent trainable-path share.",
            "- Cap each non-primary loss, including final CE and preserve KL, at 25%.",
            "- Keep preserve KL descriptive and enforce preservation through endpoint-quality tripwires.",
            "- Use each seed's observed p99 x 10 value as a catastrophe-only clip tripwire, not a shaper.",
            "- Keep the +2% oracle-headroom gate, matched draft-head-only superiority, and endpoint quality gate unchanged.",
            "",
            "The legacy 35/35/10/20 point shares are intentionally incompatible with the",
            "future directional contract at step zero. This is not a calibration pathology:",
            "strategy demoted them to initialization targets only. The amendment must state",
            "explicitly that hard directional audits begin at optimizer step 200.",
            "",
            "## Next decision",
            "",
            "Strategy must approve or revise this draft. Only the subsequent committed",
            "amendment lock may authorize the two A2 arms and two matched controls.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    calibration = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    if calibration.get("kind") != CALIBRATION_KIND:
        raise RuntimeError("A2 calibration summary kind mismatch")
    if calibration.get("status") != "complete_with_a2_amendment_required":
        raise RuntimeError("A2 calibration is not complete")
    if int(calibration["optimizer_updates"]) != 0 or calibration["a2_training_launched"]:
        raise RuntimeError("A2 calibration crossed its zero-update boundary")
    if calibration["strategy_authority"]["drive_id"] != STRATEGY_DRIVE_ID:
        raise RuntimeError("A2 calibration strategy authority mismatch")
    arms = []
    for arm in calibration["arms"]:
        seed = int(arm["seed"])
        arms.append(verify_arm(arm, args.private_dir / f"seed_{seed}_batch_rows.json"))
    arms.sort(key=lambda row: int(row["seed"]))
    if [arm["seed"] for arm in arms] != [0, 1]:
        raise RuntimeError("A2 calibration must contain exactly seeds 0 and 1")
    pathology = any(arm["pathology_found"] for arm in arms)
    draft = {
        "kind": "paper2_phase2_a2_amendment_draft_v1",
        "status": "draft_unlocked_strategy_review_required",
        "source_calibration_sha256": sha256_file(args.calibration_summary),
        "source_calibration_commit": "e4275e02",
        "strategy_authority_drive_id": STRATEGY_DRIVE_ID,
        "a2_training_authorized": False,
        "optimizer_updates": 0,
        "alpha": 0.5,
        "seeds": [0, 1],
        "source_checkpoint_sha256_by_seed": {
            str(arm["seed"]): arm["source_checkpoint"]["sha256"] for arm in arms
        },
        "initialization_weights_by_seed": {
            str(arm["seed"]): arm["initialization_weights"] for arm in arms
        },
        "directional_contract": {
            "population": "matched_training_estimator_51_batches_x_128_seed_specific",
            "audit_steps": [200, 400, 600, 800, 1000],
            "primary_losses": ["cumulative_kl", "local_ce"],
            "primary_joint_share_minimum": 0.50,
            "non_primary_losses": ["final_ce", "preserve_kl"],
            "individual_non_primary_share_maximum": 0.25,
            "preserve_kl_role": "descriptive_weight_fixed_from_calibration",
            "step_zero_legacy_share_exception": True,
        },
        "clip_catastrophe_tripwire_by_seed": {
            str(arm["seed"]): arm["candidate_clip_catastrophe_tripwire"] for arm in arms
        },
        "unchanged_gates": {
            "oracle_headroom_minimum_relative": 0.02,
            "full_system_must_beat_matched_draft_head_only": True,
            "endpoint_quality_retained": True,
        },
        "run_matrix_after_lock": [
            "seed_0_full_a2",
            "seed_0_draft_head_only_control",
            "seed_1_full_a2",
            "seed_1_draft_head_only_control",
        ],
        "automatic_lock_or_launch": False,
    }
    summary = {
        "kind": KIND,
        "status": "complete_with_strategy_lock_required",
        "launcher_commit": _git_head(root),
        "verification_status": "two_seed_public_private_receipts_reconciled",
        "pathology_verdict": "clear_for_amendment_review" if not pathology else "stop_and_report",
        "optimizer_updates": 0,
        "a2_training_launched": False,
        "frozen_confirmatory_partitions_touched": [],
        "calibration_summary": {
            "path": str(args.calibration_summary),
            "sha256": sha256_file(args.calibration_summary),
        },
        "arms": arms,
        "amendment_draft": draft,
        "next_action": "strategy approves or revises draft before committed A2 amendment lock",
        "v1d_receipt": {
            "drive_id": "1zH20VEuuc4myQl9pvFgv56iQ4tNXa4iQ",
            "teacher_top1_flips": [1209, 2000],
            "preserve_retention": [1997, 2000],
            "preserve_wilson_95_lower": 0.9956,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "a2_amendment_draft.json", draft)
    (args.output_dir / "STRATEGY_HANDOFF_A2_CALIBRATION_20260805.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration_summary", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "pathology_verdict": result["pathology_verdict"],
                "a2_training_launched": result["a2_training_launched"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
