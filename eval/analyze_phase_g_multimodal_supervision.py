"""Audit Phase G multimodal exposure and posterior target fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def problem_signature(row: dict[str, Any]) -> str:
    payload = {
        "depth": int(row["depth"]),
        "question": str(row["question"]),
        "start": str(row["start"]),
        "successors": row["successors"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def curriculum_exposure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[problem_signature(row)].append(row)

    group_rows: list[dict[str, Any]] = []
    for signature, members in groups.items():
        reachable_sets = {
            tuple(sorted(str(value) for value in row["reachable_symbols"]))
            for row in members
        }
        if len(reachable_sets) != 1:
            raise ValueError(f"Problem group {signature} has inconsistent reachable sets")
        reachable = set(next(iter(reachable_sets)))
        targets = {str(row["target"]) for row in members}
        if not targets.issubset(reachable):
            raise ValueError(f"Problem group {signature} contains an invalid target")
        group_rows.append(
            {
                "rows": len(members),
                "distinct_targets": len(targets),
                "reachable_targets": len(reachable),
                "target_support_fraction": len(targets) / len(reachable),
            }
        )

    repeated_rows = sum(row["rows"] for row in group_rows if row["rows"] > 1)
    multi_target_rows = sum(
        row["rows"] for row in group_rows if row["distinct_targets"] > 1
    )
    return {
        "rows": len(rows),
        "problem_groups": len(group_rows),
        "groups_with_repeated_prompt": sum(row["rows"] > 1 for row in group_rows),
        "groups_with_multiple_targets": sum(
            row["distinct_targets"] > 1 for row in group_rows
        ),
        "rows_in_repeated_prompt_groups": repeated_rows,
        "rows_in_multi_target_groups": multi_target_rows,
        "repeated_prompt_row_fraction": repeated_rows / max(len(rows), 1),
        "multi_target_row_fraction": multi_target_rows / max(len(rows), 1),
        "mean_target_support_fraction": (
            fmean(row["target_support_fraction"] for row in group_rows)
            if group_rows
            else 0.0
        ),
        "max_distinct_targets_per_problem": max(
            (row["distinct_targets"] for row in group_rows),
            default=0,
        ),
    }


def _fidelity_row(
    row: dict[str, Any],
    cache: dict[str, Any],
    *,
    arm: str,
    count: int,
) -> dict[str, Any]:
    predictions = [
        str(value) for value in cache["arms"][arm][str(count)]["predictions"]
    ]
    target = str(row["target"])
    return {
        "id": str(row["id"]),
        "depth": int(row["depth"]),
        "stratum": str(row["reachable_set_stratum"]),
        "first_matches_target": bool(predictions and predictions[0] == target),
        "contains_target": target in predictions,
        "target_sample_rate": (
            sum(value == target for value in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
    }


def _aggregate_fidelity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "first_target_rate": fmean(
            float(row["first_matches_target"]) for row in rows
        )
        if rows
        else 0.0,
        "target_in_k_rate": fmean(float(row["contains_target"]) for row in rows)
        if rows
        else 0.0,
        "mean_target_sample_rate": fmean(row["target_sample_rate"] for row in rows)
        if rows
        else 0.0,
    }


def posterior_target_fidelity(
    rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if [str(row["id"]) for row in rows] != [
        str(row["id"]) for row in cache_rows
    ]:
        raise ValueError("Phase G test rows and row cache do not have identical IDs")
    if not cache_rows:
        raise ValueError("Phase G row cache is empty")

    required_arms = {"prior", "posterior_teacher"}
    observed_arms = set(cache_rows[0]["arms"])
    if not required_arms.issubset(observed_arms):
        raise ValueError(
            f"Phase G row cache lacks required arms: {required_arms - observed_arms}"
        )
    sample_counts = sorted(
        int(value) for value in cache_rows[0]["arms"]["prior"]
    )
    k_values = sorted({1, max(sample_counts)})
    fidelity: dict[str, Any] = {"sample_counts": sample_counts, "by_k": {}}

    for count in k_values:
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for arm in ("prior", "posterior_teacher"):
            by_arm[arm] = [
                _fidelity_row(row, cache, arm=arm, count=count)
                for row, cache in zip(rows, cache_rows)
            ]
        prior = by_arm["prior"]
        teacher = by_arm["posterior_teacher"]
        paired = {
            "helped": sum(
                teacher_row["contains_target"] and not prior_row["contains_target"]
                for prior_row, teacher_row in zip(prior, teacher)
            ),
            "hurt": sum(
                prior_row["contains_target"] and not teacher_row["contains_target"]
                for prior_row, teacher_row in zip(prior, teacher)
            ),
        }
        paired["tied"] = len(rows) - paired["helped"] - paired["hurt"]
        fidelity["by_k"][str(count)] = {
            "prior": _aggregate_fidelity(prior),
            "posterior_teacher": _aggregate_fidelity(teacher),
            "posterior_minus_prior_target_in_k": (
                _aggregate_fidelity(teacher)["target_in_k_rate"]
                - _aggregate_fidelity(prior)["target_in_k_rate"]
            ),
            "paired_target_in_k": paired,
        }
    return fidelity


def posterior_target_conditioning(
    rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure whether posterior K=1 predictions change with a selected target.

    This is meaningful only on a repeated-prompt test set where the same
    problem is paired with more than one valid target chain.
    """

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, cache in zip(rows, cache_rows):
        grouped[problem_signature(row)].append((row, cache))
    multi_target = [
        members
        for members in grouped.values()
        if len(members) > 1 and len({str(row["target"]) for row, _ in members}) > 1
    ]
    result: dict[str, Any] = {
        "multi_target_groups": len(multi_target),
        "status": (
            "measured" if multi_target else "not_applicable_no_repeated_test_prompts"
        ),
    }
    for arm in ("prior", "posterior_teacher"):
        distinct_predictions: list[float] = []
        all_variants_match: list[float] = []
        for members in multi_target:
            first_predictions = [
                str(cache["arms"][arm]["1"]["predictions"][0])
                if cache["arms"][arm]["1"]["predictions"]
                else ""
                for _, cache in members
            ]
            targets = [str(row["target"]) for row, _ in members]
            distinct_predictions.append(float(len(set(first_predictions))))
            all_variants_match.append(
                float(all(prediction == target for prediction, target in zip(first_predictions, targets)))
            )
        result[arm] = {
            "mean_distinct_first_predictions": (
                fmean(distinct_predictions) if distinct_predictions else 0.0
            ),
            "all_variants_match_target_rate": (
                fmean(all_variants_match) if all_variants_match else 0.0
            ),
        }
    return result


def analyze(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exposure = curriculum_exposure(train_rows)
    fidelity = posterior_target_fidelity(test_rows, cache_rows)
    max_k = str(max(fidelity["sample_counts"]))
    teacher_lift = fidelity["by_k"][max_k][
        "posterior_minus_prior_target_in_k"
    ]
    if exposure["groups_with_multiple_targets"] == 0:
        interpretation = "single_target_per_problem_curriculum"
    elif teacher_lift < 0.05:
        interpretation = "posterior_not_materially_target_selective"
    else:
        interpretation = "posterior_target_signal_present"
    return {
        "kind": "phase_g_multimodal_supervision_audit",
        "status": "finished",
        "curriculum_exposure": exposure,
        "posterior_target_fidelity": fidelity,
        "posterior_target_conditioning": posterior_target_conditioning(
            test_rows,
            cache_rows,
        ),
        "interpretation": interpretation,
    }


def markdown(summary: dict[str, Any]) -> str:
    exposure = summary["curriculum_exposure"]
    fidelity = summary["posterior_target_fidelity"]
    max_k = str(max(fidelity["sample_counts"]))
    k1 = fidelity["by_k"]["1"]
    kmax = fidelity["by_k"][max_k]
    conditioning = summary["posterior_target_conditioning"]
    return "\n".join(
        [
            "# Phase G Multimodal Supervision Audit",
            "",
            f"- Interpretation: `{summary['interpretation']}`",
            f"- Training rows / problem groups: `{exposure['rows']}` / "
            f"`{exposure['problem_groups']}`",
            f"- Groups with repeated prompts: "
            f"`{exposure['groups_with_repeated_prompt']}`",
            f"- Groups with multiple supervised targets: "
            f"`{exposure['groups_with_multiple_targets']}`",
            f"- Mean supervised target support: "
            f"`{exposure['mean_target_support_fraction']:.4f}`",
            f"- K=1 posterior/prior target rates: "
            f"`{k1['posterior_teacher']['target_in_k_rate']:.4f}` / "
            f"`{k1['prior']['target_in_k_rate']:.4f}`",
            f"- K={max_k} posterior/prior target-in-K rates: "
            f"`{kmax['posterior_teacher']['target_in_k_rate']:.4f}` / "
            f"`{kmax['prior']['target_in_k_rate']:.4f}`",
            f"- K={max_k} posterior-minus-prior target lift: "
            f"`{kmax['posterior_minus_prior_target_in_k']:.4f}`",
            f"- Repeated-prompt target-conditioning groups: "
            f"`{conditioning['multi_target_groups']}`",
            f"- Posterior/prior distinct first predictions per target group: "
            f"`{conditioning['posterior_teacher']['mean_distinct_first_predictions']:.4f}` / "
            f"`{conditioning['prior']['mean_distinct_first_predictions']:.4f}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--row_cache_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    summary = analyze(
        read_jsonl(args.train_jsonl),
        read_jsonl(args.test_jsonl),
        read_jsonl(args.row_cache_jsonl),
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
