"""Analyze retained recirculation Phase-A generation receipts without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


KIND = "paper2_recirculation_phase_a_harm_channel_desk_rider_v1"
LOCK_KIND = "paper2_recirculation_phase_a_desk_rider_lock_v1"


def file_receipt(path: str | Path) -> dict[str, int | str]:
    source = Path(path)
    payload = source.read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="")
    return {
        "path": path.name,
        "bytes": len(payload.encode("utf-8")),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    """Return the zero-based first differing position, including prefix exhaustion."""

    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    if lower == upper:
        return ordered[lower]
    weight = coordinate - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    return {
        "rows": len(numeric),
        "minimum": min(numeric) if numeric else None,
        "q25": quantile(numeric, 0.25),
        "median": quantile(numeric, 0.5),
        "mean": statistics.fmean(numeric) if numeric else None,
        "q75": quantile(numeric, 0.75),
        "maximum": max(numeric) if numeric else None,
    }


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("correlation inputs must have equal length")
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [float(value) - left_mean for value in left]
    right_delta = [float(value) - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def classify_token_piece(
    piece: str | None,
    *,
    token_id: int | None,
    special_token_ids: set[int] | None = None,
) -> str:
    if token_id is None or piece is None:
        return "sequence_end"
    if special_token_ids and int(token_id) in special_token_ids:
        return "special"
    if not piece.strip():
        return "whitespace"
    has_digit = any(character.isdigit() for character in piece)
    has_operator = any(character in "+-*/=^%<>" for character in piece)
    if has_digit and has_operator:
        return "numeric_operator"
    if has_operator:
        return "operator"
    if has_digit:
        return "numeric"
    if all(
        character.isspace()
        or unicodedata.category(character).startswith(("P", "S"))
        for character in piece
    ):
        return "punctuation"
    return "lexical"


def _validate_row(row: Mapping[str, Any], *, source: str) -> None:
    required = {
        "item_id",
        "battery",
        "reader",
        "augmented_correct",
        "generated_text",
        "generated_token_ids",
        "generated_tokens",
        "answer_token_margins",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RuntimeError(f"{source} row is missing required fields: {missing}")
    tokens = list(row["generated_token_ids"])
    margins = list(row["answer_token_margins"])
    if int(row["generated_tokens"]) != len(tokens) or len(tokens) != len(margins):
        raise RuntimeError(
            f"{source} token/margin alignment failed for {row['item_id']}: "
            f"declared={row['generated_tokens']} tokens={len(tokens)} margins={len(margins)}"
        )
    if not tokens:
        raise RuntimeError(f"{source} row has no generated tokens: {row['item_id']}")


def validate_population(
    baseline_rows: Sequence[Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
    *,
    arm_name: str,
) -> None:
    if len(baseline_rows) != len(arm_rows):
        raise RuntimeError(f"{arm_name} row count differs from baseline")
    for baseline, arm in zip(baseline_rows, arm_rows):
        _validate_row(baseline, source="baseline")
        _validate_row(arm, source=arm_name)
        for field in ("item_id", "battery", "reader"):
            if baseline[field] != arm[field]:
                raise RuntimeError(
                    f"{arm_name} population mismatch at {field}: "
                    f"{baseline[field]!r} != {arm[field]!r}"
                )


def correctness_transition(baseline_correct: bool, arm_correct: bool) -> str:
    if baseline_correct and not arm_correct:
        return "regression"
    if not baseline_correct and arm_correct:
        return "fix"
    if baseline_correct:
        return "preserved_correct"
    return "preserved_incorrect"


def _empirical_percentile(values: Sequence[float], value: float) -> float:
    return sum(float(candidate) <= value for candidate in values) / len(values)


def analyze_arm(
    baseline_rows: Sequence[Mapping[str, Any]],
    arm_rows: Sequence[Mapping[str, Any]],
    *,
    arm_name: str,
    decode_token: Callable[[int], str],
    special_token_ids: set[int],
    pooled_baseline_gsm8k_margins: Sequence[float],
    low_margin_threshold: float,
) -> list[dict[str, Any]]:
    validate_population(baseline_rows, arm_rows, arm_name=arm_name)
    output = []
    for baseline, arm in zip(baseline_rows, arm_rows):
        baseline_tokens = [int(value) for value in baseline["generated_token_ids"]]
        arm_tokens = [int(value) for value in arm["generated_token_ids"]]
        baseline_margins = [float(value) for value in baseline["answer_token_margins"]]
        arm_margins = [float(value) for value in arm["answer_token_margins"]]
        divergence = first_divergence(baseline_tokens, arm_tokens)
        transition = correctness_transition(
            bool(baseline["augmented_correct"]), bool(arm["augmented_correct"])
        )
        if divergence is None:
            if (
                bool(baseline["augmented_correct"]) != bool(arm["augmented_correct"])
                or baseline.get("prediction") != arm.get("prediction")
                or baseline.get("generated_text") != arm.get("generated_text")
            ):
                raise RuntimeError(
                    f"identical token sequence changed reader outcome for {baseline['item_id']}"
                )
            prefix_fraction = 1.0
            baseline_token_id = None
            arm_token_id = None
            baseline_margin = None
            arm_margin = None
            baseline_piece = None
            arm_piece = None
        else:
            prefix_fraction = divergence / len(baseline_tokens)
            baseline_token_id = (
                baseline_tokens[divergence] if divergence < len(baseline_tokens) else None
            )
            arm_token_id = arm_tokens[divergence] if divergence < len(arm_tokens) else None
            baseline_margin = (
                baseline_margins[divergence] if divergence < len(baseline_margins) else None
            )
            arm_margin = arm_margins[divergence] if divergence < len(arm_margins) else None
            baseline_piece = (
                decode_token(baseline_token_id) if baseline_token_id is not None else None
            )
            arm_piece = decode_token(arm_token_id) if arm_token_id is not None else None
        baseline_class = classify_token_piece(
            baseline_piece,
            token_id=baseline_token_id,
            special_token_ids=special_token_ids,
        )
        arm_class = classify_token_piece(
            arm_piece,
            token_id=arm_token_id,
            special_token_ids=special_token_ids,
        )
        baseline_first_token_id = baseline_tokens[0]
        arm_first_token_id = arm_tokens[0]
        baseline_first_token_piece = decode_token(baseline_first_token_id)
        arm_first_token_piece = decode_token(arm_first_token_id)
        output.append(
            {
                "item_id": str(baseline["item_id"]),
                "battery": str(baseline["battery"]),
                "transition": transition,
                "baseline_correct": bool(baseline["augmented_correct"]),
                "arm_correct": bool(arm["augmented_correct"]),
                "baseline_length": len(baseline_tokens),
                "arm_length": len(arm_tokens),
                "arm_to_baseline_length_ratio": len(arm_tokens) / len(baseline_tokens),
                "baseline_first_token_id": baseline_first_token_id,
                "arm_first_token_id": arm_first_token_id,
                "baseline_first_token_piece": baseline_first_token_piece,
                "arm_first_token_piece": arm_first_token_piece,
                "first_token_changed": baseline_first_token_id != arm_first_token_id,
                "first_divergence_index_zero_based": divergence,
                "first_divergence_position_one_based": (
                    None if divergence is None else divergence + 1
                ),
                "baseline_prefix_fraction_retained": prefix_fraction,
                "baseline_onset_token_id": baseline_token_id,
                "arm_onset_token_id": arm_token_id,
                "baseline_onset_token_piece": baseline_piece,
                "arm_onset_token_piece": arm_piece,
                "baseline_onset_token_class": baseline_class,
                "arm_onset_token_class": arm_class,
                "either_onset_numeric_or_operator": bool(
                    baseline_class in {"numeric", "operator", "numeric_operator"}
                    or arm_class in {"numeric", "operator", "numeric_operator"}
                ),
                "baseline_onset_top1_margin": baseline_margin,
                "arm_onset_top1_margin": arm_margin,
                "baseline_onset_margin_pooled_percentile": (
                    None
                    if baseline_margin is None
                    else _empirical_percentile(
                        pooled_baseline_gsm8k_margins, baseline_margin
                    )
                ),
                "baseline_onset_margin_within_row_percentile": (
                    None
                    if baseline_margin is None
                    else _empirical_percentile(baseline_margins, baseline_margin)
                ),
                "baseline_onset_low_margin_q25": bool(
                    baseline_margin is not None and baseline_margin <= low_margin_threshold
                ),
            }
        )
    return output


def _transition_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["transition"]) for row in rows)
    return {
        name: int(counts.get(name, 0))
        for name in ("regression", "fix", "preserved_correct", "preserved_incorrect")
    }


def onset_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    transition: str,
    absolute_early_positions: int,
    normalized_early_fraction: float,
    normalized_late_fraction: float,
) -> dict[str, Any]:
    selected = [row for row in rows if row["transition"] == transition]
    if any(row["first_divergence_index_zero_based"] is None for row in selected):
        raise RuntimeError(f"{transition} row lacks token divergence")
    positions = [float(row["first_divergence_position_one_based"]) for row in selected]
    fractions = [float(row["baseline_prefix_fraction_retained"]) for row in selected]
    baseline_margins = [
        float(row["baseline_onset_top1_margin"])
        for row in selected
        if row["baseline_onset_top1_margin"] is not None
    ]
    arm_margins = [
        float(row["arm_onset_top1_margin"])
        for row in selected
        if row["arm_onset_top1_margin"] is not None
    ]
    baseline_classes = Counter(str(row["baseline_onset_token_class"]) for row in selected)
    arm_classes = Counter(str(row["arm_onset_token_class"]) for row in selected)
    baseline_ids = Counter(
        int(row["baseline_onset_token_id"])
        for row in selected
        if row["baseline_onset_token_id"] is not None
    )
    top_baseline_id_count = max(baseline_ids.values(), default=0)
    return {
        "rows": len(selected),
        "baseline_generation_length": describe(
            [float(row["baseline_length"]) for row in selected]
        ),
        "arm_generation_length": describe(
            [float(row["arm_length"]) for row in selected]
        ),
        "arm_to_baseline_generation_length_ratio": describe(
            [float(row["arm_to_baseline_length_ratio"]) for row in selected]
        ),
        "first_divergence_position_one_based": describe(positions),
        "first_position_rows": sum(
            int(row["first_divergence_position_one_based"]) == 1 for row in selected
        ),
        "first_position_fraction": (
            sum(int(row["first_divergence_position_one_based"]) == 1 for row in selected)
            / len(selected)
            if selected
            else None
        ),
        "baseline_prefix_fraction_retained": describe(fractions),
        "absolute_early_first_positions": absolute_early_positions,
        "absolute_early_rows": sum(
            int(row["first_divergence_position_one_based"]) <= absolute_early_positions
            for row in selected
        ),
        "absolute_early_fraction": (
            sum(
                int(row["first_divergence_position_one_based"])
                <= absolute_early_positions
                for row in selected
            )
            / len(selected)
            if selected
            else None
        ),
        "normalized_early_threshold": normalized_early_fraction,
        "normalized_early_rows": sum(
            float(row["baseline_prefix_fraction_retained"])
            <= normalized_early_fraction
            for row in selected
        ),
        "normalized_early_fraction": (
            sum(
                float(row["baseline_prefix_fraction_retained"])
                <= normalized_early_fraction
                for row in selected
            )
            / len(selected)
            if selected
            else None
        ),
        "normalized_late_threshold": normalized_late_fraction,
        "normalized_late_rows": sum(
            float(row["baseline_prefix_fraction_retained"])
            >= normalized_late_fraction
            for row in selected
        ),
        "normalized_late_fraction": (
            sum(
                float(row["baseline_prefix_fraction_retained"])
                >= normalized_late_fraction
                for row in selected
            )
            / len(selected)
            if selected
            else None
        ),
        "baseline_onset_top1_margin": describe(baseline_margins),
        "arm_onset_top1_margin": describe(arm_margins),
        "baseline_onset_low_margin_q25_rows": sum(
            bool(row["baseline_onset_low_margin_q25"]) for row in selected
        ),
        "baseline_onset_low_margin_q25_fraction": (
            sum(bool(row["baseline_onset_low_margin_q25"]) for row in selected)
            / len(selected)
            if selected
            else None
        ),
        "either_onset_numeric_or_operator_rows": sum(
            bool(row["either_onset_numeric_or_operator"]) for row in selected
        ),
        "either_onset_numeric_or_operator_fraction": (
            sum(bool(row["either_onset_numeric_or_operator"]) for row in selected)
            / len(selected)
            if selected
            else None
        ),
        "baseline_onset_token_classes": dict(sorted(baseline_classes.items())),
        "arm_onset_token_classes": dict(sorted(arm_classes.items())),
        "unique_baseline_onset_token_ids": len(baseline_ids),
        "largest_baseline_onset_token_id_share": (
            top_baseline_id_count / len(selected) if selected else None
        ),
    }


def first_token_transition_summary(
    rows: Sequence[Mapping[str, Any]], *, top_limit: int = 8
) -> dict[str, Any]:
    groups: dict[tuple[int, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["baseline_first_token_id"]),
            str(row["baseline_first_token_piece"]),
            int(row["arm_first_token_id"]),
            str(row["arm_first_token_piece"]),
        )
        groups.setdefault(key, []).append(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][2], item[0][1], item[0][3]),
    )
    top = []
    for key, selected in ordered[:top_limit]:
        baseline_correct = [row for row in selected if bool(row["baseline_correct"])]
        baseline_incorrect = [row for row in selected if not bool(row["baseline_correct"])]
        top.append(
            {
                "baseline_token_id": key[0],
                "baseline_token_piece": key[1],
                "arm_token_id": key[2],
                "arm_token_piece": key[3],
                "rows": len(selected),
                "transitions": _transition_counts(selected),
                "baseline_correct_rows": len(baseline_correct),
                "regression_rate_given_baseline_correct": (
                    sum(row["transition"] == "regression" for row in baseline_correct)
                    / len(baseline_correct)
                    if baseline_correct
                    else None
                ),
                "baseline_incorrect_rows": len(baseline_incorrect),
                "fix_rate_given_baseline_incorrect": (
                    sum(row["transition"] == "fix" for row in baseline_incorrect)
                    / len(baseline_incorrect)
                    if baseline_incorrect
                    else None
                ),
                "arm_to_baseline_generation_length_ratio": describe(
                    [float(row["arm_to_baseline_length_ratio"]) for row in selected]
                ),
            }
        )
    return {
        "rows": len(rows),
        "first_token_changed_rows": sum(bool(row["first_token_changed"]) for row in rows),
        "first_token_changed_fraction": (
            sum(bool(row["first_token_changed"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "unique_transition_pairs": len(groups),
        "top_transition_pairs": top,
        "status": "post_primary_receipt_descriptive_extension",
    }


def length_conditioning(
    rows: Sequence[Mapping[str, Any]],
    *,
    transition: str,
    eligibility: str,
    bins: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if eligibility == "baseline_correct":
        eligible = [row for row in rows if bool(row["baseline_correct"])]
    elif eligibility == "baseline_incorrect":
        eligible = [row for row in rows if not bool(row["baseline_correct"])]
    else:
        raise ValueError(f"unknown eligibility: {eligibility}")
    table = []
    assigned = set()
    for lower, upper in bins:
        selected = [
            row
            for row in eligible
            if int(lower) <= int(row["baseline_length"]) <= int(upper)
        ]
        assigned.update(str(row["item_id"]) for row in selected)
        events = sum(row["transition"] == transition for row in selected)
        table.append(
            {
                "baseline_length_inclusive": [int(lower), int(upper)],
                "eligible_rows": len(selected),
                "event_rows": events,
                "event_rate": events / len(selected) if selected else None,
            }
        )
    outside = [row for row in eligible if str(row["item_id"]) not in assigned]
    if outside:
        events = sum(row["transition"] == transition for row in outside)
        table.append(
            {
                "baseline_length_inclusive": "outside_registered_bins",
                "eligible_rows": len(outside),
                "event_rows": events,
                "event_rate": events / len(outside),
            }
        )
    lengths = [float(row["baseline_length"]) for row in eligible]
    indicators = [float(row["transition"] == transition) for row in eligible]
    event_lengths = [
        float(row["baseline_length"]) for row in eligible if row["transition"] == transition
    ]
    non_event_lengths = [
        float(row["baseline_length"]) for row in eligible if row["transition"] != transition
    ]
    nonempty_rates = [
        float(row["event_rate"]) for row in table if row["event_rate"] is not None
    ]
    return {
        "eligibility": eligibility,
        "eligible_rows": len(eligible),
        "event": transition,
        "event_rows": sum(indicators),
        "length_event_point_biserial": pearson(lengths, indicators),
        "event_baseline_length": describe(event_lengths),
        "non_event_baseline_length": describe(non_event_lengths),
        "fixed_bins": table,
        "nonempty_bin_rates_monotone_non_decreasing": all(
            right >= left for left, right in zip(nonempty_rates, nonempty_rates[1:])
        ),
    }


def arm_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    estimator = lock["estimator"]
    by_battery = {}
    for battery in sorted({str(row["battery"]) for row in rows}):
        selected = [row for row in rows if row["battery"] == battery]
        by_battery[battery] = {
            "rows": len(selected),
            "transitions": _transition_counts(selected),
            "baseline_correct_rows": sum(bool(row["baseline_correct"]) for row in selected),
            "regression_rate_given_baseline_correct": (
                sum(row["transition"] == "regression" for row in selected)
                / sum(bool(row["baseline_correct"]) for row in selected)
                if any(bool(row["baseline_correct"]) for row in selected)
                else None
            ),
            "baseline_correct_generation_length": describe(
                [
                    float(row["baseline_length"])
                    for row in selected
                    if bool(row["baseline_correct"])
                ]
            ),
        }
    gsm8k = [row for row in rows if row["battery"] == "gsm8k"]
    return {
        "rows": len(rows),
        "transitions": _transition_counts(rows),
        "by_battery": by_battery,
        "all_battery_regression_length_conditioning": length_conditioning(
            rows,
            transition="regression",
            eligibility="baseline_correct",
            bins=estimator["length_bins_inclusive"],
        ),
        "gsm8k": {
            "first_token_transitions": first_token_transition_summary(gsm8k),
            "regression_onset": onset_summary(
                gsm8k,
                transition="regression",
                absolute_early_positions=int(estimator["absolute_early_positions"]),
                normalized_early_fraction=float(estimator["normalized_early_fraction"]),
                normalized_late_fraction=float(estimator["normalized_late_fraction"]),
            ),
            "fix_onset": onset_summary(
                gsm8k,
                transition="fix",
                absolute_early_positions=int(estimator["absolute_early_positions"]),
                normalized_early_fraction=float(estimator["normalized_early_fraction"]),
                normalized_late_fraction=float(estimator["normalized_late_fraction"]),
            ),
            "regression_length_conditioning": length_conditioning(
                gsm8k,
                transition="regression",
                eligibility="baseline_correct",
                bins=estimator["length_bins_inclusive"],
            ),
            "fix_length_conditioning": length_conditioning(
                gsm8k,
                transition="fix",
                eligibility="baseline_incorrect",
                bins=estimator["length_bins_inclusive"],
            ),
        },
    }


def overlap_summary(
    rank1_rows: Sequence[Mapping[str, Any]],
    rank2_rows: Sequence[Mapping[str, Any]],
    *,
    transition: str,
    battery: str,
) -> dict[str, Any]:
    left_rows = {
        str(row["item_id"]): row
        for row in rank1_rows
        if row["battery"] == battery and row["transition"] == transition
    }
    right_rows = {
        str(row["item_id"]): row
        for row in rank2_rows
        if row["battery"] == battery and row["transition"] == transition
    }
    left = set(left_rows)
    right = set(right_rows)
    intersection = left & right
    union = left | right
    onset_left = [
        float(left_rows[item_id]["first_divergence_position_one_based"])
        for item_id in sorted(intersection)
    ]
    onset_right = [
        float(right_rows[item_id]["first_divergence_position_one_based"])
        for item_id in sorted(intersection)
    ]
    differences = [abs(a - b) for a, b in zip(onset_left, onset_right)]
    return {
        "battery": battery,
        "transition": transition,
        "rank_1_rows": len(left),
        "rank_2_rows": len(right),
        "intersection_rows": len(intersection),
        "union_rows": len(union),
        "jaccard": len(intersection) / len(union) if union else None,
        "rank_1_contained_fraction": len(intersection) / len(left) if left else None,
        "rank_2_contained_fraction": len(intersection) / len(right) if right else None,
        "shared_row_onset_correlation": pearson(onset_left, onset_right),
        "shared_row_onset_absolute_difference": describe(differences),
        "shared_row_exact_onset_matches": sum(
            left_value == right_value
            for left_value, right_value in zip(onset_left, onset_right)
        ),
    }


def _verify_source(path: Path, expected: Mapping[str, Any], *, label: str) -> None:
    actual = file_receipt(path)
    required = {"bytes": int(expected["bytes"]), "sha256": str(expected["sha256"])}
    if actual != required:
        raise RuntimeError(f"{label} source identity changed: {actual} != {required}")


def _load_token_decoder(lock: Mapping[str, Any]) -> tuple[Callable[[int], str], set[int], str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        lock["model"]["tokenizer_id"],
        revision=lock["model"]["revision"],
        local_files_only=bool(lock["model"]["local_files_only"]),
    )

    def decode(token_id: int) -> str:
        return str(
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    return decode, {int(value) for value in tokenizer.all_special_ids}, type(tokenizer).__name__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--baseline-rows", type=Path, required=True)
    parser.add_argument("--rank1-rows", type=Path, required=True)
    parser.add_argument("--rank2-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("kind") != LOCK_KIND or lock.get("status") != "LOCKED_CPU_ONLY_RECEIPT_ANALYSIS":
        raise RuntimeError("desk-rider lock is invalid")
    if not all(bool(value) for value in lock["prohibitions"].values()):
        raise RuntimeError("desk-rider prohibitions are not fully armed")
    _verify_source(args.baseline_rows, lock["sources"]["baseline"], label="baseline")
    _verify_source(
        args.rank1_rows,
        lock["sources"]["rank_1_additive_norm_matched"],
        label="rank_1_additive_norm_matched",
    )
    _verify_source(
        args.rank2_rows,
        lock["sources"]["rank_2_convex_identity"],
        label="rank_2_convex_identity",
    )

    baseline_rows = read_jsonl(args.baseline_rows)
    rank1_source = read_jsonl(args.rank1_rows)
    rank2_source = read_jsonl(args.rank2_rows)
    for label, rows in (
        ("baseline", baseline_rows),
        ("rank_1_additive_norm_matched", rank1_source),
        ("rank_2_convex_identity", rank2_source),
    ):
        if len(rows) != int(lock["sources"][label]["rows"]):
            raise RuntimeError(f"{label} row count changed")

    pooled_margins = [
        float(margin)
        for row in baseline_rows
        if row["battery"] == lock["estimator"]["primary_battery"]
        for margin in row["answer_token_margins"]
    ]
    low_margin_threshold = quantile(
        pooled_margins, float(lock["estimator"]["low_margin_quantile"])
    )
    if low_margin_threshold is None:
        raise RuntimeError("baseline GSM8K margin reference is empty")
    decode_token, special_token_ids, tokenizer_class = _load_token_decoder(lock)
    rank1_rows = analyze_arm(
        baseline_rows,
        rank1_source,
        arm_name="rank_1_additive_norm_matched",
        decode_token=decode_token,
        special_token_ids=special_token_ids,
        pooled_baseline_gsm8k_margins=pooled_margins,
        low_margin_threshold=low_margin_threshold,
    )
    rank2_rows = analyze_arm(
        baseline_rows,
        rank2_source,
        arm_name="rank_2_convex_identity",
        decode_token=decode_token,
        special_token_ids=special_token_ids,
        pooled_baseline_gsm8k_margins=pooled_margins,
        low_margin_threshold=low_margin_threshold,
    )

    private_receipts = {
        "rank_1_rows": write_jsonl(args.private_dir / "rank_1_row_analysis.jsonl", rank1_rows),
        "rank_2_rows": write_jsonl(args.private_dir / "rank_2_row_analysis.jsonl", rank2_rows),
    }
    summary = {
        "kind": KIND,
        "status": "complete_cpu_only_retained_receipts",
        "authority": dict(lock["authority"]),
        "phase_a_key": lock["phase_a_key"],
        "sources": {
            "baseline": {**file_receipt(args.baseline_rows), "rows": len(baseline_rows)},
            "rank_1_additive_norm_matched": {
                **file_receipt(args.rank1_rows),
                "rows": len(rank1_source),
            },
            "rank_2_convex_identity": {
                **file_receipt(args.rank2_rows),
                "rows": len(rank2_source),
            },
        },
        "estimator": dict(lock["estimator"]),
        "instrument": {
            "tokenizer_id": lock["model"]["tokenizer_id"],
            "tokenizer_revision": lock["model"]["revision"],
            "tokenizer_class": tokenizer_class,
            "local_files_only": True,
            "model_weights_loaded": False,
            "position_aligned_margin_rows_verified": len(baseline_rows) * 3,
            "margin_semantics": lock["estimator"]["margin_semantics"],
        },
        "baseline_gsm8k_position_margin_reference": {
            "positions": len(pooled_margins),
            "distribution": describe(pooled_margins),
            "low_margin_quantile": lock["estimator"]["low_margin_quantile"],
            "low_margin_threshold": low_margin_threshold,
        },
        "arms": {
            "rank_1_additive_norm_matched": arm_summary(rank1_rows, lock=lock),
            "rank_2_convex_identity": arm_summary(rank2_rows, lock=lock),
        },
        "cross_arm": {
            "gsm8k_regression_overlap": overlap_summary(
                rank1_rows, rank2_rows, transition="regression", battery="gsm8k"
            ),
            "gsm8k_fix_overlap": overlap_summary(
                rank1_rows, rank2_rows, transition="fix", battery="gsm8k"
            ),
        },
        "private_row_receipts": private_receipts,
        "caveats": [
            "First token divergence localizes output-trajectory departure, not the hidden-state event that caused it.",
            "The onset margin is top-1 minus runner-up at the generated position, not a gold-token margin.",
            "Token class is assigned from the pinned tokenizer's single-token decoded piece.",
            "All Phase-A writes were always on; this receipt contains no learned-gate labels.",
        ],
        "governance": {
            "gpu_used": False,
            "model_loaded": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_steps": 0,
            "new_generation": False,
            "confirm_scored": False,
            "eval_e_scored": False,
            "static_score_only_cells_added": 0,
            "phase_b_authorized": False,
            "branch_assignment": "withheld_for_strategy",
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        "recirculation_phase_a_desk_rider_complete "
        f"rank1_regressions={summary['arms']['rank_1_additive_norm_matched']['transitions']['regression']} "
        f"rank2_regressions={summary['arms']['rank_2_convex_identity']['transitions']['regression']} "
        f"summary={args.output_dir / 'summary.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
