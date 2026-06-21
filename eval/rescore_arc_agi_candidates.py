"""Rescore saved ARC-AGI candidate JSONL rows without regenerating candidates.

This is useful when selector logic changes faster than model generation. The
input is the candidate JSONL emitted by ``eval/eval_arc_agi.py``. If the
matching summary JSON is provided, stored inferred output shapes are reused for
shape-aware selectors.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_arc_agi import (  # noqa: E402
    grid_shape,
    grid_vote_key,
    grid_to_json_text,
    score_grid_prediction,
    summarize_candidate_sources,
    summarize_difficulty_buckets,
    summarize_examples,
    summarize_parse_methods,
    summarize_program_verifier,
    summarize_task_families,
    write_summary,
    write_summary_md,
)

Shape = tuple[int, int]
GroupKey = tuple[str, int]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_key(row: dict[str, Any]) -> GroupKey:
    return str(row["task_id"]), int(row.get("test_index", 0))


def group_candidate_rows(rows: list[dict[str, Any]]) -> dict[GroupKey, list[dict[str, Any]]]:
    groups: dict[GroupKey, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row.get("candidate_index", 0)))
    return groups


def shape_from_any(value: Any) -> Shape | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            rows = int(value[0])
            cols = int(value[1])
        except (TypeError, ValueError):
            return None
        if rows > 0 and cols > 0:
            return rows, cols
    return None


def read_inferred_shapes(summary_payload: dict[str, Any] | None) -> dict[GroupKey, list[Shape]]:
    if not summary_payload:
        return {}
    shapes_by_key: dict[GroupKey, list[Shape]] = {}
    for item in summary_payload.get("examples", []):
        key = (str(item["task_id"]), int(item.get("test_index", 0)))
        shapes: list[Shape] = []
        for raw_shape in item.get("inferred_shapes", []):
            shape = shape_from_any(raw_shape)
            if shape is not None and shape not in shapes:
                shapes.append(shape)
        shapes_by_key[key] = shapes
    return shapes_by_key


def read_example_metadata(summary_payload: dict[str, Any] | None) -> dict[GroupKey, dict[str, Any]]:
    if not summary_payload:
        return {}
    metadata_by_key: dict[GroupKey, dict[str, Any]] = {}
    for item in summary_payload.get("examples", []):
        key = (str(item["task_id"]), int(item.get("test_index", 0)))
        metadata_by_key[key] = {
            "difficulty_bucket": item.get("difficulty_bucket") or item.get("difficulty") or "unknown",
            "difficulty_score": item.get("difficulty_score"),
            "difficulty_source": item.get("difficulty_source"),
            "difficulty_features": item.get("difficulty_features"),
        }
    return metadata_by_key


def row_example_metadata(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in candidate_rows:
        if row.get("difficulty_bucket") or row.get("difficulty"):
            return {
                "difficulty_bucket": row.get("difficulty_bucket") or row.get("difficulty") or "unknown",
                "difficulty_score": row.get("difficulty_score"),
                "difficulty_source": row.get("difficulty_source"),
                "difficulty_features": row.get("difficulty_features"),
            }
    return {
        "difficulty_bucket": "unknown",
        "difficulty_score": None,
        "difficulty_source": None,
        "difficulty_features": None,
    }


def verified_program_index(candidate_rows: list[dict[str, Any]]) -> int | None:
    for idx, row in enumerate(candidate_rows):
        if (
            row.get("parsed_grid") is not None
            and row.get("parse_method") == "program"
            and row.get("program_fits_train")
        ):
            return idx
    return None


def select_heuristic_candidate_index(candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]) -> int:
    verified_index = verified_program_index(candidate_rows)
    if verified_index is not None:
        return verified_index

    first_valid: int | None = None
    for idx, row in enumerate(candidate_rows):
        parsed = row.get("parsed_grid")
        if parsed is None:
            continue
        if first_valid is None:
            first_valid = idx
        if preferred_shapes and grid_shape(parsed) in preferred_shapes:
            return idx
    if first_valid is not None:
        return first_valid
    return 0


def select_self_consistency_candidate_index(
    candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]
) -> int:
    verified_index = verified_program_index(candidate_rows)
    if verified_index is not None:
        return verified_index

    valid_rows: list[tuple[int, dict[str, Any]]] = [
        (idx, row) for idx, row in enumerate(candidate_rows) if row.get("parsed_grid") is not None
    ]
    if not valid_rows:
        return 0

    shape_rows = [
        (idx, row)
        for idx, row in valid_rows
        if preferred_shapes and grid_shape(row["parsed_grid"]) in preferred_shapes
    ]
    selection_pool = shape_rows or valid_rows

    grid_counts: dict[str, int] = {}
    grid_sources: dict[str, set[str]] = {}
    for _idx, row in selection_pool:
        key = grid_vote_key(row["parsed_grid"])
        grid_counts[key] = grid_counts.get(key, 0) + 1
        grid_sources.setdefault(key, set()).add(str(row.get("candidate_source", "unknown")))

    return max(
        selection_pool,
        key=lambda item: (
            grid_counts[grid_vote_key(item[1]["parsed_grid"])],
            len(grid_sources[grid_vote_key(item[1]["parsed_grid"])]),
            -item[0],
        ),
    )[0]


def reliability_vote_weight(row: dict[str, Any], preferred_shapes: set[Shape]) -> float:
    parsed = row.get("parsed_grid")
    if parsed is None:
        return 0.0

    weight = 1.0
    source = str(row.get("candidate_source", "unknown"))
    parse_method = str(row.get("parse_method", ""))
    if source.startswith("symbolic_"):
        weight += 2.0
    if source.startswith("model_tta_"):
        weight += 0.25
    if parse_method == "program":
        weight += 1.5
    if row.get("program_fits_train"):
        weight += 4.0
    else:
        try:
            matches = int(row.get("program_train_matches", 0))
            total = int(row.get("program_train_total", 0))
        except (TypeError, ValueError):
            matches = 0
            total = 0
        if total > 0 and matches > 0:
            weight += matches / total
    if preferred_shapes and grid_shape(parsed) in preferred_shapes:
        weight += 0.75
    return weight


def select_reliability_vote_candidate_index(
    candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]
) -> int:
    valid_rows: list[tuple[int, dict[str, Any], float]] = []
    for idx, row in enumerate(candidate_rows):
        weight = reliability_vote_weight(row, preferred_shapes)
        if weight > 0.0:
            valid_rows.append((idx, row, weight))
    if not valid_rows:
        return 0

    grid_stats: dict[str, dict[str, Any]] = {}
    for idx, row, weight in valid_rows:
        key = grid_vote_key(row["parsed_grid"])
        stats = grid_stats.setdefault(
            key,
            {
                "weight": 0.0,
                "count": 0,
                "sources": set(),
                "program_fits": 0,
                "symbolic": 0,
                "shape_matches": 0,
                "first_index": idx,
            },
        )
        stats["weight"] += weight
        stats["count"] += 1
        stats["sources"].add(str(row.get("candidate_source", "unknown")))
        stats["program_fits"] += int(bool(row.get("program_fits_train")))
        stats["symbolic"] += int(str(row.get("candidate_source", "")).startswith("symbolic_"))
        stats["shape_matches"] += int(bool(preferred_shapes and grid_shape(row["parsed_grid"]) in preferred_shapes))
        stats["first_index"] = min(int(stats["first_index"]), idx)

    winning_key = max(
        grid_stats,
        key=lambda key: (
            float(grid_stats[key]["weight"]),
            int(grid_stats[key]["count"]),
            len(grid_stats[key]["sources"]),
            int(grid_stats[key]["program_fits"]),
            int(grid_stats[key]["symbolic"]),
            int(grid_stats[key]["shape_matches"]),
            -int(grid_stats[key]["first_index"]),
        ),
    )
    return max(
        [(idx, row, weight) for idx, row, weight in valid_rows if grid_vote_key(row["parsed_grid"]) == winning_key],
        key=lambda item: (item[2], -item[0]),
    )[0]


def symbolic_candidate_index(candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]) -> int | None:
    valid_symbolic_rows = [
        (idx, row)
        for idx, row in enumerate(candidate_rows)
        if row.get("parsed_grid") is not None and str(row.get("candidate_source", "")).startswith("symbolic_")
    ]
    if not valid_symbolic_rows:
        return None

    for idx, row in valid_symbolic_rows:
        if preferred_shapes and grid_shape(row["parsed_grid"]) in preferred_shapes:
            return idx
    return valid_symbolic_rows[0][0]


def select_symbolic_priority_candidate_index(
    candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]
) -> int:
    verified_index = verified_program_index(candidate_rows)
    if verified_index is not None:
        return verified_index
    symbolic_index = symbolic_candidate_index(candidate_rows, preferred_shapes)
    if symbolic_index is not None:
        return symbolic_index
    return select_heuristic_candidate_index(candidate_rows, preferred_shapes)


def _candidate_weight(row: dict[str, Any], preferred_shapes: set[Shape]) -> float:
    return max(reliability_vote_weight(row, preferred_shapes), 0.0)


def cell_vote_grid(candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]) -> list[list[int]] | None:
    weighted_rows: list[tuple[int, dict[str, Any], float, Shape]] = []
    for idx, row in enumerate(candidate_rows):
        parsed = row.get("parsed_grid")
        if parsed is None:
            continue
        shape = grid_shape(parsed)
        if preferred_shapes and shape not in preferred_shapes:
            continue
        weight = _candidate_weight(row, preferred_shapes)
        if weight > 0.0:
            weighted_rows.append((idx, row, weight, shape))

    if not weighted_rows and preferred_shapes:
        return cell_vote_grid(candidate_rows, set())
    if not weighted_rows:
        return None

    shape_stats: dict[Shape, dict[str, float | int]] = {}
    for idx, _row, weight, shape in weighted_rows:
        stats = shape_stats.setdefault(shape, {"weight": 0.0, "count": 0, "first_index": idx})
        stats["weight"] = float(stats["weight"]) + weight
        stats["count"] = int(stats["count"]) + 1
        stats["first_index"] = min(int(stats["first_index"]), idx)

    winning_shape = max(
        shape_stats,
        key=lambda shape: (
            float(shape_stats[shape]["weight"]),
            int(shape_stats[shape]["count"]),
            -int(shape_stats[shape]["first_index"]),
        ),
    )
    shape_rows = [(idx, row, weight) for idx, row, weight, shape in weighted_rows if shape == winning_shape]
    rows, cols = winning_shape
    voted: list[list[int]] = []
    for row_idx in range(rows):
        voted_row: list[int] = []
        for col_idx in range(cols):
            color_stats: dict[int, dict[str, float | int]] = {}
            for idx, row, weight in shape_rows:
                color = int(row["parsed_grid"][row_idx][col_idx])
                stats = color_stats.setdefault(color, {"weight": 0.0, "count": 0, "first_index": idx})
                stats["weight"] = float(stats["weight"]) + weight
                stats["count"] = int(stats["count"]) + 1
                stats["first_index"] = min(int(stats["first_index"]), idx)
            voted_row.append(
                max(
                    color_stats,
                    key=lambda color: (
                        float(color_stats[color]["weight"]),
                        int(color_stats[color]["count"]),
                        -int(color_stats[color]["first_index"]),
                    ),
                )
            )
        voted.append(voted_row)
    return voted


def append_cell_vote_candidate(candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]) -> int | None:
    if not candidate_rows:
        return None
    voted = cell_vote_grid(candidate_rows, preferred_shapes)
    if voted is None:
        return None

    max_candidate_index = max(int(row.get("candidate_index", idx)) for idx, row in enumerate(candidate_rows))
    template = copy.deepcopy(candidate_rows[0])
    target = template.get("target_grid")
    template.update(
        {
            "candidate_index": max_candidate_index + 1,
            "candidate_source": "selector_cell_vote",
            "candidate_text": grid_to_json_text(voted),
            "parsed_grid": voted,
            "parse_method": "cell_vote",
            "program_train_matches": 0,
            "program_train_total": 0,
            "program_fits_train": False,
            "score": score_grid_prediction(voted, target),
            "selector_generated": True,
            "selected": False,
            "previous_selected": False,
        }
    )
    candidate_rows.append(template)
    return len(candidate_rows) - 1


def select_cell_vote_candidate_index(
    candidate_rows: list[dict[str, Any]], preferred_shapes: set[Shape]
) -> int:
    synthetic_index = append_cell_vote_candidate(candidate_rows, preferred_shapes)
    if synthetic_index is not None:
        return synthetic_index
    return select_reliability_vote_candidate_index(candidate_rows, preferred_shapes)


def select_candidate_index(
    candidate_rows: list[dict[str, Any]],
    preferred_shapes: list[Shape],
    *,
    selection_strategy: str,
) -> int:
    preferred_shape_set = set(preferred_shapes)
    if selection_strategy == "heuristic":
        return select_heuristic_candidate_index(candidate_rows, preferred_shape_set)
    if selection_strategy == "self_consistency":
        return select_self_consistency_candidate_index(candidate_rows, preferred_shape_set)
    if selection_strategy == "reliability_vote":
        return select_reliability_vote_candidate_index(candidate_rows, preferred_shape_set)
    if selection_strategy == "symbolic_priority":
        return select_symbolic_priority_candidate_index(candidate_rows, preferred_shape_set)
    if selection_strategy == "cell_vote":
        return select_cell_vote_candidate_index(candidate_rows, preferred_shape_set)
    raise ValueError(f"Unknown selection_strategy={selection_strategy!r}")


def has_target(candidate_rows: list[dict[str, Any]]) -> bool:
    if not candidate_rows:
        return False
    first_score = candidate_rows[0].get("score", {})
    return candidate_rows[0].get("target_grid") is not None or first_score.get("exact") is not None


def score_exact(row: dict[str, Any]) -> bool:
    return bool(row.get("score", {}).get("exact"))


def score_valid(row: dict[str, Any]) -> bool:
    return bool(row.get("score", {}).get("valid"))


def rescore_groups(
    rows: list[dict[str, Any]],
    *,
    inferred_shapes_by_key: dict[GroupKey, list[Shape]] | None = None,
    example_metadata_by_key: dict[GroupKey, dict[str, Any]] | None = None,
    selection_strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inferred_shapes_by_key = inferred_shapes_by_key or {}
    example_metadata_by_key = example_metadata_by_key or {}
    rescored_rows: list[dict[str, Any]] = []
    example_summaries: list[dict[str, Any]] = []

    for key, group_rows in sorted(group_candidate_rows(rows).items()):
        candidate_rows = [copy.deepcopy(row) for row in group_rows]
        preferred_shapes = inferred_shapes_by_key.get(key, [])
        example_metadata = example_metadata_by_key.get(key) or row_example_metadata(candidate_rows)
        selected_index = select_candidate_index(
            candidate_rows,
            preferred_shapes,
            selection_strategy=selection_strategy,
        )

        for idx, row in enumerate(candidate_rows):
            row["previous_selected"] = bool(row.get("selected"))
            row["selected"] = idx == selected_index
            row["selection_strategy"] = selection_strategy
            row["difficulty_bucket"] = example_metadata.get("difficulty_bucket", "unknown")
            row["difficulty_score"] = example_metadata.get("difficulty_score")
            row["difficulty_source"] = example_metadata.get("difficulty_source")
            rescored_rows.append(row)

        target_available = has_target(candidate_rows)
        selected_row = candidate_rows[selected_index] if candidate_rows else {}
        generated_rows = [row for row in candidate_rows if not row.get("selector_generated")]
        example_summaries.append(
            {
                "task_id": key[0],
                "test_index": key[1],
                "has_target": target_available,
                "first_exact": score_exact(candidate_rows[0]) if target_available and candidate_rows else None,
                "selected_exact": score_exact(selected_row) if target_available and candidate_rows else None,
                "selected_index": selected_index,
                "selection_strategy": selection_strategy,
                "inferred_shapes": [list(shape) for shape in preferred_shapes],
                "best_of_k_exact": any(score_exact(row) for row in generated_rows) if target_available else None,
                "valid_candidates": sum(1 for row in generated_rows if score_valid(row)),
                "num_candidates": len(generated_rows),
                "selector_generated_candidates": len(candidate_rows) - len(generated_rows),
                **example_metadata,
            }
        )

    return rescored_rows, example_summaries


def build_payload(
    *,
    candidates_jsonl: str | Path,
    summary_payload: dict[str, Any] | None,
    rescored_rows: list[dict[str, Any]],
    example_summaries: list[dict[str, Any]],
    selection_strategy: str,
) -> dict[str, Any]:
    source = summary_payload or {}
    return {
        "mode": f"rescore_{selection_strategy}",
        "checkpoint": source.get("checkpoint"),
        "tasks_path": source.get("tasks_path", "unknown"),
        "solutions_path": source.get("solutions_path"),
        "limit": source.get("limit"),
        "num_candidates": source.get("num_candidates"),
        "num_trajectories": source.get("num_trajectories"),
        "max_new_tokens": source.get("max_new_tokens"),
        "temperature": source.get("temperature"),
        "grid_format": source.get("grid_format", "already_parsed"),
        "geometry_tta": source.get("geometry_tta", "unknown"),
        "program_parse_mode": source.get("program_parse_mode", "already_parsed"),
        "selection_strategy": selection_strategy,
        "include_symbolic_candidates": source.get("include_symbolic_candidates"),
        "symbolic_position": source.get("symbolic_position"),
        "symbolic_candidate_format": source.get("symbolic_candidate_format", "grid"),
        "candidates_jsonl": str(candidates_jsonl),
        "source_summary": source.get("summary"),
        "summary": summarize_examples(example_summaries),
        "candidate_source_summary": summarize_candidate_sources(rescored_rows),
        "task_family_summary": summarize_task_families(example_summaries),
        "difficulty_summary": summarize_difficulty_buckets(example_summaries),
        "parse_method_summary": summarize_parse_methods(rescored_rows),
        "program_verifier_summary": summarize_program_verifier(rescored_rows),
        "examples": example_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--summary_json", help="Optional original summary JSON for inferred output shapes and metadata.")
    parser.add_argument(
        "--selection_strategy",
        default="heuristic",
        choices=("heuristic", "self_consistency", "reliability_vote", "symbolic_priority", "cell_vote"),
    )
    parser.add_argument("--output_jsonl")
    parser.add_argument("--output_summary_json")
    parser.add_argument("--output_summary_md")
    args = parser.parse_args()

    rows = read_jsonl(args.candidates_jsonl)
    summary_payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8")) if args.summary_json else None
    inferred_shapes_by_key = read_inferred_shapes(summary_payload)
    example_metadata_by_key = read_example_metadata(summary_payload)

    rescored_rows, example_summaries = rescore_groups(
        rows,
        inferred_shapes_by_key=inferred_shapes_by_key,
        example_metadata_by_key=example_metadata_by_key,
        selection_strategy=args.selection_strategy,
    )
    payload = build_payload(
        candidates_jsonl=args.candidates_jsonl,
        summary_payload=summary_payload,
        rescored_rows=rescored_rows,
        example_summaries=example_summaries,
        selection_strategy=args.selection_strategy,
    )
    write_jsonl(args.output_jsonl, rescored_rows)
    write_summary(args.output_summary_json, payload)
    write_summary_md(args.output_summary_md, payload)

    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
