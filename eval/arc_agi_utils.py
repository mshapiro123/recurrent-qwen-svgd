"""Utilities for ARC-AGI grid-task evaluation.

The canonical ARC-AGI task format is a JSON object with ``train`` and ``test``
pairs. Kaggle-style releases may also store many tasks in one JSON file and keep
test outputs in a separate solutions file. These helpers keep that data handling
separate from model execution so the evaluator can be tested cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable


Grid = list[list[int]]


@dataclass(frozen=True)
class ArcPair:
    input: Grid
    output: Grid | None = None


@dataclass(frozen=True)
class ArcAgiExample:
    task_id: str
    test_index: int
    train: tuple[ArcPair, ...]
    test_input: Grid
    test_output: Grid | None


def validate_grid(value: Any) -> Grid:
    if not isinstance(value, list) or not value:
        raise ValueError("grid must be a non-empty list of rows")
    width: int | None = None
    grid: Grid = []
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("grid rows must be non-empty lists")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("grid must be rectangular")
        parsed_row = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise ValueError("grid cells must be integers from 0 to 9")
            parsed_row.append(cell)
        grid.append(parsed_row)
    if len(grid) > 30 or (width or 0) > 30:
        raise ValueError("ARC-AGI grids must be at most 30x30")
    return grid


def _pair_from_json(row: dict[str, Any], *, require_output: bool) -> ArcPair:
    output = row.get("output")
    if require_output and output is None:
        raise ValueError("training pairs must include output grids")
    return ArcPair(
        input=validate_grid(row["input"]),
        output=validate_grid(output) if output is not None else None,
    )


def _task_examples(task_id: str, payload: dict[str, Any], solutions: Any | None = None) -> list[ArcAgiExample]:
    train = tuple(_pair_from_json(row, require_output=True) for row in payload["train"])
    tests = [_pair_from_json(row, require_output=False) for row in payload["test"]]
    solution_grids: list[Grid | None] = [None] * len(tests)

    if solutions is not None:
        if isinstance(solutions, list):
            if len(solutions) != len(tests):
                raise ValueError(f"solution count mismatch for {task_id}: {len(solutions)} != {len(tests)}")
            solution_grids = [validate_grid(grid) for grid in solutions]
        elif isinstance(solutions, dict):
            raw = solutions.get(task_id)
            if raw is not None:
                if not isinstance(raw, list):
                    raise ValueError(f"solutions[{task_id!r}] must be a list of grids")
                if len(raw) != len(tests):
                    raise ValueError(f"solution count mismatch for {task_id}: {len(raw)} != {len(tests)}")
                solution_grids = [validate_grid(grid) for grid in raw]
        else:
            raise ValueError("solutions must be a task_id dictionary or a list for a single task")

    examples: list[ArcAgiExample] = []
    for idx, pair in enumerate(tests):
        embedded_output = pair.output
        solved_output = solution_grids[idx] if idx < len(solution_grids) else None
        examples.append(
            ArcAgiExample(
                task_id=task_id,
                test_index=idx,
                train=train,
                test_input=pair.input,
                test_output=embedded_output or solved_output,
            )
        )
    return examples


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_arc_agi_examples(
    tasks_path: str | Path,
    *,
    solutions_path: str | Path | None = None,
    limit: int | None = None,
) -> list[ArcAgiExample]:
    """Load ARC-AGI examples from a directory, a single task, or a task dictionary."""

    path = Path(tasks_path)
    solutions = _load_json(solutions_path) if solutions_path else None
    examples: list[ArcAgiExample] = []

    if path.is_dir():
        for task_file in sorted(path.glob("*.json")):
            task_solutions = solutions.get(task_file.stem) if isinstance(solutions, dict) else None
            examples.extend(_task_examples(task_file.stem, _load_json(task_file), task_solutions))
    else:
        payload = _load_json(path)
        if isinstance(payload, dict) and "train" in payload and "test" in payload:
            examples.extend(_task_examples(path.stem, payload, solutions))
        elif isinstance(payload, dict):
            for task_id, task_payload in sorted(payload.items()):
                task_solutions = solutions.get(task_id) if isinstance(solutions, dict) else None
                examples.extend(_task_examples(str(task_id), task_payload, task_solutions))
        else:
            raise ValueError(f"Unsupported ARC-AGI task file shape: {path}")

    if limit is not None:
        examples = examples[:limit]
    return examples


def grid_to_compact_text(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def grid_to_row_text(grid: Grid) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def grid_to_json_text(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))


def format_grid_completion(grid: Grid, output_format: str = "json") -> str:
    if output_format == "json":
        return grid_to_json_text(grid)
    if output_format == "compact":
        return grid_to_row_text(grid)
    if output_format == "tagged":
        return f"<grid>\n{grid_to_row_text(grid)}\n</grid>"
    raise ValueError(f"Unknown output_format={output_format!r}")


def render_arc_prompt(example: ArcAgiExample, output_format: str = "json") -> str:
    if output_format == "json":
        instruction = "Colors are integers 0 through 9. Return only the output grid as JSON, with no prose."
        output_marker = "Output JSON grid:"
    elif output_format == "compact":
        instruction = (
            "Colors are integers 0 through 9. Return only the output grid rows, "
            "one row per line, using digits with no spaces and no prose."
        )
        output_marker = "Output grid rows:"
    elif output_format == "tagged":
        instruction = (
            "Colors are integers 0 through 9. Return only the output grid rows between "
            "<grid> and </grid>, using digits with no spaces."
        )
        output_marker = "Output tagged grid:"
    else:
        raise ValueError(f"Unknown output_format={output_format!r}")

    parts = [
        "You are solving an ARC-AGI grid transformation task.",
        "Infer the rule from the training examples and apply it to the test input.",
        instruction,
        "",
    ]
    for idx, pair in enumerate(example.train, start=1):
        assert pair.output is not None
        parts.extend(
            [
                f"Training example {idx} input:",
                grid_to_compact_text(pair.input),
                f"Training example {idx} output:",
                grid_to_compact_text(pair.output),
                "",
            ]
        )
    parts.extend(
        [
            "Test input:",
            grid_to_compact_text(example.test_input),
            output_marker,
        ]
    )
    return "\n".join(parts)


def _candidate_json_regions(text: str) -> Iterable[str]:
    stripped = text.strip()
    yield stripped
    if "```" in stripped:
        chunks = stripped.split("```")
        for idx, chunk in enumerate(chunks):
            if idx % 2 == 1:
                if chunk.lstrip().startswith("json"):
                    chunk = chunk.lstrip()[4:]
                yield chunk.strip()


def _candidate_text_regions(text: str) -> Iterable[str]:
    stripped = text.strip()
    yield stripped
    for match in re.finditer(r"<grid>\s*(.*?)\s*</grid>", stripped, flags=re.IGNORECASE | re.DOTALL):
        yield match.group(1).strip()
    if "```" in stripped:
        chunks = stripped.split("```")
        for idx, chunk in enumerate(chunks):
            if idx % 2 == 1:
                lines = chunk.strip().splitlines()
                if lines and lines[0].strip().lower() in {"text", "grid", "json"}:
                    chunk = "\n".join(lines[1:])
                yield chunk.strip()


def _parse_compact_grid_region(region: str) -> Grid | None:
    blocks: list[list[list[int]]] = []
    current: list[list[int]] = []
    for raw_line in region.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if re.fullmatch(r"[0-9](?:\s*[0-9]){0,29}", line):
            current.append([int(cell) for cell in re.findall(r"[0-9]", line)])
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    for block in blocks:
        try:
            return validate_grid(block)
        except ValueError:
            continue
    return None


def _parse_json_grid_from_text(text: str) -> Grid | None:
    """Extract the first valid ARC grid from generated text."""

    decoder = json.JSONDecoder()
    for region in _candidate_json_regions(text):
        for idx, char in enumerate(region):
            if char != "[":
                continue
            try:
                value, _ = decoder.raw_decode(region[idx:])
            except json.JSONDecodeError:
                continue
            try:
                return validate_grid(value)
            except ValueError:
                continue
    return None


def parse_grid_from_text(text: str, output_format: str = "auto") -> Grid | None:
    if output_format not in {"auto", "json", "compact", "tagged"}:
        raise ValueError(f"Unknown output_format={output_format!r}")
    if output_format == "json":
        return _parse_json_grid_from_text(text) or parse_grid_from_text(text, "compact")
    if output_format in {"compact", "tagged"}:
        for region in _candidate_text_regions(text):
            parsed = _parse_compact_grid_region(region)
            if parsed is not None:
                return parsed
        return _parse_json_grid_from_text(text)
    return _parse_json_grid_from_text(text) or parse_grid_from_text(text, "compact")


def score_grid_prediction(prediction: Grid | None, target: Grid | None) -> dict[str, Any]:
    if target is None:
        return {"has_target": False, "valid": prediction is not None, "exact": None}
    if prediction is None:
        return {"has_target": True, "valid": False, "exact": False}
    return {
        "has_target": True,
        "valid": True,
        "exact": prediction == target,
        "shape_match": len(prediction) == len(target)
        and all(len(left) == len(right) for left, right in zip(prediction, target)),
    }
