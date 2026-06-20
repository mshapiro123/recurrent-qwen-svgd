"""Parse and execute the tiny ARC program traces used for curriculum SFT.

This is intentionally a small DSL, not Python execution. It supports the
program lines emitted by ``format_symbolic_program_trace``:

    grid = constant_output_from_demonstrations(train_outputs)
    grid = transform(test_input, 'rot90')
    grid = crop_non_background(test_input, background=0)
    grid = recolor(grid, {1: 2, 0: 3})
    return grid
"""

from __future__ import annotations

import re

from eval.arc_agi_utils import ArcAgiExample, GEOMETRY_TRANSFORMS, Grid, apply_geometry_transform, validate_grid
from eval.arc_agi_symbolic import all_outputs_equal, apply_color_map, crop_non_background


_TRANSFORM_RE = re.compile(r"grid\s*=\s*transform\s*\(\s*test_input\s*,\s*['\"]?([a-z0-9_]+)['\"]?\s*\)")
_RECOLOR_RE = re.compile(r"grid\s*=\s*recolor\s*\(\s*grid\s*,\s*\{([^}]*)\}\s*\)")
_CONSTANT_RE = re.compile(r"grid\s*=\s*constant_output_from_demonstrations\s*\(\s*train_outputs\s*\)")
_CROP_RE = re.compile(r"grid\s*=\s*crop_non_background\s*\(\s*test_input\s*,\s*background\s*=\s*([0-9])\s*\)")


def _program_regions(text: str) -> list[str]:
    regions = [text]
    regions.extend(match.group(1) for match in re.finditer(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL | re.I))
    if "```" in text:
        chunks = text.split("```")
        for idx, chunk in enumerate(chunks):
            if idx % 2 == 1:
                lines = chunk.strip().splitlines()
                if lines and lines[0].strip().lower() in {"python", "text", "program"}:
                    chunk = "\n".join(lines[1:])
                regions.append(chunk)
    return regions


def parse_color_map_literal(value: str) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    if not value.strip():
        return mapping
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"['\"]?([0-9])['\"]?\s*:\s*([0-9])", item)
        if not match:
            return None
        mapping[int(match.group(1))] = int(match.group(2))
    return mapping


def execute_arc_program(example: ArcAgiExample, program_text: str) -> Grid | None:
    grid: Grid | None = None
    saw_return = False
    lines = [line.strip() for line in program_text.splitlines() if line.strip()]
    if not any("grid =" in line or line.startswith("return") for line in lines):
        return None

    for line in lines:
        if line == "program:":
            continue
        if _CONSTANT_RE.fullmatch(line):
            train_outputs = [pair.output for pair in example.train if pair.output is not None]
            if not all_outputs_equal(train_outputs):
                return None
            grid = [row[:] for row in train_outputs[0]]
            continue
        transform_match = _TRANSFORM_RE.fullmatch(line)
        if transform_match:
            transform = transform_match.group(1)
            if transform not in GEOMETRY_TRANSFORMS:
                return None
            grid = apply_geometry_transform(example.test_input, transform)
            continue
        crop_match = _CROP_RE.fullmatch(line)
        if crop_match:
            grid = crop_non_background(example.test_input, int(crop_match.group(1)))
            if grid is None:
                return None
            continue
        recolor_match = _RECOLOR_RE.fullmatch(line)
        if recolor_match:
            if grid is None:
                return None
            color_map = parse_color_map_literal(recolor_match.group(1))
            if color_map is None:
                return None
            grid = apply_color_map(grid, color_map)
            continue
        if line == "return grid":
            saw_return = True
            continue
        if line.startswith("#"):
            continue
        if "grid =" in line or line.startswith("return"):
            return None

    if not saw_return or grid is None:
        return None
    try:
        return validate_grid(grid)
    except ValueError:
        return None


def parse_arc_program_from_text(example: ArcAgiExample, text: str) -> Grid | None:
    for region in _program_regions(text):
        grid = execute_arc_program(example, region)
        if grid is not None:
            return grid
    return None


def execute_arc_program_on_input(example: ArcAgiExample, program_text: str, input_grid: Grid) -> Grid | None:
    probe = ArcAgiExample(
        task_id=example.task_id,
        test_index=example.test_index,
        train=example.train,
        test_input=input_grid,
        test_output=None,
    )
    return execute_arc_program(probe, program_text)


def arc_program_training_match_count(example: ArcAgiExample, text: str) -> tuple[int, int]:
    train_pairs = [pair for pair in example.train if pair.output is not None]
    if not train_pairs:
        return 0, 0

    best_matches = 0
    saw_executable_program = False
    for region in _program_regions(text):
        if execute_arc_program(example, region) is None:
            continue
        saw_executable_program = True
        matches = 0
        for pair in train_pairs:
            predicted = execute_arc_program_on_input(example, region, pair.input)
            matches += int(predicted == pair.output)
        best_matches = max(best_matches, matches)
        if best_matches == len(train_pairs):
            break
    if not saw_executable_program:
        return 0, 0
    return best_matches, len(train_pairs)


def arc_program_fits_training_examples(example: ArcAgiExample, text: str) -> bool:
    matches, total = arc_program_training_match_count(example, text)
    return total > 0 and matches == total
