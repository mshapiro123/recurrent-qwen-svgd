"""Small symbolic candidate generator for easy ARC-AGI transformations.

This is not intended to be a full ARC solver. It gives the evaluation harness a
stronger non-neural baseline and a hybrid candidate source for transformations
that can be inferred exactly from demonstrations:

- dihedral geometry transforms;
- consistent per-color maps after a geometry transform;
- non-background object bounding-box crops;
- crop, then transform, then recolor compositions;
- same-canvas non-background object moves, optionally with recoloring;
- constant-output tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.arc_agi_utils import ArcAgiExample, GEOMETRY_TRANSFORMS, Grid, apply_geometry_transform


@dataclass(frozen=True)
class SymbolicCandidate:
    name: str
    grid: Grid
    trace: tuple[str, ...] = ()
    program: tuple[str, ...] = ()


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def apply_color_map(grid: Grid, color_map: dict[int, int]) -> Grid:
    return [[color_map.get(cell, cell) for cell in row] for row in grid]


def crop_non_background(grid: Grid, background: int) -> Grid | None:
    rows: list[int] = []
    cols: list[int] = []
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell != background:
                rows.append(row_idx)
                cols.append(col_idx)
    if not rows or not cols:
        return None
    row_start, row_end = min(rows), max(rows)
    col_start, col_end = min(cols), max(cols)
    return [row[col_start : col_end + 1] for row in grid[row_start : row_end + 1]]


def move_non_background(grid: Grid, background: int, delta_row: int, delta_col: int) -> Grid | None:
    height, width = grid_shape(grid)
    moved = [[background for _ in range(width)] for _ in range(height)]
    saw_foreground = False
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell == background:
                continue
            saw_foreground = True
            out_row = row_idx + delta_row
            out_col = col_idx + delta_col
            if not (0 <= out_row < height and 0 <= out_col < width):
                return None
            moved[out_row][out_col] = cell
    return moved if saw_foreground else None


def learn_color_map(inputs: list[Grid], outputs: list[Grid]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for input_grid, output_grid in zip(inputs, outputs):
        if grid_shape(input_grid) != grid_shape(output_grid):
            return None
        for in_row, out_row in zip(input_grid, output_grid):
            for source, target in zip(in_row, out_row):
                existing = mapping.get(source)
                if existing is not None and existing != target:
                    return None
                mapping[source] = target
    return mapping


def color_map_trace(color_map: dict[int, int]) -> str:
    if not color_map:
        return "Color map: identity."
    pairs = ", ".join(f"{source}->{target}" for source, target in sorted(color_map.items()))
    return f"Color map: {pairs}."


def color_map_program_literal(color_map: dict[int, int]) -> str:
    pairs = ", ".join(f"{source}: {target}" for source, target in sorted(color_map.items()))
    return "{" + pairs + "}"


def all_outputs_equal(outputs: list[Grid]) -> bool:
    return bool(outputs) and all(output == outputs[0] for output in outputs)


def learn_crop_background(inputs: list[Grid], outputs: list[Grid]) -> int | None:
    for background in range(10):
        predicted = [crop_non_background(grid, background) for grid in inputs]
        if all(item is not None for item in predicted) and predicted == outputs:
            return background
    return None


def learn_crop_background_and_color_map(inputs: list[Grid], outputs: list[Grid]) -> tuple[int, dict[int, int]] | None:
    for background in range(10):
        cropped = [crop_non_background(grid, background) for grid in inputs]
        if any(item is None for item in cropped):
            continue
        color_map = learn_color_map([grid for grid in cropped if grid is not None], outputs)
        if color_map is not None and any(source != target for source, target in color_map.items()):
            return background, color_map
    return None


def learn_crop_transform_and_color_map(inputs: list[Grid], outputs: list[Grid]) -> tuple[int, str, dict[int, int]] | None:
    for background in range(10):
        cropped = [crop_non_background(grid, background) for grid in inputs]
        if any(item is None for item in cropped):
            continue
        cropped_grids = [grid for grid in cropped if grid is not None]
        for transform in GEOMETRY_TRANSFORMS:
            transformed = [apply_geometry_transform(grid, transform) for grid in cropped_grids]
            color_map = learn_color_map(transformed, outputs)
            if color_map is None:
                continue
            is_nontrivial = transform != "identity" or any(source != target for source, target in color_map.items())
            if is_nontrivial:
                return background, transform, color_map
    return None


def learn_move_background_delta_and_color_map(
    inputs: list[Grid],
    outputs: list[Grid],
) -> tuple[int, int, int, dict[int, int]] | None:
    if not inputs or any(grid_shape(input_grid) != grid_shape(output_grid) for input_grid, output_grid in zip(inputs, outputs)):
        return None
    max_height = max(len(grid) for grid in inputs)
    max_width = max(len(grid[0]) for grid in inputs)
    for background in range(10):
        for delta_row in range(-(max_height - 1), max_height):
            for delta_col in range(-(max_width - 1), max_width):
                moved = [move_non_background(grid, background, delta_row, delta_col) for grid in inputs]
                if any(item is None for item in moved):
                    continue
                moved_grids = [grid for grid in moved if grid is not None]
                color_map = learn_color_map(moved_grids, outputs)
                if color_map is None:
                    continue
                is_nontrivial = (
                    delta_row != 0
                    or delta_col != 0
                    or any(source != target for source, target in color_map.items())
                )
                if is_nontrivial:
                    return background, delta_row, delta_col, color_map
    return None


def dedupe_candidates(candidates: list[SymbolicCandidate]) -> list[SymbolicCandidate]:
    seen: set[str] = set()
    deduped: list[SymbolicCandidate] = []
    for candidate in candidates:
        key = repr(candidate.grid)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def symbolic_candidates(example: ArcAgiExample) -> list[SymbolicCandidate]:
    train_pairs = [pair for pair in example.train if pair.output is not None]
    if not train_pairs:
        return []

    outputs = [pair.output for pair in train_pairs if pair.output is not None]
    candidates: list[SymbolicCandidate] = []

    if all_outputs_equal(outputs):
        candidates.append(
            SymbolicCandidate(
                "constant_output",
                [row[:] for row in outputs[0]],
                ("Rule: all demonstrations share the same output grid.", "Action: copy that output grid."),
                (
                    "program:",
                    "  grid = constant_output_from_demonstrations(train_outputs)",
                    "  return grid",
                ),
            )
        )

    crop_background = learn_crop_background([pair.input for pair in train_pairs], outputs)
    if crop_background is not None:
        predicted_crop = crop_non_background(example.test_input, crop_background)
        if predicted_crop is not None:
            candidates.append(
                SymbolicCandidate(
                    f"crop_non_background_bg{crop_background}",
                    predicted_crop,
                    (
                        f"Background color: {crop_background}.",
                        "Rule: crop the minimal bounding box containing all non-background cells.",
                        "Action: apply the same crop to the test input.",
                    ),
                    (
                        "program:",
                        f"  grid = crop_non_background(test_input, background={crop_background})",
                        "  return grid",
                    ),
                )
            )

    crop_recolor = learn_crop_background_and_color_map([pair.input for pair in train_pairs], outputs)
    if crop_recolor is not None:
        crop_background, color_map = crop_recolor
        cropped_test = crop_non_background(example.test_input, crop_background)
        if cropped_test is not None:
            predicted = apply_color_map(cropped_test, color_map)
            candidates.append(
                SymbolicCandidate(
                    f"crop_non_background_bg{crop_background}+color_map",
                    predicted,
                    (
                        f"Background color: {crop_background}.",
                        "Rule: crop the minimal bounding box around non-background cells, then apply a consistent color map.",
                        color_map_trace(color_map),
                        "Action: crop and recolor the test input.",
                    ),
                    (
                        "program:",
                        f"  grid = crop_non_background(test_input, background={crop_background})",
                        f"  grid = recolor(grid, {color_map_program_literal(color_map)})",
                        "  return grid",
                    ),
                )
            )

    for transform in GEOMETRY_TRANSFORMS:
        transformed_inputs = [apply_geometry_transform(pair.input, transform) for pair in train_pairs]
        color_map = learn_color_map(transformed_inputs, outputs)
        if color_map is None:
            continue
        transformed_test = apply_geometry_transform(example.test_input, transform)
        predicted = apply_color_map(transformed_test, color_map)
        label = f"{transform}+color_map" if color_map else transform
        candidates.append(
            SymbolicCandidate(
                label,
                predicted,
                (
                    f"Geometry transform: {transform}.",
                    color_map_trace(color_map),
                    "Action: apply the transform and color map to the test input.",
                ),
                (
                    "program:",
                    f"  grid = transform(test_input, {transform!r})",
                    f"  grid = recolor(grid, {color_map_program_literal(color_map)})",
                    "  return grid",
                ),
            )
        )

    move_recolor = learn_move_background_delta_and_color_map([pair.input for pair in train_pairs], outputs)
    if move_recolor is not None:
        background, delta_row, delta_col, color_map = move_recolor
        moved_test = move_non_background(example.test_input, background, delta_row, delta_col)
        if moved_test is not None:
            predicted = apply_color_map(moved_test, color_map)
            candidates.append(
                SymbolicCandidate(
                    f"move_non_background_bg{background}_dr{delta_row}_dc{delta_col}+color_map",
                    predicted,
                    (
                        f"Background color: {background}.",
                        f"Rule: move every non-background cell by row offset {delta_row} and column offset {delta_col}.",
                        color_map_trace(color_map),
                        "Action: move and recolor the test input on the same canvas.",
                    ),
                    (
                        "program:",
                        (
                            f"  grid = move_non_background(test_input, background={background}, "
                            f"delta_row={delta_row}, delta_col={delta_col})"
                        ),
                        f"  grid = recolor(grid, {color_map_program_literal(color_map)})",
                        "  return grid",
                    ),
                )
            )

    crop_transform_recolor = learn_crop_transform_and_color_map([pair.input for pair in train_pairs], outputs)
    if crop_transform_recolor is not None:
        crop_background, transform, color_map = crop_transform_recolor
        cropped_test = crop_non_background(example.test_input, crop_background)
        if cropped_test is not None:
            transformed_test = apply_geometry_transform(cropped_test, transform)
            predicted = apply_color_map(transformed_test, color_map)
            candidates.append(
                SymbolicCandidate(
                    f"crop_non_background_bg{crop_background}+{transform}+color_map",
                    predicted,
                    (
                        f"Background color: {crop_background}.",
                        "Rule: crop the minimal bounding box around non-background cells.",
                        f"Geometry transform after crop: {transform}.",
                        color_map_trace(color_map),
                        "Action: crop, transform, and recolor the test input.",
                    ),
                    (
                        "program:",
                        f"  grid = crop_non_background(test_input, background={crop_background})",
                        f"  grid = transform(grid, {transform!r})",
                        f"  grid = recolor(grid, {color_map_program_literal(color_map)})",
                        "  return grid",
                    ),
                )
            )

    return dedupe_candidates(candidates)


def exact_symbolic_candidate(example: ArcAgiExample) -> SymbolicCandidate | None:
    if example.test_output is None:
        return None
    for candidate in symbolic_candidates(example):
        if candidate.grid == example.test_output:
            return candidate
    return None


def format_symbolic_trace(candidate: SymbolicCandidate) -> str:
    lines = ["<think>", *candidate.trace, "</think>"]
    return "\n".join(lines) + "\n"


def format_symbolic_program_trace(candidate: SymbolicCandidate) -> str:
    lines = ["<think>", *(candidate.program or candidate.trace), "</think>"]
    return "\n".join(lines) + "\n"
