"""Small symbolic candidate generator for easy ARC-AGI transformations.

This is not intended to be a full ARC solver. It gives the evaluation harness a
stronger non-neural baseline and a hybrid candidate source for transformations
that can be inferred exactly from demonstrations:

- dihedral geometry transforms;
- consistent per-color maps after a geometry transform;
- constant-output tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.arc_agi_utils import ArcAgiExample, GEOMETRY_TRANSFORMS, Grid, apply_geometry_transform


@dataclass(frozen=True)
class SymbolicCandidate:
    name: str
    grid: Grid


def grid_shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def apply_color_map(grid: Grid, color_map: dict[int, int]) -> Grid:
    return [[color_map.get(cell, cell) for cell in row] for row in grid]


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


def all_outputs_equal(outputs: list[Grid]) -> bool:
    return bool(outputs) and all(output == outputs[0] for output in outputs)


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
        candidates.append(SymbolicCandidate("constant_output", [row[:] for row in outputs[0]]))

    for transform in GEOMETRY_TRANSFORMS:
        transformed_inputs = [apply_geometry_transform(pair.input, transform) for pair in train_pairs]
        color_map = learn_color_map(transformed_inputs, outputs)
        if color_map is None:
            continue
        transformed_test = apply_geometry_transform(example.test_input, transform)
        predicted = apply_color_map(transformed_test, color_map)
        label = f"{transform}+color_map" if color_map else transform
        candidates.append(SymbolicCandidate(label, predicted))

    return dedupe_candidates(candidates)
