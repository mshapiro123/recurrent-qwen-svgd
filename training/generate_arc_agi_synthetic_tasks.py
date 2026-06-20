"""Generate synthetic ARC-AGI-format tasks covered by the symbolic trace solver."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.arc_agi_utils import GEOMETRY_TRANSFORMS, Grid, apply_geometry_transform  # noqa: E402
from eval.arc_agi_symbolic import apply_color_map, crop_non_background, frame_non_background, move_non_background  # noqa: E402


SHAPE_CHANGING_TRANSFORMS = ("rot90", "rot270", "transpose", "anti_transpose")


def random_palette(rng: random.Random, min_colors: int = 2, max_colors: int = 5) -> list[int]:
    count = rng.randint(min_colors, max_colors)
    return rng.sample(range(10), count)


def random_grid(
    rng: random.Random,
    height: int,
    width: int,
    palette: list[int],
    *,
    ensure_palette: bool = False,
) -> Grid:
    grid = [[rng.choice(palette) for _ in range(width)] for _ in range(height)]
    if ensure_palette:
        cells = [(row, col) for row in range(height) for col in range(width)]
        rng.shuffle(cells)
        for color, (row, col) in zip(palette, cells):
            grid[row][col] = color
    return grid


def random_color_map(rng: random.Random, palette: list[int]) -> dict[int, int]:
    targets = rng.sample(range(10), len(palette))
    return dict(zip(palette, targets))


def random_non_identity_color_map(rng: random.Random, palette: list[int]) -> dict[int, int]:
    for _ in range(100):
        color_map = random_color_map(rng, palette)
        if any(source != target for source, target in color_map.items()):
            return color_map
    source = palette[0]
    target = (source + 1) % 10
    return {**{color: color for color in palette}, source: target}


def random_object_grid(
    rng: random.Random,
    *,
    min_size: int,
    max_size: int,
    background: int,
    object_palette: list[int],
    ensure_object_palette: bool = False,
    object_shape: tuple[int, int] | None = None,
) -> Grid:
    height = rng.randint(max(min_size + 1, 3), max_size)
    width = rng.randint(max(min_size + 1, 3), max_size)
    if object_shape is not None:
        obj_h, obj_w = object_shape
        height = max(height, obj_h + 1)
        width = max(width, obj_w + 1)
    grid = [[background for _ in range(width)] for _ in range(height)]
    if object_shape is None:
        min_area = len(object_palette) if ensure_object_palette else 1
        for _ in range(100):
            obj_h = rng.randint(1, max(1, height - 1))
            obj_w = rng.randint(1, max(1, width - 1))
            if obj_h * obj_w >= min_area:
                break
    row_start = rng.randint(0, height - obj_h)
    col_start = rng.randint(0, width - obj_w)
    object_cells: list[tuple[int, int]] = []
    for row in range(row_start, row_start + obj_h):
        for col in range(col_start, col_start + obj_w):
            object_cells.append((row, col))
            grid[row][col] = rng.choice(object_palette)
    if ensure_object_palette:
        rng.shuffle(object_cells)
        for color, (row, col) in zip(object_palette, object_cells):
            grid[row][col] = color
    return grid


def transform_grid(grid: Grid, transform: str, color_map: dict[int, int]) -> Grid:
    return apply_color_map(apply_geometry_transform(grid, transform), color_map)


def can_move_non_background(grid: Grid, background: int, delta_row: int, delta_col: int) -> bool:
    return move_non_background(grid, background, delta_row, delta_col) is not None


def generate_geometry_color_task(
    rng: random.Random,
    task_id: str,
    *,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    palette = random_palette(rng)
    color_map = random_color_map(rng, palette)
    transform = rng.choice(GEOMETRY_TRANSFORMS)

    def make_pair(idx: int, *, is_train: bool) -> dict[str, Grid]:
        height = rng.randint(max(min_size, 2), max_size)
        width = rng.randint(max(min_size, 2), max_size)
        grid = random_grid(rng, height, width, palette, ensure_palette=is_train and idx == 0)
        return {"input": grid, "output": transform_grid(grid, transform, color_map)}

    return {
        "train": [make_pair(idx, is_train=True) for idx in range(train_examples)],
        "test": [make_pair(idx, is_train=False) for idx in range(test_examples)],
    }


def generate_constant_output_task(
    rng: random.Random,
    task_id: str,
    *,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    del task_id
    input_palette = random_palette(rng)
    output_palette = random_palette(rng)
    out_h = rng.randint(min_size, max_size)
    out_w = rng.randint(min_size, max_size)
    output = random_grid(rng, out_h, out_w, output_palette)

    def make_pair() -> dict[str, Grid]:
        height = rng.randint(min_size, max_size)
        width = rng.randint(min_size, max_size)
        return {"input": random_grid(rng, height, width, input_palette), "output": [row[:] for row in output]}

    return {
        "train": [make_pair() for _ in range(train_examples)],
        "test": [make_pair() for _ in range(test_examples)],
    }


def generate_crop_non_background_task(
    rng: random.Random,
    task_id: str,
    *,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    del task_id
    background = rng.randrange(10)
    object_palette = rng.sample([color for color in range(10) if color != background], rng.randint(1, 4))

    def make_pair() -> dict[str, Grid]:
        grid = random_object_grid(
            rng,
            min_size=min_size,
            max_size=max(max_size, min_size + 1),
            background=background,
            object_palette=object_palette,
        )
        output = crop_non_background(grid, background)
        if output is None:
            raise RuntimeError("generated crop task without foreground")
        return {"input": grid, "output": output}

    return {
        "train": [make_pair() for _ in range(train_examples)],
        "test": [make_pair() for _ in range(test_examples)],
    }


def generate_crop_recolor_task(
    rng: random.Random,
    task_id: str,
    *,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    del task_id
    background = rng.randrange(10)
    object_palette = rng.sample([color for color in range(10) if color != background], rng.randint(1, 4))
    color_map = random_non_identity_color_map(rng, object_palette)

    def make_pair(idx: int, *, is_train: bool) -> dict[str, Grid]:
        grid = random_object_grid(
            rng,
            min_size=min_size,
            max_size=max(max_size, min_size + 1),
            background=background,
            object_palette=object_palette,
            ensure_object_palette=is_train and idx == 0,
        )
        cropped = crop_non_background(grid, background)
        if cropped is None:
            raise RuntimeError("generated crop-recolor task without foreground")
        return {"input": grid, "output": apply_color_map(cropped, color_map)}

    return {
        "train": [make_pair(idx, is_train=True) for idx in range(train_examples)],
        "test": [make_pair(idx, is_train=False) for idx in range(test_examples)],
    }


def generate_crop_transform_recolor_task(
    rng: random.Random,
    task_id: str,
    *,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    del task_id
    background = rng.randrange(10)
    object_palette = rng.sample([color for color in range(10) if color != background], rng.randint(1, 4))
    color_map = random_non_identity_color_map(rng, object_palette)
    transform = rng.choice(SHAPE_CHANGING_TRANSFORMS)
    effective_max_size = max(max_size, min_size + 2, 4)
    shape_limit = max(2, effective_max_size - 1)

    def object_shape(idx: int) -> tuple[int, int]:
        height = rng.randint(1, shape_limit)
        width = rng.randint(1, shape_limit)
        if height == width:
            width = min(shape_limit, width + 1) if width < shape_limit else max(1, width - 1)
        if idx == 0:
            while height * width < len(object_palette):
                width += 1
                if width > shape_limit:
                    height += 1
                    width = 1
        return height, width

    def make_pair(idx: int, *, is_train: bool) -> dict[str, Grid]:
        grid = random_object_grid(
            rng,
            min_size=min_size,
            max_size=effective_max_size,
            background=background,
            object_palette=object_palette,
            ensure_object_palette=is_train and idx == 0,
            object_shape=object_shape(idx),
        )
        cropped = crop_non_background(grid, background)
        if cropped is None:
            raise RuntimeError("generated crop-transform-recolor task without foreground")
        transformed = apply_geometry_transform(cropped, transform)
        return {"input": grid, "output": apply_color_map(transformed, color_map)}

    return {
        "train": [make_pair(idx, is_train=True) for idx in range(train_examples)],
        "test": [make_pair(idx, is_train=False) for idx in range(test_examples)],
    }


def generate_move_recolor_task(
    rng: random.Random,
    task_id: str,
    *,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    del task_id
    background = rng.randrange(10)
    object_palette = rng.sample([color for color in range(10) if color != background], rng.randint(1, 4))
    color_map = random_non_identity_color_map(rng, [background, *object_palette])
    delta_choices = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if dr or dc]
    delta_row, delta_col = rng.choice(delta_choices)
    effective_max_size = max(max_size, min_size + 3, 5)

    def random_movable_grid(*, ensure_object_palette: bool) -> Grid:
        for _ in range(200):
            height = rng.randint(max(min_size + 2, 4), effective_max_size)
            width = rng.randint(max(min_size + 2, 4), effective_max_size)
            max_obj_h = height - abs(delta_row)
            max_obj_w = width - abs(delta_col)
            if max_obj_h < 1 or max_obj_w < 1:
                continue
            obj_h = rng.randint(1, max_obj_h)
            obj_w = rng.randint(1, max_obj_w)
            if ensure_object_palette and obj_h * obj_w < len(object_palette):
                continue

            row_low = max(0, -delta_row)
            row_high = min(height - obj_h, height - obj_h - delta_row)
            col_low = max(0, -delta_col)
            col_high = min(width - obj_w, width - obj_w - delta_col)
            if row_low > row_high or col_low > col_high:
                continue

            row_start = rng.randint(row_low, row_high)
            col_start = rng.randint(col_low, col_high)
            grid = [[background for _ in range(width)] for _ in range(height)]
            object_cells: list[tuple[int, int]] = []
            for row in range(row_start, row_start + obj_h):
                for col in range(col_start, col_start + obj_w):
                    object_cells.append((row, col))
                    grid[row][col] = rng.choice(object_palette)
            if ensure_object_palette:
                rng.shuffle(object_cells)
                for color, (row, col) in zip(object_palette, object_cells):
                    grid[row][col] = color
            if can_move_non_background(grid, background, delta_row, delta_col):
                return grid
        raise RuntimeError("could not generate movable object grid")

    def make_pair(idx: int, *, is_train: bool) -> dict[str, Grid]:
        grid = random_movable_grid(ensure_object_palette=is_train and idx == 0)
        moved = move_non_background(grid, background, delta_row, delta_col)
        if moved is None:
            raise RuntimeError("generated immovable object task")
        return {"input": grid, "output": apply_color_map(moved, color_map)}

    return {
        "train": [make_pair(idx, is_train=True) for idx in range(train_examples)],
        "test": [make_pair(idx, is_train=False) for idx in range(test_examples)],
    }


def generate_frame_object_task(
    rng: random.Random,
    task_id: str,
    *,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
) -> dict[str, Any]:
    del task_id
    background = rng.randrange(10)
    object_palette = rng.sample([color for color in range(10) if color != background], rng.randint(1, 4))
    frame_candidates = [color for color in range(10) if color != background and color not in object_palette]
    frame_color = rng.choice(frame_candidates or [color for color in range(10) if color != background])
    effective_max_size = max(max_size, min_size + 3, 5)

    def make_pair(idx: int, *, is_train: bool) -> dict[str, Grid]:
        for _ in range(200):
            height = rng.randint(max(min_size + 3, 5), effective_max_size)
            width = rng.randint(max(min_size + 3, 5), effective_max_size)
            obj_h = rng.randint(1, max(1, height - 2))
            obj_w = rng.randint(1, max(1, width - 2))
            if is_train and idx == 0 and obj_h * obj_w < len(object_palette):
                continue
            row_start = rng.randint(1, height - obj_h - 1)
            col_start = rng.randint(1, width - obj_w - 1)
            grid = [[background for _ in range(width)] for _ in range(height)]
            object_cells: list[tuple[int, int]] = []
            for row in range(row_start, row_start + obj_h):
                for col in range(col_start, col_start + obj_w):
                    object_cells.append((row, col))
                    grid[row][col] = rng.choice(object_palette)
            if is_train and idx == 0:
                rng.shuffle(object_cells)
                for color, (row, col) in zip(object_palette, object_cells):
                    grid[row][col] = color
            framed = frame_non_background(grid, background, frame_color)
            if framed is not None:
                return {"input": grid, "output": framed}
        raise RuntimeError("could not generate frame object task")

    return {
        "train": [make_pair(idx, is_train=True) for idx in range(train_examples)],
        "test": [make_pair(idx, is_train=False) for idx in range(test_examples)],
    }


def generate_tasks(
    *,
    num_tasks: int,
    seed: int,
    train_examples: int,
    test_examples: int,
    min_size: int,
    max_size: int,
    modes: list[str],
) -> dict[str, Any]:
    rng = random.Random(seed)
    tasks: dict[str, Any] = {}
    for idx in range(num_tasks):
        mode = rng.choice(modes)
        task_id = f"synthetic_{mode}_{idx:06d}"
        if mode == "geometry_color":
            tasks[task_id] = generate_geometry_color_task(
                rng,
                task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                min_size=min_size,
                max_size=max_size,
            )
        elif mode == "constant_output":
            tasks[task_id] = generate_constant_output_task(
                rng,
                task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                min_size=min_size,
                max_size=max_size,
            )
        elif mode == "crop_non_background":
            tasks[task_id] = generate_crop_non_background_task(
                rng,
                task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                min_size=min_size,
                max_size=max_size,
            )
        elif mode == "crop_recolor":
            tasks[task_id] = generate_crop_recolor_task(
                rng,
                task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                min_size=min_size,
                max_size=max_size,
            )
        elif mode == "crop_transform_recolor":
            tasks[task_id] = generate_crop_transform_recolor_task(
                rng,
                task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                min_size=min_size,
                max_size=max_size,
            )
        elif mode == "move_recolor":
            tasks[task_id] = generate_move_recolor_task(
                rng,
                task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                min_size=min_size,
                max_size=max_size,
            )
        elif mode == "frame_object":
            tasks[task_id] = generate_frame_object_task(
                rng,
                task_id,
                train_examples=train_examples,
                test_examples=test_examples,
                min_size=min_size,
                max_size=max_size,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return tasks


def parse_modes(value: str) -> list[str]:
    if value == "all":
        return [
            "geometry_color",
            "constant_output",
            "crop_non_background",
            "crop_recolor",
            "crop_transform_recolor",
            "move_recolor",
            "frame_object",
        ]
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(modes) - {
        "geometry_color",
        "constant_output",
        "crop_non_background",
        "crop_recolor",
        "crop_transform_recolor",
        "move_recolor",
        "frame_object",
    }
    if unknown:
        raise ValueError(f"Unknown synthetic modes: {sorted(unknown)}")
    return modes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train_examples", type=int, default=3)
    parser.add_argument("--test_examples", type=int, default=1)
    parser.add_argument("--min_size", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=6)
    parser.add_argument("--modes", default="all")
    args = parser.parse_args()

    if args.num_tasks < 1:
        raise SystemExit("--num_tasks must be positive")
    if args.train_examples < 2:
        raise SystemExit("--train_examples must be at least 2 for leave-one-out rows")
    if args.test_examples < 1:
        raise SystemExit("--test_examples must be positive")
    if not 1 <= args.min_size <= args.max_size <= 30:
        raise SystemExit("Expected 1 <= min_size <= max_size <= 30")

    tasks = generate_tasks(
        num_tasks=args.num_tasks,
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        min_size=args.min_size,
        max_size=args.max_size,
        modes=parse_modes(args.modes),
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, separators=(",", ":")), encoding="utf-8")
    print(f"synthetic_tasks={len(tasks)}")
    print(f"output_json={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
