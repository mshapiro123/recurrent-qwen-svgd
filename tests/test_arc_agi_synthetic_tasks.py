from __future__ import annotations

import json

from eval.analyze_arc_agi_symbolic import analyze_examples
from eval.arc_agi_utils import load_arc_agi_examples
from training.generate_arc_agi_synthetic_tasks import generate_tasks


def test_generate_tasks_are_symbolically_covered(tmp_path) -> None:
    tasks = generate_tasks(
        num_tasks=20,
        seed=123,
        train_examples=3,
        test_examples=1,
        min_size=2,
        max_size=4,
        modes=["geometry_color", "constant_output", "crop_non_background", "crop_recolor", "crop_transform_recolor"],
    )
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    examples = load_arc_agi_examples(path)
    summary = analyze_examples(examples)["summary"]
    assert summary["examples_with_targets"] == 20
    assert summary["exact_symbolic"] == 20


def test_generated_task_file_loads(tmp_path) -> None:
    tasks = generate_tasks(
        num_tasks=3,
        seed=7,
        train_examples=3,
        test_examples=1,
        min_size=2,
        max_size=3,
        modes=["geometry_color"],
    )
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    examples = load_arc_agi_examples(path)
    assert len(examples) == 3
    assert analyze_examples(examples)["summary"]["exact_symbolic"] == 3


def test_generate_crop_non_background_tasks_are_symbolically_covered(tmp_path) -> None:
    tasks = generate_tasks(
        num_tasks=5,
        seed=99,
        train_examples=3,
        test_examples=1,
        min_size=2,
        max_size=5,
        modes=["crop_non_background"],
    )
    path = tmp_path / "synthetic_crop.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    examples = load_arc_agi_examples(path)
    summary = analyze_examples(examples)["summary"]
    assert summary["examples_with_targets"] == 5
    assert summary["exact_symbolic"] == 5
    assert any(source.startswith("crop_non_background_bg") for source in summary["exact_by_source"])


def test_generate_crop_recolor_tasks_are_symbolically_covered(tmp_path) -> None:
    tasks = generate_tasks(
        num_tasks=5,
        seed=100,
        train_examples=3,
        test_examples=1,
        min_size=2,
        max_size=5,
        modes=["crop_recolor"],
    )
    path = tmp_path / "synthetic_crop_recolor.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    examples = load_arc_agi_examples(path)
    summary = analyze_examples(examples)["summary"]
    assert summary["examples_with_targets"] == 5
    assert summary["exact_symbolic"] == 5
    assert any(source.startswith("crop_non_background_bg") and "+color_map" in source for source in summary["exact_by_source"])


def test_generate_crop_transform_recolor_tasks_are_symbolically_covered(tmp_path) -> None:
    tasks = generate_tasks(
        num_tasks=5,
        seed=101,
        train_examples=3,
        test_examples=1,
        min_size=2,
        max_size=5,
        modes=["crop_transform_recolor"],
    )
    path = tmp_path / "synthetic_crop_transform_recolor.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    examples = load_arc_agi_examples(path)
    summary = analyze_examples(examples)["summary"]
    assert summary["examples_with_targets"] == 5
    assert summary["exact_symbolic"] == 5
    assert any(
        source.startswith("crop_non_background_bg") and "+color_map" in source and "+identity+" not in source
        for source in summary["exact_by_source"]
    )
