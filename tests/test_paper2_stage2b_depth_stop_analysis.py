from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from analysis.analyze_paper2_stage2b_depth_stop import EXPECTED_LOCK_SHA256, analyze


BATTERY = {
    "arc_challenge": {"rows": 76, "base_correct": 41, "initialization_correct": 41, "current_correct": 36},
    "arc_easy": {"rows": 243, "base_correct": 189, "initialization_correct": 192, "current_correct": 172},
    "gsm8k": {"rows": 369, "base_correct": 107, "initialization_correct": 105, "current_correct": 2},
    "mbpp": {"rows": 67, "base_correct": 28, "initialization_correct": 35, "current_correct": 0},
    "mmlu": {"rows": 244, "base_correct": 115, "initialization_correct": 117, "current_correct": 84},
    "tier1": {"rows": 25, "base_correct": 22, "initialization_correct": 20, "current_correct": 0},
}


def fixture(seed: int) -> dict:
    battery = copy.deepcopy(BATTERY)
    for cell in battery.values():
        cell["delta_vs_base_rows"] = cell["current_correct"] - cell["base_correct"]
        cell["delta_vs_initialization_rows"] = cell["current_correct"] - cell["initialization_correct"]
    return {
        "seed": seed,
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "status": "stopped",
        "stop_reason": "dev1_hard_floor",
        "step": 1000,
        "target_step": 5000,
        "confirm_scored": False,
        "eval_e_scored": False,
        "frozen_digest": f"digest-{seed}",
        "teacher_cache_index_sha256": "teacher-cache",
        "history": [
            {
                "step": 1000,
                "look": 1,
                "stage": "M2",
                "confirm_scored": False,
                "eval_e_scored": False,
                "pass_one_max_abs_difference": 0.0,
                "ema_checkpoint": {"path": f"seed-{seed}.pt", "sha256": f"checkpoint-{seed}"},
                "dev1": {
                    "both_comparators_reported": True,
                    "battery": battery,
                    "safety": {
                        "pass": False,
                        "gsm8k_correct": 2,
                        "gsm8k_floor": 91,
                        "gsm8k_pass": False,
                        "tier1_correct": 0,
                        "tier1_floor": 19,
                        "tier1_pass": False,
                    },
                },
                "dev2": {
                    "rows": 2048,
                    "per_loop_mean_teacher_token_margin": [2.7, 1.0, 0.25, -0.08],
                    "transition_means": {"k1_to_k2": -1.7, "k2_to_k3": -0.75, "k3_to_k4": -0.33},
                },
                "objective_components": {"ce": 1.9, "kl": 1.0, "monotonicity": 0.9},
                "finite_horizon": {
                    "catastrophe": False,
                    "catastrophe_threshold": 100.0,
                    "centered_directional_gains": [1.0, 1.0, 1.0, 1.0],
                    "lambda2_mean": 1.0,
                    "lane_effective_rank_mean": 3.0,
                    "sinkhorn_column_residual_max": 0.0,
                    "sinkhorn_row_residual_max": 0.0,
                },
            }
        ],
        "last_gradient_audit": {
            "pass": True,
            "finite_parameter_tensors": 36,
            "missing_active": [],
            "missing_expected": [],
            "stage": "M2",
        },
    }


def write_fixture(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_stage2b_stop_analysis_builds_receipt_and_figures(tmp_path: Path) -> None:
    inputs = [tmp_path / "seed_0.json", tmp_path / "seed_1.json"]
    for seed, path in enumerate(inputs):
        write_fixture(path, fixture(seed))
    output = tmp_path / "analysis.json"
    figure = tmp_path / "figure"
    result = analyze(inputs, output, figure)
    assert result["registered_verdict"] == "REPLICATED_DEV1_HARD_FLOOR_STOP_AT_STEP_1000"
    assert not result["cross_seed"]["step_5000_adjudication_eligible"]
    assert output.stat().st_size > 1000
    assert figure.with_suffix(".svg").stat().st_size > 1000
    assert figure.with_suffix(".png").stat().st_size > 1000


def test_stage2b_stop_analysis_rejects_sealed_partition_contact(tmp_path: Path) -> None:
    inputs = [tmp_path / "seed_0.json", tmp_path / "seed_1.json"]
    for seed, path in enumerate(inputs):
        payload = fixture(seed)
        if seed == 1:
            payload["confirm_scored"] = True
        write_fixture(path, payload)
    with pytest.raises(AssertionError, match="sealed partition"):
        analyze(inputs, tmp_path / "analysis.json", tmp_path / "figure")
