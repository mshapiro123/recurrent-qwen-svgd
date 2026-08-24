from __future__ import annotations

import json
from pathlib import Path

from models.bicameral import OPERATING_GATE_VALUE, SEQUENTIAL_EXECUTION_SCHEDULE


ROOT = Path(__file__).resolve().parents[1]


def test_w0_lock_is_forward_only_and_binds_schedule() -> None:
    lock = json.loads(
        (ROOT / "training/paper2_bicameral_w0_lock.json").read_text(encoding="utf-8")
    )
    assert lock["status"] == "LOCKED_W0_ONLY"
    assert lock["mark_ratified"] is True
    assert lock["training_authorized"] is False
    assert lock["optimizer_steps_allowed"] == 0
    assert lock["sealed_partitions_authorized"] is False
    assert lock["step1_training_authorized"] is False
    assert lock["step2_training_authorized"] is False
    assert lock["execution"]["schedule"] == SEQUENTIAL_EXECUTION_SCHEDULE
    assert lock["execution"]["batch_concat_prohibited"] is True
    for key in (
        "callosum_gate_a",
        "callosum_gate_b",
        "bank_gate_a",
        "bank_gate_b",
    ):
        assert lock["execution"][key] == OPERATING_GATE_VALUE


def test_w0_cost_probe_declares_pinned_runtime_and_inputs() -> None:
    source = (ROOT / "colab/run_paper2_bicameral_step1_cost_probe.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "NVIDIA A100-SXM4-40GB",
        "SEQUENTIAL_EXECUTION_SCHEDULE",
        "--manifest",
        "--probe-batch",
        "--initializer-seed-0",
        "--initializer-seed-1",
        "actual_t2_contract",
        "operating_point_divergence",
        "manifest byte hash does not match its locked summary",
    ):
        assert marker in source
