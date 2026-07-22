from __future__ import annotations

import json
import math
from pathlib import Path

from eval.build_paper_one_figure5 import ARM_ORDER, load_and_validate


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/figures/figure5_data.json"


def test_figure5_data_contains_no_placeholders_and_valid_counts() -> None:
    payload = load_and_validate(DATA)

    assert payload["placeholder_data_present"] is False
    assert tuple(payload["arms"]) == ARM_ORDER
    assert all(item["placeholder"] is False for item in payload["arms"].values())
    assert payload["rows_per_depth"] == 128


def test_figure5_counts_match_canonical_phase_a_and_arm_e_receipts() -> None:
    payload = load_and_validate(DATA)
    phase_a = json.loads((ROOT / payload["sources"]["phase_a"]).read_text(encoding="utf-8"))
    dense_audit = json.loads(
        (ROOT / payload["sources"]["phase_a_dense_reader_audit"]).read_text(encoding="utf-8")
    )
    arm_e = json.loads((ROOT / payload["sources"]["adapter_arm_e"]).read_text(encoding="utf-8"))

    for depth in payload["depths"]:
        assert payload["arms"]["A"]["correct"][str(depth)] == phase_a["scoring"]["counts"]["A"][str(depth)]
        assert phase_a["scoring"]["depth_totals"][str(depth)] == 128
    for arm, receipt_key in {"B": "B_step4000", "C": "C_step4000", "D": "D_step4000"}.items():
        for depth in payload["depths"]:
            observed = dense_audit["arms"][receipt_key]["by_depth"][str(depth)]["corrected_correct"]
            assert payload["arms"][arm]["correct"][str(depth)] == observed
    for depth in payload["depths"]:
        assert payload["arms"]["E"]["correct"][str(depth)] == arm_e["final_eval"]["by_depth"][str(depth)]["same_reader_correct"]
        assert arm_e["final_eval"]["by_depth"][str(depth)]["total"] == 128


def test_figure5_latency_matches_complete_registered_model_calls() -> None:
    payload = load_and_validate(DATA)
    latency = json.loads((ROOT / payload["sources"]["wall_clock"]).read_text(encoding="utf-8"))

    for arm in ARM_ORDER:
        for depth in payload["depths"]:
            observed = payload["arms"][arm]["wall_clock_ms"][str(depth)]
            expected = latency["arms"][arm]["by_depth"][str(depth)]["model_total_ms"]["median"]
            assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)


def test_figure5_artifacts_are_declared_without_watermark() -> None:
    for suffix in ("svg", "pdf", "png"):
        assert (ROOT / f"docs/figures/figure5_accuracy_depth_wall_clock.{suffix}").exists()
    svg = (ROOT / "docs/figures/figure5_accuracy_depth_wall_clock.svg").read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in svg.upper()
    assert "Accuracy by composition depth" in svg
