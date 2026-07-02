import json

import pytest
import torch

from eval.eval_gradient_path_audit import (
    CoherenceAccumulator,
    SignatureThresholds,
    interpret_gradient_signature,
    multiplier_consumption_check,
    numeric_summary,
    select_audit_rows,
    static_source_audit,
    target_validity_summary,
)


def row(loop, active, bridge, recurrent=1e-4, coda=1e-4):
    return {
        "loop": loop,
        "active_label_tokens": active,
        "bridge_prelude_weight_grad_rms": bridge,
        "bridge_state_weight_grad_rms": 0.0,
        "bridge_prelude_norm_weight_grad_rms": 0.0,
        "bridge_prelude_norm_bias_grad_rms": 0.0,
        "recurrent_block_grad_rms": recurrent,
        "coda_grad_rms": coda,
    }


def fd(loop, delta):
    return {"loop": loop, "abs_delta": delta}


def test_interpret_gradient_signature_marks_connected_graph():
    out = interpret_gradient_signature(
        [
            row(1, 1, 0.0),
            row(2, 1, 2e-6),
            row(3, 1, 5e-6),
        ],
        [fd(1, 0.0), fd(2, 1e-4), fd(3, 2e-4)],
        thresholds=SignatureThresholds(grad_tol=1e-12, fd_tol=1e-6),
    )

    assert out["status"] == "graph_connected"
    assert out["any_bridge_autograd"] is True
    assert out["deep_loops_analyzed"] == 2


def test_interpret_gradient_signature_flags_autograd_cut():
    out = interpret_gradient_signature(
        [
            row(1, 1, 0.0),
            row(2, 1, 0.0),
            row(3, 1, 0.0),
        ],
        [fd(1, 0.0), fd(2, 1e-4), fd(3, 2e-4)],
        thresholds=SignatureThresholds(grad_tol=1e-12, fd_tol=1e-6),
    )

    assert out["status"] == "autograd_cut_suspected"
    assert "finite_difference_dependence_with_zero_bridge_autograd" in out["issues"]


def test_interpret_gradient_signature_flags_structural_independence():
    out = interpret_gradient_signature(
        [
            row(1, 1, 0.0),
            row(2, 1, 0.0),
            row(3, 1, 0.0),
        ],
        [fd(1, 0.0), fd(2, 0.0), fd(3, 0.0)],
        thresholds=SignatureThresholds(grad_tol=1e-12, fd_tol=1e-6),
    )

    assert out["status"] == "structural_independence_or_decode_bypass_suspected"
    assert "deep_losses_do_not_functionally_depend_on_bridge_prelude" in out["issues"]


def test_interpret_gradient_signature_ignores_loop_one_bridge_zero():
    out = interpret_gradient_signature(
        [
            row(1, 1, 0.0),
            row(2, 1, 1e-5),
        ],
        [fd(1, 0.0), fd(2, 1e-4)],
        thresholds=SignatureThresholds(grad_tol=1e-12, fd_tol=1e-6),
    )

    assert out["status"] == "graph_connected"
    assert out["deep_loops_analyzed"] == 1


def test_interpret_gradient_signature_requires_active_deep_labels():
    out = interpret_gradient_signature(
        [row(1, 1, 0.0), row(2, 0, 0.0)],
        [fd(1, 0.0), fd(2, 0.0)],
    )

    assert out["status"] == "no_active_deep_loop_loss"


def test_static_source_audit_sees_bridge_and_per_loop_decode_path():
    out = static_source_audit()

    assert out["bridge_call_in_loop_body"] is True
    assert out["loop_logits_after_coda"] is True
    assert out["per_loop_labels_path"] is True
    assert out["loop_body_contains_data_access"] is False


def test_select_audit_rows_scans_past_depth_one_prefix(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = [
        {"id": f"d1_{idx}", "depth": 1, "loop_completions": [" A"]}
        for idx in range(300)
    ]
    rows.append({"id": "d2", "depth": 2, "loop_completions": [" A", " B"]})
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    out = select_audit_rows(
        source,
        tmp_path / "audit.jsonl",
        max_loops=2,
        max_scan_rows=512,
        row_id=None,
        min_active_loop_labels=2,
    )

    assert out["selected_id"] == "d2"
    assert out["selected_depth"] == 2


def test_select_audit_rows_can_require_all_four_loop_labels(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = [
        {"id": "d2", "depth": 2, "loop_completions": [" A", " B"]},
        {"id": "d4", "depth": 4, "loop_completions": [" A", " B", " C", " D"]},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    out = select_audit_rows(
        source,
        tmp_path / "audit.jsonl",
        max_loops=4,
        max_scan_rows=10,
        row_id=None,
        min_active_loop_labels=4,
    )

    assert out["selected_id"] == "d4"
    assert out["selected_depth"] == 4
    assert out["min_active_loop_labels"] == 4


def test_select_audit_rows_balances_depths_with_auto_active_requirement(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = []
    for depth in (1, 2, 3, 4):
        for idx in range(3):
            rows.append(
                {
                    "id": f"d{depth}_{idx}",
                    "depth": depth,
                    "loop_completions": [" A", " B", " C", " D"][:depth],
                }
            )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    out = select_audit_rows(
        source,
        tmp_path / "audit.jsonl",
        max_loops=4,
        max_scan_rows=100,
        row_id=None,
        min_active_loop_labels="auto",
        num_rows=8,
        depths=[1, 2, 3, 4],
    )

    assert out["selected_rows"] == 8
    assert out["depth_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2}


def test_target_validity_summary_accepts_chain_labels_matching_orbit():
    rows = [
        {
            "id": "ok",
            "depth": 2,
            "choices": {"A": "9", "C": "8"},
            "orbit": ["13", "9", "8"],
            "loop_completions": [" A", " C"],
            "chain_answer_by_loop": {"1": "A", "2": "C"},
        }
    ]

    out = target_validity_summary(rows, max_loops=2)

    assert out["checked_loop_targets"] == 2
    assert out["invalid_loop_targets"] == 0


def test_numeric_summary_reports_zero_fraction_and_quantiles():
    out = numeric_summary([0.0, 0.0, 2.0, 4.0])

    assert out["count"] == 4
    assert out["zero_fraction"] == 0.5
    assert out["median"] == pytest.approx(1.0)


def test_coherence_accumulator_distinguishes_agreement_from_cancellation():
    aligned = CoherenceAccumulator()
    cancelled = CoherenceAccumulator()
    for _ in range(4):
        aligned.add(2, "bridge_prelude", torch.tensor([1.0, 0.0]))
    for value in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        cancelled.add(2, "bridge_prelude", torch.tensor(value))

    aligned_summary = aligned.summary()["by_loop"]["2"]["bridge_prelude"]
    cancelled_summary = cancelled.summary()["by_loop"]["2"]["bridge_prelude"]

    assert aligned_summary["coherence"] == pytest.approx(1.0)
    assert cancelled_summary["coherence"] == pytest.approx(0.0)
    assert cancelled_summary["random_cancellation_floor"] == pytest.approx(0.5)


def test_multiplier_consumption_check_flags_gradient_scale_inert_risk():
    out = multiplier_consumption_check(
        {
            "optimizer": "adamw",
            "bridge_prelude_grad_multiplier": 8.0,
        }
    )

    assert out["implementation"] == "slice_gradient_scaled_before_optimizer_step"
    assert out["inert_under_adamw_or_muon_risk"] is True
    assert "param group" in out["reason"]
