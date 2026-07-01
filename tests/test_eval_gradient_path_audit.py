import json

import pytest

from eval.eval_gradient_path_audit import (
    SignatureThresholds,
    interpret_gradient_signature,
    select_audit_rows,
    static_source_audit,
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

    out = select_audit_rows(source, tmp_path / "audit.jsonl", max_loops=2, max_scan_rows=512, row_id=None)

    assert out["selected_id"] == "d2"
    assert out["selected_depth"] == 2
