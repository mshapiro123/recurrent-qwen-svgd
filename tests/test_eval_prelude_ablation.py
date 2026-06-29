from __future__ import annotations

import torch

from eval.eval_prelude_ablation import aggregate_records, logit_comparison_metrics, parse_loop_counts, prelude_variant


def test_parse_loop_counts_rejects_out_of_range() -> None:
    assert parse_loop_counts("4,1,2,2", max_loops=4) == [1, 2, 4]

    try:
        parse_loop_counts("1,5", max_loops=4)
    except ValueError as exc:
        assert "within" in str(exc)
    else:
        raise AssertionError("expected out-of-range loop count to fail")


def test_prelude_variant_zero_and_shuffle_preserve_shape() -> None:
    entry = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])

    zero = prelude_variant(entry, mask, "zero")
    shuffled = prelude_variant(entry, mask, "shuffled")

    assert zero.shape == entry.shape
    assert shuffled.shape == entry.shape
    assert torch.count_nonzero(zero) == 0
    assert torch.allclose(shuffled[0, 0], entry[0, 2])
    assert torch.allclose(shuffled[0, 1], entry[0, 0])
    assert torch.allclose(shuffled[0, 2], entry[0, 1])
    assert torch.allclose(shuffled[0, 3], entry[0, 3])
    assert torch.allclose(shuffled[1], entry[1])


def test_logit_comparison_metrics_detects_top1_change() -> None:
    normal = torch.tensor([[0.0, 2.0, 1.0]])
    variant = torch.tensor([[3.0, 1.0, 0.0]])

    metrics = logit_comparison_metrics(normal, variant)

    assert metrics["normal_argmax"] == 1
    assert metrics["variant_argmax"] == 0
    assert metrics["top1_changed"] is True
    assert metrics["logit_max_abs_delta"] == 3.0


def test_aggregate_records_by_loop_and_variant() -> None:
    rows = [
        {"loop": 1, "variant": "zero", "top1_changed": False, "logit_mean_abs_delta": 0.0, "logit_max_abs_delta": 0.0},
        {"loop": 2, "variant": "zero", "top1_changed": True, "logit_mean_abs_delta": 2.0, "logit_max_abs_delta": 5.0},
        {"loop": 2, "variant": "zero", "top1_changed": False, "logit_mean_abs_delta": 4.0, "logit_max_abs_delta": 3.0},
    ]

    aggregate = aggregate_records(rows)

    assert aggregate["1"]["zero"]["top1_changed_fraction"] == 0.0
    assert aggregate["2"]["zero"]["top1_changed_fraction"] == 0.5
    assert aggregate["2"]["zero"]["logit_mean_abs_delta"] == 3.0
    assert aggregate["2"]["zero"]["logit_max_abs_delta"] == 5.0
