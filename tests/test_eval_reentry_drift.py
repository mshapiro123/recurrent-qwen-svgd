import pytest
import torch

from eval.eval_reentry_drift import (
    aggregate_prompt_records,
    bridge_gradient_liveness,
    masked_token_matrix,
    reentry_adapter_gradient_liveness,
    reentry_adapter_stats,
    rms,
    subspace_overlap,
)
from models.bridge import IdentityGatedBridge
from models.reentry_adapter import ReentryAffineAdapter


def test_rms_respects_attention_mask():
    hidden = torch.tensor(
        [
            [[3.0, 4.0], [100.0, 100.0]],
        ]
    )
    mask = torch.tensor([[1, 0]])

    assert rms(hidden, mask).item() == pytest.approx((12.5) ** 0.5)


def test_masked_token_matrix_filters_padding_tokens():
    hidden = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]])

    out = masked_token_matrix(hidden, mask)

    assert out.tolist() == [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]]


def test_subspace_overlap_identical_spaces_is_one():
    states = torch.randn(12, 6)

    out = subspace_overlap(states, states.clone(), rank=4)

    assert out["rank"] == 4
    assert out["overlap"] == pytest.approx(1.0, abs=1e-5)
    assert out["aligned_dims_cos_ge_0p9"] == 4


def test_bridge_liveness_detects_dead_identity_gate_zero():
    bridge = IdentityGatedBridge(hidden_size=4, gate_init=0.0)
    sample = torch.randn(2, 3, 4)

    out = bridge_gradient_liveness(bridge, sample)

    assert out["gate_grad_abs"] == pytest.approx(0.0)
    assert out["weight_grad_rms"] == pytest.approx(0.0)
    assert out["bias_grad_rms"] == pytest.approx(0.0)


def test_bridge_liveness_detects_live_identity_gate_one():
    bridge = IdentityGatedBridge(hidden_size=4, gate_init=1.0)
    sample = torch.randn(2, 3, 4)

    out = bridge_gradient_liveness(bridge, sample)

    assert out["gate_grad_abs"] == pytest.approx(0.0)
    assert out["weight_grad_rms"] > 0.0
    assert out["bias_grad_rms"] > 0.0


def test_reentry_adapter_stats_are_identity_at_init():
    adapter = ReentryAffineAdapter(hidden_size=4)
    sample = torch.randn(2, 3, 4)

    out = reentry_adapter_stats(adapter, sample)

    assert out["scale_identity_max_abs_diff"] == pytest.approx(0.0)
    assert out["bias_max_abs"] == pytest.approx(0.0)
    assert out["sample_adapter_delta_rms"] == pytest.approx(0.0)
    assert out["sample_state_rms"] > 0.0


def test_reentry_adapter_liveness_reports_scale_and_bias_gradients():
    adapter = ReentryAffineAdapter(hidden_size=4)
    sample = torch.randn(2, 3, 4)

    out = reentry_adapter_gradient_liveness(adapter, sample)

    assert out["loss"] > 0.0
    assert out["scale_grad_rms"] > 0.0
    assert out["bias_grad_rms"] > 0.0


def test_aggregate_prompt_records_summarizes_loop_drift():
    records = [
        {
            "tokens": 2,
            "entry_rms": 1.0,
            "exit_rms": 2.0,
            "exit_over_entry_rms": 2.0,
            "pooled_entry_exit_cosine": 0.5,
            "entry_tokens": torch.randn(2, 4),
            "exit_tokens": torch.randn(2, 4),
            "loop_records": [
                {
                    "loop": 1,
                    "raw_input_rms": 1.0,
                    "input_rms": 1.0,
                    "output_rms": 2.0,
                    "input_over_entry_rms": 1.0,
                    "output_over_entry_rms": 2.0,
                    "output_over_input_rms": 2.0,
                    "bridge_delta_rms": 0.0,
                    "pooled_input_output_cosine": 0.5,
                }
            ],
        },
        {
            "tokens": 2,
            "entry_rms": 3.0,
            "exit_rms": 6.0,
            "exit_over_entry_rms": 2.0,
            "pooled_entry_exit_cosine": 0.25,
            "entry_tokens": torch.randn(2, 4),
            "exit_tokens": torch.randn(2, 4),
            "loop_records": [
                {
                    "loop": 1,
                    "raw_input_rms": 3.0,
                    "input_rms": 3.0,
                    "output_rms": 6.0,
                    "input_over_entry_rms": 1.0,
                    "output_over_entry_rms": 2.0,
                    "output_over_input_rms": 2.0,
                    "bridge_delta_rms": 0.0,
                    "pooled_input_output_cosine": 0.25,
                }
            ],
        },
    ]

    out = aggregate_prompt_records(records, subspace_rank=2)

    assert out["prompts"] == 2
    assert out["tokens"] == 4
    assert out["mean_exit_over_entry_rms"] == pytest.approx(2.0)
    assert out["loop_summary"]["1"]["output_over_input_rms"] == pytest.approx(2.0)
