from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from models.ablation_lm.accounting import composition_receipt
from models.ablation_lm.bicameral_combiner import PerBandUnitCircleCombiner
from models.ablation_lm.bicameral_core import BicameralTransformerBlock
from models.ablation_lm.bicameral_recurrent import expected_bicameral_visit_schedules
from models.ablation_lm.config import AblationLMConfig
from models.ablation_lm.layers import TransformerBlock
from models.ablation_lm.model import AblationLM


def _config(*, kv_policy: str = "live", steps: int = 1, scratch: bool = True) -> AblationLMConfig:
    return AblationLMConfig(
        vocab_size=32,
        d_model=64,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        n_prelude_layers=1,
        n_core_blocks=2,
        n_coda_layers=1,
        use_recurrence=True,
        recurrent_steps=steps,
        max_recurrent_steps=8,
        use_bicameral_core=True,
        kv_policy=kv_policy,
        max_sequence_length=16,
        use_scratch=scratch,
        scratch_width=8,
    )


def _tokens() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    document_ids = torch.tensor([[10, 10, 11, 11], [20, 20, 20, -1]])
    return input_ids, attention_mask, document_ids


def _copy_bicameral_consensus_into_dense(
    bicameral: AblationLM,
    dense: AblationLM,
) -> None:
    """Realize the structural-OFF dense anchor without a second initializer.

    ``SwapLinear.mu`` is the physical dense map when both disagreement factors
    are zero.  Copying it into the ordinary Transformer makes the comparison
    about graph equivalence rather than two unrelated random initializations.
    """

    with torch.no_grad():
        dense.token_embedding.weight.copy_(bicameral.token_embedding.weight)
        dense.final_norm.weight.copy_(bicameral.final_norm.weight)
        for source, target in zip(
            bicameral.prelude_blocks,
            dense.prelude_blocks,
            strict=True,
        ):
            target.load_state_dict(source.state_dict())
        for source, target in zip(
            bicameral.coda_blocks,
            dense.coda_blocks,
            strict=True,
        ):
            target.load_state_dict(source.state_dict())
        for source, target in zip(
            bicameral.core_blocks,
            dense.core_blocks,
            strict=True,
        ):
            assert isinstance(source, BicameralTransformerBlock)
            assert isinstance(target, TransformerBlock)
            for paired in source.swap_linears:
                paired.dU.zero_()
                paired.dV.zero_()
            target.attention_norm.weight.copy_(source.attention_norm.weight)
            target.ffn_norm.weight.copy_(source.ffn_norm.weight)
            target.attention.query_norm.weight.copy_(source.query_norm.weight)
            target.attention.key_norm.weight.copy_(source.key_norm.weight)
            target.attention.q_proj.weight.copy_(source.q_proj.mu)
            target.attention.k_proj.weight.copy_(source.k_proj.weight)
            target.attention.v_proj.weight.copy_(source.v_proj.weight)
            target.attention.output_proj.weight.copy_(source.o_proj.mu)
            target.feed_forward.gate_proj.weight.copy_(source.gate_proj.mu)
            target.feed_forward.up_proj.weight.copy_(source.up_proj.mu)
            target.feed_forward.down_proj.weight.copy_(source.down_proj.mu)
        assert dense.loop_embedding is not None
        dense.loop_embedding.weight.zero_()


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_a7_integrated_structural_off_matches_ordinary_dense_anchor(
    steps: int,
) -> None:
    """A7/T1: zero disagreement and theta recover the ordinary dense graph."""

    bicameral = AblationLM(
        _config(kv_policy="static", steps=steps, scratch=False)
    ).eval()
    dense = AblationLM(
        replace(
            _config(kv_policy="live", steps=steps, scratch=False),
            use_bicameral_core=False,
            use_static_kv_core=True,
        )
    ).eval()
    _copy_bicameral_consensus_into_dense(bicameral, dense)
    assert bicameral.bicameral_combiner is not None
    assert torch.count_nonzero(bicameral.bicameral_combiner.theta) == 0

    input_ids, attention_mask, document_ids = _tokens()
    with torch.no_grad():
        bicameral_output = bicameral(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            labels=input_ids,
            return_diagnostics=True,
        )
        dense_output = dense(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            labels=input_ids,
            return_diagnostics=True,
        )

    # The S-2 WHT round trip is float32-isometric but not bit-exact; its
    # measured round-trip error is therefore covered by a numerical anchor.
    torch.testing.assert_close(
        bicameral_output.logits,
        dense_output.logits,
        rtol=2e-5,
        atol=3e-6,
    )
    assert bicameral_output.loss is not None
    assert dense_output.loss is not None
    torch.testing.assert_close(
        bicameral_output.loss,
        dense_output.loss,
        rtol=2e-6,
        atol=2e-7,
    )


def test_composition_receipt_rejects_incomplete_or_reordered_executed_schedule() -> None:
    model = AblationLM(
        replace(
            _config(kv_policy="live", steps=2),
            use_lane_carrier=True,
        )
    ).eval()
    input_ids, attention_mask, document_ids = _tokens()
    with torch.no_grad():
        schedule = model(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_diagnostics=True,
        ).diagnostics["visit_schedule"]

    missing_lane = tuple(
        item
        for item in schedule
        if item != "visit[0].block[0].TwoLaneBirkhoffMixer"
    )
    with pytest.raises(ValueError, match="not the exact Step-2 recurrence trace"):
        composition_receipt(
            model,
            requested_visits=2,
            executed_visits=2,
            kv_policy="live",
            kv_cache_multiplier_at_serving=4,
            visit_schedule=missing_lane,
        )

    terminal_first = (schedule[-1], *schedule[:-1])
    with pytest.raises(ValueError, match="not the exact Step-2 recurrence trace"):
        composition_receipt(
            model,
            requested_visits=2,
            executed_visits=2,
            kv_policy="live",
            kv_cache_multiplier_at_serving=4,
            visit_schedule=terminal_first,
        )


def test_composition_receipt_rejects_seam_only_caller_preprojected_trace() -> None:
    model = AblationLM(_config(kv_policy="static", steps=2, scratch=False)).eval()
    seam_candidates = expected_bicameral_visit_schedules(
        recurrent_steps=2,
        unique_core_blocks=2,
        kv_policy="static",
    )
    assert len(seam_candidates) == 2
    caller_preprojected = (
        *seam_candidates[1],
        "terminal.PerBandUnitCircleCombiner",
    )

    with pytest.raises(ValueError, match="not the exact Step-2 recurrence trace"):
        composition_receipt(
            model,
            requested_visits=2,
            executed_visits=2,
            kv_policy="static",
            kv_cache_multiplier_at_serving=1,
            visit_schedule=caller_preprojected,
        )


def test_integrated_k1_live_and_static_are_bit_identical() -> None:
    live = AblationLM(_config(kv_policy="live", steps=1)).eval()
    static = AblationLM(_config(kv_policy="static", steps=1)).eval()
    assert live.state_dict().keys() == static.state_dict().keys()
    assert all(
        torch.equal(live.state_dict()[name], static.state_dict()[name])
        for name in live.state_dict()
    )
    assert all(isinstance(block, BicameralTransformerBlock) for block in live.core_blocks)
    assert isinstance(live.bicameral_combiner, PerBandUnitCircleCombiner)
    assert live.loop_embedding is live.front_hadamard is live.reentry_bridge is None

    input_ids, attention_mask, document_ids = _tokens()
    with torch.no_grad():
        live_output = live(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_diagnostics=True,
        )
        static_output = static(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_diagnostics=True,
        )

    assert torch.equal(live_output.logits, static_output.logits)
    assert live_output.diagnostics["kv_cache_multiplier_at_serving"] == 2
    assert static_output.diagnostics["kv_cache_multiplier_at_serving"] == 1
    assert live_output.diagnostics["visit_schedule"]
    assert static_output.diagnostics["visit_schedule"]


def test_integrated_live_receipt_and_after_each_block_lane_schedule() -> None:
    model = AblationLM(
        replace(_config(kv_policy="live", steps=3), use_lane_carrier=True)
    ).eval()
    assert model.scratch is not None
    input_ids, attention_mask, document_ids = _tokens()

    with (
        torch.no_grad(),
        patch.object(
            model.scratch,
            "step_bicameral",
            wraps=model.scratch.step_bicameral,
        ) as lane_step,
    ):
        output = model(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_diagnostics=True,
        )

    diagnostics = output.diagnostics
    receipt = diagnostics["composition_receipt"]
    assert lane_step.call_count == 6
    assert [call.kwargs["step_index"] for call in lane_step.call_args_list] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert all(call.kwargs["residual_scale"] == 1.0 / 3.0 for call in lane_step.call_args_list)
    assert diagnostics["bicameral_scratch_update_events"] == 6
    assert diagnostics["loop_state_basis"] == "bicameral_consensus_step2"
    assert diagnostics["main_graph_core_kv_projection_events"] == 3
    assert diagnostics["main_graph_core_kv_linear_projection_calls"] == 24
    assert diagnostics["terminal_s2_combiner_executed"] is True
    assert receipt["kv_policy"] == "live"
    assert receipt["kv_cache_multiplier_at_serving"] == 6
    assert receipt["visit_schedule"] == diagnostics["visit_schedule"]
    assert receipt["visit_schedule"][0] == "visit[0].block[0].project_kv.live"
    assert receipt["visit_schedule"][-1] == "terminal.PerBandUnitCircleCombiner"
    assert receipt["visit_schedule"].index("visit[0].block[0].attention") < receipt[
        "visit_schedule"
    ].index("visit[0].block[0].feed_forward")
    assert receipt["visit_schedule"].index("visit[0].block[0].feed_forward") < receipt[
        "visit_schedule"
    ].index("visit[0].block[0].PositionAlignedScratch.step_bicameral")
    assert any("TwoLaneBirkhoffMixer" in item for item in receipt["visit_schedule"])
    assert receipt["coda_decodes_per_step"] == 1
    assert receipt["lstage_sampled_visit"] is None


@pytest.mark.parametrize(
    ("policy", "cache_multiplier", "projection_events", "linear_calls"),
    [
        ("live", 8, 4, 32),
        ("static", 1, 1, 4),
        ("midpoint", 2, 2, 12),
    ],
)
def test_integrated_policy_accounting_is_exact(
    policy: str,
    cache_multiplier: int,
    projection_events: int,
    linear_calls: int,
) -> None:
    model = AblationLM(_config(kv_policy=policy, steps=4, scratch=False)).eval()
    input_ids, attention_mask, document_ids = _tokens()
    with torch.no_grad():
        diagnostics = model(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_diagnostics=True,
        ).diagnostics

    assert diagnostics["kv_policy"] == policy
    assert diagnostics["kv_cache_multiplier_at_serving"] == cache_multiplier
    assert diagnostics["main_graph_core_kv_projection_events"] == projection_events
    assert diagnostics["main_graph_core_kv_linear_projection_calls"] == linear_calls
    assert diagnostics["executed_core_block_passes"] == 8
    expected_first_event = {
        "live": "visit[0].block[0].project_kv.live",
        "static": "setup.block[0].project_kv.static_shared",
        "midpoint": "setup.block[0].project_kv.midpoint_shared",
    }[policy]
    assert diagnostics["visit_schedule"][0] == expected_first_event
    if policy == "midpoint":
        assert "visit[2].block[0].project_kv.midpoint_refresh" in diagnostics[
            "visit_schedule"
        ]


def test_step2_fails_closed_on_not_yet_integrated_separated_state_probes() -> None:
    model = AblationLM(_config()).eval()
    input_ids, attention_mask, document_ids = _tokens()
    with pytest.raises(NotImplementedError, match="Step-7 live-K/V instrument"):
        model(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_diagnostics=True,
            jacobian_probe_iterations=1,
        )
    with pytest.raises(NotImplementedError, match="Step-7 live-K/V instrument"):
        model(
            input_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_diagnostics=True,
            capture_loop_gradients=True,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"use_front_hadamard_experts": True},
        {"use_reentry_bridge": True},
    ],
)
def test_bicameral_path_rejects_retired_legacy_modules(
    updates: dict[str, bool],
) -> None:
    with pytest.raises(ValueError, match="structurally retires"):
        AblationLM(replace(_config(), **updates))


def test_legacy_core_structure_is_unchanged() -> None:
    config = replace(
        _config(scratch=False),
        use_bicameral_core=False,
        kv_policy="live",
    )
    model = AblationLM(config)
    assert all(isinstance(block, TransformerBlock) for block in model.core_blocks)
    assert model.bicameral_combiner is None
    assert model.loop_embedding is not None
