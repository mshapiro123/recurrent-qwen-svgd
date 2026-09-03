from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.ablation_lm import AblationLM, AblationLMConfig
from models.ablation_lm.liveness import (
    GradientLivenessBlocked,
    PF1_NOT_MATERIALIZED_INTEGRATIONS,
    gradient_liveness_receipt,
    inverse_k1_k4_liveness,
    parameter_eligibility_matrix,
)
from models.ablation_lm.memory import ReadOnlyLatentMemory


def _active_config() -> AblationLMConfig:
    return AblationLMConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_prelude_layers=1,
        n_core_blocks=2,
        n_coda_layers=1,
        use_recurrence=True,
        recurrent_steps=4,
        max_recurrent_steps=4,
        use_static_kv_core=True,
        use_front_hadamard_experts=True,
        hadamard_experts=4,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        scratch_width=8,
        use_engram=True,
        engram_hashes_per_order=2,
        engram_table_size=31,
        engram_row_dim=2,
        use_long_term_memory=True,
        long_term_memory_slots=16,
        long_term_memory_width=8,
        max_sequence_length=16,
    )


def _active_model() -> AblationLM:
    config = _active_config()
    generator = torch.Generator().manual_seed(9_771)
    memory = ReadOnlyLatentMemory(
        config.d_model,
        keys=torch.randn(
            config.long_term_memory_slots,
            config.long_term_memory_width,
            generator=generator,
        ),
        values=torch.randn(
            config.long_term_memory_slots,
            config.long_term_memory_width,
            generator=generator,
        ),
        provenance_ids=torch.arange(config.long_term_memory_slots),
        layer_scale=config.long_term_memory_layer_scale,
        norm_eps=config.norm_eps,
        initialization_seed=config.initialization_seed,
    )
    return AblationLM(config, long_term_memory=memory)


def _bicameral_config(
    *,
    kv_policy: str = "live",
    recurrent_steps: int = 4,
    use_scratch: bool = False,
) -> AblationLMConfig:
    return AblationLMConfig(
        vocab_size=64,
        d_model=64,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        n_prelude_layers=1,
        n_core_blocks=2,
        n_coda_layers=1,
        use_recurrence=True,
        recurrent_steps=recurrent_steps,
        max_recurrent_steps=4,
        use_bicameral_core=True,
        kv_policy=kv_policy,
        use_scratch=use_scratch,
        scratch_width=8,
        max_sequence_length=16,
    )


def _bicameral_backward_receipt(
    *,
    recurrent_steps: int,
    use_scratch: bool,
):
    model = AblationLM(
        _bicameral_config(
            recurrent_steps=recurrent_steps,
            use_scratch=use_scratch,
        )
    ).train()
    tokens = torch.tensor([[1, 7, 12, 19], [5, 11, 18, 22]])
    output = model(tokens, labels=tokens, recurrent_steps=recurrent_steps)
    assert output.loss is not None
    output.loss.backward()
    return model, gradient_liveness_receipt(
        model,
        recurrent_steps=recurrent_steps,
    )


def _backward_receipt(recurrent_steps: int) -> tuple[AblationLM, object]:
    model = _active_model().train()
    tokens = torch.tensor(
        [
            [1, 7, 12, 19, 23, 31, 42, 55],
            [5, 11, 18, 22, 29, 37, 46, 61],
        ]
    )
    record_ids = torch.tensor([[0] * 8, [1] * 8])
    output = model(
        tokens,
        labels=tokens,
        memory_record_ids=record_ids,
        recurrent_steps=recurrent_steps,
    )
    assert output.loss is not None
    output.loss.backward()
    return model, gradient_liveness_receipt(
        model,
        recurrent_steps=recurrent_steps,
    )


def test_pf1_eligibility_matrix_is_explicit_by_module_k_and_visit() -> None:
    model = _active_model()
    k1 = parameter_eligibility_matrix(model, recurrent_steps=1)
    k4 = parameter_eligibility_matrix(model, recurrent_steps=4)

    k1_reentry = tuple(row for row in k1 if row.module_name == "reentry_bridge")
    assert k1_reentry
    assert {row.visit_index for row in k1_reentry} == {0}
    assert all(not row.eligible and not row.executed for row in k1_reentry)

    k4_reentry = tuple(row for row in k4 if row.module_name == "reentry_bridge")
    assert {row.visit_index for row in k4_reentry} == {0, 1, 2, 3}
    assert all(not row.eligible for row in k4_reentry if row.visit_index == 0)
    assert all(
        row.eligible and row.executed
        for row in k4_reentry
        if row.visit_index in (1, 2, 3)
    )

    k4_loop = tuple(row for row in k4 if row.module_name == "loop_embedding")
    assert {row.visit_index for row in k4_loop} == {0, 1, 2, 3}
    assert all(row.eligible and row.executed for row in k4_loop)


def test_pf1_k1_k4_liveness_and_inverse_gate_cover_materialized_graph() -> None:
    _k1_model, k1 = _backward_receipt(1)
    _k4_model, k4 = _backward_receipt(4)

    k1.require_passed()
    k4.require_passed()
    assert all(item.minimum_gradient_norm > 0.0 for item in k1.module_minimums)
    assert all(item.minimum_gradient_norm > 0.0 for item in k4.module_minimums)
    assert {item.module_name for item in k1.ineligible_parameters} == {
        "reentry_bridge"
    }
    assert not k4.ineligible_parameters
    assert k1.not_materialized_integrations == PF1_NOT_MATERIALIZED_INTEGRATIONS

    inverse = inverse_k1_k4_liveness(k1, k4)
    inverse.require_passed()
    assert inverse.k1_ineligible_parameter_names
    assert inverse.activated_and_live_at_k4 == inverse.k1_ineligible_parameter_names


def test_pf1_liveness_fails_closed_for_missing_and_zero_eligible_gradients() -> None:
    model, _receipt = _backward_receipt(4)
    parameters = dict(model.named_parameters())
    parameters["core_blocks.0.attention.q_proj.weight"].grad = None
    parameters["reentry_bridge.projection.weight"].grad = torch.zeros_like(
        parameters["reentry_bridge.projection.weight"]
    )

    failed = gradient_liveness_receipt(model, recurrent_steps=4)

    assert failed.eligible_missing_gradients == (
        "core_blocks.0.attention.q_proj.weight",
    )
    assert failed.eligible_zero_gradients == (
        "reentry_bridge.projection.weight",
    )
    with pytest.raises(GradientLivenessBlocked, match="eligible gradient liveness failed"):
        failed.require_passed()


def test_pf1_inverse_gate_rejects_a_k1_ineligible_tensor_that_stays_frozen() -> None:
    _k1_model, k1 = _backward_receipt(1)
    k4_model, _k4 = _backward_receipt(4)
    parameters = dict(k4_model.named_parameters())
    parameters["reentry_bridge.layer_scale"].grad = None
    k4_with_frozen_reentry = gradient_liveness_receipt(
        k4_model,
        recurrent_steps=4,
    )

    inverse = inverse_k1_k4_liveness(k1, k4_with_frozen_reentry)

    assert inverse.not_live_at_k4 == ("reentry_bridge.layer_scale",)
    with pytest.raises(GradientLivenessBlocked, match="did not become eligible and live"):
        inverse.require_passed()


def test_pf1_unknown_trainable_module_requires_an_authored_rule() -> None:
    model = _active_model()
    model.unregistered_trainable = torch.nn.Linear(16, 16, bias=False)

    with pytest.raises(GradientLivenessBlocked, match="no PF-1.5 eligibility rule"):
        parameter_eligibility_matrix(model, recurrent_steps=1)


def test_pf1_nonrecurrent_model_rejects_k4_matrix() -> None:
    config = replace(
        _active_config(),
        use_recurrence=False,
        recurrent_steps=1,
        use_static_kv_core=False,
        use_reentry_bridge=False,
        use_long_term_memory=False,
    )
    model = AblationLM(config)

    with pytest.raises(ValueError, match="K > 1 requires structural recurrence"):
        parameter_eligibility_matrix(model, recurrent_steps=4)


@pytest.mark.parametrize(
    ("kv_policy", "expected_visits"),
    [
        ("live", {0, 1, 2, 3}),
        ("static", {None}),
        ("midpoint", {None, 2}),
    ],
)
def test_pf1_bicameral_kv_and_combiner_follow_the_executed_schedule(
    kv_policy: str,
    expected_visits: set[int | None],
) -> None:
    model = AblationLM(_bicameral_config(kv_policy=kv_policy))
    matrix = parameter_eligibility_matrix(model, recurrent_steps=4)

    kv_names = {
        name
        for name, _parameter in model.named_parameters()
        if name.endswith(".k_proj.weight")
        or name.endswith(".v_proj.weight")
        or name.endswith(".key_norm.weight")
        if name.startswith("core_blocks.")
    }
    assert kv_names
    for name in kv_names:
        rows = tuple(row for row in matrix if row.parameter_name == name)
        assert {row.visit_index for row in rows} == expected_visits
        assert all(row.eligible and row.executed and not row.deferred for row in rows)

    combiner_rows = tuple(
        row for row in matrix if row.parameter_name == "bicameral_combiner.theta"
    )
    assert len(combiner_rows) == 1
    assert combiner_rows[0].visit_index is None
    assert combiner_rows[0].eligible and combiner_rows[0].executed
    assert not combiner_rows[0].deferred


def test_pf1_bicameral_midpoint_k1_records_only_the_initial_projection() -> None:
    model = AblationLM(
        _bicameral_config(kv_policy="midpoint", recurrent_steps=1)
    )
    matrix = parameter_eligibility_matrix(model, recurrent_steps=1)
    kv_rows = tuple(
        row
        for row in matrix
        if row.parameter_name.startswith("core_blocks.")
        and (
            row.parameter_name.endswith(".k_proj.weight")
            or row.parameter_name.endswith(".v_proj.weight")
            or row.parameter_name.endswith(".key_norm.weight")
        )
    )
    assert kv_rows
    assert {row.visit_index for row in kv_rows} == {None}


def test_pf1_step2_scratch_is_a_typed_deferred_nonpass_until_it_reaches_logits() -> None:
    _model_k1, k1 = _bicameral_backward_receipt(
        recurrent_steps=1,
        use_scratch=True,
    )
    model_k4, k4 = _bicameral_backward_receipt(
        recurrent_steps=4,
        use_scratch=True,
    )

    expected = tuple(
        sorted(
            name
            for name, _parameter in model_k4.named_parameters()
            if name.startswith("scratch.")
        )
    )
    assert tuple(item.parameter_name for item in k4.deferred_parameters) == expected
    assert all(item.deferred for item in k4.deferred_parameters)
    assert not set(expected).intersection(k4.eligible_parameter_names)
    assert not k4.eligible_missing_gradients
    assert not k4.eligible_zero_gradients
    assert not k4.passed
    with pytest.raises(GradientLivenessBlocked, match="deferred=.*scratch"):
        k4.require_passed()

    inverse = inverse_k1_k4_liveness(k1, k4)
    assert not inverse.k1_ineligible_parameter_names
    inverse.require_passed()


def test_pf1_bicameral_step2_without_scratch_has_live_core_and_combiner() -> None:
    _model, receipt = _bicameral_backward_receipt(
        recurrent_steps=4,
        use_scratch=False,
    )
    receipt.require_passed()
    assert not receipt.deferred_parameters
    assert "bicameral_combiner.theta" in receipt.live_parameter_names
    assert "core_blocks.0.k_proj.weight" in receipt.live_parameter_names
