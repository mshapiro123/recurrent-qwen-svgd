from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from models.ablation_lm.accounting import composition_receipt
from models.ablation_lm.config import AblationLMConfig
from models.ablation_lm.model import AblationLM
from models.ablation_lm.rng import ModuleRNGStream, derive_draw_seed


CPU = torch.device("cpu")


def _config(
    *,
    steps: int = 4,
    enabled: bool = True,
    bicameral: bool = False,
) -> AblationLMConfig:
    d_model = 64 if bicameral else 16
    n_heads = 2 if bicameral else 4
    n_kv_heads = 1 if bicameral else 2
    d_ff = 64 if bicameral else 32
    return AblationLMConfig(
        vocab_size=32,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        n_prelude_layers=1,
        n_core_blocks=1,
        n_coda_layers=1,
        use_recurrence=True,
        recurrent_steps=steps,
        max_recurrent_steps=8,
        use_bicameral_core=bicameral,
        kv_policy="live",
        max_sequence_length=8,
        use_scratch=False,
        scratch_width=4,
        use_lstage_sampled_decode=enabled,
        run_seed=1_337,
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]])
    labels = torch.tensor([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]])
    return tokens, labels


def _single_stream_visit_states(
    model: AblationLM,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    hidden = model.token_embedding(tokens)
    for block in model.prelude_blocks:
        hidden = block(hidden)
    prelude = hidden
    alpha = model.config.recurrence_scale(model.config.recurrent_steps)
    states: list[torch.Tensor] = []
    for step_index in range(model.config.recurrent_steps):
        hidden, lanes = model._run_recurrent_visit(
            hidden,
            prelude=prelude,
            lanes=None,
            step_index=step_index,
            alpha=alpha,
            attention_mask=None,
            position_ids=None,
            document_ids=None,
        )
        assert lanes is None
        states.append(hidden)
    return tuple(states)


def _gradient_l1(
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return sum(gradient.abs().sum() for gradient in gradients)


def test_dmc1_is_structural_off_by_default_and_does_not_perturb_final_path() -> None:
    tokens, labels = _batch()
    control = AblationLM(_config(enabled=False)).train()
    treatment = AblationLM(_config(enabled=True)).train()
    treatment_parameters = dict(treatment.named_parameters())
    assert all(
        torch.equal(parameter, treatment_parameters[name])
        for name, parameter in control.named_parameters()
    )
    assert control.config.use_lstage_sampled_decode is False
    assert control.lstage_sample_rng is None

    with patch.object(
        control.coda_blocks[0],
        "forward",
        wraps=control.coda_blocks[0].forward,
    ) as control_coda:
        control_output = control(tokens, labels=labels)
    treatment_output = treatment(tokens, labels=labels)

    assert control_coda.call_count == 1
    assert control_output.sampled_lstage_logits is None
    assert control_output.diagnostics["composition_receipt"]["coda_decodes_per_step"] == 1
    assert torch.equal(control_output.logits, treatment_output.logits)
    assert control_output.loss is not None and treatment_output.loss is not None
    assert torch.equal(control_output.loss, treatment_output.loss)


def test_dmc1_bfloat16_final_path_is_exact_and_second_decode_is_serial() -> None:
    tokens, labels = _batch()
    control = AblationLM(_config(enabled=False)).to(dtype=torch.bfloat16).train()
    treatment = AblationLM(_config(enabled=True)).to(dtype=torch.bfloat16).train()

    with patch.object(
        treatment.coda_blocks[0],
        "forward",
        wraps=treatment.coda_blocks[0].forward,
    ) as treatment_coda:
        control_output = control(tokens, labels=labels)
        treatment_output = treatment(tokens, labels=labels)

    assert treatment_coda.call_count == 2
    assert all(
        call.args[0].shape[0] == tokens.shape[0]
        for call in treatment_coda.call_args_list
    )
    assert treatment_output.sampled_lstage_logits is not None
    assert treatment_output.sampled_lstage_logits.dtype is torch.bfloat16
    assert torch.equal(control_output.logits, treatment_output.logits)
    assert control_output.loss is not None and treatment_output.loss is not None
    assert control_output.loss.dtype is torch.float32
    assert treatment_output.loss.dtype is torch.float32
    assert torch.equal(control_output.loss, treatment_output.loss)


def test_dmc1_replay_receipts_exact_draw_and_isolates_other_streams() -> None:
    tokens, _labels = _batch()
    first = AblationLM(_config()).train()
    replay = AblationLM(_config()).train()

    first_receipts = []
    replay_receipts = []
    for _micro_batch in range(2):
        first_receipts.append(first(tokens).diagnostics["composition_receipt"])
        replay_receipts.append(replay(tokens).diagnostics["composition_receipt"])

    for index, (actual, expected) in enumerate(
        zip(first_receipts, replay_receipts, strict=True)
    ):
        assert actual["lstage_sample_draw_index"] == index
        assert actual["lstage_sample_source_key"] == "weft.lstage.sample"
        assert actual["lstage_sample_rng_coordinate"] == 0
        assert actual["lstage_sample_seed"] == expected["lstage_sample_seed"]
        assert actual["lstage_sampled_visit"] == expected["lstage_sampled_visit"]
        assert 0 <= actual["lstage_sampled_visit"] <= 2

    treatment = AblationLM(_config()).train()
    control = AblationLM(_config()).train()
    treatment_lstage = ModuleRNGStream(
        treatment.config.run_seed,
        "weft.lstage.sample",
        replica=1,
    )
    control_lstage = ModuleRNGStream(
        control.config.run_seed,
        "weft.lstage.sample",
        replica=0,
    )
    _treatment_generator, treatment_seed = treatment_lstage.next_generator_with_seed(CPU)
    _control_generator, control_seed = control_lstage.next_generator_with_seed(CPU)
    assert treatment_seed != control_seed

    treatment_other = treatment.coda_blocks[0].attention.dropout_rng
    control_other = control.coda_blocks[0].attention.dropout_rng
    treatment_draw = torch.rand(
        17,
        generator=treatment_other.next_generator(CPU, coordinate=0),
    )
    control_draw = torch.rand(
        17,
        generator=control_other.next_generator(CPU, coordinate=0),
    )
    torch.testing.assert_close(treatment_draw, control_draw, rtol=0, atol=0)


def test_dmc1_cpu_draws_cover_every_eligible_visit_without_global_rng_use() -> None:
    stream = ModuleRNGStream(1_337, "weft.lstage.sample")
    torch.manual_seed(91)
    ambient = torch.random.get_rng_state().clone()
    counts = [0, 0, 0]
    for _index in range(600):
        generator, _seed = stream.next_generator_with_seed(CPU)
        visit = int(torch.randint(3, (1,), generator=generator).item())
        counts[visit] += 1

    torch.testing.assert_close(torch.random.get_rng_state(), ambient, rtol=0, atol=0)
    assert all(count > 0 for count in counts)
    # Deterministic implementation smoke test, not a research decision band.
    assert max(counts) - min(counts) < 75


def test_dmc1_decodes_serially_and_exposes_direct_sampled_loss_gradient() -> None:
    tokens, labels = _batch()
    model = AblationLM(_config()).train()

    with patch.object(
        model.coda_blocks[0],
        "forward",
        wraps=model.coda_blocks[0].forward,
    ) as coda_call:
        output = model(tokens, labels=labels)

    receipt = output.diagnostics["composition_receipt"]
    assert coda_call.call_count == 2
    assert all(call.args[0].shape[0] == tokens.shape[0] for call in coda_call.call_args_list)
    assert receipt["coda_decodes_per_step"] == 2
    sampled_visit = receipt["lstage_sampled_visit"]
    assert sampled_visit is not None
    assert output.sampled_lstage_logits is not None

    visit_states = _single_stream_visit_states(model, tokens)
    direct_logits = model._decode_with_shared_coda(
        visit_states[sampled_visit],
        attention_mask=None,
        position_ids=None,
        document_ids=None,
    )
    torch.testing.assert_close(
        output.sampled_lstage_logits,
        direct_logits,
        rtol=0,
        atol=0,
    )

    final_loss = model._language_model_loss(output.logits, labels, None, None)
    sampled_loss = model._language_model_loss(
        output.sampled_lstage_logits,
        labels,
        None,
        None,
    )
    assert output.loss is not None
    torch.testing.assert_close(output.loss, final_loss, rtol=0, atol=0)
    coda_parameters = tuple(model.coda_blocks.parameters())
    assert _gradient_l1(final_loss, coda_parameters, retain_graph=True) > 0
    assert _gradient_l1(sampled_loss, coda_parameters, retain_graph=True) > 0

    # lambda_stage is deliberately test-local: its production value and the
    # staged target remain unbound curriculum inputs.
    (final_loss + 0.25 * sampled_loss).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in coda_parameters
    )


def test_dmc1_k1_and_inference_decode_once_and_consume_no_sample_draw() -> None:
    tokens, labels = _batch()
    k1 = AblationLM(_config(steps=1)).train()
    assert k1.lstage_sample_rng is not None
    with patch.object(
        k1.coda_blocks[0],
        "forward",
        wraps=k1.coda_blocks[0].forward,
    ) as coda_call:
        output = k1(tokens, labels=labels)
    receipt = output.diagnostics["composition_receipt"]
    assert coda_call.call_count == 1
    assert k1.lstage_sample_rng.draw_index == 0
    assert output.sampled_lstage_logits is None
    assert receipt["coda_decodes_per_step"] == 1
    assert receipt["lstage_sampled_visit"] is None
    assert receipt["lstage_sample_source_key"] is None
    assert receipt["lstage_sample_draw_index"] is None
    assert receipt["lstage_sample_seed"] is None

    inference = AblationLM(_config(steps=4)).eval()
    assert inference.lstage_sample_rng is not None
    with patch.object(
        inference.coda_blocks[0],
        "forward",
        wraps=inference.coda_blocks[0].forward,
    ) as inference_coda:
        with torch.no_grad():
            inference_output = inference(tokens)
    inference_receipt = inference_output.diagnostics["composition_receipt"]
    assert inference_coda.call_count == 1
    assert inference.lstage_sample_rng.draw_index == 0
    assert inference_output.sampled_lstage_logits is None
    assert inference_receipt["coda_decodes_per_step"] == 1
    assert inference_receipt["lstage_sampled_visit"] is None


def test_dmc1_bicameral_sample_uses_s2_and_active_train_counts_extra_coda() -> None:
    tokens, _labels = _batch()
    model = AblationLM(_config(steps=3, bicameral=True)).train()
    assert model.bicameral_combiner is not None
    with patch.object(
        model.bicameral_combiner,
        "forward",
        wraps=model.bicameral_combiner.forward,
    ) as combine_call:
        output = model(tokens)

    receipt = output.diagnostics["composition_receipt"]
    assert combine_call.call_count == 2
    assert output.sampled_lstage_logits is not None
    assert receipt["active_eval_exact"] is True
    assert receipt["active_train_exact"] is True
    repeated_decode_parameters = {
        id(parameter): parameter
        for module in (
            model.coda_blocks,
            model.final_norm,
            model.bicameral_combiner,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    assert receipt["n_coda"] == sum(
        parameter.numel() for parameter in repeated_decode_parameters.values()
    )
    assert receipt["n_active_train"] == receipt["n_active_eval"] + receipt["n_coda"]


def test_dmc1_bicameral_diagnostic_states_detach_but_sample_stays_live() -> None:
    model = AblationLM(_config(steps=3, bicameral=True)).train()
    prelude = torch.randn(2, 5, model.config.d_model, requires_grad=True)
    result = model._run_bicameral_core(
        prelude,
        lanes=None,
        steps=3,
        alpha=model.config.recurrence_scale(3),
        attention_mask=None,
        position_ids=None,
        document_ids=None,
        capture_trajectory=True,
        sampled_visit=1,
    )
    diagnostic_states = result[4]
    sampled_hemisphere_states = result[5]
    assert len(diagnostic_states) == 3
    assert all(not state.requires_grad for state in diagnostic_states)
    assert sampled_hemisphere_states is not None
    assert all(state.requires_grad for state in sampled_hemisphere_states)


def test_dmc1_receipt_rejects_missing_or_orphaned_rng_metadata() -> None:
    model = AblationLM(_config(steps=3, bicameral=True)).eval()
    with pytest.raises(ValueError, match="coordinate must be exactly zero"):
        composition_receipt(
            model,
            requested_visits=3,
            executed_visits=3,
            coda_decodes_per_step=2,
            lstage_sampled_visit=0,
            lstage_sample_source_key="weft.lstage.sample",
            lstage_sample_draw_index=0,
            lstage_sample_seed=1,
            kv_policy="live",
            kv_cache_multiplier_at_serving=6,
            visit_schedule=model(
                _batch()[0],
                return_diagnostics=True,
            ).diagnostics["visit_schedule"],
        )
    with pytest.raises(ValueError, match="metadata requires a sampled visit"):
        composition_receipt(
            model,
            requested_visits=3,
            executed_visits=3,
            lstage_sample_seed=1,
            kv_policy="live",
            kv_cache_multiplier_at_serving=6,
            visit_schedule=model(
                _batch()[0],
                return_diagnostics=True,
            ).diagnostics["visit_schedule"],
        )
    with pytest.raises(ValueError, match="multiple coda decodes require"):
        composition_receipt(
            model,
            requested_visits=3,
            executed_visits=3,
            coda_decodes_per_step=2,
            kv_policy="live",
            kv_cache_multiplier_at_serving=6,
            visit_schedule=model(
                _batch()[0],
                return_diagnostics=True,
            ).diagnostics["visit_schedule"],
        )
    valid_schedule = model(
        _batch()[0],
        return_diagnostics=True,
    ).diagnostics["visit_schedule"]
    valid_seed = derive_draw_seed(
        model.config.run_seed,
        "weft.lstage.sample",
        model.config.rng_replica,
        coordinate=0,
        draw_index=0,
    )
    valid_generator = torch.Generator(device="cpu").manual_seed(valid_seed)
    valid_visit = int(torch.randint(2, (1,), generator=valid_generator).item())
    valid_replay = composition_receipt(
        model,
        requested_visits=3,
        executed_visits=3,
        coda_decodes_per_step=2,
        lstage_sampled_visit=valid_visit,
        lstage_sample_source_key="weft.lstage.sample",
        lstage_sample_rng_coordinate=0,
        lstage_sample_draw_index=0,
        lstage_sample_seed=valid_seed,
        kv_policy="live",
        kv_cache_multiplier_at_serving=6,
        visit_schedule=valid_schedule,
    )
    assert valid_replay.lstage_sampled_visit == valid_visit
    with pytest.raises(ValueError, match="does not replay"):
        composition_receipt(
            model,
            requested_visits=3,
            executed_visits=3,
            coda_decodes_per_step=2,
            lstage_sampled_visit=0,
            lstage_sample_source_key="weft.lstage.sample",
            lstage_sample_rng_coordinate=0,
            lstage_sample_draw_index=0,
            lstage_sample_seed=1,
            kv_policy="live",
            kv_cache_multiplier_at_serving=6,
            visit_schedule=valid_schedule,
        )
    forged_seed = 1
    assert forged_seed != valid_seed
    forged_generator = torch.Generator(device="cpu").manual_seed(forged_seed)
    forged_visit = int(torch.randint(2, (1,), generator=forged_generator).item())
    with pytest.raises(ValueError, match="configured O-9 derivation"):
        composition_receipt(
            model,
            requested_visits=3,
            executed_visits=3,
            coda_decodes_per_step=2,
            lstage_sampled_visit=forged_visit,
            lstage_sample_source_key="weft.lstage.sample",
            lstage_sample_rng_coordinate=0,
            lstage_sample_draw_index=0,
            lstage_sample_seed=forged_seed,
            kv_policy="live",
            kv_cache_multiplier_at_serving=6,
            visit_schedule=valid_schedule,
        )
    sample_metadata = {
        "coda_decodes_per_step": 2,
        "lstage_sampled_visit": 2,
        "lstage_sample_source_key": "weft.lstage.sample",
        "lstage_sample_rng_coordinate": 0,
        "lstage_sample_draw_index": 0,
        "lstage_sample_seed": 1,
        "kv_policy": "live",
        "kv_cache_multiplier_at_serving": 6,
        "visit_schedule": valid_schedule,
    }
    with pytest.raises(ValueError, match="executed_visits - 2"):
        composition_receipt(
            model,
            requested_visits=3,
            executed_visits=1,
            **sample_metadata,
        )
    with pytest.raises(ValueError, match="aggregate or fractional"):
        composition_receipt(
            model,
            requested_visits=3,
            executed_visits=2.5,
            **sample_metadata,
        )
