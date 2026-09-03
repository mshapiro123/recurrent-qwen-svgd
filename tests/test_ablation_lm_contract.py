from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from models.ablation_lm.accounting import (
    TokenizerScreenAccounting,
    TokenizerTargetDecisionContract,
    composition_receipt,
    estimate_dense_unique_parameters,
    lexical_parameter_share,
    tokenizer_screen_accounting,
)
from models.ablation_lm.model import AblationLM
from models.ablation_lm.config import (
    REGISTERED_CORE_BLOCK_COUNTS,
    REGISTERED_PROXY_BLOCK_SPLITS,
    REGISTERED_TARGET_BLOCK_SPLITS,
    RATIFIED_TARGET_D_MODEL,
    RATIFIED_TARGET_AUTHORITY,
    RATIFIED_TARGET_AUTHORITY_SHA256,
    RATIFIED_TARGET_ROUNDED_UNIQUE_PARAMETERS_BY_CORE,
    TOKENIZER_VOCAB_CANDIDATES,
    AblationLMConfig,
    registered_mu_r_configs,
    registered_target_configs,
)


def _small_config() -> AblationLMConfig:
    return AblationLMConfig(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_prelude_layers=1,
        n_core_blocks=2,
        n_coda_layers=1,
        recurrent_steps=4,
        max_recurrent_steps=8,
        max_sequence_length=32,
        scratch_width=8,
        long_term_memory_width=8,
    )


def _target_config(vocab_size: int = 32_768) -> AblationLMConfig:
    return replace(registered_target_configs()[1], vocab_size=vocab_size)


def _fixed_total_contract() -> TokenizerTargetDecisionContract:
    candidates = tuple(_target_config(vocab_size) for vocab_size in TOKENIZER_VOCAB_CANDIDATES)
    locked_total = estimate_dense_unique_parameters(_target_config())
    tolerance = max(
        abs(estimate_dense_unique_parameters(config) - locked_total)
        for config in candidates
    )
    return TokenizerTargetDecisionContract(
        config=_target_config(),
        authority=RATIFIED_TARGET_AUTHORITY,
        budget_semantics="fixed_total",
        fixed_total_parameters=locked_total,
        fixed_total_tolerance_parameters=tolerance,
        candidate_topologies=candidates,
    )


def test_recurrence_default_is_the_one_over_t_boundedness_anchor() -> None:
    config = _small_config()

    assert config.recurrence_exponent == 1.0
    assert config.recurrence_scale(1) == 1.0
    assert config.recurrence_scale(4) == 0.25
    assert config.recurrence_scale(8) == 0.125
    with pytest.raises(TypeError, match="exact integer"):
        config.recurrence_scale(2.5)


def test_proxy_mu_r_sweep_reallocates_a_constant_ten_blocks() -> None:
    config = AblationLMConfig()
    with pytest.raises(ValueError, match="structural recurrence"):
        registered_mu_r_configs(config)
    sweep = registered_mu_r_configs(replace(config, use_recurrence=True))

    assert config.n_core_blocks == 2
    assert REGISTERED_CORE_BLOCK_COUNTS == (4, 6)
    assert REGISTERED_PROXY_BLOCK_SPLITS == ((4, 2, 4), (3, 4, 3), (2, 6, 2))
    assert tuple(
        (item.n_prelude_layers, item.n_core_blocks, item.n_coda_layers)
        for item in sweep
    ) == REGISTERED_PROXY_BLOCK_SPLITS
    assert {
        item.n_prelude_layers + item.n_core_blocks + item.n_coda_layers
        for item in sweep
    } == {10}


def test_target_rungs_encode_exact_width_gqa_and_block_mapping() -> None:
    targets = registered_target_configs()

    assert REGISTERED_TARGET_BLOCK_SPLITS == ((9, 4, 9), (8, 6, 8))
    assert tuple(
        (item.n_prelude_layers, item.n_core_blocks, item.n_coda_layers)
        for item in targets
    ) == REGISTERED_TARGET_BLOCK_SPLITS
    assert all(item.d_model == 1_024 for item in targets)
    assert all((item.n_heads, item.n_kv_heads) == (16, 8) for item in targets)
    assert all(item.d_ff == 2_816 and item.scratch_width == 256 for item in targets)
    assert all(
        item.n_prelude_layers + item.n_core_blocks + item.n_coda_layers == 22
        for item in targets
    )


def test_tokenizer_screen_includes_both_directions_around_32k() -> None:
    assert TOKENIZER_VOCAB_CANDIDATES == (16_384, 24_576, 32_768, 49_152)
    assert lexical_parameter_share(32_768, 512, 50_000_000) == pytest.approx(0.33554432)
    assert lexical_parameter_share(
        32_768,
        RATIFIED_TARGET_D_MODEL,
        RATIFIED_TARGET_ROUNDED_UNIQUE_PARAMETERS_BY_CORE[6],
    ) == pytest.approx((32_768 * 1_024) / 305_800_000)


def test_active_layer_scales_and_callosum_cannot_be_initialized_dead() -> None:
    config = _small_config()
    with pytest.raises(ValueError, match="strictly positive"):
        replace(config, scratch_layer_scale=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        replace(config, lane_carrier_rho_init=0.0)
    with pytest.raises(ValueError, match="retention cap"):
        replace(config, lane_carrier_rho_init=0.1)
    with pytest.raises(ValueError, match="successor arms"):
        replace(config, recurrence_exponent=0.5)
    with pytest.raises(ValueError, match="requires structural recurrence"):
        replace(config, use_reentry_bridge=True)
    with pytest.raises(ValueError, match="static core KV requires structural recurrence"):
        replace(config, use_static_kv_core=True)
    with pytest.raises(ValueError, match="midpoint KV refresh"):
        replace(config, static_kv_midpoint_refresh=True)
    with pytest.raises(TypeError, match="exact bool"):
        replace(config, use_static_kv_core=1)
    with pytest.raises(ValueError, match="positive suffix"):
        replace(config, engram_orders=(2.5, 3))
    with pytest.raises(ValueError, match="exact integer"):
        replace(config, engram_hash_seed=17.9)
    with pytest.raises(ValueError, match="run_seed must be an exact integer"):
        replace(config, run_seed=True)
    with pytest.raises(ValueError, match="rng_replica must be an exact integer"):
        replace(config, rng_replica=1.5)
    with pytest.raises(ValueError, match="rng_replica must be non-negative"):
        replace(config, rng_replica=-1)
    with pytest.raises(ValueError, match="0.1 scale bound"):
        replace(config, engram_layer_scale=0.1)
    with pytest.raises(ValueError, match="positive integer"):
        replace(config, n_prelude_layers=1.5)


def test_full_bringup_profile_enables_recurrence_and_separate_lane_carrier() -> None:
    active = _small_config().with_innovations()

    assert active.use_recurrence is True
    assert active.recurrent_steps == 4
    assert active.use_static_kv_core is True
    assert active.use_scratch is True
    assert active.use_lane_carrier is True


def test_proxy_execution_accounting_matches_allocated_dense_model() -> None:
    config = _small_config()
    model = AblationLM(config)

    assert estimate_dense_unique_parameters(config) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_composition_receipt_partitions_fixed_and_recurrent_capacity() -> None:
    baseline = AblationLM(_small_config())
    baseline_receipt = composition_receipt(
        baseline,
        requested_visits=1,
        executed_visits=1,
    )
    assert baseline_receipt.n_unique == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert baseline_receipt.n_recurrent == 0
    assert baseline_receipt.n_fixed == baseline_receipt.n_body
    assert baseline_receipt.n_active_eval == baseline_receipt.n_body
    assert baseline_receipt.composition_exact is True
    assert baseline_receipt.active_eval_exact is True
    assert baseline_receipt.coda_decodes_per_step == 1
    assert baseline_receipt.lstage_sampled_visit is None
    json.dumps(baseline_receipt.as_dict())

    recurrent = AblationLM(replace(_small_config(), use_recurrence=True))
    receipt = composition_receipt(
        recurrent,
        requested_visits=4,
        executed_visits=3.5,
    )
    assert receipt.n_fixed + receipt.n_recurrent == receipt.n_body
    assert receipt.n_recurrent > 0
    assert receipt.n_active_eval is None
    assert receipt.composition_exact is False
    assert receipt.active_eval_exact is False
    with pytest.raises(ValueError, match="materialized sidecar"):
        composition_receipt(
            recurrent,
            requested_visits=4,
            executed_visits=4,
            sidecar_firing_fraction_by_step=(0.0, 0.25, 0.5, 0.75),
        )

    sampled = composition_receipt(
        recurrent,
        requested_visits=4,
        executed_visits=4,
        coda_decodes_per_step=2,
        lstage_sampled_visit=1,
    )
    assert sampled.coda_decodes_per_step == 2
    assert sampled.lstage_sampled_visit == 1
    json.dumps(sampled.as_dict())
    with pytest.raises(ValueError, match="earlier visit"):
        composition_receipt(
            recurrent,
            requested_visits=4,
            executed_visits=4,
            coda_decodes_per_step=2,
            lstage_sampled_visit=3,
        )
    with pytest.raises(ValueError, match="exactly two coda decodes"):
        composition_receipt(
            recurrent,
            requested_visits=4,
            executed_visits=4,
            coda_decodes_per_step=1,
            lstage_sampled_visit=0,
        )
    with pytest.raises(ValueError, match=r"\[1, requested_visits\]"):
        composition_receipt(
            recurrent,
            requested_visits=1,
            executed_visits=1,
            coda_decodes_per_step=2,
        )

    wrapper = nn.Module()
    wrapper.model = baseline
    wrapper.extra = nn.Parameter(torch.ones(7))
    with pytest.raises(ValueError, match="outside the AblationLM"):
        composition_receipt(wrapper, requested_visits=1, executed_visits=1)


def test_diagnostic_forward_emits_the_composition_receipt() -> None:
    config = replace(_small_config(), vocab_size=64)
    model = AblationLM(config).eval()
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = model(inputs, return_diagnostics=True)

    receipt = output.diagnostics["composition_receipt"]
    assert receipt["requested_visits"] == 1
    assert receipt["executed_visits"] == 1.0
    assert receipt["coda_decodes_per_step"] == 1
    assert receipt["lstage_sampled_visit"] is None
    assert receipt["n_unique"] == sum(parameter.numel() for parameter in model.parameters())
    json.dumps(receipt)

    ordinary = model(inputs)
    assert ordinary.diagnostics["composition_receipt"] == receipt


@pytest.mark.parametrize("vocab_size", TOKENIZER_VOCAB_CANDIDATES)
def test_tokenizer_screen_separates_proxy_runs_from_target_decision_columns(
    vocab_size: int,
) -> None:
    proxy = replace(AblationLMConfig(), vocab_size=vocab_size, n_core_blocks=4)
    screen = tokenizer_screen_accounting(proxy)

    assert screen.execution_proxy.d_model == 512
    assert screen.decision_target.d_model == 1_024
    assert screen.decision_target.vocabulary_parameters == vocab_size * 1_024
    assert screen.decision_target.total_unique_parameters == (
        RATIFIED_TARGET_ROUNDED_UNIQUE_PARAMETERS_BY_CORE[6]
    )
    assert screen.decision_target.core_blocks == 6
    with pytest.raises(RuntimeError, match="resolved G-TOK selector"):
        _ = screen.selection_vocabulary_share


def test_tokenizer_freeze_requires_exact_authority_and_budget_semantics() -> None:
    proxy = replace(AblationLMConfig(), vocab_size=32_768, n_core_blocks=6)
    target = _target_config()
    contract = _fixed_total_contract()
    derived = tokenizer_screen_accounting(proxy, target_contract=contract)

    with pytest.raises(ValueError, match="authority"):
        TokenizerTargetDecisionContract(
            config=target,
            authority="",
            budget_semantics="fixed_total",
            fixed_total_parameters=estimate_dense_unique_parameters(target),
            candidate_topologies=contract.candidate_topologies,
        )
    assert RATIFIED_TARGET_AUTHORITY_SHA256 == (
        "c5df74297594e75697ffb71d8d05d75efcf94f7857d55ddd357043200efb6d3a"
    )
    assert RATIFIED_TARGET_AUTHORITY.endswith(RATIFIED_TARGET_AUTHORITY_SHA256)
    with pytest.raises(ValueError, match="exact WEFT-1 ratification receipt"):
        TokenizerTargetDecisionContract(
            config=target,
            authority="weft1_ratification_forged",
            budget_semantics="fixed_total",
            fixed_total_parameters=contract.fixed_total_parameters,
            fixed_total_tolerance_parameters=contract.fixed_total_tolerance_parameters,
            candidate_topologies=contract.candidate_topologies,
        )
    assert derived.decision_target.total_unique_parameters == estimate_dense_unique_parameters(
        target
    )
    assert derived.decision_target.core_blocks == target.n_core_blocks
    assert derived.decision_target.exact_total is False
    with pytest.raises(RuntimeError, match="full-model target composition"):
        _ = derived.selection_vocabulary_share

    rung_a = TokenizerTargetDecisionContract(
        config=registered_target_configs()[0],
        authority=RATIFIED_TARGET_AUTHORITY,
        budget_semantics="fixed_non_vocabulary",
    )
    with pytest.raises(ValueError, match="decision contract must use rung B"):
        tokenizer_screen_accounting(proxy, target_contract=rung_a)


def test_tokenizer_selection_rederives_rows_and_rejects_public_dataclass_forgery() -> None:
    proxy = replace(AblationLMConfig(), vocab_size=32_768, n_core_blocks=6)
    approximate = tokenizer_screen_accounting(proxy)
    target = _target_config()
    forged_without_contract = replace(
        approximate,
        decision_target=replace(
            approximate.decision_target,
            topology_config=target,
            authority="forged",
            exact_total=True,
            budget_semantics="fixed_total",
        ),
    )
    with pytest.raises(RuntimeError, match="resolved G-TOK selector"):
        _ = forged_without_contract.selection_vocabulary_share

    contract = _fixed_total_contract()
    valid = tokenizer_screen_accounting(proxy, target_contract=contract)
    direct_forgery = TokenizerScreenAccounting(
        execution_proxy=valid.execution_proxy,
        decision_target=replace(
            valid.decision_target,
            total_unique_parameters=valid.decision_target.total_unique_parameters + 1,
        ),
        target_contract=contract,
    )
    with pytest.raises(RuntimeError, match="resolved G-TOK selector"):
        _ = direct_forgery.selection_vocabulary_share

    forged_proxy = replace(
        valid,
        execution_proxy=replace(valid.execution_proxy, exact_total=False),
    )
    with pytest.raises(RuntimeError, match="resolved G-TOK selector"):
        _ = forged_proxy.selection_vocabulary_share


def test_fixed_non_vocabulary_accounting_reprices_each_candidate_total() -> None:
    reference_target = _target_config()
    contract = TokenizerTargetDecisionContract(
        config=reference_target,
        authority=RATIFIED_TARGET_AUTHORITY,
        budget_semantics="fixed_non_vocabulary",
    )
    small = tokenizer_screen_accounting(
        replace(AblationLMConfig(), vocab_size=16_384),
        target_contract=contract,
    )
    large = tokenizer_screen_accounting(
        replace(AblationLMConfig(), vocab_size=49_152),
        target_contract=contract,
    )

    reference_total = estimate_dense_unique_parameters(reference_target)
    body = reference_total - 32_768 * 1_024
    assert small.decision_target.total_unique_parameters == body + 16_384 * 1_024
    assert large.decision_target.total_unique_parameters == body + 49_152 * 1_024
    assert small.decision_target.vocabulary_share == pytest.approx(
        (16_384 * 1_024) / (body + 16_384 * 1_024)
    )
    assert large.decision_target.vocabulary_share == pytest.approx(
        (49_152 * 1_024) / (body + 49_152 * 1_024)
    )
    with pytest.raises(ValueError, match="registered vocabulary candidate"):
        tokenizer_screen_accounting(
            replace(AblationLMConfig(), vocab_size=12_345),
            target_contract=contract,
        )


def test_fixed_total_contract_binds_one_budget_across_every_candidate() -> None:
    contract = _fixed_total_contract()
    small = tokenizer_screen_accounting(
        replace(AblationLMConfig(), vocab_size=16_384),
        target_contract=contract,
    )
    large = tokenizer_screen_accounting(
        replace(AblationLMConfig(), vocab_size=49_152),
        target_contract=contract,
    )

    assert small.decision_target.total_unique_parameters == contract.fixed_total_parameters
    assert large.decision_target.total_unique_parameters == contract.fixed_total_parameters
    assert small.target_contract is large.target_contract is contract

    with pytest.raises(ValueError, match="every registered tokenizer candidate"):
        TokenizerTargetDecisionContract(
            config=_target_config(16_384),
            authority=RATIFIED_TARGET_AUTHORITY,
            budget_semantics="fixed_total",
            fixed_total_parameters=estimate_dense_unique_parameters(_target_config(16_384)),
            candidate_topologies=(_target_config(16_384),),
        )

    with pytest.raises(ValueError, match="locked common budget"):
        TokenizerTargetDecisionContract(
            config=_target_config(),
            authority=RATIFIED_TARGET_AUTHORITY,
            budget_semantics="fixed_total",
            fixed_total_parameters=estimate_dense_unique_parameters(_target_config()),
            candidate_topologies=tuple(
                _target_config(vocab_size) for vocab_size in TOKENIZER_VOCAB_CANDIDATES
            ),
        )


def test_target_contract_rejects_bringup_mixed_rungs_and_impossible_capacity() -> None:
    with pytest.raises(ValueError, match="registered 4/6-core"):
        TokenizerTargetDecisionContract(
            config=replace(_target_config(), n_core_blocks=2),
            authority=RATIFIED_TARGET_AUTHORITY,
            budget_semantics="fixed_non_vocabulary",
        )
    with pytest.raises(ValueError, match="reference vocabulary must be a registered"):
        TokenizerTargetDecisionContract(
            config=replace(_target_config(), vocab_size=12_345),
            authority=RATIFIED_TARGET_AUTHORITY,
            budget_semantics="fixed_non_vocabulary",
            reference_vocab_size=12_345,
        )

    candidates = tuple(_target_config(vocab_size) for vocab_size in TOKENIZER_VOCAB_CANDIDATES)
    mixed_rungs = (
        replace(registered_target_configs()[0], vocab_size=candidates[0].vocab_size),
        *candidates[1:],
    )
    locked_total = estimate_dense_unique_parameters(_target_config())
    tolerance = max(
        abs(estimate_dense_unique_parameters(config) - locked_total)
        for config in candidates
    )
    with pytest.raises(ValueError, match="share one registered target rung"):
        TokenizerTargetDecisionContract(
            config=_target_config(),
            authority=RATIFIED_TARGET_AUTHORITY,
            budget_semantics="fixed_total",
            fixed_total_parameters=locked_total,
            fixed_total_tolerance_parameters=tolerance,
            candidate_topologies=mixed_rungs,
        )

    wrong_geometry = (replace(candidates[0], d_ff=3_008), *candidates[1:])
    with pytest.raises(ValueError, match="exact registered target topology"):
        TokenizerTargetDecisionContract(
            config=_target_config(),
            authority=RATIFIED_TARGET_AUTHORITY,
            budget_semantics="fixed_total",
            fixed_total_parameters=locked_total,
            fixed_total_tolerance_parameters=tolerance,
            candidate_topologies=wrong_geometry,
        )

    with pytest.raises(ValueError, match="exceed the largest candidate vocabulary"):
        TokenizerTargetDecisionContract(
            config=_target_config(),
            authority=RATIFIED_TARGET_AUTHORITY,
            budget_semantics="fixed_total",
            fixed_total_parameters=1,
            fixed_total_tolerance_parameters=max(
                estimate_dense_unique_parameters(config) for config in candidates
            ),
            candidate_topologies=candidates,
        )
