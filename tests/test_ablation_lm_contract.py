from __future__ import annotations

from dataclasses import replace

import pytest

from models.ablation_lm.accounting import (
    TokenizerScreenAccounting,
    TokenizerTargetDecisionContract,
    estimate_dense_unique_parameters,
    lexical_parameter_share,
    tokenizer_screen_accounting,
)
from models.ablation_lm.model import AblationLM
from models.ablation_lm.config import (
    REGISTERED_CORE_BLOCK_COUNTS,
    RATIFIED_TARGET_D_MODEL,
    RATIFIED_TARGET_PARAMETER_BUDGET,
    TOKENIZER_VOCAB_CANDIDATES,
    AblationLMConfig,
    registered_mu_r_configs,
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
    return replace(
        AblationLMConfig(),
        vocab_size=vocab_size,
        d_model=1_024,
        n_heads=16,
        n_kv_heads=4,
        d_ff=3_072,
        n_core_blocks=6,
    )


def _fixed_total_contract() -> TokenizerTargetDecisionContract:
    candidates = tuple(_target_config(vocab_size) for vocab_size in TOKENIZER_VOCAB_CANDIDATES)
    locked_total = estimate_dense_unique_parameters(_target_config())
    tolerance = max(
        abs(estimate_dense_unique_parameters(config) - locked_total)
        for config in candidates
    )
    return TokenizerTargetDecisionContract(
        config=_target_config(),
        authority="test_common_target_topology_receipt",
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


def test_two_blocks_are_bringup_but_four_and_six_are_registered_controls() -> None:
    config = _small_config()
    with pytest.raises(ValueError, match="structural recurrence"):
        registered_mu_r_configs(config)
    sweep = registered_mu_r_configs(replace(config, use_recurrence=True))

    assert config.n_core_blocks == 2
    assert REGISTERED_CORE_BLOCK_COUNTS == (4, 6)
    assert tuple(item.n_core_blocks for item in sweep) == (4, 6)


def test_tokenizer_screen_includes_both_directions_around_32k() -> None:
    assert TOKENIZER_VOCAB_CANDIDATES == (16_384, 24_576, 32_768, 49_152)
    assert lexical_parameter_share(32_768, 512, 50_000_000) == pytest.approx(0.33554432)
    assert lexical_parameter_share(
        32_768,
        RATIFIED_TARGET_D_MODEL,
        RATIFIED_TARGET_PARAMETER_BUDGET,
    ) == pytest.approx(0.11570494)


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
    with pytest.raises(ValueError, match="positive suffix"):
        replace(config, engram_orders=(2.5, 3))
    with pytest.raises(ValueError, match="exact integer"):
        replace(config, engram_hash_seed=17.9)
    with pytest.raises(ValueError, match="0.1 scale bound"):
        replace(config, engram_layer_scale=0.1)
    with pytest.raises(ValueError, match="positive integer"):
        replace(config, n_prelude_layers=1.5)


def test_full_bringup_profile_enables_recurrence_and_separate_lane_carrier() -> None:
    active = _small_config().with_innovations()

    assert active.use_recurrence is True
    assert active.recurrent_steps == 4
    assert active.use_scratch is True
    assert active.use_lane_carrier is True


def test_proxy_execution_accounting_matches_allocated_dense_model() -> None:
    config = _small_config()
    model = AblationLM(config)

    assert estimate_dense_unique_parameters(config) == sum(
        parameter.numel() for parameter in model.parameters()
    )


@pytest.mark.parametrize("vocab_size", TOKENIZER_VOCAB_CANDIDATES)
def test_tokenizer_screen_separates_proxy_runs_from_target_decision_columns(
    vocab_size: int,
) -> None:
    proxy = replace(AblationLMConfig(), vocab_size=vocab_size, n_core_blocks=4)
    screen = tokenizer_screen_accounting(proxy)

    assert screen.execution_proxy.d_model == 512
    assert screen.decision_target.d_model == 1_024
    assert screen.decision_target.vocabulary_parameters == vocab_size * 1_024
    assert screen.decision_target.total_unique_parameters == 290_000_000
    with pytest.raises(RuntimeError, match="exact target topology"):
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
    assert derived.decision_target.total_unique_parameters == estimate_dense_unique_parameters(
        target
    )
    assert derived.decision_target.core_blocks == target.n_core_blocks
    assert derived.selection_vocabulary_share == pytest.approx(
        (32_768 * 1_024) / estimate_dense_unique_parameters(target)
    )


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
    with pytest.raises(RuntimeError, match="topology contract"):
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
    with pytest.raises(RuntimeError, match="does not match its typed contract"):
        _ = direct_forgery.selection_vocabulary_share

    forged_proxy = replace(
        valid,
        execution_proxy=replace(valid.execution_proxy, exact_total=False),
    )
    with pytest.raises(RuntimeError, match="does not match its topology"):
        _ = forged_proxy.selection_vocabulary_share


def test_fixed_non_vocabulary_accounting_reprices_each_candidate_total() -> None:
    reference_target = replace(
        AblationLMConfig(),
        vocab_size=32_768,
        d_model=1_024,
        n_heads=16,
        n_kv_heads=4,
        d_ff=3_072,
        n_core_blocks=6,
    )
    contract = TokenizerTargetDecisionContract(
        config=reference_target,
        authority="test_fixed_body_receipt",
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
    assert small.selection_vocabulary_share == pytest.approx(
        (16_384 * 1_024) / (body + 16_384 * 1_024)
    )
    assert large.selection_vocabulary_share == pytest.approx(
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
            authority="per_candidate_budget_is_not_authority",
            budget_semantics="fixed_total",
            fixed_total_parameters=estimate_dense_unique_parameters(_target_config(16_384)),
            candidate_topologies=(_target_config(16_384),),
        )

    with pytest.raises(ValueError, match="locked common budget"):
        TokenizerTargetDecisionContract(
            config=_target_config(),
            authority="zero_tolerance_rejects_same_body_vocab_drift",
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
            authority="bringup_is_not_selection_authority",
            budget_semantics="fixed_non_vocabulary",
        )
    with pytest.raises(ValueError, match="reference vocabulary must be a registered"):
        TokenizerTargetDecisionContract(
            config=replace(_target_config(), vocab_size=12_345),
            authority="unregistered_reference_is_not_selection_authority",
            budget_semantics="fixed_non_vocabulary",
            reference_vocab_size=12_345,
        )

    candidates = tuple(_target_config(vocab_size) for vocab_size in TOKENIZER_VOCAB_CANDIDATES)
    mixed_rungs = (
        replace(candidates[0], n_core_blocks=4),
        *candidates[1:],
    )
    locked_total = estimate_dense_unique_parameters(_target_config())
    tolerance = max(
        abs(estimate_dense_unique_parameters(config) - locked_total)
        for config in candidates
    )
    with pytest.raises(ValueError, match="share one registered core rung"):
        TokenizerTargetDecisionContract(
            config=_target_config(),
            authority="mixed_rungs_are_not_one_selection_column",
            budget_semantics="fixed_total",
            fixed_total_parameters=locked_total,
            fixed_total_tolerance_parameters=tolerance,
            candidate_topologies=mixed_rungs,
        )

    with pytest.raises(ValueError, match="exceed the largest candidate vocabulary"):
        TokenizerTargetDecisionContract(
            config=_target_config(),
            authority="tolerance_cannot_authorize_impossible_capacity",
            budget_semantics="fixed_total",
            fixed_total_parameters=1,
            fixed_total_tolerance_parameters=max(
                estimate_dense_unique_parameters(config) for config in candidates
            ),
            candidate_topologies=candidates,
        )
