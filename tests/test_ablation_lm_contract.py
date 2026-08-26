from __future__ import annotations

from dataclasses import replace

import pytest

from models.ablation_lm.accounting import (
    estimate_dense_unique_parameters,
    lexical_parameter_share,
    tokenizer_target_accounting,
)
from models.ablation_lm.model import AblationLM
from models.ablation_lm.config import (
    REGISTERED_CORE_BLOCK_COUNTS,
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
    assert lexical_parameter_share(32_768, 1_024, 300_000_000) == pytest.approx(0.1118481)


def test_active_layer_scales_and_callosum_cannot_be_initialized_dead() -> None:
    config = _small_config()
    with pytest.raises(ValueError, match="strictly positive"):
        replace(config, scratch_layer_scale=0.0)
    with pytest.raises(ValueError, match="strictly between"):
        replace(config, lane_carrier_rho_init=0.0)
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


def test_analytic_target_accounting_matches_allocated_dense_model() -> None:
    config = _small_config()
    model = AblationLM(config)
    target = tokenizer_target_accounting(config)

    assert estimate_dense_unique_parameters(config) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert target.vocabulary_parameters == config.vocab_size * config.d_model
    assert target.core_blocks == config.n_core_blocks
