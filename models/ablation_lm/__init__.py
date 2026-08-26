"""Ablation-first recursive language-model substrate."""

from .accounting import (
    ParameterAccounting,
    TokenizerTargetAccounting,
    estimate_dense_unique_parameters,
    lexical_parameter_share,
    parameter_accounting,
    tokenizer_target_accounting,
)
from .config import (
    REGISTERED_CORE_BLOCK_COUNTS,
    TOKENIZER_VOCAB_CANDIDATES,
    AblationLMConfig,
    registered_mu_r_configs,
)
from .model import AblationLM, AblationLMOutput

__all__ = [
    "AblationLM",
    "AblationLMConfig",
    "AblationLMOutput",
    "ParameterAccounting",
    "REGISTERED_CORE_BLOCK_COUNTS",
    "TOKENIZER_VOCAB_CANDIDATES",
    "TokenizerTargetAccounting",
    "estimate_dense_unique_parameters",
    "lexical_parameter_share",
    "parameter_accounting",
    "registered_mu_r_configs",
    "tokenizer_target_accounting",
]
