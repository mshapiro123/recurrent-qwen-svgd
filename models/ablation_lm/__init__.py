"""Ablation-first recursive language-model substrate."""

from .accounting import (
    ParameterAccounting,
    TokenizerScaleAccounting,
    TokenizerScreenAccounting,
    TokenizerTargetDecisionContract,
    estimate_dense_unique_parameters,
    lexical_parameter_share,
    parameter_accounting,
    tokenizer_screen_accounting,
)
from .config import (
    RATIFIED_TARGET_AUTHORITY,
    RATIFIED_TARGET_D_MODEL,
    RATIFIED_TARGET_PARAMETER_BUDGET,
    RATIFIED_TARGET_REFERENCE_VOCAB_SIZE,
    REGISTERED_CORE_BLOCK_COUNTS,
    TOKENIZER_VOCAB_CANDIDATES,
    AblationLMConfig,
    registered_mu_r_configs,
)
from .diagnostics import (
    RouterCalibrationDecision,
    RouterMomentSnapshot,
    router_calibration_stability,
)
from .geometry import Cl20Rotor, LaneModeState, cl20_rotate, lanes_to_modes
from .model import AblationLM, AblationLMOutput

__all__ = [
    "AblationLM",
    "AblationLMConfig",
    "AblationLMOutput",
    "Cl20Rotor",
    "LaneModeState",
    "ParameterAccounting",
    "RATIFIED_TARGET_AUTHORITY",
    "RATIFIED_TARGET_D_MODEL",
    "RATIFIED_TARGET_PARAMETER_BUDGET",
    "RATIFIED_TARGET_REFERENCE_VOCAB_SIZE",
    "RouterCalibrationDecision",
    "RouterMomentSnapshot",
    "REGISTERED_CORE_BLOCK_COUNTS",
    "TOKENIZER_VOCAB_CANDIDATES",
    "TokenizerScaleAccounting",
    "TokenizerScreenAccounting",
    "TokenizerTargetDecisionContract",
    "estimate_dense_unique_parameters",
    "lexical_parameter_share",
    "cl20_rotate",
    "lanes_to_modes",
    "parameter_accounting",
    "registered_mu_r_configs",
    "router_calibration_stability",
    "tokenizer_screen_accounting",
]
