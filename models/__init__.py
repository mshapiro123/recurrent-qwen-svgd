"""GRAM-inspired recurrent-depth Qwen components."""

from .bridge import IdentityGatedBridge
from .coconut_composite import (
    CoconutRecurrentQwen,
    CompositeCoconutOutput,
    HorizontalIdentityBridge,
)
from .halting import SequenceHaltingPredictor, masked_mean, pondernet_halting_probabilities
from .latent_policy import LatentPolicyHead, LatentTrajectoryModule
from .lora import LoRALinear, apply_lora_to_qwen_layers, apply_lora_to_recurrent_block
from .reentry_adapter import ReentryAffineAdapter
from .oracle_reentry_conditioner import OracleReentryConditioner
from .paper2_dc2_student import (
    AnchoredBridge,
    ControlState,
    Phase2StudentModules,
    ResidualDraftHead,
    ScratchpadInitializer,
    SharedResidualFlow,
)
from .recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM, RecurrentQwenOutput
from .sidecar_v2 import LiteralNGramMemory, ProbePool, fast_wht

__all__ = [
    "IdentityGatedBridge",
    "AnchoredBridge",
    "CoconutRecurrentQwen",
    "CompositeCoconutOutput",
    "HorizontalIdentityBridge",
    "LayerSplit",
    "LatentPolicyHead",
    "LatentTrajectoryModule",
    "LiteralNGramMemory",
    "ProbePool",
    "LoRALinear",
    "ControlState",
    "Phase2StudentModules",
    "ResidualDraftHead",
    "ScratchpadInitializer",
    "SharedResidualFlow",
    "ReentryAffineAdapter",
    "RecurrentQwenForCausalLM",
    "RecurrentQwenOutput",
    "SequenceHaltingPredictor",
    "apply_lora_to_qwen_layers",
    "apply_lora_to_recurrent_block",
    "masked_mean",
    "pondernet_halting_probabilities",
    "fast_wht",
]
