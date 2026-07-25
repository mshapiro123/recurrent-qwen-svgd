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
from .recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM, RecurrentQwenOutput

__all__ = [
    "IdentityGatedBridge",
    "CoconutRecurrentQwen",
    "CompositeCoconutOutput",
    "HorizontalIdentityBridge",
    "LayerSplit",
    "LatentPolicyHead",
    "LatentTrajectoryModule",
    "LoRALinear",
    "ReentryAffineAdapter",
    "RecurrentQwenForCausalLM",
    "RecurrentQwenOutput",
    "SequenceHaltingPredictor",
    "apply_lora_to_qwen_layers",
    "apply_lora_to_recurrent_block",
    "masked_mean",
    "pondernet_halting_probabilities",
]
