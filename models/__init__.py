"""GRAM-inspired recurrent-depth Qwen components."""

from .bridge import IdentityGatedBridge
from .halting import SequenceHaltingPredictor, masked_mean, pondernet_halting_probabilities
from .latent_policy import LatentPolicyHead, LatentTrajectoryModule
from .lora import LoRALinear, apply_lora_to_qwen_layers, apply_lora_to_recurrent_block
from .recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM, RecurrentQwenOutput

__all__ = [
    "IdentityGatedBridge",
    "LayerSplit",
    "LatentPolicyHead",
    "LatentTrajectoryModule",
    "LoRALinear",
    "RecurrentQwenForCausalLM",
    "RecurrentQwenOutput",
    "SequenceHaltingPredictor",
    "apply_lora_to_qwen_layers",
    "apply_lora_to_recurrent_block",
    "masked_mean",
    "pondernet_halting_probabilities",
]
