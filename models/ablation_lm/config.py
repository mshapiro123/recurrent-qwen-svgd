"""Validated configuration contracts for the ablation-first language model."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


TOKENIZER_VOCAB_CANDIDATES = (16_384, 24_576, 32_768, 49_152)
REGISTERED_CORE_BLOCK_COUNTS = (4, 6)


@dataclass(frozen=True)
class AblationLMConfig:
    """One explicit model graph; innovations are structural on/off switches.

    The defaults describe the inexpensive two-core-block bring-up graph.  They
    are intentionally not the registered recurrent-scaling sweep, whose core
    block counts are exposed by :func:`registered_mu_r_configs`.
    """

    vocab_size: int = 32_768
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1_408
    n_prelude_layers: int = 2
    n_core_blocks: int = 2
    n_coda_layers: int = 2
    use_recurrence: bool = False
    recurrent_steps: int = 1
    max_recurrent_steps: int = 8
    recurrence_coefficient: float = 1.0
    recurrence_exponent: float = 1.0
    max_sequence_length: int = 4_096
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    tie_embeddings: bool = True
    initialization_seed: int = 20_260_826

    use_front_hadamard_experts: bool = False
    hadamard_experts: int = 4
    hadamard_layer_scale: float = 1e-3
    hadamard_seed: int = 20_260_826

    use_reentry_bridge: bool = False
    bridge_layer_scale: float = 1e-3
    use_scratch: bool = False
    use_lane_carrier: bool = False
    scratch_lanes: int = 2
    scratch_width: int = 128
    scratch_layer_scale: float = 1e-3
    lane_carrier_rho_init: float = 0.01

    use_engram: bool = False
    engram_orders: tuple[int, ...] = (2, 3)
    engram_hashes_per_order: int = 4
    engram_table_size: int = 65_521
    engram_row_dim: int = 8
    engram_layer_scale: float = 1e-3
    engram_hash_seed: int = 20_260_826

    use_long_term_memory: bool = False
    long_term_memory_slots: int = 1_024
    long_term_memory_width: int = 128
    long_term_memory_layer_scale: float = 1e-3

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
            "n_prelude_layers": self.n_prelude_layers,
            "n_core_blocks": self.n_core_blocks,
            "recurrent_steps": self.recurrent_steps,
            "max_recurrent_steps": self.max_recurrent_steps,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in positive_ints.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.n_coda_layers) is not int or self.n_coda_layers < 0:
            raise ValueError("n_coda_layers must be non-negative")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.recurrent_steps > self.max_recurrent_steps:
            raise ValueError("recurrent_steps exceeds max_recurrent_steps")
        if not math.isfinite(self.recurrence_coefficient) or self.recurrence_coefficient <= 0:
            raise ValueError("recurrence_coefficient must be finite and positive")
        if self.recurrence_exponent != 1.0:
            raise ValueError(
                "this substrate locks recurrence_exponent=1; alternatives are successor arms"
            )
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
            raise ValueError("rope_theta must be finite and greater than one")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must lie in [0, 1)")
        if not self.tie_embeddings:
            raise ValueError("the substrate contract requires tied input/output embeddings")
        if self.use_front_hadamard_experts and not self._is_power_of_two(self.d_model):
            raise ValueError("Hadamard experts require power-of-two d_model")
        if type(self.hadamard_experts) is not int or self.hadamard_experts < 1:
            raise ValueError("hadamard_experts must be positive")
        if type(self.scratch_lanes) is not int or self.scratch_lanes != 2:
            raise ValueError("the initial bicameral contract requires exactly two scratch lanes")
        if type(self.scratch_width) is not int or self.scratch_width < 1:
            raise ValueError("scratch_width must be positive")
        if not 0 < self.lane_carrier_rho_init < 0.5:
            raise ValueError("lane_carrier_rho_init must lie strictly between 0 and 0.5")
        if self.use_lane_carrier and not self.use_scratch:
            raise ValueError("the lane carrier requires the position-aligned scratch arm")
        if self.use_reentry_bridge and not self.use_recurrence:
            raise ValueError("the re-entry bridge requires structural recurrence")
        if self.use_reentry_bridge and self.recurrent_steps < 2:
            raise ValueError("the re-entry bridge requires at least two recurrent visits")
        if not self.engram_orders or any(
            type(order) is not int or order < 1 for order in self.engram_orders
        ):
            raise ValueError("engram_orders must contain positive suffix lengths")
        if len(set(self.engram_orders)) != len(self.engram_orders):
            raise ValueError("engram_orders must be unique")
        for name, value in {
            "engram_hashes_per_order": self.engram_hashes_per_order,
            "engram_table_size": self.engram_table_size,
            "engram_row_dim": self.engram_row_dim,
            "long_term_memory_slots": self.long_term_memory_slots,
            "long_term_memory_width": self.long_term_memory_width,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in {
            "initialization_seed": self.initialization_seed,
            "hadamard_seed": self.hadamard_seed,
            "engram_hash_seed": self.engram_hash_seed,
        }.items():
            if type(value) is not int:
                raise ValueError(f"{name} must be an exact integer")
        for name, value in {
            "hadamard_layer_scale": self.hadamard_layer_scale,
            "scratch_layer_scale": self.scratch_layer_scale,
            "bridge_layer_scale": self.bridge_layer_scale,
            "engram_layer_scale": self.engram_layer_scale,
            "long_term_memory_layer_scale": self.long_term_memory_layer_scale,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive for an active arm")
        if self.engram_layer_scale >= 0.1:
            raise ValueError("engram_layer_scale must be smaller than the 0.1 scale bound")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and value & (value - 1) == 0

    def recurrence_scale(self, steps: int | None = None) -> float:
        """Return ``c / T**p``; ``p=1`` is the boundedness anchor."""

        if steps is not None and type(steps) is not int:
            raise TypeError("steps must be an exact integer")
        actual_steps = self.recurrent_steps if steps is None else steps
        if actual_steps < 1 or actual_steps > self.max_recurrent_steps:
            raise ValueError("steps must lie within the configured recurrence cap")
        return self.recurrence_coefficient / actual_steps**self.recurrence_exponent

    def with_innovations(self) -> "AblationLMConfig":
        """Return full switches; model construction still requires a frozen LTM store."""

        return replace(
            self,
            use_front_hadamard_experts=True,
            use_recurrence=True,
            recurrent_steps=4,
            use_reentry_bridge=True,
            use_scratch=True,
            use_lane_carrier=True,
            use_engram=True,
            use_long_term_memory=True,
        )


def registered_mu_r_configs(base: AblationLMConfig) -> tuple[AblationLMConfig, ...]:
    """Materialize the registered 4/6-core-block control axis.

    Recurrent step count remains a separate registered axis.  The two-block
    default is deliberately excluded because it is a bring-up configuration.
    """

    if not base.use_recurrence:
        raise ValueError("the mu-R core-block sweep requires structural recurrence")
    return tuple(replace(base, n_core_blocks=count) for count in REGISTERED_CORE_BLOCK_COUNTS)
