"""Parameter accounting that keeps lexical and memory capacity explicit."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .config import AblationLMConfig
from .engram import CausalTokenEngram
from .memory import ReadOnlyLatentMemory


@dataclass(frozen=True)
class ParameterAccounting:
    total: int
    vocabulary: int
    engram_tables: int
    engram_frozen_table_elements: int
    engram_interface: int
    long_term_memory_trainable: int
    long_term_memory_store_elements: int
    non_vocabulary_dense: int

    @property
    def vocabulary_share(self) -> float:
        return self.vocabulary / self.total if self.total else 0.0

    @property
    def memory_share(self) -> float:
        memory = (
            self.engram_tables
            + self.engram_interface
            + self.long_term_memory_trainable
        )
        return memory / self.total if self.total else 0.0


@dataclass(frozen=True)
class TokenizerTargetAccounting:
    """Vocabulary cost at one intended model geometry."""

    vocab_size: int
    d_model: int
    core_blocks: int
    total_unique_parameters: int
    vocabulary_parameters: int

    @property
    def vocabulary_share(self) -> float:
        return self.vocabulary_parameters / self.total_unique_parameters


def _unique_parameters(module: nn.Module) -> dict[int, nn.Parameter]:
    return {id(parameter): parameter for parameter in module.parameters() if parameter.requires_grad}


def parameter_accounting(model: nn.Module) -> ParameterAccounting:
    """Count tied parameters once and separate vocab/static-memory capacity."""

    all_parameters = _unique_parameters(model)
    vocabulary_ids: set[int] = set()
    embedding = getattr(model, "token_embedding", None)
    if isinstance(embedding, nn.Embedding) and embedding.weight.requires_grad:
        vocabulary_ids.add(id(embedding.weight))

    engram_ids: set[int] = set()
    engram_interface_ids: set[int] = set()
    engram_frozen_table_elements = 0
    long_term_ids: set[int] = set()
    long_term_store_elements = 0
    for module in model.modules():
        if isinstance(module, CausalTokenEngram):
            module_ids = set(_unique_parameters(module))
            table_ids = {id(table.weight) for table in module.tables.values()}
            engram_ids.update(table_ids & module_ids)
            engram_interface_ids.update(module_ids - table_ids)
            engram_frozen_table_elements += sum(
                table.weight.numel()
                for table in module.tables.values()
                if not table.weight.requires_grad
            )
        elif isinstance(module, ReadOnlyLatentMemory):
            long_term_ids.update(_unique_parameters(module))
            long_term_store_elements += module.memory_keys.numel() + module.memory_values.numel()

    total = sum(parameter.numel() for parameter in all_parameters.values())
    vocabulary = sum(all_parameters[index].numel() for index in vocabulary_ids)
    engram = sum(all_parameters[index].numel() for index in engram_ids)
    engram_interface = sum(
        all_parameters[index].numel() for index in engram_interface_ids
    )
    long_term = sum(all_parameters[index].numel() for index in long_term_ids)
    dense = total - vocabulary - engram - engram_interface - long_term
    return ParameterAccounting(
        total,
        vocabulary,
        engram,
        engram_frozen_table_elements,
        engram_interface,
        long_term,
        long_term_store_elements,
        dense,
    )


def lexical_parameter_share(vocab_size: int, d_model: int, total_parameters: int) -> float:
    """Target-geometry lexical share for a tied input/output vocabulary."""

    if min(vocab_size, d_model, total_parameters) < 1:
        raise ValueError("all accounting inputs must be positive")
    return vocab_size * d_model / total_parameters


def estimate_dense_unique_parameters(config: AblationLMConfig) -> int:
    """Analytic count for a pillar-free graph without allocating its tensors."""

    if any(
        (
            config.use_front_hadamard_experts,
            config.use_reentry_bridge,
            config.use_scratch,
            config.use_lane_carrier,
            config.use_engram,
            config.use_long_term_memory,
        )
    ):
        raise ValueError("tokenizer target accounting requires the pillar-free dense graph")
    d_model = config.d_model
    head_dim = config.head_dim
    kv_width = config.n_kv_heads * head_dim
    attention = 2 * d_model * d_model + 2 * d_model * kv_width
    feed_forward = 3 * d_model * config.d_ff
    norms = 2 * d_model + 2 * head_dim
    blocks = config.n_prelude_layers + config.n_core_blocks + config.n_coda_layers
    loop_positions = (
        config.max_recurrent_steps * d_model if config.use_recurrence else 0
    )
    return (
        config.vocab_size * d_model
        + blocks * (attention + feed_forward + norms)
        + loop_positions
        + d_model
    )


def tokenizer_target_accounting(config: AblationLMConfig) -> TokenizerTargetAccounting:
    """Return target-ratio accounting for one candidate tokenizer/model pair."""

    total = estimate_dense_unique_parameters(config)
    vocabulary = config.vocab_size * config.d_model
    return TokenizerTargetAccounting(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        core_blocks=config.n_core_blocks,
        total_unique_parameters=total,
        vocabulary_parameters=vocabulary,
    )
