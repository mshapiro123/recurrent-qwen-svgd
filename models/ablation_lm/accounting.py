"""Parameter accounting that keeps lexical and memory capacity explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from torch import nn

from .config import (
    RATIFIED_TARGET_AUTHORITY,
    RATIFIED_TARGET_D_MODEL,
    RATIFIED_TARGET_ROUNDED_UNIQUE_PARAMETERS_BY_CORE,
    RATIFIED_TARGET_REFERENCE_VOCAB_SIZE,
    REGISTERED_CORE_BLOCK_COUNTS,
    REGISTERED_TARGET_BLOCK_SPLITS,
    TOKENIZER_VOCAB_CANDIDATES,
    AblationLMConfig,
)
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
class CompositionReceipt:
    """Capacity taxonomy bound to requested and actually executed recurrence."""

    requested_visits: int
    executed_visits: float
    n_unique: int
    n_body: int
    n_fixed: int
    n_recurrent: int
    n_sparse_addressed: int
    vocabulary_parameters: int
    vocabulary_fraction: float
    recurrent_fraction: float
    fixed_to_recurrent: float | None
    n_active_eval: float | None
    composition_exact: bool
    active_eval_exact: bool
    sidecar_firing_fraction_by_step: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable machine receipt."""

        return asdict(self)


@dataclass(frozen=True)
class TokenizerScaleAccounting:
    """Vocabulary cost at one explicitly named execution or decision scale."""

    label: str
    vocab_size: int
    d_model: int
    total_unique_parameters: int
    vocabulary_parameters: int
    core_blocks: int | None
    authority: str | None
    exact_total: bool
    budget_semantics: str | None
    reference_vocab_size: int | None
    reference_total_unique_parameters: int | None
    topology_config: AblationLMConfig | None
    topology_unique_parameters: int | None

    @property
    def vocabulary_share(self) -> float:
        return self.vocabulary_parameters / self.total_unique_parameters


@dataclass(frozen=True)
class TokenizerScreenAccounting:
    """Keep cheap proxy execution separate from target-scale selection math.

    The current object is deliberately non-selecting.  The target decision is
    priced on rung B and serves both rungs under R-G4h; exact complete-model
    composition and a separately ratified selector are still required.
    """

    execution_proxy: TokenizerScaleAccounting
    decision_target: TokenizerScaleAccounting
    target_contract: TokenizerTargetDecisionContract | None

    @property
    def selection_vocabulary_share(self) -> float:
        raise RuntimeError(
            "WEFT-1 tokenizer selection requires exact full-model target composition "
            "and a resolved G-TOK selector; S0 dense rows cannot freeze a tokenizer"
        )


@dataclass(frozen=True)
class TokenizerTargetDecisionContract:
    """Written authority plus a topology from which target totals are derived."""

    config: AblationLMConfig
    authority: str
    budget_semantics: str
    reference_vocab_size: int = RATIFIED_TARGET_REFERENCE_VOCAB_SIZE
    fixed_total_parameters: int | None = None
    fixed_total_tolerance_parameters: int = 0
    candidate_topologies: tuple[AblationLMConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.config.d_model != RATIFIED_TARGET_D_MODEL:
            raise ValueError(
                f"target topology must use d_model={RATIFIED_TARGET_D_MODEL}"
            )
        if self.config.n_core_blocks not in REGISTERED_CORE_BLOCK_COUNTS:
            raise ValueError(
                "target topology must use a registered 4/6-core selection rung"
            )
        if (
            self.config.n_prelude_layers,
            self.config.n_core_blocks,
            self.config.n_coda_layers,
        ) not in REGISTERED_TARGET_BLOCK_SPLITS:
            raise ValueError("target topology must be exactly 9/4/9 or 8/6/8")
        if (
            self.config.n_heads,
            self.config.n_kv_heads,
            self.config.d_ff,
        ) != (16, 8, 2_816):
            raise ValueError("target topology must use 16Q/8KV and d_ff=2816")
        if not isinstance(self.authority, str) or not self.authority.strip():
            raise ValueError("target topology authority must be a nonempty string")
        if self.budget_semantics not in {"fixed_total", "fixed_non_vocabulary"}:
            raise ValueError("target budget semantics must be ratified")
        if type(self.reference_vocab_size) is not int or self.reference_vocab_size < 1:
            raise ValueError("target reference vocabulary size must be a positive integer")
        if self.reference_vocab_size not in TOKENIZER_VOCAB_CANDIDATES:
            raise ValueError("target reference vocabulary must be a registered candidate")
        if (
            type(self.fixed_total_tolerance_parameters) is not int
            or self.fixed_total_tolerance_parameters < 0
        ):
            raise ValueError("fixed-total tolerance must be a non-negative integer")
        if not isinstance(self.candidate_topologies, tuple) or any(
            not isinstance(config, AblationLMConfig)
            for config in self.candidate_topologies
        ):
            raise TypeError("candidate topologies must be a tuple of AblationLMConfig values")
        if self.budget_semantics == "fixed_total":
            if type(self.fixed_total_parameters) is not int or self.fixed_total_parameters < 1:
                raise ValueError("fixed-total semantics require a positive locked total")
            by_vocab = {config.vocab_size: config for config in self.candidate_topologies}
            if len(by_vocab) != len(self.candidate_topologies):
                raise ValueError("fixed-total candidate topologies must have unique vocabularies")
            if set(by_vocab) != set(TOKENIZER_VOCAB_CANDIDATES):
                raise ValueError(
                    "fixed-total authority must bind every registered tokenizer candidate"
                )
            if any(config.d_model != RATIFIED_TARGET_D_MODEL for config in by_vocab.values()):
                raise ValueError(
                    f"all fixed-total topologies must use d_model={RATIFIED_TARGET_D_MODEL}"
                )
            if any(
                (
                    config.n_prelude_layers,
                    config.n_core_blocks,
                    config.n_coda_layers,
                )
                not in REGISTERED_TARGET_BLOCK_SPLITS
                or (config.n_heads, config.n_kv_heads, config.d_ff) != (16, 8, 2_816)
                for config in by_vocab.values()
            ):
                raise ValueError(
                    "every fixed-total candidate must use an exact registered target topology"
                )
            if any(
                (
                    config.n_prelude_layers,
                    config.n_core_blocks,
                    config.n_coda_layers,
                )
                != (
                    self.config.n_prelude_layers,
                    self.config.n_core_blocks,
                    self.config.n_coda_layers,
                )
                for config in by_vocab.values()
            ):
                raise ValueError(
                    "fixed-total candidate topologies must share one registered target rung"
                )
            if self.reference_vocab_size not in by_vocab:
                raise ValueError("reference vocabulary must be a registered candidate")
            if self.config != by_vocab[self.reference_vocab_size]:
                raise ValueError(
                    "fixed-total reference config must equal its candidate topology"
                )
            largest_vocabulary = (
                max(TOKENIZER_VOCAB_CANDIDATES) * RATIFIED_TARGET_D_MODEL
            )
            if self.fixed_total_parameters <= largest_vocabulary:
                raise ValueError(
                    "fixed-total budget must exceed the largest candidate vocabulary matrix"
                )
            for candidate_config in by_vocab.values():
                topology_total = estimate_dense_unique_parameters(candidate_config)
                if (
                    abs(topology_total - self.fixed_total_parameters)
                    > self.fixed_total_tolerance_parameters
                ):
                    raise ValueError(
                        "fixed-total candidate topology differs from the locked common budget"
                    )
        elif self.fixed_total_parameters is not None:
            raise ValueError("fixed-total parameters apply only to fixed-total semantics")
        elif self.candidate_topologies:
            raise ValueError(
                "candidate topologies apply only to fixed-total semantics"
            )
        if (
            self.budget_semantics == "fixed_non_vocabulary"
            and self.config.vocab_size != self.reference_vocab_size
        ):
            raise ValueError(
                "fixed-non-vocabulary topology must use the reference vocabulary size"
            )


def _unique_parameters(module: nn.Module) -> dict[int, nn.Parameter]:
    return {
        id(parameter): parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    }


def parameter_accounting(model: nn.Module) -> ParameterAccounting:
    """Count tied parameters once and separate vocab/static-memory capacity."""

    all_parameters = _unique_parameters(model)
    vocabulary_ids: set[int] = set()
    for module_name, module in model.named_modules(remove_duplicate=False):
        if (
            module_name.rsplit(".", 1)[-1] == "token_embedding"
            and isinstance(module, nn.Embedding)
            and module.weight.requires_grad
        ):
            parameter_id = id(module.weight)
            if parameter_id not in all_parameters:
                raise RuntimeError("token embedding is missing from the trainable inventory")
            vocabulary_ids.add(parameter_id)

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


def _capacity_root(model: nn.Module) -> nn.Module:
    candidates = tuple(
        module
        for module in model.modules()
        if all(
            hasattr(module, name)
            for name in ("token_embedding", "core_blocks", "prelude_blocks", "coda_blocks")
        )
    )
    if len(candidates) != 1:
        raise ValueError("composition accounting requires exactly one AblationLM graph")
    root = candidates[0]
    outside = set(_unique_parameters(model)) - set(_unique_parameters(root))
    if outside:
        raise ValueError("wrapper contains trainable parameters outside the AblationLM graph")
    return root


def _trainable_parameter_ids(module: nn.Module | None) -> set[int]:
    if module is None:
        return set()
    return {id(parameter) for parameter in module.parameters() if parameter.requires_grad}


def composition_receipt(
    model: nn.Module,
    *,
    requested_visits: int,
    executed_visits: float,
    sidecar_firing_fraction_by_step: tuple[float, ...] = (),
) -> CompositionReceipt:
    """Derive the binding WEFT-1 capacity receipt without double-counting ties.

    Dynamic active-equivalent accounting fails visibly when the current legacy
    recurrent auxiliaries do not fit the final WEFT-1 ``N_fixed + K*N_recurrent``
    taxonomy.  This prevents a provisional estimate from becoming a receipt.
    """

    if type(requested_visits) is not int or requested_visits < 1:
        raise ValueError("requested_visits must be a positive integer")
    if (
        not math.isfinite(float(executed_visits))
        or not 0 < float(executed_visits) <= requested_visits
    ):
        raise ValueError("executed_visits must lie in (0, requested_visits]")
    if not isinstance(sidecar_firing_fraction_by_step, tuple):
        raise TypeError("sidecar firing fractions must be a tuple")
    if len(sidecar_firing_fraction_by_step) > requested_visits:
        raise ValueError("sidecar firing fractions cannot exceed requested visits")
    if any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in sidecar_firing_fraction_by_step
    ):
        raise ValueError("sidecar firing fractions must be finite values in [0, 1]")

    root = _capacity_root(model)
    sidecar = getattr(root, "sidecar", None)
    if sidecar is None and sidecar_firing_fraction_by_step:
        raise ValueError("sidecar firing fractions require a materialized sidecar")
    inventory = _unique_parameters(root)
    accounting = parameter_accounting(root)
    vocabulary_ids = {id(root.token_embedding.weight)}
    sparse_ids: set[int] = set()
    for module in root.modules():
        if isinstance(module, CausalTokenEngram):
            sparse_ids.update(
                id(table.weight)
                for table in module.tables.values()
                if table.weight.requires_grad
            )

    recurrent_ids: set[int] = set()
    config = getattr(root, "config", None)
    if config is not None and bool(getattr(config, "use_recurrence", False)):
        for name in (
            "core_blocks",
            "loop_embedding",
            "reentry_bridge",
            "sidecar",
            "rotor_a",
            "rotor_b",
            "carrier_write_a",
            "carrier_write_b",
            "callosum",
        ):
            recurrent_ids.update(_trainable_parameter_ids(getattr(root, name, None)))
        scratch = getattr(root, "scratch", None)
        recurrent_ids.update(_trainable_parameter_ids(scratch))
        if scratch is not None:
            recurrent_ids.difference_update(
                _trainable_parameter_ids(getattr(scratch, "initializer", None))
            )

    body_ids = set(inventory) - vocabulary_ids - sparse_ids
    recurrent_ids &= body_ids
    n_recurrent = sum(inventory[index].numel() for index in recurrent_ids)
    n_body = sum(inventory[index].numel() for index in body_ids)
    n_fixed = n_body - n_recurrent
    n_sparse = sum(inventory[index].numel() for index in sparse_ids)
    if n_body != accounting.total - accounting.vocabulary - accounting.engram_tables:
        raise RuntimeError("composition body partition disagrees with parameter accounting")
    vocabulary_denominator = accounting.vocabulary + n_body
    vocabulary_fraction = (
        accounting.vocabulary / vocabulary_denominator if vocabulary_denominator else 0.0
    )
    recurrent_fraction = n_recurrent / n_body if n_body else 0.0
    fixed_to_recurrent = n_fixed / n_recurrent if n_recurrent else None
    has_step_indexed_auxiliaries = bool(
        config is not None
        and getattr(config, "use_recurrence", False)
        and any(
            getattr(root, name, None) is not None
            for name in ("loop_embedding", "reentry_bridge", "scratch")
        )
    )
    active_eval_exact = not has_step_indexed_auxiliaries
    n_active_eval = (
        n_fixed + float(executed_visits) * n_recurrent
        if active_eval_exact
        else None
    )
    return CompositionReceipt(
        requested_visits=requested_visits,
        executed_visits=float(executed_visits),
        n_unique=accounting.total,
        n_body=n_body,
        n_fixed=n_fixed,
        n_recurrent=n_recurrent,
        n_sparse_addressed=n_sparse,
        vocabulary_parameters=accounting.vocabulary,
        vocabulary_fraction=vocabulary_fraction,
        recurrent_fraction=recurrent_fraction,
        fixed_to_recurrent=fixed_to_recurrent,
        n_active_eval=n_active_eval,
        composition_exact=active_eval_exact,
        active_eval_exact=active_eval_exact,
        sidecar_firing_fraction_by_step=tuple(
            float(value) for value in sidecar_firing_fraction_by_step
        ),
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


def _proxy_scale_accounting(config: AblationLMConfig) -> TokenizerScaleAccounting:
    if config.d_model != 512:
        raise ValueError("tokenizer execution accounting requires the d=512 muProxy graph")
    total = estimate_dense_unique_parameters(config)
    vocabulary = config.vocab_size * config.d_model
    return TokenizerScaleAccounting(
        label="muProxy_d512",
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        core_blocks=config.n_core_blocks,
        total_unique_parameters=total,
        vocabulary_parameters=vocabulary,
        authority=None,
        exact_total=True,
        budget_semantics="materialized_proxy",
        reference_vocab_size=config.vocab_size,
        reference_total_unique_parameters=total,
        topology_config=config,
        topology_unique_parameters=total,
    )


def _approximate_target_scale_accounting(
    candidate_vocab_size: int,
    core_blocks: int,
) -> TokenizerScaleAccounting:
    if core_blocks not in RATIFIED_TARGET_ROUNDED_UNIQUE_PARAMETERS_BY_CORE:
        raise ValueError("approximate target accounting requires the 4- or 6-core target rung")
    target_total = RATIFIED_TARGET_ROUNDED_UNIQUE_PARAMETERS_BY_CORE[core_blocks]
    rung = "A" if core_blocks == 4 else "B"
    return TokenizerScaleAccounting(
        label=f"target_rung_{rung}_d1024_rounded_unratified_topology",
        vocab_size=candidate_vocab_size,
        d_model=RATIFIED_TARGET_D_MODEL,
        core_blocks=core_blocks,
        total_unique_parameters=target_total,
        vocabulary_parameters=candidate_vocab_size * RATIFIED_TARGET_D_MODEL,
        authority=RATIFIED_TARGET_AUTHORITY,
        exact_total=False,
        budget_semantics=None,
        reference_vocab_size=RATIFIED_TARGET_REFERENCE_VOCAB_SIZE,
        reference_total_unique_parameters=target_total,
        topology_config=None,
        topology_unique_parameters=None,
    )


def _target_scale_accounting(
    candidate_vocab_size: int,
    contract: TokenizerTargetDecisionContract,
) -> TokenizerScaleAccounting:
    if candidate_vocab_size not in TOKENIZER_VOCAB_CANDIDATES:
        raise ValueError("tokenizer selection requires a registered vocabulary candidate")
    if contract.budget_semantics == "fixed_total":
        matching = tuple(
            config
            for config in contract.candidate_topologies
            if config.vocab_size == candidate_vocab_size
        )
        if len(matching) != 1:
            raise ValueError(
                "fixed-total authority does not bind exactly one candidate topology"
            )
        config = matching[0]
    else:
        config = contract.config
    topology_total = estimate_dense_unique_parameters(config)
    vocabulary = candidate_vocab_size * RATIFIED_TARGET_D_MODEL
    target_total = topology_total
    reference_total = topology_total
    if contract.budget_semantics == "fixed_total":
        assert contract.fixed_total_parameters is not None
        if (
            abs(topology_total - contract.fixed_total_parameters)
            > contract.fixed_total_tolerance_parameters
        ):
            raise ValueError("fixed-total target topology differs from the locked common budget")
        target_total = contract.fixed_total_parameters
        reference_total = contract.fixed_total_parameters
    else:
        non_vocabulary = (
            topology_total
            - contract.reference_vocab_size * RATIFIED_TARGET_D_MODEL
        )
        if non_vocabulary <= 0:
            raise ValueError("target reference total must exceed its vocabulary matrix")
        target_total = non_vocabulary + vocabulary
    if target_total <= vocabulary:
        raise ValueError("target total must exceed its vocabulary matrix")
    return TokenizerScaleAccounting(
        label=f"target_d1024_s0_dense_nonselection_{contract.budget_semantics}",
        vocab_size=candidate_vocab_size,
        d_model=RATIFIED_TARGET_D_MODEL,
        core_blocks=config.n_core_blocks,
        total_unique_parameters=target_total,
        vocabulary_parameters=vocabulary,
        authority=contract.authority,
        exact_total=False,
        budget_semantics=contract.budget_semantics,
        reference_vocab_size=contract.reference_vocab_size,
        reference_total_unique_parameters=reference_total,
        topology_config=config,
        topology_unique_parameters=topology_total,
    )


def tokenizer_screen_accounting(
    proxy_config: AblationLMConfig,
    *,
    target_contract: TokenizerTargetDecisionContract | None = None,
) -> TokenizerScreenAccounting:
    """Return separate muProxy execution and target-scale decision columns.

    The provisional decision column is always target rung B and is independent
    of whichever proxy arm executes.  All target rows remain non-selection S0
    accounting until the complete model body and the resolved G-TOK selector
    are represented by a fail-closed gate.
    """

    if target_contract is not None and target_contract.config.n_core_blocks != 6:
        raise ValueError(
            "the tokenizer decision contract must use rung B; one decision "
            "then serves both target rungs under R-G4h"
        )
    execution_proxy = _proxy_scale_accounting(proxy_config)
    decision_target = (
        _approximate_target_scale_accounting(
            proxy_config.vocab_size,
            6,
        )
        if target_contract is None
        else _target_scale_accounting(proxy_config.vocab_size, target_contract)
    )
    return TokenizerScreenAccounting(
        execution_proxy=execution_proxy,
        decision_target=decision_target,
        target_contract=target_contract,
    )
