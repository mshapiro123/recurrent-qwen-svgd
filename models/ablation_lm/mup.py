"""Closed PF-3 muP parameterization for the WEFT-1 pre-flight graph.

This module is deliberately separate from the production optimizer partition.
PF-3 authorizes provisional AdamW numerics for the C1 coordinate check; it does
not replace the later S2 calibration or authorize a production optimizer.

Every trainable tensor must match one explicit semantic rule.  An unknown
tensor raises :class:`MuPClassificationError` before initialization or optimizer
construction, preserving PF-3.1's catch clause.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Final

import torch
from torch import nn

from .rng import derive_module_seed


MUP_BASE_WIDTH: Final[int] = 512
MUP_BASE_HEAD_DIM: Final[int] = 64
MUP_BASE_Q_HEADS: Final[int] = 8
MUP_BASE_KV_HEADS: Final[int] = 4
MUP_BASE_FFN_WIDTH: Final[int] = 1_408
MUP_BASE_LANES: Final[int] = 2
MUP_BASE_LANE_WIDTH: Final[int] = 128
MUP_BASE_VOCAB_SIZE: Final[int] = 32_768

MUP_SIGMA_BASE: Final[float] = 1.0
MUP_SIGMA_EMBEDDING: Final[float] = 0.02
MUP_ETA_BASE: Final[float] = 3.0e-4
MUP_ALPHA_OUTPUT: Final[float] = 1.0
MUP_ALPHA_EMBEDDING: Final[float] = 1.0
MUP_RESIDUAL_MULTIPLIER: Final[float] = 1.0
MUP_WEIGHT_DECAY: Final[float] = 0.1
MUP_BETAS: Final[tuple[float, float]] = (0.9, 0.95)
MUP_EPSILON: Final[float] = 1.0e-8

MUP_NUMERICS_STATUS: Final[str] = "PROVISIONAL-PF_owned_by_S2_calibration"
MUP_DECAY_IMPLEMENTATION: Final[str] = (
    "torch_AdamW_hidden_weight_decay_equals_base_wd_times_width_multiplier"
)


class MuPClassificationError(RuntimeError):
    """Raised before mutation when a trainable tensor lacks a PF-3 class."""


class MuPParameterClass(str, Enum):
    """The three stored-tensor classes in PF-3.1.

    The output/readout rule is a forward multiplier on the tied input tensor,
    not a fourth stored-parameter class.
    """

    HIDDEN = "hidden"
    INPUT = "input"
    VECTOR = "vector"


@dataclass(frozen=True)
class MuPParameterAssignment:
    """One unique trainable tensor's exhaustive PF-3 assignment."""

    canonical_name: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    parameter_class: MuPParameterClass
    initialization_rule: str
    learning_rate: float
    weight_decay: float
    per_step_decay_product: float


@dataclass(frozen=True)
class MuPParameterization:
    """Complete parameter map and provisional optimizer constants."""

    width: int
    width_multiplier: float
    assignments: tuple[MuPParameterAssignment, ...]
    output_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    recurrence_residual_rule: str
    numerics_status: str
    decay_implementation: str

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item.canonical_name for item in self.assignments)

    def assignment_for(self, name: str) -> MuPParameterAssignment:
        for item in self.assignments:
            if name == item.canonical_name or name in item.aliases:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class MuPClassificationIssue:
    """One exact tensor rejected by PF-3.1's catch clause."""

    canonical_name: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class MuPClassificationAudit:
    """Non-mutating exhaustive inventory, including typed non-passes."""

    width: int
    width_multiplier: float
    assignments: tuple[MuPParameterAssignment, ...]
    issues: tuple[MuPClassificationIssue, ...]


@dataclass(frozen=True)
class _Owner:
    alias: str
    module: nn.Module
    local_name: str


_TRANSFORMER_HIDDEN_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:prelude_blocks|core_blocks|coda_blocks)\.\d+\."
    r"(?:attention\.(?:q_proj|k_proj|v_proj|output_proj)|"
    r"feed_forward\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
_BICAMERAL_SWAP_RE: Final[re.Pattern[str]] = re.compile(
    r"^core_blocks\.\d+\."
    r"(?P<projection>q_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"(?P<component>mu|dU|dV)$"
)
_BICAMERAL_SHARED_KV_RE: Final[re.Pattern[str]] = re.compile(
    r"^core_blocks\.\d+\.(?P<projection>k_proj|v_proj)\.weight$"
)
_BICAMERAL_COMBINER_THETA: Final[str] = "bicameral_combiner.theta"
_EXPLICIT_HIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "engram.query_proj.weight",
        "reentry_bridge.projection.weight",
        "scratch.initializer.weight",
        "scratch.context_projection.weight",
        "scratch.update_in.weight",
        "scratch.update_out.weight",
        "scratch.readout.weight",
        "long_term_memory.query.weight",
        "long_term_memory.output.weight",
    }
)
_FIXED_FAN_IN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "engram.value_proj.weight",
    }
)
_EXPLICIT_VECTOR_NAMES: Final[frozenset[str]] = frozenset(
    {
        # The gain rows are explicitly vector-valued parameters.  The router
        # weight is intentionally absent: shape E x d has scaling fan-in and
        # fixed fan-out, so PF-3 assigns it to none of its four classes.
        "front_hadamard.expert_gains",
        "front_hadamard.layer_scale",
        "engram.gate_bias",
        "engram.raw_residual_scale",
        "reentry_bridge.layer_scale",
        "scratch.layer_scale",
        "scratch.carrier.raw_rho",
        "long_term_memory.layer_scale",
    }
)


def _normalization_types() -> tuple[type[nn.Module], ...]:
    candidates: list[type[nn.Module]] = [
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ]
    rms_norm = getattr(nn, "RMSNorm", None)
    if isinstance(rms_norm, type):
        candidates.append(rms_norm)
    return tuple(candidates)


_NORMALIZATION_TYPES: Final[tuple[type[nn.Module], ...]] = _normalization_types()


def width_multiplier(width: int) -> float:
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive exact integer")
    return width / MUP_BASE_WIDTH


def output_multiplier(width: int) -> float:
    """Return PF-3's tied-readout multiplier ``alpha_out / m``."""

    return MUP_ALPHA_OUTPUT / width_multiplier(width)


def _parameter_owners(model: nn.Module) -> dict[int, tuple[nn.Parameter, tuple[_Owner, ...]]]:
    records: dict[int, dict[str, object]] = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for local_name, parameter in module._parameters.items():
            if parameter is None or not parameter.requires_grad:
                continue
            alias = f"{module_name}.{local_name}" if module_name else local_name
            record = records.setdefault(
                id(parameter),
                {"parameter": parameter, "owners": []},
            )
            record["owners"].append(_Owner(alias, module, local_name))  # type: ignore[union-attr]

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(records) != expected:
        raise MuPClassificationError(
            "PF-3 classifier did not observe every unique trainable tensor"
        )
    return {
        identity: (
            record["parameter"],  # type: ignore[dict-item]
            tuple(record["owners"]),  # type: ignore[arg-type]
        )
        for identity, record in records.items()
    }


def _canonical_name(aliases: tuple[str, ...]) -> str:
    if "token_embedding.weight" in aliases:
        return "token_embedding.weight"
    return min(aliases)


def _is_normalization(module: nn.Module) -> bool:
    # ``models.ablation_lm.layers.RMSNorm`` is intentionally identified by its
    # semantic class name so this closed map does not import layers and create
    # an optimizer/layer import cycle.
    return isinstance(module, _NORMALIZATION_TYPES) or type(module).__name__ == "RMSNorm"


def _all_bicameral_component(
    owners: tuple[_Owner, ...],
    *,
    component: str,
) -> bool:
    return bool(owners) and all(
        type(owner.module).__name__ == "SwapLinear"
        and owner.local_name == component
        and (match := _BICAMERAL_SWAP_RE.fullmatch(owner.alias)) is not None
        and match.group("component") == component
        for owner in owners
    )


def _all_bicameral_shared_kv(owners: tuple[_Owner, ...]) -> bool:
    return bool(owners) and all(
        isinstance(owner.module, nn.Linear)
        and owner.local_name == "weight"
        and _BICAMERAL_SHARED_KV_RE.fullmatch(owner.alias) is not None
        for owner in owners
    )


def _classify(owners: tuple[_Owner, ...]) -> MuPParameterClass:
    aliases = tuple(sorted(owner.alias for owner in owners))
    alias_set = set(aliases)

    # The shared readout tensor is stored and optimized once as the input
    # embedding.  The output rule is realized only by ``output_multiplier``.
    if "token_embedding.weight" in alias_set:
        if "lm_head.weight" not in alias_set:
            raise MuPClassificationError(
                "PF-3 tied input tensor is missing the lm_head.weight alias"
            )
        return MuPParameterClass.INPUT

    if any(isinstance(owner.module, nn.Embedding) for owner in owners):
        if not all(
            isinstance(owner.module, nn.Embedding)
            for owner in owners
        ):
            raise MuPClassificationError(
                f"embedding aliases conflict with another tensor role: {aliases!r}"
            )
        return MuPParameterClass.INPUT

    if all(_is_normalization(owner.module) for owner in owners):
        return MuPParameterClass.VECTOR

    if all(owner.local_name == "bias" for owner in owners):
        return MuPParameterClass.VECTOR

    if alias_set and alias_set <= _EXPLICIT_VECTOR_NAMES:
        return MuPParameterClass.VECTOR

    if alias_set == {_BICAMERAL_COMBINER_THETA}:
        owner = owners[0]
        if (
            len(owners) != 1
            or type(owner.module).__name__ != "PerBandUnitCircleCombiner"
            or owner.local_name != "theta"
        ):
            raise MuPClassificationError(
                f"bicameral combiner theta alias has the wrong owner: {aliases!r}"
            )
        return MuPParameterClass.VECTOR

    if _all_bicameral_component(owners, component="dV"):
        raise MuPClassificationError(
            "PF-3 catch clause: SwapLinear dV has width-scaling fan-in and "
            "fixed-rank fan-out; PF-3.1 binds no stored parameter class for "
            f"this shape: {aliases!r}"
        )

    if _all_bicameral_component(owners, component="dU"):
        # dU maps from the fixed disagreement rank to a width-scaled output.
        # It therefore follows PF-3.1's width-independent-fan-in input rule.
        return MuPParameterClass.INPUT

    if alias_set and alias_set <= _FIXED_FAN_IN_NAMES:
        if not all(isinstance(owner.module, nn.Linear) for owner in owners):
            raise MuPClassificationError(
                f"fixed-fan-in aliases are not Linear weights: {aliases!r}"
            )
        return MuPParameterClass.INPUT

    if alias_set and all(
        alias in _EXPLICIT_HIDDEN_NAMES
        or _TRANSFORMER_HIDDEN_RE.fullmatch(alias)
        or (
            (match := _BICAMERAL_SWAP_RE.fullmatch(alias)) is not None
            and match.group("component") == "mu"
        )
        or _BICAMERAL_SHARED_KV_RE.fullmatch(alias)
        for alias in alias_set
    ):
        if not all(
            (
                isinstance(owner.module, nn.Linear)
                and owner.local_name == "weight"
            )
            or (
                type(owner.module).__name__ == "SwapLinear"
                and owner.local_name == "mu"
            )
            for owner in owners
        ):
            raise MuPClassificationError(
                f"hidden aliases are not Linear weights: {aliases!r}"
            )
        return MuPParameterClass.HIDDEN

    raise MuPClassificationError(
        "PF-3 catch clause: unclassifiable trainable tensor aliases="
        f"{aliases!r}, shape={tuple(owners[0].module._parameters[owners[0].local_name].shape)!r}"
    )


def _validate_bound_shape(
    assignment_class: MuPParameterClass,
    *,
    aliases: tuple[str, ...],
    parameter: nn.Parameter,
    width: int,
    owners: tuple[_Owner, ...],
) -> None:
    """Prove that each mapped matrix has the scaling property its class claims."""

    shape = tuple(parameter.shape)
    alias_set = set(aliases)
    if "token_embedding.weight" in alias_set:
        if shape != (MUP_BASE_VOCAB_SIZE, width):
            raise MuPClassificationError(
                f"PF-3 base vocabulary/tied embedding shape mismatch: {aliases!r}, {shape!r}"
            )
        return

    if assignment_class is MuPParameterClass.VECTOR:
        if "front_hadamard.expert_gains" in alias_set and shape != (8, width):
            raise MuPClassificationError(
                f"Hadamard gain bank must be eight width-scaled vectors: {shape!r}"
            )
        return

    if assignment_class is MuPParameterClass.INPUT:
        if _all_bicameral_component(owners, component="dU"):
            owner = owners[0]
            projection = _BICAMERAL_SWAP_RE.fullmatch(owner.alias)
            assert projection is not None
            expected_output = {
                "q_proj": width,
                "o_proj": width,
                "gate_proj": 11 * width // 4,
                "up_proj": 11 * width // 4,
                "down_proj": width,
            }[projection.group("projection")]
            expected = (expected_output, owner.module.rank)  # type: ignore[attr-defined]
            if shape != expected:
                raise MuPClassificationError(
                    "SwapLinear dU no longer maps from its fixed rank to the "
                    f"bound width-scaled output: {aliases!r}, shape={shape!r}, "
                    f"expected={expected!r}"
                )
            return
        if alias_set <= _FIXED_FAN_IN_NAMES:
            # The regular engram has two orders, four hash heads and row width
            # eight: a fixed 64-coordinate address independent of model width.
            if shape != (width, 64):
                raise MuPClassificationError(
                    "engram fixed-fan-in projection no longer has the bound "
                    f"64-coordinate input: {aliases!r}, shape={shape!r}"
                )
        return

    assert assignment_class is MuPParameterClass.HIDDEN
    alias = aliases[0]
    expected: tuple[int, int] | None = None
    bicameral_swap_match = _BICAMERAL_SWAP_RE.fullmatch(alias)
    bicameral_kv_match = _BICAMERAL_SHARED_KV_RE.fullmatch(alias)
    if bicameral_swap_match:
        projection = bicameral_swap_match.group("projection")
        if bicameral_swap_match.group("component") != "mu":
            raise MuPClassificationError(
                f"only SwapLinear mu may enter the hidden class: {aliases!r}"
            )
        expected = {
            "q_proj": (width, width),
            "o_proj": (width, width),
            "gate_proj": (11 * width // 4, width),
            "up_proj": (11 * width // 4, width),
            "down_proj": (width, 11 * width // 4),
        }[projection]
    elif bicameral_kv_match:
        expected = (width // 2, width)
    transformer_match = re.fullmatch(
        r"(?:prelude_blocks|core_blocks|coda_blocks)\.\d+\."
        r"(?:attention\.(q_proj|k_proj|v_proj|output_proj)|"
        r"feed_forward\.(gate_proj|up_proj|down_proj))\.weight",
        alias,
    )
    if transformer_match and expected is None:
        projection = transformer_match.group(1) or transformer_match.group(2)
        expected = {
            "q_proj": (width, width),
            "k_proj": (width // 2, width),
            "v_proj": (width // 2, width),
            "output_proj": (width, width),
            "gate_proj": (11 * width // 4, width),
            "up_proj": (11 * width // 4, width),
            "down_proj": (width, 11 * width // 4),
        }[projection]
    elif expected is None:
        expected = {
            # PF-3 keeps the memory-space query in the hidden class even
            # though its fixed 64-coordinate output does not widen with d.
            "engram.query_proj.weight": (64, width),
            "reentry_bridge.projection.weight": (width, width),
            "scratch.initializer.weight": (width // 2, width),
            "scratch.context_projection.weight": (width // 4, width),
            "scratch.update_in.weight": (width // 2, width // 2),
            "scratch.update_out.weight": (width // 4, width // 2),
            "scratch.readout.weight": (width, width // 2),
            # LTM width is the base configuration's 128 scaled by m=d/512.
            "long_term_memory.query.weight": (width // 4, width),
            "long_term_memory.output.weight": (width, width // 4),
        }.get(alias)
    if expected is None or shape != expected:
        if alias == "engram.query_proj.weight":
            raise MuPClassificationError(
                "hidden tensor does not match the authorized fixed-output "
                f"memory-query rule: {aliases!r}, shape={shape!r}, expected={expected!r}"
            )
        raise MuPClassificationError(
            "hidden tensor does not prove both fan-in and fan-out scale with "
            f"{aliases!r}, shape={shape!r}, expected={expected!r}"
        )


def audit_mup_parameters(model: nn.Module, *, width: int) -> MuPClassificationAudit:
    """Inventory every tensor without mutating, retaining every catch."""

    multiplier = width_multiplier(width)
    items: list[MuPParameterAssignment] = []
    issues: list[MuPClassificationIssue] = []
    for parameter, owners in _parameter_owners(model).values():
        aliases = tuple(sorted(owner.alias for owner in owners))
        canonical_name = _canonical_name(aliases)
        try:
            parameter_class = _classify(owners)
            _validate_bound_shape(
                parameter_class,
                aliases=aliases,
                parameter=parameter,
                width=width,
                owners=owners,
            )
        except MuPClassificationError as error:
            issues.append(
                MuPClassificationIssue(
                    canonical_name=canonical_name,
                    aliases=aliases,
                    shape=tuple(parameter.shape),
                    reason=str(error),
                )
            )
            continue
        if parameter_class is MuPParameterClass.HIDDEN:
            if parameter.ndim != 2:
                raise MuPClassificationError(
                    f"hidden tensor must be a matrix: {aliases!r}, shape={tuple(parameter.shape)!r}"
                )
            learning_rate = MUP_ETA_BASE / multiplier
            weight_decay = MUP_WEIGHT_DECAY * multiplier
            init_rule = "normal(mean=0,std=sigma_base/sqrt(fan_in))"
        elif parameter_class is MuPParameterClass.INPUT:
            learning_rate = MUP_ETA_BASE
            weight_decay = 0.0
            init_rule = "normal(mean=0,std=sigma_emb)"
        else:
            learning_rate = MUP_ETA_BASE
            weight_decay = 0.0
            init_rule = "module_design_state_preserved"
        items.append(
            MuPParameterAssignment(
                canonical_name=canonical_name,
                aliases=aliases,
                shape=tuple(parameter.shape),
                parameter_class=parameter_class,
                initialization_rule=init_rule,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                per_step_decay_product=learning_rate * weight_decay,
            )
        )
    items.sort(key=lambda item: item.canonical_name)
    issues.sort(key=lambda item: item.canonical_name)
    if len({item.canonical_name for item in items}) != len(items):
        raise MuPClassificationError("PF-3 canonical parameter names are not unique")
    return MuPClassificationAudit(
        width=width,
        width_multiplier=multiplier,
        assignments=tuple(items),
        issues=tuple(issues),
    )


def classify_mup_parameters(model: nn.Module, *, width: int) -> MuPParameterization:
    """Return the complete, duplicate-free PF-3 map or fail closed."""

    audit = audit_mup_parameters(model, width=width)
    if audit.issues:
        details = "; ".join(
            f"{item.canonical_name} shape={item.shape}: {item.reason}"
            for item in audit.issues
        )
        raise MuPClassificationError(f"PF-3 unclassified tensors: {details}")
    return MuPParameterization(
        width=width,
        width_multiplier=audit.width_multiplier,
        assignments=audit.assignments,
        output_multiplier=output_multiplier(width),
        embedding_multiplier=MUP_ALPHA_EMBEDDING,
        residual_multiplier=MUP_RESIDUAL_MULTIPLIER,
        recurrence_residual_rule="alpha_T=c/T_only_no_additional_muP_alpha",
        numerics_status=MUP_NUMERICS_STATUS,
        decay_implementation=MUP_DECAY_IMPLEMENTATION,
    )


def initialize_mup_parameters(
    model: nn.Module,
    *,
    width: int,
    base_seed: int,
) -> MuPParameterization:
    """Apply PF-3 initialization with one isolated O-9 stream per tensor."""

    if type(base_seed) is not int:
        raise TypeError("base_seed must be an exact integer")
    parameterization = classify_mup_parameters(model, width=width)
    by_identity = _parameter_owners(model)
    by_canonical = {
        _canonical_name(tuple(sorted(owner.alias for owner in owners))): parameter
        for parameter, owners in by_identity.values()
    }
    with torch.no_grad():
        for assignment in parameterization.assignments:
            if assignment.parameter_class is MuPParameterClass.VECTOR:
                continue
            parameter = by_canonical[assignment.canonical_name]
            std = (
                MUP_SIGMA_BASE / math.sqrt(parameter.shape[-1])
                if assignment.parameter_class is MuPParameterClass.HIDDEN
                else MUP_SIGMA_EMBEDDING
            )
            generator = torch.Generator(device="cpu").manual_seed(
                derive_module_seed(
                    base_seed,
                    f"preflight.c1.mup.{assignment.canonical_name}.initialization",
                    0,
                )
            )
            values = torch.randn(
                tuple(parameter.shape),
                dtype=torch.float32,
                generator=generator,
            ) * std
            parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))
    return parameterization


def build_provisional_mup_adamw(
    model: nn.Module,
    *,
    width: int,
) -> tuple[torch.optim.AdamW, MuPParameterization]:
    """Build the PF-3 C1-only AdamW groups after exhaustive classification."""

    parameterization = classify_mup_parameters(model, width=width)
    owners = _parameter_owners(model)
    by_canonical = {
        _canonical_name(tuple(sorted(owner.alias for owner in item_owners))): parameter
        for parameter, item_owners in owners.values()
    }
    groups: list[dict[str, object]] = []
    for parameter_class in MuPParameterClass:
        assignments = tuple(
            item
            for item in parameterization.assignments
            if item.parameter_class is parameter_class
        )
        if not assignments:
            continue
        learning_rates = {item.learning_rate for item in assignments}
        weight_decays = {item.weight_decay for item in assignments}
        if len(learning_rates) != 1 or len(weight_decays) != 1:
            raise RuntimeError(f"PF-3 {parameter_class.value} group is internally inconsistent")
        groups.append(
            {
                "params": [by_canonical[item.canonical_name] for item in assignments],
                "lr": next(iter(learning_rates)),
                "weight_decay": next(iter(weight_decays)),
                "mup_class": parameter_class.value,
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        lr=MUP_ETA_BASE,
        betas=MUP_BETAS,
        eps=MUP_EPSILON,
        weight_decay=0.0,
        foreach=False,
    )
    unique_optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if unique_optimizer_parameters != expected:
        raise RuntimeError("PF-3 optimizer did not consume every tensor exactly once")
    return optimizer, parameterization


def scale_tied_readout(logits: torch.Tensor, *, width: int) -> torch.Tensor:
    """Apply PF-3's output/readout multiplier without creating another tensor."""

    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating tensor")
    return logits * output_multiplier(width)


__all__ = [
    "MUP_ALPHA_EMBEDDING",
    "MUP_ALPHA_OUTPUT",
    "MUP_BASE_FFN_WIDTH",
    "MUP_BASE_HEAD_DIM",
    "MUP_BASE_KV_HEADS",
    "MUP_BASE_LANES",
    "MUP_BASE_LANE_WIDTH",
    "MUP_BASE_Q_HEADS",
    "MUP_BASE_VOCAB_SIZE",
    "MUP_BASE_WIDTH",
    "MUP_BETAS",
    "MUP_DECAY_IMPLEMENTATION",
    "MUP_EPSILON",
    "MUP_ETA_BASE",
    "MUP_NUMERICS_STATUS",
    "MUP_RESIDUAL_MULTIPLIER",
    "MUP_SIGMA_BASE",
    "MUP_SIGMA_EMBEDDING",
    "MUP_WEIGHT_DECAY",
    "MuPClassificationError",
    "MuPClassificationAudit",
    "MuPClassificationIssue",
    "MuPParameterAssignment",
    "MuPParameterClass",
    "MuPParameterization",
    "build_provisional_mup_adamw",
    "audit_mup_parameters",
    "classify_mup_parameters",
    "initialize_mup_parameters",
    "output_multiplier",
    "scale_tied_readout",
    "width_multiplier",
]
