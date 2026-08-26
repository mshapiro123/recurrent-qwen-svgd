"""Semantic optimizer roles for the ablation-first language model.

This module deliberately does not construct optimizers.  It produces an
auditable partition that a training layer can later use to construct a Muon
optimizer for ordinary dense hidden matrices and an auxiliary AdamW optimizer
for every other parameter family. Coupled ``mu/delta`` or factored mode paths
must remain together on AdamW until a mode-wise Muon derivation is registered.

Muon eligibility is opt-in and semantic; tensor rank alone is never enough.
In particular, embeddings, normalization parameters, gates, Engram state, and
new/experimental modules stay on auxiliary AdamW until a separate registered
optimizer ablation promotes them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from torch import nn


FULL_MATRIX_MUON_GEOMETRY: Final[str] = "full_matrix"
"""The only Muon update geometry represented by this partition contract."""

MODE_WISE_MUON_SUPPORTED: Final[bool] = False
"""There is intentionally no mode-wise or tensor-slice Muon arm."""

LEGACY_RANK_ONLY_MUON_SPLITTER_SUPPORTED: Final[bool] = False
"""``training/muon.py`` must not partition an ``AblationLM`` by tensor rank."""

REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR: Final[str] = (
    "_ablation_lm_require_closed_muon_allowlist"
)
"""Model marker that makes the semantic partition enforce the closed allowlist."""

RANK_ONLY_MUON_PROHIBITED_ATTR: Final[str] = (
    "_ablation_lm_rank_only_muon_prohibited"
)
"""Parameter marker consumed by the legacy rank-only splitter's fail-closed guard."""

_ABLATION_LM_MUON_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:prelude_blocks|core_blocks|coda_blocks)\.\d+\."
    r"(?:attention\.(?:q_proj|k_proj|v_proj|output_proj)|"
    r"feed_forward\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


class ParameterRole(str, Enum):
    """Semantic role of a trainable parameter."""

    DENSE_HIDDEN_WEIGHT = "dense_hidden_weight"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    GATE = "gate"
    ENGRAM = "engram"
    COUPLED_MODE = "coupled_mode"
    NOVEL_MODULE = "novel_module"
    BIAS = "bias"
    AUXILIARY = "auxiliary"


class OptimizerTarget(str, Enum):
    """Optimizer family to which a parameter may later be supplied."""

    MUON_ELIGIBLE = "muon_eligible"
    AUXILIARY_ADAMW = "auxiliary_adamw"


@dataclass(frozen=True)
class _RoleTag:
    role: ParameterRole
    post_muon_multiplier: float | None = None


@dataclass(frozen=True)
class ParameterAssignment:
    """Resolved semantic assignment for one unique trainable tensor."""

    canonical_name: str
    aliases: tuple[str, ...]
    parameter: nn.Parameter
    role: ParameterRole
    observed_roles: tuple[ParameterRole, ...]
    target: OptimizerTarget
    post_muon_multiplier: float | None


@dataclass(frozen=True)
class OptimizerGroupSpec:
    """Optimizer-independent metadata for a deterministic parameter group.

    ``post_muon_multiplier`` is applied *after* Muon normalization.  It is not
    a gradient multiplier and therefore remains effective under the positive
    scale invariance of ideal polar normalization.
    """

    name: str
    target: OptimizerTarget
    parameters: tuple[nn.Parameter, ...]
    parameter_names: tuple[str, ...]
    post_muon_multiplier: float | None
    update_geometry: str | None


@dataclass(frozen=True)
class OptimizerPartition:
    """Complete, duplicate-free trainable-parameter partition."""

    assignments: tuple[ParameterAssignment, ...]
    muon_groups: tuple[OptimizerGroupSpec, ...]
    auxiliary_adamw_group: OptimizerGroupSpec

    @property
    def groups(self) -> tuple[OptimizerGroupSpec, ...]:
        return (*self.muon_groups, self.auxiliary_adamw_group)

    @property
    def muon_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for group in self.muon_groups for parameter in group.parameters)

    @property
    def auxiliary_adamw_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.auxiliary_adamw_group.parameters

    def assignment_for(self, name: str) -> ParameterAssignment:
        """Return the unique assignment containing a canonical name or alias."""

        for assignment in self.assignments:
            if name == assignment.canonical_name or name in assignment.aliases:
                return assignment
        raise KeyError(name)


_ROLE_TAGS_ATTR: Final[str] = "_ablation_lm_optimizer_role_tags"

# Any auxiliary observation wins over Muon eligibility for a tied tensor.  The
# order among auxiliary roles only chooses a stable audit label; it does not
# change the optimizer target.
_ROLE_PRECEDENCE: Final[dict[ParameterRole, int]] = {
    ParameterRole.EMBEDDING: 0,
    ParameterRole.NORMALIZATION: 1,
    ParameterRole.GATE: 2,
    ParameterRole.ENGRAM: 3,
    ParameterRole.COUPLED_MODE: 4,
    ParameterRole.NOVEL_MODULE: 5,
    ParameterRole.BIAS: 6,
    ParameterRole.AUXILIARY: 7,
    ParameterRole.DENSE_HIDDEN_WEIGHT: 8,
}


def _validate_post_muon_multiplier(value: float) -> float:
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("post_muon_multiplier must be finite and positive")
    return multiplier


def tag_optimizer_role(
    module: nn.Module,
    parameter_name: str,
    role: ParameterRole,
    *,
    post_muon_multiplier: float | None = None,
) -> nn.Module:
    """Attach a semantic optimizer role to one direct module parameter.

    The tag belongs to the owning module rather than the tensor, so tied
    parameters retain every alias-specific observation.  Resolution is
    conservative: if any owner identifies a tied tensor as auxiliary, the
    tensor is assigned to AdamW exactly once.
    """

    if parameter_name not in module._parameters or module._parameters[parameter_name] is None:
        raise ValueError(f"{type(module).__name__} has no direct parameter {parameter_name!r}")
    if role is ParameterRole.DENSE_HIDDEN_WEIGHT:
        if post_muon_multiplier is None:
            post_muon_multiplier = 1.0
        post_muon_multiplier = _validate_post_muon_multiplier(post_muon_multiplier)
    elif post_muon_multiplier is not None:
        raise ValueError("post_muon_multiplier is only valid for dense hidden weights")

    tags = dict(getattr(module, _ROLE_TAGS_ATTR, {}))
    tags[parameter_name] = _RoleTag(role, post_muon_multiplier)
    setattr(module, _ROLE_TAGS_ATTR, tags)
    return module


def mark_muon_eligible(
    linear: nn.Linear,
    *,
    post_muon_multiplier: float = 1.0,
) -> nn.Linear:
    """Mark an existing dense hidden ``nn.Linear`` weight as Muon-eligible.

    The bias, when present, remains auxiliary AdamW.  This helper lets model
    construction use ordinary ``nn.Linear`` objects without subclass coupling.
    """

    if not isinstance(linear, nn.Linear):
        raise TypeError("mark_muon_eligible expects an nn.Linear module")
    tag_optimizer_role(
        linear,
        "weight",
        ParameterRole.DENSE_HIDDEN_WEIGHT,
        post_muon_multiplier=post_muon_multiplier,
    )
    if linear.bias is not None:
        tag_optimizer_role(linear, "bias", ParameterRole.BIAS)
    return linear


def mark_coupled_mode_adamw(linear: nn.Linear) -> nn.Linear:
    """Keep a dense member of a coupled mode/factor system on AdamW.

    A matrix can look ordinary in isolation while its scientific meaning is
    tied to a low-rank or differently parameterized companion. Marking every
    member explicitly prevents the ``mu`` path from drifting into Muon while a
    ``delta`` factorization remains on AdamW.
    """

    if not isinstance(linear, nn.Linear):
        raise TypeError("mark_coupled_mode_adamw expects an nn.Linear module")
    tag_optimizer_role(linear, "weight", ParameterRole.COUPLED_MODE)
    if linear.bias is not None:
        tag_optimizer_role(linear, "bias", ParameterRole.BIAS)
    return linear


class DenseHiddenLinear(nn.Linear):
    """Convenience ``Linear`` whose weight opts into full-matrix Muon."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        post_muon_multiplier: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        mark_muon_eligible(self, post_muon_multiplier=post_muon_multiplier)


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


def _role_for_owner(module: nn.Module, local_name: str, parameter: nn.Parameter) -> _RoleTag:
    # Structural safety roles override explicit eligibility on another alias of
    # a tied tensor (most importantly a tied embedding / language-model head).
    if isinstance(module, nn.Embedding):
        return _RoleTag(ParameterRole.EMBEDDING)
    if isinstance(module, _NORMALIZATION_TYPES):
        return _RoleTag(ParameterRole.NORMALIZATION)

    explicit_tags: dict[str, _RoleTag] = getattr(module, _ROLE_TAGS_ATTR, {})
    if local_name in explicit_tags:
        tag = explicit_tags[local_name]
        if tag.role is ParameterRole.DENSE_HIDDEN_WEIGHT and parameter.ndim != 2:
            raise ValueError(
                f"Muon-eligible dense hidden parameter {local_name!r} must be 2D, got {parameter.ndim}D"
            )
        return tag

    if local_name == "bias":
        return _RoleTag(ParameterRole.BIAS)
    # Safe default: an unregistered matrix is a new optimizer hypothesis, not
    # evidence that rank alone makes it a suitable hidden Linear operator.
    return _RoleTag(ParameterRole.NOVEL_MODULE if parameter.ndim >= 2 else ParameterRole.AUXILIARY)


def _resolved_role(tags: list[_RoleTag]) -> ParameterRole:
    return min((tag.role for tag in tags), key=_ROLE_PRECEDENCE.__getitem__)


def _enforce_ablation_lm_muon_allowlist(
    assignments: tuple[ParameterAssignment, ...],
    model_prefixes: tuple[str, ...],
) -> None:
    """Require the exact Transformer-only Muon inventory for ``AblationLM``.

    This closes the opt-in tagging gap: adding ``mark_muon_eligible`` to a new
    module cannot silently create a new optimizer hypothesis, and removing a
    tag from an ordinary Transformer matrix cannot silently alter the control.
    """

    def allowed_alias(alias: str) -> bool:
        for prefix in model_prefixes:
            if prefix:
                prefix_with_separator = f"{prefix}."
                if not alias.startswith(prefix_with_separator):
                    continue
                relative_name = alias[len(prefix_with_separator) :]
            else:
                relative_name = alias
            if _ABLATION_LM_MUON_NAME_PATTERN.fullmatch(relative_name):
                return True
        return False

    expected = {
        assignment.canonical_name
        for assignment in assignments
        if any(allowed_alias(alias) for alias in assignment.aliases)
    }
    actual = {
        assignment.canonical_name
        for assignment in assignments
        if assignment.target is OptimizerTarget.MUON_ELIGIBLE
    }
    if actual != expected:
        unexpected = tuple(sorted(actual - expected))
        missing = tuple(sorted(expected - actual))
        raise RuntimeError(
            "AblationLM closed Muon allowlist differs: "
            f"unexpected={unexpected!r}, missing={missing!r}"
        )


def partition_optimizer_parameters(model: nn.Module) -> OptimizerPartition:
    """Build a deterministic semantic partition of all trainable tensors.

    Each unique trainable ``Parameter`` appears in exactly one returned group.
    Aliases from weight tying are retained for audit, and any auxiliary alias
    conservatively routes the shared tensor to AdamW.
    """

    observations: dict[int, dict[str, object]] = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for local_name, parameter in module._parameters.items():
            if parameter is None or not parameter.requires_grad:
                continue
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            record = observations.setdefault(
                id(parameter),
                {"parameter": parameter, "aliases": set(), "tags": []},
            )
            record["aliases"].add(full_name)  # type: ignore[union-attr]
            record["tags"].append(_role_for_owner(module, local_name, parameter))  # type: ignore[union-attr]

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(observations) != expected:
        raise RuntimeError("semantic optimizer partition failed to observe every trainable parameter")

    assignments: list[ParameterAssignment] = []
    for record in observations.values():
        parameter = record["parameter"]
        aliases = tuple(sorted(record["aliases"]))  # type: ignore[arg-type]
        tags = record["tags"]  # type: ignore[assignment]
        role = _resolved_role(tags)
        observed_roles = tuple(sorted({tag.role for tag in tags}, key=lambda item: item.value))
        target = (
            OptimizerTarget.MUON_ELIGIBLE
            if role is ParameterRole.DENSE_HIDDEN_WEIGHT
            else OptimizerTarget.AUXILIARY_ADAMW
        )
        multiplier: float | None = None
        if target is OptimizerTarget.MUON_ELIGIBLE:
            multipliers = {tag.post_muon_multiplier for tag in tags if tag.role is role}
            if len(multipliers) != 1 or None in multipliers:
                raise ValueError(
                    f"tied Muon-eligible parameter {aliases!r} has conflicting post-Muon multipliers"
                )
            multiplier = next(iter(multipliers))
        assignments.append(
            ParameterAssignment(
                canonical_name=aliases[0],
                aliases=aliases,
                parameter=parameter,  # type: ignore[arg-type]
                role=role,
                observed_roles=observed_roles,
                target=target,
                post_muon_multiplier=multiplier,
            )
        )
    assignments.sort(key=lambda item: item.canonical_name)

    by_multiplier: dict[float, list[ParameterAssignment]] = {}
    auxiliary: list[ParameterAssignment] = []
    for assignment in assignments:
        if assignment.target is OptimizerTarget.MUON_ELIGIBLE:
            assert assignment.post_muon_multiplier is not None
            by_multiplier.setdefault(assignment.post_muon_multiplier, []).append(assignment)
        else:
            auxiliary.append(assignment)

    muon_groups = tuple(
        OptimizerGroupSpec(
            name=f"muon_dense_hidden_post_x{multiplier:g}",
            target=OptimizerTarget.MUON_ELIGIBLE,
            parameters=tuple(item.parameter for item in by_multiplier[multiplier]),
            parameter_names=tuple(item.canonical_name for item in by_multiplier[multiplier]),
            post_muon_multiplier=multiplier,
            update_geometry=FULL_MATRIX_MUON_GEOMETRY,
        )
        for multiplier in sorted(by_multiplier)
    )
    auxiliary_group = OptimizerGroupSpec(
        name="auxiliary_adamw",
        target=OptimizerTarget.AUXILIARY_ADAMW,
        parameters=tuple(item.parameter for item in auxiliary),
        parameter_names=tuple(item.canonical_name for item in auxiliary),
        post_muon_multiplier=None,
        update_geometry=None,
    )
    partition = OptimizerPartition(tuple(assignments), muon_groups, auxiliary_group)
    closed_model_prefixes = tuple(
        module_name
        for module_name, module in model.named_modules(remove_duplicate=False)
        if bool(getattr(module, REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR, False))
    )
    if closed_model_prefixes:
        _enforce_ablation_lm_muon_allowlist(
            partition.assignments,
            closed_model_prefixes,
        )
    return partition


__all__ = [
    "DenseHiddenLinear",
    "FULL_MATRIX_MUON_GEOMETRY",
    "LEGACY_RANK_ONLY_MUON_SPLITTER_SUPPORTED",
    "MODE_WISE_MUON_SUPPORTED",
    "OptimizerGroupSpec",
    "OptimizerPartition",
    "OptimizerTarget",
    "ParameterAssignment",
    "ParameterRole",
    "REQUIRE_CLOSED_MUON_ALLOWLIST_ATTR",
    "RANK_ONLY_MUON_PROHIBITED_ATTR",
    "mark_muon_eligible",
    "mark_coupled_mode_adamw",
    "partition_optimizer_parameters",
    "tag_optimizer_role",
]
