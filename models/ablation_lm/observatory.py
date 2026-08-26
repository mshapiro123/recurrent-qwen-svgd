"""Fail-closed ordering for WEFT-1 observatory statistics.

T14b is an ordering gate, not a diagnostic statistic.  A graph-bound causality
receipt must pass before any downstream accumulator is allowed to observe model
state.  Keeping this gate free of statistic implementations makes the unsafe
ordering (measure first, validate later) unavailable through this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Callable, ParamSpec, TypeVar

from .config import RATIFIED_TARGET_AUTHORITY_SHA256


_RECEIPT_KIND = "weft1.t14b.sequence_axis_causality.v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
P = ParamSpec("P")
R = TypeVar("R")


class ObservatoryBlocked(RuntimeError):
    """Raised before observatory code can run without exact T14b authority."""


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_k_values(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("tested_k_values must be an exact tuple")
    if not value:
        raise ValueError("tested_k_values must not be empty")
    if any(type(item) is not int for item in value):
        raise TypeError("every tested K value must be an exact int")
    if any(item < 1 for item in value):
        raise ValueError("every tested K value must be positive")
    if tuple(sorted(set(value))) != value:
        raise ValueError("tested K values must be unique and strictly increasing")
    return value


def _require_stage_names(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("sequence_axis_stages must be an exact tuple")
    if not value:
        raise ValueError("sequence_axis_stages must not be empty")
    if any(type(item) is not str for item in value):
        raise TypeError("every sequence-axis stage name must be an exact str")
    if any(not item or item != item.strip() for item in value):
        raise ValueError("sequence-axis stage names must be non-empty and normalized")
    if len(set(value)) != len(value):
        raise ValueError("sequence-axis stage names must be unique")
    return value


def _canonical_payload(receipt: "T14bReceipt") -> bytes:
    payload = {
        "authority_sha256": receipt.authority_sha256,
        "config_fingerprint": receipt.config_fingerprint,
        "graph_fingerprint": receipt.graph_fingerprint,
        "max_future_gradient": receipt.max_future_gradient,
        "packed": receipt.packed,
        "padded": receipt.padded,
        "passed": receipt.passed,
        "receipt_kind": _RECEIPT_KIND,
        "sequence_axis_stages": list(receipt.sequence_axis_stages),
        "tested_k_values": list(receipt.tested_k_values),
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(receipt: "T14bReceipt") -> str:
    return hashlib.sha256(_canonical_payload(receipt)).hexdigest()


@dataclass(frozen=True, slots=True)
class T14bReceipt:
    """Immutable causality receipt bound to one graph and configuration.

    ``passed=True`` is representable only when the maximum observed gradient
    from every future position into every tested earlier position is exactly
    zero.  Coverage itself is policy-bound by :class:`ObservatoryGuard`; this
    prevents a locally valid receipt for a smaller graph or K panel from being
    reused for a larger observation run.
    """

    authority_sha256: str
    graph_fingerprint: str
    config_fingerprint: str
    tested_k_values: tuple[int, ...]
    sequence_axis_stages: tuple[str, ...]
    packed: bool
    padded: bool
    max_future_gradient: float
    passed: bool
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256("authority_sha256", self.authority_sha256)
        if self.authority_sha256 != RATIFIED_TARGET_AUTHORITY_SHA256:
            raise ValueError("T14b authority must equal the WEFT-1 ratification SHA")
        _require_sha256("graph_fingerprint", self.graph_fingerprint)
        _require_sha256("config_fingerprint", self.config_fingerprint)
        _require_k_values(self.tested_k_values)
        _require_stage_names(self.sequence_axis_stages)
        if type(self.packed) is not bool or type(self.padded) is not bool:
            raise TypeError("packed and padded coverage flags must be exact bool values")
        if type(self.max_future_gradient) is not float:
            raise TypeError("max_future_gradient must be an exact float")
        if not math.isfinite(self.max_future_gradient) or self.max_future_gradient < 0.0:
            raise ValueError("max_future_gradient must be finite and non-negative")
        if type(self.passed) is not bool:
            raise TypeError("passed must be an exact bool")
        if self.passed and self.max_future_gradient != 0.0:
            raise ValueError("a passing T14b receipt requires exact zero future gradient")
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(self))

    def canonical_payload(self) -> bytes:
        """Return the byte-exact payload covered by ``receipt_sha256``."""

        return _canonical_payload(self)


@dataclass(frozen=True, slots=True)
class ObservatoryGuard:
    """Authorize exactly one T14b-tested graph before running a callback."""

    expected_graph_fingerprint: str
    expected_config_fingerprint: str
    expected_receipt_sha256: str
    required_k_values: tuple[int, ...]
    required_sequence_axis_stages: tuple[str, ...]
    require_packed: bool = True
    require_padded: bool = True
    expected_authority_sha256: str = RATIFIED_TARGET_AUTHORITY_SHA256

    def __post_init__(self) -> None:
        _require_sha256("expected_graph_fingerprint", self.expected_graph_fingerprint)
        _require_sha256("expected_config_fingerprint", self.expected_config_fingerprint)
        _require_sha256("expected_receipt_sha256", self.expected_receipt_sha256)
        _require_sha256("expected_authority_sha256", self.expected_authority_sha256)
        if self.expected_authority_sha256 != RATIFIED_TARGET_AUTHORITY_SHA256:
            raise ValueError("observatory authority must equal the WEFT-1 ratification SHA")
        _require_k_values(self.required_k_values)
        _require_stage_names(self.required_sequence_axis_stages)
        if type(self.require_packed) is not bool or type(self.require_padded) is not bool:
            raise TypeError("required coverage flags must be exact bool values")

    def validate(self, receipt: T14bReceipt | None) -> None:
        """Fail closed unless ``receipt`` exactly matches the bound graph panel."""

        if type(receipt) is not T14bReceipt:
            raise ObservatoryBlocked("an exact T14bReceipt is required before observation")
        if receipt.receipt_sha256 != _canonical_sha256(receipt):
            raise ObservatoryBlocked("T14b receipt integrity check failed")
        if receipt.authority_sha256 != self.expected_authority_sha256:
            raise ObservatoryBlocked("T14b authority is stale")
        if not receipt.passed or receipt.max_future_gradient != 0.0:
            raise ObservatoryBlocked("T14b did not pass with exact zero future gradient")
        if receipt.graph_fingerprint != self.expected_graph_fingerprint:
            raise ObservatoryBlocked("T14b graph fingerprint is stale")
        if receipt.config_fingerprint != self.expected_config_fingerprint:
            raise ObservatoryBlocked("T14b config fingerprint is stale")
        if receipt.tested_k_values != self.required_k_values:
            raise ObservatoryBlocked("T14b K coverage is missing or stale")
        if receipt.sequence_axis_stages != self.required_sequence_axis_stages:
            raise ObservatoryBlocked("T14b sequence-axis stage coverage is missing or stale")
        if receipt.packed is not self.require_packed:
            raise ObservatoryBlocked("T14b packed-sequence coverage does not match policy")
        if receipt.padded is not self.require_padded:
            raise ObservatoryBlocked("T14b padded-sequence coverage does not match policy")
        if receipt.receipt_sha256 != self.expected_receipt_sha256:
            raise ObservatoryBlocked("T14b receipt is not the exact authorized receipt")

    def invoke(
        self,
        receipt: T14bReceipt | None,
        callback: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Validate first, then and only then invoke an observatory callback."""

        self.validate(receipt)
        if not callable(callback):
            raise TypeError("observatory callback must be callable")
        return callback(*args, **kwargs)
