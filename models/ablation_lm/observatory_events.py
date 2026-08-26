"""Versioned, causality-first event records for the WEFT-1 observatory.

The event schema records an instrument's pre-registered tier alongside every
value.  Tiers are derived from the ratified registry rather than accepted from
callers, so RESP-LEAK cannot drift out of Tier 1 and ``A_state`` cannot silently
be promoted back into it.  Event minting is available only through an
``ObservatoryGuard`` that validates the exact T14b receipt first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

from .config import RATIFIED_TARGET_AUTHORITY_SHA256
from .observatory import ObservatoryGuard, T14bReceipt


EVENT_KIND = "weft1.observatory.event.v1"
_NORMALIZED_ID = re.compile(r"[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*")
_HEX_DIGITS = frozenset("0123456789abcdef")


class InstrumentTier(str, Enum):
    """The three assertion tiers ratified by ruling O-7."""

    DECISION = "tier_1_decision"
    DIAGNOSTIC = "tier_2_diagnostic"
    EXPLORATORY = "tier_3_exploratory"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    instrument_id: str
    tier: InstrumentTier
    required_measurements: frozenset[str]
    branch_rule: str
    t14_receipt_is_record: bool = False


def _spec(
    instrument_id: str,
    tier: InstrumentTier,
    measurements: tuple[str, ...],
    branch_rule: str,
    *,
    t14_receipt_is_record: bool = False,
) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id=instrument_id,
        tier=tier,
        required_measurements=frozenset(measurements),
        branch_rule=branch_rule,
        t14_receipt_is_record=t14_receipt_is_record,
    )


# This registry is an implementation of ratification record section 3.1, not a
# menu.  In particular, Tier 1 is capped at eight entries.
INSTRUMENT_REGISTRY: Mapping[str, InstrumentSpec] = MappingProxyType({
    "t14_causality": _spec(
        "t14_causality",
        InstrumentTier.DECISION,
        ("max_future_gradient", "minimum_allowed_gradient_max"),
        "any non-zero forbidden backward gradient halts the line",
        t14_receipt_is_record=True,
    ),
    "obs_inv": _spec(
        "obs_inv",
        InstrumentTier.DECISION,
        ("bit_identical",),
        "inequality invalidates all attribution",
    ),
    "carrier_retention": _spec(
        "carrier_retention",
        InstrumentTier.DECISION,
        ("retention", "retention_floor", "executed_visits"),
        "retention below 0.9 means the loop is not carrying state",
    ),
    "mu_r_exponent": _spec(
        "mu_r_exponent",
        InstrumentTier.DECISION,
        ("p_hat", "ci_low", "ci_high", "jacobian_anchor"),
        "p=1 is the bounded-Jacobian anchor; report the registered branch",
    ),
    "loop_marginal_gain": _spec(
        "loop_marginal_gain",
        InstrumentTier.DECISION,
        ("visit_index", "eta_k", "flop_matched_control"),
        "no gain at k>=2 ends the recurrence thesis",
    ),
    "resp_leak": _spec(
        "resp_leak",
        InstrumentTier.DECISION,
        (
            "sidecar_only_delta_q_disabled",
            "checkpoint_delta_q_disabled",
            "retained_fraction",
        ),
        "more than half decay withdraws the selective-memory claim",
    ),
    "g_inv": _spec(
        "g_inv",
        InstrumentTier.DECISION,
        ("invocation_rate_by_step", "kappa_alignment"),
        "uniform invocation reports an ordinary MoE, not a conditional one",
    ),
    "t5_rotor_signature": _spec(
        "t5_rotor_signature",
        InstrumentTier.DECISION,
        ("max_relative_norm_error", "tolerance"),
        "failure means the carrier is an amplifier",
    ),
    "a_state": _spec(
        "a_state",
        InstrumentTier.DIAGNOSTIC,
        ("intervention", "numerator", "denominator", "ratio_unclipped"),
        "diagnostic only; may explain but may not select",
    ),
    "delta_mode_energy": _spec(
        "delta_mode_energy",
        InstrumentTier.DIAGNOSTIC,
        (
            "band_index",
            "visit_index",
            "rho",
            "disagreement_eigenvalue",
            "steps",
            "amplitude_law_exponent",
            "energy_law_exponent",
            "expected_amplitude_retention",
            "observed_amplitude_retention",
            "expected_energy_retention",
            "observed_energy_retention",
            "amplitude_absolute_error",
            "energy_absolute_error",
            "scope",
        ),
        (
            "compare amplitude with (1-2rho)^K and squared energy with "
            "(1-2rho)^(2K)"
        ),
    ),
    "engram_address_health": _spec(
        "engram_address_health",
        InstrumentTier.DIAGNOSTIC,
        (
            "hit_rate",
            "address_utilization_entropy",
            "template_hit_rate",
            "content_hit_rate",
            "token_rarity_stratum",
        ),
        "diagnose boilerplate capture and rare-token allocation",
    ),
    "router_calibration": _spec(
        "router_calibration",
        InstrumentTier.DIAGNOSTIC,
        ("calibration_step", "m_sha256", "s_sha256", "flip_rate"),
        "record the stability-selected freeze moment and frozen buffers",
    ),
    "expert_substitution": _spec(
        "expert_substitution",
        InstrumentTier.DIAGNOSTIC,
        ("source_expert", "substitute_expert", "loss_delta"),
        "distinguish specialized experts with multiplicity control",
    ),
    "gate_hysteresis": _spec(
        "gate_hysteresis",
        InstrumentTier.EXPLORATORY,
        ("multiplier", "forward_value", "reverse_value"),
        "exploratory until registered on a fresh cohort",
    ),
})


if sum(
    spec.tier is InstrumentTier.DECISION for spec in INSTRUMENT_REGISTRY.values()
) != 8:
    raise RuntimeError("the ratified Tier-1 observatory set must contain exactly eight instruments")
if INSTRUMENT_REGISTRY["resp_leak"].tier is not InstrumentTier.DECISION:
    raise RuntimeError("RESP-LEAK must remain Tier 1")
if INSTRUMENT_REGISTRY["a_state"].tier is not InstrumentTier.DIAGNOSTIC:
    raise RuntimeError("A_state must remain Tier 2")


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_identifier(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if _NORMALIZED_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a normalized identifier")
    return value


@dataclass(frozen=True, slots=True)
class _FrozenJSONArray:
    """Type-tagged immutable JSON array representation."""

    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _FrozenJSONObject:
    """Type-tagged immutable JSON object representation."""

    items: tuple[tuple[str, Any], ...]


_EVENT_FACTORY_TOKEN = object()


def _json_value(value: Any, *, path: str) -> Any:
    """Return an immutable JSON-compatible value, rejecting NaN and tensors."""

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if type(value) in (tuple, list):
        return _FrozenJSONArray(
            tuple(
                _json_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        )
    if type(value) is dict:
        items = []
        for key, item in value.items():
            _require_identifier(f"{path} key", key)
            items.append((key, _json_value(item, path=f"{path}.{key}")))
        return _FrozenJSONObject(tuple(sorted(items)))
    raise TypeError(f"{path} must contain only exact JSON scalar/list/object values")


def _thaw(value: Any) -> Any:
    if type(value) is _FrozenJSONObject:
        return {key: _thaw(item) for key, item in value.items}
    if type(value) is _FrozenJSONArray:
        return [_thaw(item) for item in value.items]
    return value


@dataclass(frozen=True, slots=True)
class PrecisionRecord:
    """The mandatory precision disclosure for Tier-2/3 measurements."""

    effective_n: float
    null_id: str
    interval_low: float
    interval_high: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.effective_n, bool) or not math.isfinite(float(self.effective_n)):
            raise ValueError("effective_n must be a finite positive scalar")
        if float(self.effective_n) <= 0.0:
            raise ValueError("effective_n must be a finite positive scalar")
        _require_identifier("null_id", self.null_id)
        for name, value in (
            ("interval_low", self.interval_low),
            ("interval_high", self.interval_high),
            ("confidence_level", self.confidence_level),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.interval_low) > float(self.interval_high):
            raise ValueError("precision interval must be ordered")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")

    def as_dict(self) -> dict[str, float | str]:
        return {
            "effective_n": float(self.effective_n),
            "null_id": self.null_id,
            "interval_low": float(self.interval_low),
            "interval_high": float(self.interval_high),
            "confidence_level": float(self.confidence_level),
        }


@dataclass(frozen=True, slots=True)
class ObservatoryEvent:
    """One immutable, hash-bound observatory row."""

    instrument_id: str
    tier: InstrumentTier
    run_id: str
    model_rung: str
    checkpoint_step: int
    seed: int
    cohort_id: str
    graph_fingerprint: str
    config_fingerprint: str
    t14b_receipt_sha256: str
    authority_sha256: str
    measurements_frozen: _FrozenJSONObject = field(repr=False)
    _factory_token: InitVar[object | None] = field(default=None, repr=False)
    registration_sha256: str | None = None
    branch_outcome: str | None = None
    precision: PrecisionRecord | None = None
    event_sha256: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _EVENT_FACTORY_TOKEN:
            raise TypeError(
                "ObservatoryEvent must be minted by the guarded internal factory"
            )
        _require_identifier("instrument_id", self.instrument_id)
        spec = INSTRUMENT_REGISTRY.get(self.instrument_id)
        if spec is None:
            raise ValueError("instrument_id is absent from the ratified registry")
        if spec.t14_receipt_is_record:
            raise ValueError("T14 causality is recorded by T14bReceipt, not a metric event")
        if self.tier is not spec.tier:
            raise ValueError("event tier must be derived from the ratified registry")
        _require_identifier("run_id", self.run_id)
        _require_identifier("model_rung", self.model_rung)
        _require_identifier("cohort_id", self.cohort_id)
        if type(self.checkpoint_step) is not int or self.checkpoint_step < 0:
            raise ValueError("checkpoint_step must be a non-negative exact integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative exact integer")
        _require_sha256("graph_fingerprint", self.graph_fingerprint)
        _require_sha256("config_fingerprint", self.config_fingerprint)
        _require_sha256("t14b_receipt_sha256", self.t14b_receipt_sha256)
        _require_sha256("authority_sha256", self.authority_sha256)
        if self.authority_sha256 != RATIFIED_TARGET_AUTHORITY_SHA256:
            raise ValueError("event authority is not the WEFT-1 ratification authority")
        if type(self.measurements_frozen) is not _FrozenJSONObject:
            raise TypeError("measurements_frozen must be an internal frozen JSON object")
        measurement_keys = {key for key, _value in self.measurements_frozen.items}
        if len(measurement_keys) != len(self.measurements_frozen.items):
            raise ValueError("measurement keys must be unique")
        missing = spec.required_measurements - measurement_keys
        if missing:
            raise ValueError(f"event measurements missing fields: {sorted(missing)}")
        if self.instrument_id == "delta_mode_energy":
            delta_measurements = self.measurements
            steps = delta_measurements["steps"]
            amplitude_exponent = delta_measurements["amplitude_law_exponent"]
            energy_exponent = delta_measurements["energy_law_exponent"]
            if type(steps) is not int or steps < 1:
                raise ValueError("delta-mode steps must be a positive exact integer")
            if type(amplitude_exponent) is not int or amplitude_exponent != steps:
                raise ValueError("delta-mode amplitude law exponent must equal K")
            if type(energy_exponent) is not int or energy_exponent != 2 * steps:
                raise ValueError("delta-mode energy law exponent must equal 2K")
            if (
                delta_measurements["scope"]
                != "callosum_only_excludes_intervening_core_dynamics"
            ):
                raise ValueError("delta-mode scope must exclude intervening core dynamics")
        if self.tier is InstrumentTier.DECISION:
            if self.registration_sha256 is None:
                raise ValueError("Tier-1 events require a pre-registration SHA")
            _require_sha256("registration_sha256", self.registration_sha256)
            _require_identifier("branch_outcome", self.branch_outcome)
        else:
            if self.registration_sha256 is not None or self.branch_outcome is not None:
                raise ValueError("Tier-2/3 events cannot carry decision branch fields")
            if self.precision is None:
                raise ValueError("Tier-2/3 events require effective n, null, and interval")
        object.__setattr__(self, "event_sha256", self._canonical_sha256())

    @property
    def measurements(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self.measurements_frozen.items}

    def as_dict(self, *, include_event_sha256: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": EVENT_KIND,
            "instrument_id": self.instrument_id,
            "instrument_tier": self.tier.value,
            "run_id": self.run_id,
            "model_rung": self.model_rung,
            "checkpoint_step": self.checkpoint_step,
            "seed": self.seed,
            "cohort_id": self.cohort_id,
            "graph_fingerprint": self.graph_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "t14b_receipt_sha256": self.t14b_receipt_sha256,
            "authority_sha256": self.authority_sha256,
            "measurements": self.measurements,
            "registration_sha256": self.registration_sha256,
            "branch_outcome": self.branch_outcome,
            "precision": self.precision.as_dict() if self.precision is not None else None,
        }
        if include_event_sha256:
            payload["event_sha256"] = self.event_sha256
        return payload

    def canonical_payload(self) -> bytes:
        return json.dumps(
            self.as_dict(include_event_sha256=False),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()


def emit_observatory_event(
    guard: ObservatoryGuard,
    receipt: T14bReceipt | None,
    *,
    instrument_id: str,
    run_id: str,
    model_rung: str,
    checkpoint_step: int,
    seed: int,
    cohort_id: str,
    measurements: Mapping[str, Any],
    registration_sha256: str | None = None,
    branch_outcome: str | None = None,
    precision: PrecisionRecord | None = None,
) -> ObservatoryEvent:
    """Validate T14b first, then mint one schema-bound event.

    ``ObservatoryGuard.invoke`` is deliberately the first executable operation.
    A missing, failed, stale, or mismatched causality receipt therefore prevents
    even measurement normalization from running.
    """

    if type(guard) is not ObservatoryGuard:
        raise TypeError("guard must be an exact ObservatoryGuard")

    def mint() -> ObservatoryEvent:
        _require_identifier("instrument_id", instrument_id)
        spec = INSTRUMENT_REGISTRY.get(instrument_id)
        if spec is None:
            raise ValueError("instrument_id is absent from the ratified registry")
        if type(measurements) is not dict:
            raise TypeError("measurements must be an exact dict")
        frozen = _json_value(measurements, path="measurements")
        assert type(frozen) is _FrozenJSONObject
        assert receipt is not None
        return ObservatoryEvent(
            instrument_id=instrument_id,
            tier=spec.tier,
            run_id=run_id,
            model_rung=model_rung,
            checkpoint_step=checkpoint_step,
            seed=seed,
            cohort_id=cohort_id,
            graph_fingerprint=receipt.graph_fingerprint,
            config_fingerprint=receipt.config_fingerprint,
            t14b_receipt_sha256=receipt.receipt_sha256,
            authority_sha256=receipt.authority_sha256,
            measurements_frozen=frozen,
            _factory_token=_EVENT_FACTORY_TOKEN,
            registration_sha256=registration_sha256,
            branch_outcome=branch_outcome,
            precision=precision,
        )

    return guard.invoke(receipt, mint)
