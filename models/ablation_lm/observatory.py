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
import re
from typing import Callable, ParamSpec, TypeVar

import torch

from .config import RATIFIED_TARGET_AUTHORITY_SHA256


_RECEIPT_KIND = "weft1.t14b.sequence_axis_causality.v1"
_EVIDENCE_KIND = "weft1.t14b.gradient_evidence.v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_COVERAGE_MODES = ("packed", "padded")
_HOOK_IDENTITY_PATTERN = re.compile(
    r"[a-z][a-z0-9_]*(?:\.(?:[a-z][a-z0-9_]*|[0-9]+))*"
)
T14B_REGISTERED_K_VALUES = (1, 2, 4, 8)
_MEASUREMENT_ALGORITHM = (
    "torch.autograd.grad/full-channel-basis/all-valid-output-positions/"
    "forbidden-sequence-and-batch-mask/v3"
)
T14B_MEASUREMENT_ALGORITHM_SHA256 = hashlib.sha256(
    _MEASUREMENT_ALGORITHM.encode("ascii")
).hexdigest()
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


def _require_causal_segment_layout(causal_segment_ids: torch.Tensor) -> None:
    """Require unique contiguous document runs and one edge-padding run."""

    for row in causal_segment_ids.detach().to(device="cpu", dtype=torch.int64):
        values = row.tolist()
        runs = [
            value
            for index, value in enumerate(values)
            if index == 0 or value != values[index - 1]
        ]
        non_padding_runs = [value for value in runs if value >= 0]
        if len(non_padding_runs) != len(set(non_padding_runs)):
            raise ValueError(
                "each causal segment ID must occupy exactly one contiguous run"
            )
        padding_positions = [
            position for position, value in enumerate(values) if value == -1
        ]
        if padding_positions:
            padding_is_contiguous = (
                padding_positions[-1] - padding_positions[0] + 1
                == len(padding_positions)
            )
            padding_is_at_edge = (
                padding_positions[0] == 0
                or padding_positions[-1] == len(values) - 1
            )
            if not padding_is_contiguous or not padding_is_at_edge:
                raise ValueError("padding must occupy one contiguous edge run")


def _canonical_payload(receipt: "T14bReceipt") -> bytes:
    payload = {
        "authority_sha256": receipt.authority_sha256,
        "config_fingerprint": receipt.config_fingerprint,
        "evidence_sha256": receipt.evidence_sha256,
        "graph_fingerprint": receipt.graph_fingerprint,
        "liveness_passed": receipt.liveness_passed,
        "max_future_gradient": receipt.max_future_gradient,
        "minimum_allowed_gradient_max": receipt.minimum_allowed_gradient_max,
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


@dataclass(frozen=True, slots=True, init=False)
class T14bGradientEvidence:
    """One autograd-measured tensor for a stage, K, and packing mode.

    Evidence has no public constructor.  Use
    :func:`measure_t14b_future_gradients`, which derives the tensor from an
    autograd graph and binds the probe definition into ``probe_manifest_sha256``.
    """

    stage: str
    k_value: int
    coverage_mode: str
    hook_identity: str
    input_manifest_sha256: str
    causal_segment_ids_sha256: str
    probe_manifest_sha256: str
    measurement_algorithm_sha256: str
    probe_positions: tuple[tuple[int, int], ...]
    future_gradients: torch.Tensor = field(repr=False, compare=False)
    allowed_gradients: torch.Tensor = field(repr=False, compare=False)

    def __new__(cls, *_args, **_kwargs):
        raise TypeError("T14bGradientEvidence must be created by the autograd measurer")


def _probe_manifest_sha256(
    *,
    stage: str,
    k_value: int,
    coverage_mode: str,
    hook_identity: str,
    input_manifest_sha256: str,
    causal_segment_ids_sha256: str,
    probe_positions: tuple[tuple[int, int], ...],
    stage_output: torch.Tensor,
    sequence_source: torch.Tensor,
) -> str:
    payload = {
        "coverage_mode": coverage_mode,
        "hook_identity": hook_identity,
        "input_manifest_sha256": input_manifest_sha256,
        "causal_segment_ids_sha256": causal_segment_ids_sha256,
        "k_value": k_value,
        "measurement_algorithm_sha256": T14B_MEASUREMENT_ALGORITHM_SHA256,
        "probe_positions": [list(position) for position in probe_positions],
        "sequence_source_dtype": str(sequence_source.dtype),
        "sequence_source_shape": list(sequence_source.shape),
        "stage": stage,
        "stage_output_dtype": str(stage_output.dtype),
        "stage_output_shape": list(stage_output.shape),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def measure_t14b_future_gradients(
    *,
    stage: str,
    k_value: int,
    coverage_mode: str,
    hook_identity: str,
    input_manifest_sha256: str,
    causal_segment_ids: torch.Tensor,
    stage_output: torch.Tensor,
    sequence_source: torch.Tensor,
) -> T14bGradientEvidence:
    """Measure future-position gradients through a live autograd graph.

    Every output channel at every non-padding output position is differentiated
    against the full sequence source.  The full position panel is derived from
    ``causal_segment_ids`` rather than selected by the caller, so an unprobed
    token or packed segment cannot hide a dependency.  Only the same-batch,
    same-segment causal prefix is allowed; every other source position is
    forbidden evidence and the allowed prefix is a liveness control.  The
    complete channel basis cannot hide a dependency through projection
    cancellation.
    """

    _require_stage_names((stage,))
    if type(hook_identity) is not str:
        raise TypeError("hook_identity must be an exact str")
    if _HOOK_IDENTITY_PATTERN.fullmatch(hook_identity) is None:
        raise ValueError("hook_identity must be a canonical dotted module path")
    if type(k_value) is not int:
        raise TypeError("evidence K must be an exact int")
    if k_value not in T14B_REGISTERED_K_VALUES:
        raise ValueError("evidence K is outside the registered T14b panel")
    if type(coverage_mode) is not str:
        raise TypeError("coverage_mode must be an exact str")
    if coverage_mode not in _COVERAGE_MODES:
        raise ValueError("coverage_mode must be packed or padded")
    _require_sha256("input_manifest_sha256", input_manifest_sha256)
    if type(stage_output) is not torch.Tensor or type(sequence_source) is not torch.Tensor:
        raise TypeError("stage_output and sequence_source must be exact tensors")
    if stage_output.layout != torch.strided or sequence_source.layout != torch.strided:
        raise TypeError("T14b measurement requires strided dense tensors")
    if not stage_output.is_floating_point() or not sequence_source.is_floating_point():
        raise TypeError("T14b measurement tensors must use floating dtypes")
    if stage_output.ndim < 3 or sequence_source.ndim < 3:
        raise ValueError("T14b tensors must have [batch, sequence, channels...] shape")
    if stage_output.numel() < 1 or sequence_source.numel() < 1:
        raise ValueError("T14b tensors must contain channel values")
    if stage_output.shape[:2] != sequence_source.shape[:2]:
        raise ValueError("stage output and sequence source must share batch and sequence")
    if type(causal_segment_ids) is not torch.Tensor:
        raise TypeError("causal_segment_ids must be an exact tensor")
    if causal_segment_ids.shape != sequence_source.shape[:2]:
        raise ValueError("causal_segment_ids must match batch and sequence")
    if causal_segment_ids.device != sequence_source.device:
        raise ValueError("causal_segment_ids must share the sequence-source device")
    if causal_segment_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("causal_segment_ids must use an integer dtype")
    if causal_segment_ids.numel() and int(causal_segment_ids.min()) < -1:
        raise ValueError("causal_segment_ids may use only -1 or non-negative IDs")
    _require_causal_segment_layout(causal_segment_ids)
    if coverage_mode == "padded" and not bool(causal_segment_ids.eq(-1).any()):
        raise ValueError("padded T14b evidence requires at least one padded source position")
    if coverage_mode == "packed" and not any(
        len(set(row[row.ge(0)].detach().cpu().tolist())) >= 2
        for row in causal_segment_ids
    ):
        raise ValueError("packed T14b evidence requires at least two causal segments")
    if not stage_output.requires_grad or not sequence_source.requires_grad:
        raise ValueError("T14b tensors must belong to a live autograd graph")
    positions = tuple(
        (batch_index, position)
        for batch_index in range(stage_output.shape[0])
        for position in range(stage_output.shape[1])
        if causal_segment_ids[batch_index, position].item() >= 0
    )
    if not positions:
        raise ValueError("T14b requires at least one valid non-padding output position")
    segment_ids_cpu = causal_segment_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    segment_bytes = segment_ids_cpu.view(torch.uint8).numpy().tobytes()
    causal_segment_ids_sha256 = hashlib.sha256(segment_bytes).hexdigest()
    tasks = tuple(
        (batch_index, position, channel)
        for batch_index, position in positions
        for channel in range(stage_output[batch_index, position].numel())
    )
    forbidden_measurements: list[torch.Tensor] = []
    allowed_measurements: list[torch.Tensor] = []
    for task_index, (batch_index, position, channel) in enumerate(tasks):
        output_vector = stage_output[batch_index, position].float().reshape(-1)
        gradient = torch.autograd.grad(
            output_vector[channel],
            sequence_source,
            retain_graph=task_index < len(tasks) - 1,
            create_graph=False,
            allow_unused=False,
        )[0]
        allowed_mask = torch.zeros_like(causal_segment_ids, dtype=torch.bool)
        probe_segment = causal_segment_ids[batch_index, position]
        allowed_mask[batch_index, : position + 1] = causal_segment_ids[
            batch_index, : position + 1
        ].eq(probe_segment)
        allowed_measurements.append(gradient[allowed_mask].detach().reshape(-1).cpu())
        forbidden_measurements.append(
            gradient[~allowed_mask].detach().reshape(-1).cpu()
        )
    future_gradients = torch.cat(forbidden_measurements).contiguous()
    allowed_gradients = torch.cat(allowed_measurements).contiguous()
    if not bool(torch.isfinite(future_gradients).all()) or not bool(
        torch.isfinite(allowed_gradients).all()
    ):
        raise ValueError("measured T14b gradients must be finite")
    evidence = object.__new__(T14bGradientEvidence)
    values = {
        "stage": stage,
        "k_value": k_value,
        "coverage_mode": coverage_mode,
        "hook_identity": hook_identity,
        "input_manifest_sha256": input_manifest_sha256,
        "causal_segment_ids_sha256": causal_segment_ids_sha256,
        "probe_manifest_sha256": _probe_manifest_sha256(
            stage=stage,
            k_value=k_value,
            coverage_mode=coverage_mode,
            hook_identity=hook_identity,
            input_manifest_sha256=input_manifest_sha256,
            causal_segment_ids_sha256=causal_segment_ids_sha256,
            probe_positions=positions,
            stage_output=stage_output,
            sequence_source=sequence_source,
        ),
        "measurement_algorithm_sha256": T14B_MEASUREMENT_ALGORITHM_SHA256,
        "probe_positions": positions,
        "future_gradients": future_gradients,
        "allowed_gradients": allowed_gradients,
    }
    for name, value in values.items():
        object.__setattr__(evidence, name, value)
    return evidence


def _gradient_evidence_sha256(evidence: tuple[T14bGradientEvidence, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_EVIDENCE_KIND.encode("ascii") + b"\x00")
    for row in evidence:
        future = row.future_gradients.detach().cpu().contiguous()
        allowed = row.allowed_gradients.detach().cpu().contiguous()
        metadata = {
            "allowed_dtype": str(allowed.dtype),
            "allowed_shape": list(allowed.shape),
            "coverage_mode": row.coverage_mode,
            "causal_segment_ids_sha256": row.causal_segment_ids_sha256,
            "future_dtype": str(future.dtype),
            "future_shape": list(future.shape),
            "hook_identity": row.hook_identity,
            "input_manifest_sha256": row.input_manifest_sha256,
            "k_value": row.k_value,
            "measurement_algorithm_sha256": row.measurement_algorithm_sha256,
            "probe_manifest_sha256": row.probe_manifest_sha256,
            "probe_positions": [list(position) for position in row.probe_positions],
            "stage": row.stage,
        }
        encoded = json.dumps(
            metadata,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big") + encoded)
        for tensor in (future, allowed):
            raw = tensor.view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class T14bReceipt:
    """Immutable causality receipt bound to one graph and configuration.

    There is no public value constructor: callers must use
    :meth:`from_gradient_evidence`.  That factory derives ``passed`` and the
    gradient extrema from measured tensors rather than caller booleans, and it
    hashes the complete evidence.  Integrated hook ownership and derivation of
    the graph/config/input fingerprints remain prerequisites for a production
    receipt; this generic scaffold does not attest those caller-supplied labels.
    """

    authority_sha256: str
    graph_fingerprint: str
    config_fingerprint: str
    evidence_sha256: str
    tested_k_values: tuple[int, ...]
    sequence_axis_stages: tuple[str, ...]
    packed: bool
    padded: bool
    max_future_gradient: float
    minimum_allowed_gradient_max: float
    liveness_passed: bool
    passed: bool
    receipt_sha256: str = field(init=False)

    def __new__(cls, *_args, **_kwargs):
        raise TypeError("T14bReceipt must be minted from gradient evidence")

    @classmethod
    def from_gradient_evidence(
        cls,
        *,
        authority_sha256: str,
        graph_fingerprint: str,
        config_fingerprint: str,
        tested_k_values: tuple[int, ...],
        sequence_axis_stages: tuple[str, ...],
        evidence: tuple[T14bGradientEvidence, ...],
    ) -> "T14bReceipt":
        """Derive an immutable receipt from a complete tensor-evidence panel."""

        _require_sha256("authority_sha256", authority_sha256)
        if authority_sha256 != RATIFIED_TARGET_AUTHORITY_SHA256:
            raise ValueError("T14b authority must equal the WEFT-1 ratification SHA")
        _require_sha256("graph_fingerprint", graph_fingerprint)
        _require_sha256("config_fingerprint", config_fingerprint)
        _require_k_values(tested_k_values)
        if tested_k_values != T14B_REGISTERED_K_VALUES:
            raise ValueError("T14b receipt requires the registered K=(1,2,4,8) panel")
        _require_stage_names(sequence_axis_stages)
        if type(evidence) is not tuple:
            raise TypeError("evidence must be an exact tuple")
        if not evidence:
            raise ValueError("evidence must not be empty")
        if any(type(row) is not T14bGradientEvidence for row in evidence):
            raise TypeError("every evidence row must be an exact T14bGradientEvidence")
        if any(
            row.measurement_algorithm_sha256 != T14B_MEASUREMENT_ALGORITHM_SHA256
            for row in evidence
        ):
            raise ValueError("T14b evidence uses an unregistered measurement algorithm")
        coordinates = tuple(
            (row.k_value, row.stage, row.coverage_mode) for row in evidence
        )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("T14b evidence contains duplicate coverage coordinates")
        observed_modes = {row.coverage_mode for row in evidence}
        modes = tuple(mode for mode in _COVERAGE_MODES if mode in observed_modes)
        if modes != _COVERAGE_MODES:
            raise ValueError("T14b receipt requires both packed and padded evidence")
        expected = {
            (k_value, stage, mode)
            for k_value in tested_k_values
            for stage in sequence_axis_stages
            for mode in modes
        }
        if set(coordinates) != expected:
            raise ValueError("T14b evidence does not form the declared coverage cross-product")
        for mode in modes:
            mode_rows = tuple(row for row in evidence if row.coverage_mode == mode)
            panel_identities = {
                (
                    row.input_manifest_sha256,
                    row.causal_segment_ids_sha256,
                    row.probe_positions,
                )
                for row in mode_rows
            }
            if len(panel_identities) != 1:
                raise ValueError(
                    f"T14b {mode} rows must share one input and probe panel"
                )
        for stage in sequence_axis_stages:
            hook_identities = {
                row.hook_identity for row in evidence if row.stage == stage
            }
            if len(hook_identities) != 1:
                raise ValueError(f"T14b stage {stage!r} must use one hook identity")
        ordered = tuple(
            next(
                row
                for row in evidence
                if (row.k_value, row.stage, row.coverage_mode) == coordinate
            )
            for coordinate in sorted(
                expected,
                key=lambda value: (
                    tested_k_values.index(value[0]),
                    sequence_axis_stages.index(value[1]),
                    _COVERAGE_MODES.index(value[2]),
                ),
            )
        )
        maxima = tuple(
            float(row.future_gradients.detach().abs().max().cpu().item())
            for row in ordered
        )
        allowed_maxima = tuple(
            float(row.allowed_gradients.detach().abs().max().cpu().item())
            for row in ordered
        )
        max_future_gradient = float(max(maxima))
        minimum_allowed_gradient_max = float(min(allowed_maxima))
        liveness_passed = minimum_allowed_gradient_max > 0.0
        receipt = object.__new__(cls)
        values = {
            "authority_sha256": authority_sha256,
            "graph_fingerprint": graph_fingerprint,
            "config_fingerprint": config_fingerprint,
            "evidence_sha256": _gradient_evidence_sha256(ordered),
            "tested_k_values": tested_k_values,
            "sequence_axis_stages": sequence_axis_stages,
            "packed": "packed" in modes,
            "padded": "padded" in modes,
            "max_future_gradient": max_future_gradient,
            "minimum_allowed_gradient_max": minimum_allowed_gradient_max,
            "liveness_passed": liveness_passed,
            "passed": max_future_gradient == 0.0 and liveness_passed,
        }
        for name, value in values.items():
            object.__setattr__(receipt, name, value)
        object.__setattr__(receipt, "receipt_sha256", _canonical_sha256(receipt))
        return receipt

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
        if self.required_k_values != T14B_REGISTERED_K_VALUES:
            raise ValueError("observatory policy requires K=(1,2,4,8)")
        _require_stage_names(self.required_sequence_axis_stages)
        if type(self.require_packed) is not bool or type(self.require_padded) is not bool:
            raise TypeError("required coverage flags must be exact bool values")
        if not self.require_packed or not self.require_padded:
            raise ValueError("observatory policy cannot waive packed or padded T14b")

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
