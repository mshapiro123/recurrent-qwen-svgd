"""Factory-sealed A7 observational-invariance receipts.

PF-3.4 separates the K=1 dense anchor from per-module OFF-idempotence.
Executed cells require exact identity of both logits and loss, plus an active
module positive control. Ineligible, absent, and deterministic-CUDA cells are
typed non-passes; none can be promoted by editing a status flag.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from hashlib import sha256
from itertools import product
import json
import math
from typing import Final

import torch


OBS_INV_K_VALUES = (1, 2, 4, 8)
OBS_INV_DTYPES = ("float32", "bfloat16")
OBS_INV_BACKENDS = ("cpu", "cuda_deterministic")
OBS_INV_ANCHOR_NAME = "dense_4_2_4_k1"
OBS_INV_K_AXIS_BACKGROUND = ("recurrence",)


@dataclass(frozen=True)
class ObservationalInvarianceModuleSpec:
    """Registered module identity and structural eligibility."""

    module_name: str
    integrated: bool
    minimum_recurrent_steps: int
    background_modules: tuple[str, ...]
    eligibility_reason: str

    def eligible_at(self, recurrent_steps: int) -> bool:
        if recurrent_steps not in OBS_INV_K_VALUES:
            raise ValueError("recurrent_steps is outside the A7 matrix")
        return recurrent_steps >= self.minimum_recurrent_steps


# Recurrence defines the K axis and is therefore a matched background, not a
# per-module toggle. At K>1 it cannot be removed while preserving the same K;
# at K=1 its initialized loop embedding is exactly zero, so it cannot meet the
# registered ON-non-triviality control. Static K/V remains eligible at K=1:
# with multiple core blocks, every block reads the prelude anchor rather than
# the preceding block's dynamic state, so it is already behaviorally live.
OBS_INV_MODULE_SPECS: Final = (
    ObservationalInvarianceModuleSpec(
        "front_hadamard_experts", True, 1, (),
        "executes before the prelude at every registered K",
    ),
    ObservationalInvarianceModuleSpec(
        "static_kv_core", True, 1, ("recurrence",),
        "projects every core block's K/V from the fixed prelude anchor",
    ),
    ObservationalInvarianceModuleSpec(
        "reentry_bridge", True, 2, ("recurrence",),
        "visit zero deliberately bypasses recurrent re-entry",
    ),
    ObservationalInvarianceModuleSpec(
        "scratch", True, 1, (),
        "position-aligned lanes execute on visit zero",
    ),
    ObservationalInvarianceModuleSpec(
        "lane_carrier", True, 1, ("scratch",),
        "the carrier is tested against the same scratch background",
    ),
    ObservationalInvarianceModuleSpec(
        "engram", True, 1, (),
        "the causal engram executes after prelude block zero",
    ),
    ObservationalInvarianceModuleSpec(
        "long_term_memory", True, 1, (),
        "read-only retrieval executes before the coda",
    ),
    ObservationalInvarianceModuleSpec(
        "integrated_rotor_carrier", False, 1, (),
        "ratified production integration is absent",
    ),
    ObservationalInvarianceModuleSpec(
        "per_band_callosum", False, 1, ("integrated_rotor_carrier",),
        "ratified production integration is absent",
    ),
    ObservationalInvarianceModuleSpec(
        "sidecar", False, 3, (),
        "second-order-jet consumers require at least three visits",
    ),
)
_MODULE_SPECS_BY_NAME: Final = {
    spec.module_name: spec for spec in OBS_INV_MODULE_SPECS
}
OBS_INV_MATERIALIZED_MODULES = tuple(
    spec.module_name for spec in OBS_INV_MODULE_SPECS if spec.integrated
)
OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS = tuple(
    spec.module_name for spec in OBS_INV_MODULE_SPECS if not spec.integrated
)

_ANCHOR_FACTORY_TOKEN: Final = object()
_CELL_FACTORY_TOKEN: Final = object()
_MATRIX_FACTORY_TOKEN: Final = object()
_PROMOTION_FACTORY_TOKEN: Final = object()


class ObservationalInvarianceBlocked(RuntimeError):
    """Raised when a required A7 cell is missing or is not bit-identical."""


def _dtype_name(tensor: torch.Tensor) -> str:
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in OBS_INV_DTYPES:
        raise ValueError("A7 logits support only float32 and bfloat16")
    return name


def _check_output_pair(
    logits: torch.Tensor,
    loss: torch.Tensor,
    *,
    label: str,
) -> None:
    if logits.ndim < 1:
        raise ValueError(f"{label} logits must have at least one dimension")
    _dtype_name(logits)
    if loss.numel() != 1:
        raise ValueError(f"{label} loss must be scalar")
    if logits.device != loss.device:
        raise ValueError(f"{label} logits and loss must share a device")


def _validate_backend_evidence(
    backend: str,
    *tensors: torch.Tensor,
) -> None:
    """Bind an executed A7 backend label to the tensors and runtime."""

    if backend not in OBS_INV_BACKENDS:
        raise ValueError("backend is outside A7")
    if not tensors:
        raise ValueError("executed A7 evidence requires tensors")
    device_types = {tensor.device.type for tensor in tensors}
    if backend == "cpu":
        if device_types != {"cpu"}:
            raise ValueError("A7 CPU evidence must come from CPU tensors")
        return
    if device_types != {"cuda"}:
        raise ValueError("A7 deterministic-CUDA evidence must come from CUDA tensors")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError(
            "A7 deterministic-CUDA evidence requires deterministic algorithms"
        )


def _mismatch_count(left: torch.Tensor, right: torch.Tensor) -> int:
    return int(torch.count_nonzero(left.ne(right)).detach().cpu().item())


def _max_absolute_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left.float() - right.float()).abs()
    value = float(difference.max().detach().cpu().item())
    return value if math.isfinite(value) else float("inf")


@dataclass(frozen=True)
class DenseAnchorCell:
    """One K=1 4/2/4 dense-anchor cell, constructible only by factories."""

    dtype: str
    backend: str
    status: str
    logits_bit_identical: bool | None
    loss_bit_identical: bool | None
    logits_mismatch_count: int | None
    loss_mismatch_count: int | None
    max_absolute_logit_difference: float | None
    max_absolute_loss_difference: float | None
    reason: str
    _factory_token: InitVar[object] = None

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _ANCHOR_FACTORY_TOKEN:
            raise TypeError("DenseAnchorCell is factory-sealed")
        if self.dtype not in OBS_INV_DTYPES or self.backend not in OBS_INV_BACKENDS:
            raise ValueError("anchor dtype or backend is outside A7")
        if self.status == "passed":
            if self.logits_bit_identical is not True or self.loss_bit_identical is not True:
                raise ValueError("a passed anchor requires exact logits and loss")
            if self.logits_mismatch_count != 0 or self.loss_mismatch_count != 0:
                raise ValueError("a passed anchor cannot contain mismatches")
        elif self.status == "failed":
            if self.logits_bit_identical is None or self.loss_bit_identical is None:
                raise ValueError("an executed anchor must contain comparison evidence")
            if self.logits_bit_identical and self.loss_bit_identical:
                raise ValueError("an exact anchor cannot be marked failed")
        elif self.status == "pending":
            evidence = (
                self.logits_bit_identical,
                self.loss_bit_identical,
                self.logits_mismatch_count,
                self.loss_mismatch_count,
                self.max_absolute_logit_difference,
                self.max_absolute_loss_difference,
            )
            if any(value is not None for value in evidence):
                raise ValueError("a pending anchor cannot contain fabricated evidence")
        else:
            raise ValueError("anchor status must be passed, failed, or pending")

    @property
    def coordinate(self) -> tuple[str, str]:
        return self.dtype, self.backend


@dataclass(frozen=True)
class ObservationalInvarianceCell:
    """One registered module x K x dtype x backend A7 cell."""

    module_name: str
    requested_recurrent_steps: int
    executed_recurrent_steps: int | None
    dtype: str
    backend: str
    status: str
    integrated: bool
    eligible: bool
    background_modules: tuple[str, ...]
    off_logits_bit_identical: bool | None
    off_loss_bit_identical: bool | None
    on_logits_nontrivial: bool | None
    on_loss_nontrivial: bool | None
    off_logits_mismatch_count: int | None
    off_loss_mismatch_count: int | None
    on_logits_mismatch_count: int | None
    on_loss_mismatch_count: int | None
    max_absolute_off_logit_difference: float | None
    max_absolute_off_loss_difference: float | None
    reason: str
    _factory_token: InitVar[object] = None

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _CELL_FACTORY_TOKEN:
            raise TypeError("ObservationalInvarianceCell is factory-sealed")
        if self.module_name not in _MODULE_SPECS_BY_NAME:
            raise ValueError("module is outside the registered A7 scope")
        if self.requested_recurrent_steps not in OBS_INV_K_VALUES:
            raise ValueError("requested_recurrent_steps is outside A7")
        if self.dtype not in OBS_INV_DTYPES or self.backend not in OBS_INV_BACKENDS:
            raise ValueError("dtype or backend is outside A7")
        spec = _MODULE_SPECS_BY_NAME[self.module_name]
        expected_eligibility = spec.eligible_at(self.requested_recurrent_steps)
        if self.integrated is not spec.integrated:
            raise ValueError("cell integration state contradicts the registry")
        if self.eligible is not expected_eligibility:
            raise ValueError("cell eligibility contradicts the registry")
        if self.background_modules != spec.background_modules:
            raise ValueError("cell background contradicts the registry")

        evidence = (
            self.off_logits_bit_identical,
            self.off_loss_bit_identical,
            self.on_logits_nontrivial,
            self.on_loss_nontrivial,
            self.off_logits_mismatch_count,
            self.off_loss_mismatch_count,
            self.on_logits_mismatch_count,
            self.on_loss_mismatch_count,
            self.max_absolute_off_logit_difference,
            self.max_absolute_off_loss_difference,
        )
        if self.status in ("passed", "failed"):
            if not self.integrated or not self.eligible:
                raise ValueError("only an eligible integrated cell can execute")
            if self.executed_recurrent_steps != self.requested_recurrent_steps:
                raise ValueError("executed K must equal requested K")
            if any(value is None for value in evidence):
                raise ValueError("an executed cell requires complete evidence")
            passes = bool(
                self.off_logits_bit_identical
                and self.off_loss_bit_identical
                and self.on_logits_nontrivial
                and self.on_loss_nontrivial
            )
            if (self.status == "passed") is not passes:
                raise ValueError("cell status contradicts exact comparison evidence")
        elif self.status == "ineligible":
            if not self.integrated or self.eligible:
                raise ValueError("ineligible is reserved for integrated, ineligible cells")
            if self.executed_recurrent_steps is not None:
                raise ValueError("an ineligible cell cannot claim execution")
            if any(value is not None for value in evidence):
                raise ValueError("an ineligible cell cannot contain fabricated evidence")
        elif self.status == "absent":
            if self.integrated:
                raise ValueError("an integrated module cannot be marked absent")
            if self.executed_recurrent_steps is not None:
                raise ValueError("an absent module cannot claim execution")
            if any(value is not None for value in evidence):
                raise ValueError("an absent cell cannot contain fabricated evidence")
        elif self.status == "pending":
            if not self.integrated or not self.eligible:
                raise ValueError("pending is reserved for eligible integrated cells")
            if self.executed_recurrent_steps is not None:
                raise ValueError("a pending cell cannot claim execution")
            if any(value is not None for value in evidence):
                raise ValueError("a pending cell cannot contain fabricated evidence")
        else:
            raise ValueError(
                "status must be passed, failed, ineligible, absent, or pending"
            )

    @property
    def coordinate(self) -> tuple[str, int, str, str]:
        return (
            self.module_name,
            self.requested_recurrent_steps,
            self.dtype,
            self.backend,
        )


@dataclass(frozen=True)
class ObservationalInvarianceMatrix:
    """Validated typed A7 matrix; completion is evidence-derived."""

    anchor_cells: tuple[DenseAnchorCell, ...]
    cells: tuple[ObservationalInvarianceCell, ...]
    _factory_token: InitVar[object] = None

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _MATRIX_FACTORY_TOKEN:
            raise TypeError("ObservationalInvarianceMatrix is factory-sealed")

    @property
    def excluded_absent_integrations(self) -> tuple[str, ...]:
        return OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS

    @property
    def failed_cells(self) -> tuple[ObservationalInvarianceCell, ...]:
        return tuple(cell for cell in self.cells if cell.status == "failed")

    @property
    def deferred_cells(self) -> tuple[ObservationalInvarianceCell, ...]:
        return tuple(cell for cell in self.cells if cell.status == "pending")

    @property
    def typed_nonpass_cells(self) -> tuple[ObservationalInvarianceCell, ...]:
        return tuple(
            cell
            for cell in self.cells
            if cell.status in ("failed", "ineligible", "absent", "pending")
        )

    def _required_cells(self, backend: str | None = None) -> tuple[object, ...]:
        anchors = tuple(
            cell for cell in self.anchor_cells
            if backend is None or cell.backend == backend
        )
        modules = tuple(
            cell for cell in self.cells
            if cell.integrated and cell.eligible
            and (backend is None or cell.backend == backend)
        )
        return (*anchors, *modules)

    @property
    def cpu_passed(self) -> bool:
        required = self._required_cells("cpu")
        return bool(required) and all(cell.status == "passed" for cell in required)

    @property
    def complete(self) -> bool:
        required = self._required_cells()
        return bool(required) and all(cell.status == "passed" for cell in required)

    @property
    def digest(self) -> str:
        payload = {
            "anchor": [
                {
                    "coordinate": cell.coordinate,
                    "status": cell.status,
                    "logits": cell.logits_bit_identical,
                    "loss": cell.loss_bit_identical,
                    "logits_mismatch_count": cell.logits_mismatch_count,
                    "loss_mismatch_count": cell.loss_mismatch_count,
                    "max_absolute_logit_difference": (
                        cell.max_absolute_logit_difference
                    ),
                    "max_absolute_loss_difference": (
                        cell.max_absolute_loss_difference
                    ),
                    "reason": cell.reason,
                }
                for cell in self.anchor_cells
            ],
            "modules": [
                {
                    "coordinate": cell.coordinate,
                    "status": cell.status,
                    "integrated": cell.integrated,
                    "eligible": cell.eligible,
                    "background_modules": cell.background_modules,
                    "executed_recurrent_steps": cell.executed_recurrent_steps,
                    "off_logits": cell.off_logits_bit_identical,
                    "off_loss": cell.off_loss_bit_identical,
                    "on_logits": cell.on_logits_nontrivial,
                    "on_loss": cell.on_loss_nontrivial,
                    "off_logits_mismatch_count": (
                        cell.off_logits_mismatch_count
                    ),
                    "off_loss_mismatch_count": cell.off_loss_mismatch_count,
                    "on_logits_mismatch_count": cell.on_logits_mismatch_count,
                    "on_loss_mismatch_count": cell.on_loss_mismatch_count,
                    "max_absolute_off_logit_difference": (
                        cell.max_absolute_off_logit_difference
                    ),
                    "max_absolute_off_loss_difference": (
                        cell.max_absolute_off_loss_difference
                    ),
                    "reason": cell.reason,
                }
                for cell in self.cells
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    def require_cpu_passed(self) -> None:
        incomplete = tuple(
            (
                OBS_INV_ANCHOR_NAME
                if isinstance(cell, DenseAnchorCell)
                else cell.module_name,
                getattr(cell, "requested_recurrent_steps", 1),
                cell.dtype,
                cell.backend,
                cell.status,
            )
            for cell in self._required_cells("cpu")
            if cell.status != "passed"
        )
        if incomplete:
            raise ObservationalInvarianceBlocked(
                f"A7 CPU required cells did not pass: {incomplete}"
            )

    def require_complete(self) -> None:
        incomplete = tuple(
            (
                OBS_INV_ANCHOR_NAME
                if isinstance(cell, DenseAnchorCell)
                else cell.module_name,
                getattr(cell, "requested_recurrent_steps", 1),
                cell.dtype,
                cell.backend,
                cell.status,
            )
            for cell in self._required_cells()
            if cell.status != "passed"
        )
        if incomplete:
            raise ObservationalInvarianceBlocked(
                f"A7 matrix still has failed or pending required cells: {incomplete}"
            )


@dataclass(frozen=True)
class CompleteObservationalInvarianceReceipt:
    """Factory-only proof that every required CPU and CUDA cell passed."""

    matrix_digest: str
    required_cell_count: int
    _factory_token: InitVar[object] = None

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PROMOTION_FACTORY_TOKEN:
            raise TypeError("CompleteObservationalInvarianceReceipt is factory-sealed")
        if len(self.matrix_digest) != 64:
            raise ValueError("matrix_digest must be a SHA-256 hex digest")
        if self.required_cell_count < 1:
            raise ValueError("required_cell_count must be positive")


def compare_dense_anchor(
    reference_logits: torch.Tensor,
    reference_loss: torch.Tensor,
    observed_logits: torch.Tensor,
    observed_loss: torch.Tensor,
    *,
    backend: str,
    reason: str,
) -> DenseAnchorCell:
    """Execute the registered K=1 4/2/4 dense-anchor comparison."""

    for label, logits, loss in (
        ("reference", reference_logits, reference_loss),
        ("observed", observed_logits, observed_loss),
    ):
        _check_output_pair(logits, loss, label=label)
    if reference_logits.shape != observed_logits.shape:
        raise ValueError("anchor logits must share a shape")
    if reference_logits.dtype != observed_logits.dtype:
        raise ValueError("anchor logits must share a dtype")
    if reference_loss.dtype != observed_loss.dtype:
        raise ValueError("anchor losses must share a dtype")
    if reference_logits.device != observed_logits.device:
        raise ValueError("anchor outputs must share a device")
    _validate_backend_evidence(
        backend,
        reference_logits,
        reference_loss,
        observed_logits,
        observed_loss,
    )
    logits_equal = torch.equal(reference_logits, observed_logits)
    loss_equal = torch.equal(reference_loss, observed_loss)
    return DenseAnchorCell(
        dtype=_dtype_name(reference_logits), backend=backend,
        status="passed" if logits_equal and loss_equal else "failed",
        logits_bit_identical=logits_equal, loss_bit_identical=loss_equal,
        logits_mismatch_count=_mismatch_count(reference_logits, observed_logits),
        loss_mismatch_count=_mismatch_count(reference_loss, observed_loss),
        max_absolute_logit_difference=_max_absolute_difference(
            reference_logits, observed_logits
        ),
        max_absolute_loss_difference=_max_absolute_difference(
            reference_loss, observed_loss
        ),
        reason=reason, _factory_token=_ANCHOR_FACTORY_TOKEN,
    )


def pending_dense_anchor_cell(*, dtype: str, backend: str, reason: str) -> DenseAnchorCell:
    """Record an unavailable dense-anchor backend as a typed non-pass."""

    return DenseAnchorCell(
        dtype=dtype, backend=backend, status="pending",
        logits_bit_identical=None, loss_bit_identical=None,
        logits_mismatch_count=None, loss_mismatch_count=None,
        max_absolute_logit_difference=None,
        max_absolute_loss_difference=None,
        reason=reason, _factory_token=_ANCHOR_FACTORY_TOKEN,
    )


def compare_module_off_idempotence(
    all_off_logits: torch.Tensor,
    all_off_loss: torch.Tensor,
    off_after_on_logits: torch.Tensor,
    off_after_on_loss: torch.Tensor,
    on_logits: torch.Tensor,
    on_loss: torch.Tensor,
    *,
    module_name: str,
    requested_recurrent_steps: int,
    executed_recurrent_steps: int,
    backend: str,
    reason: str,
) -> ObservationalInvarianceCell:
    """Compare OFF deletion to its matched background and test ON liveness."""

    if module_name not in _MODULE_SPECS_BY_NAME:
        raise ValueError("module is outside the registered A7 scope")
    spec = _MODULE_SPECS_BY_NAME[module_name]
    if not spec.integrated or not spec.eligible_at(requested_recurrent_steps):
        raise ValueError("only an eligible integrated module can execute an A7 cell")
    for label, logits, loss in (
        ("all_off", all_off_logits, all_off_loss),
        ("off_after_on", off_after_on_logits, off_after_on_loss),
        ("on", on_logits, on_loss),
    ):
        _check_output_pair(logits, loss, label=label)
    if not (
        all_off_logits.shape == off_after_on_logits.shape == on_logits.shape
        and all_off_logits.dtype == off_after_on_logits.dtype == on_logits.dtype
        and all_off_logits.device == off_after_on_logits.device == on_logits.device
    ):
        raise ValueError("all A7 logits must share shape, dtype, and device")
    if not (
        all_off_loss.dtype == off_after_on_loss.dtype == on_loss.dtype
        and all_off_loss.device == off_after_on_loss.device == on_loss.device
    ):
        raise ValueError("all A7 losses must share dtype and device")
    _validate_backend_evidence(
        backend,
        all_off_logits,
        all_off_loss,
        off_after_on_logits,
        off_after_on_loss,
        on_logits,
        on_loss,
    )
    off_logits_equal = torch.equal(all_off_logits, off_after_on_logits)
    off_loss_equal = torch.equal(all_off_loss, off_after_on_loss)
    on_logits_differ = not torch.equal(all_off_logits, on_logits)
    on_loss_differ = not torch.equal(all_off_loss, on_loss)
    passes = (
        off_logits_equal
        and off_loss_equal
        and on_logits_differ
        and on_loss_differ
    )
    return ObservationalInvarianceCell(
        module_name=module_name,
        requested_recurrent_steps=requested_recurrent_steps,
        executed_recurrent_steps=executed_recurrent_steps,
        dtype=_dtype_name(all_off_logits), backend=backend,
        status="passed" if passes else "failed",
        integrated=spec.integrated, eligible=True,
        background_modules=spec.background_modules,
        off_logits_bit_identical=off_logits_equal,
        off_loss_bit_identical=off_loss_equal,
        on_logits_nontrivial=on_logits_differ,
        on_loss_nontrivial=on_loss_differ,
        off_logits_mismatch_count=_mismatch_count(
            all_off_logits, off_after_on_logits
        ),
        off_loss_mismatch_count=_mismatch_count(all_off_loss, off_after_on_loss),
        on_logits_mismatch_count=_mismatch_count(all_off_logits, on_logits),
        on_loss_mismatch_count=_mismatch_count(all_off_loss, on_loss),
        max_absolute_off_logit_difference=_max_absolute_difference(
            all_off_logits, off_after_on_logits
        ),
        max_absolute_off_loss_difference=_max_absolute_difference(
            all_off_loss, off_after_on_loss
        ),
        reason=reason, _factory_token=_CELL_FACTORY_TOKEN,
    )


def typed_observational_invariance_nonpass(
    *,
    module_name: str,
    requested_recurrent_steps: int,
    dtype: str,
    backend: str,
    reason: str,
) -> ObservationalInvarianceCell:
    """Create the only valid non-executed status for one registered cell."""

    if module_name not in _MODULE_SPECS_BY_NAME:
        raise ValueError("module is outside the registered A7 scope")
    if requested_recurrent_steps not in OBS_INV_K_VALUES:
        raise ValueError("requested_recurrent_steps is outside A7")
    if dtype not in OBS_INV_DTYPES or backend not in OBS_INV_BACKENDS:
        raise ValueError("dtype or backend is outside A7")
    spec = _MODULE_SPECS_BY_NAME[module_name]
    eligible = spec.eligible_at(requested_recurrent_steps)
    status = "absent" if not spec.integrated else (
        "ineligible" if not eligible else "pending"
    )
    return ObservationalInvarianceCell(
        module_name=module_name,
        requested_recurrent_steps=requested_recurrent_steps,
        executed_recurrent_steps=None,
        dtype=dtype, backend=backend, status=status,
        integrated=spec.integrated, eligible=eligible,
        background_modules=spec.background_modules,
        off_logits_bit_identical=None, off_loss_bit_identical=None,
        on_logits_nontrivial=None, on_loss_nontrivial=None,
        off_logits_mismatch_count=None, off_loss_mismatch_count=None,
        on_logits_mismatch_count=None, on_loss_mismatch_count=None,
        max_absolute_off_logit_difference=None,
        max_absolute_off_loss_difference=None,
        reason=reason, _factory_token=_CELL_FACTORY_TOKEN,
    )


def deferred_observational_invariance_cell(
    *,
    module_name: str,
    requested_recurrent_steps: int,
    executed_recurrent_steps: int,
    dtype: str,
    backend: str,
    reason: str,
) -> ObservationalInvarianceCell:
    """Backward-compatible name for a typed pending cell."""

    if executed_recurrent_steps != requested_recurrent_steps:
        raise ValueError("planned executed K must equal requested K")
    cell = typed_observational_invariance_nonpass(
        module_name=module_name,
        requested_recurrent_steps=requested_recurrent_steps,
        dtype=dtype, backend=backend, reason=reason,
    )
    if cell.status != "pending":
        raise ValueError("deferred cells must be eligible integrated cells")
    return cell


def observational_invariance_matrix(
    anchor_cells: tuple[DenseAnchorCell, ...] | list[DenseAnchorCell],
    cells: tuple[ObservationalInvarianceCell, ...]
    | list[ObservationalInvarianceCell],
) -> ObservationalInvarianceMatrix:
    """Validate exact registered coverage and assemble a typed draft matrix."""

    if any(type(cell) is not DenseAnchorCell for cell in anchor_cells):
        raise TypeError("anchor_cells must contain exact DenseAnchorCell values")
    if any(type(cell) is not ObservationalInvarianceCell for cell in cells):
        raise TypeError(
            "cells must contain exact ObservationalInvarianceCell values"
        )
    expected_anchor = set(product(OBS_INV_DTYPES, OBS_INV_BACKENDS))
    anchor_coordinates = tuple(cell.coordinate for cell in anchor_cells)
    if len(set(anchor_coordinates)) != len(anchor_coordinates):
        raise ValueError("A7 anchor contains duplicate coordinates")
    if set(anchor_coordinates) != expected_anchor:
        raise ValueError("A7 anchor coverage mismatch")

    expected = set(product(
        tuple(_MODULE_SPECS_BY_NAME), OBS_INV_K_VALUES,
        OBS_INV_DTYPES, OBS_INV_BACKENDS,
    ))
    coordinates = tuple(cell.coordinate for cell in cells)
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("A7 matrix contains duplicate coordinates")
    actual = set(coordinates)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        unexpected = tuple(sorted(actual - expected))
        raise ValueError(
            f"A7 matrix coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    return ObservationalInvarianceMatrix(
        anchor_cells=tuple(sorted(anchor_cells, key=lambda cell: cell.coordinate)),
        cells=tuple(sorted(cells, key=lambda cell: cell.coordinate)),
        _factory_token=_MATRIX_FACTORY_TOKEN,
    )


def promote_complete_observational_invariance(
    matrix: ObservationalInvarianceMatrix,
) -> CompleteObservationalInvarianceReceipt:
    """Mint a complete receipt only after every required backend cell passes."""

    if type(matrix) is not ObservationalInvarianceMatrix:
        raise TypeError("matrix must be an exact ObservationalInvarianceMatrix")
    matrix.require_complete()
    return CompleteObservationalInvarianceReceipt(
        matrix_digest=matrix.digest,
        required_cell_count=len(matrix._required_cells()),
        _factory_token=_PROMOTION_FACTORY_TOKEN,
    )


__all__ = [
    "CompleteObservationalInvarianceReceipt",
    "DenseAnchorCell",
    "OBS_INV_ANCHOR_NAME",
    "OBS_INV_BACKENDS",
    "OBS_INV_DTYPES",
    "OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS",
    "OBS_INV_K_VALUES",
    "OBS_INV_K_AXIS_BACKGROUND",
    "OBS_INV_MATERIALIZED_MODULES",
    "OBS_INV_MODULE_SPECS",
    "ObservationalInvarianceBlocked",
    "ObservationalInvarianceCell",
    "ObservationalInvarianceMatrix",
    "ObservationalInvarianceModuleSpec",
    "compare_dense_anchor",
    "compare_module_off_idempotence",
    "deferred_observational_invariance_cell",
    "observational_invariance_matrix",
    "pending_dense_anchor_cell",
    "promote_complete_observational_invariance",
    "typed_observational_invariance_nonpass",
]
