"""Typed A7 observational-invariance matrix receipts.

The matrix compares a structural-OFF construction with an independently
constructed reference graph and requires exact tensor identity.  It records
GPU cells as deferred until they are actually executed under the PRE-FLIGHT
meter; a deferred cell can never be mistaken for a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch


OBS_INV_MATERIALIZED_MODULES = (
    "front_hadamard_experts",
    "recurrence",
    "static_kv_core",
    "reentry_bridge",
    "scratch",
    "lane_carrier",
    "engram",
    "long_term_memory",
)
OBS_INV_K_VALUES = (1, 2, 4, 8)
OBS_INV_DTYPES = ("float32", "bfloat16")
OBS_INV_BACKENDS = ("cpu", "cuda_deterministic")
OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS = (
    "integrated_rotor_carrier",
    "per_band_callosum",
    "sidecar",
)


class ObservationalInvarianceBlocked(RuntimeError):
    """Raised when a required A7 cell is missing or not bit-identical."""


@dataclass(frozen=True)
class ObservationalInvarianceCell:
    """One module × K × dtype × backend result."""

    module_name: str
    requested_recurrent_steps: int
    executed_recurrent_steps: int
    dtype: str
    backend: str
    status: str
    bit_identical: bool | None
    mismatch_count: int | None
    max_absolute_difference: float | None
    reason: str

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
    """Complete A7 matrix with passes distinct from deferred GPU work."""

    cells: tuple[ObservationalInvarianceCell, ...]
    excluded_absent_integrations: tuple[str, ...]

    @property
    def failed_cells(self) -> tuple[ObservationalInvarianceCell, ...]:
        return tuple(cell for cell in self.cells if cell.status == "failed")

    @property
    def deferred_cells(self) -> tuple[ObservationalInvarianceCell, ...]:
        return tuple(cell for cell in self.cells if cell.status == "deferred")

    @property
    def cpu_passed(self) -> bool:
        cpu = tuple(cell for cell in self.cells if cell.backend == "cpu")
        return bool(cpu) and all(cell.status == "passed" for cell in cpu)

    @property
    def complete(self) -> bool:
        return all(cell.status == "passed" for cell in self.cells)

    def require_cpu_passed(self) -> None:
        failed = tuple(
            cell.coordinate
            for cell in self.cells
            if cell.backend == "cpu" and cell.status != "passed"
        )
        if failed:
            raise ObservationalInvarianceBlocked(
                f"A7 CPU observational-invariance cells did not pass: {failed}"
            )

    def require_complete(self) -> None:
        incomplete = tuple(
            cell.coordinate for cell in self.cells if cell.status != "passed"
        )
        if incomplete:
            raise ObservationalInvarianceBlocked(
                f"A7 matrix still has failed or deferred cells: {incomplete}"
            )


def compare_observational_invariance(
    reference: torch.Tensor,
    observed: torch.Tensor,
    *,
    module_name: str,
    requested_recurrent_steps: int,
    executed_recurrent_steps: int,
    backend: str,
    reason: str,
) -> ObservationalInvarianceCell:
    """Compare one executed cell with exact equality, never ``allclose``."""

    if module_name not in OBS_INV_MATERIALIZED_MODULES:
        raise ValueError(f"module {module_name!r} is outside the materialized A7 scope")
    if requested_recurrent_steps not in OBS_INV_K_VALUES:
        raise ValueError("requested_recurrent_steps is outside the A7 matrix")
    if type(executed_recurrent_steps) is not int or executed_recurrent_steps < 1:
        raise ValueError("executed_recurrent_steps must be a positive integer")
    if backend not in OBS_INV_BACKENDS:
        raise ValueError("backend is outside the A7 matrix")
    if reference.shape != observed.shape:
        raise ValueError("A7 tensors must have identical shapes")
    if reference.dtype != observed.dtype:
        raise ValueError("A7 tensors must have identical dtypes")
    if reference.device != observed.device:
        raise ValueError("A7 tensors must share a device")
    if reference.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("A7 supports only float32 and bfloat16 outputs")

    identical = torch.equal(reference, observed)
    mismatch_count = int(torch.count_nonzero(reference.ne(observed)).cpu().item())
    max_difference = float(
        (reference.float() - observed.float()).abs().max().cpu().item()
    )
    if not math.isfinite(max_difference):
        identical = False
    return ObservationalInvarianceCell(
        module_name=module_name,
        requested_recurrent_steps=requested_recurrent_steps,
        executed_recurrent_steps=executed_recurrent_steps,
        dtype=str(reference.dtype).removeprefix("torch."),
        backend=backend,
        status="passed" if identical else "failed",
        bit_identical=identical,
        mismatch_count=mismatch_count,
        max_absolute_difference=max_difference,
        reason=reason,
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
    """Create an explicit non-pass for a governed cell not yet run."""

    if module_name not in OBS_INV_MATERIALIZED_MODULES:
        raise ValueError(f"module {module_name!r} is outside the materialized A7 scope")
    if requested_recurrent_steps not in OBS_INV_K_VALUES:
        raise ValueError("requested_recurrent_steps is outside the A7 matrix")
    if dtype not in OBS_INV_DTYPES or backend not in OBS_INV_BACKENDS:
        raise ValueError("dtype or backend is outside the A7 matrix")
    return ObservationalInvarianceCell(
        module_name=module_name,
        requested_recurrent_steps=requested_recurrent_steps,
        executed_recurrent_steps=executed_recurrent_steps,
        dtype=dtype,
        backend=backend,
        status="deferred",
        bit_identical=None,
        mismatch_count=None,
        max_absolute_difference=None,
        reason=reason,
    )


def observational_invariance_matrix(
    cells: tuple[ObservationalInvarianceCell, ...]
    | list[ObservationalInvarianceCell],
) -> ObservationalInvarianceMatrix:
    """Validate exact Cartesian coverage and mint the typed A7 matrix."""

    expected = set(
        product(
            OBS_INV_MATERIALIZED_MODULES,
            OBS_INV_K_VALUES,
            OBS_INV_DTYPES,
            OBS_INV_BACKENDS,
        )
    )
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
    ordered = tuple(sorted(cells, key=lambda cell: cell.coordinate))
    return ObservationalInvarianceMatrix(
        cells=ordered,
        excluded_absent_integrations=OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS,
    )


__all__ = [
    "OBS_INV_BACKENDS",
    "OBS_INV_DTYPES",
    "OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS",
    "OBS_INV_K_VALUES",
    "OBS_INV_MATERIALIZED_MODULES",
    "ObservationalInvarianceBlocked",
    "ObservationalInvarianceCell",
    "ObservationalInvarianceMatrix",
    "compare_observational_invariance",
    "deferred_observational_invariance_cell",
    "observational_invariance_matrix",
]
