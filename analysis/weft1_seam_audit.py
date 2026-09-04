"""Fail-closed mathematical primitives for the D-EP-2 seam audit.

The ratified audit compares sender energy captured by a receiver-sensitive
subspace.  A permutation of sender *rows* cannot be used as a null for that
statistic: row permutation leaves the sender covariance unchanged exactly.
This module records that invariant in executable form while the replacement
null remains a strategy decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class SeamNullInvariantError(RuntimeError):
    """Raised when a proposed null cannot change the registered statistic."""


@dataclass(frozen=True)
class SeamNullValidation:
    maximum_covariance_difference: float
    tolerance: float
    changes_registered_statistic: bool


def sender_covariance(samples: torch.Tensor, *, centered: bool) -> torch.Tensor:
    """Return a float64 feature covariance with an explicit centering choice."""

    if not isinstance(samples, torch.Tensor):
        raise TypeError("sender samples must be a tensor")
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("sender samples must have shape [n >= 2, d >= 1]")
    work = samples.detach().double()
    if not bool(torch.isfinite(work).all()):
        raise ValueError("sender samples must be finite")
    if centered:
        work = work - work.mean(dim=0, keepdim=True)
        denominator = work.shape[0] - 1
    else:
        denominator = work.shape[0]
    return work.mT.matmul(work).div(float(denominator))


def captured_sender_fraction(
    sender_covariance_matrix: torch.Tensor,
    receiver_projector: torch.Tensor,
) -> torch.Tensor:
    """Return ``tr(P_r C_s) / tr(C_s)`` in float64.

    Projector construction (GGN definition, rank and Lanczos details) belongs
    to the later component-checkpoint runner.  This primitive validates only
    the algebra required to make a receipt value meaningful.
    """

    if sender_covariance_matrix.ndim != 2:
        raise ValueError("sender covariance must be a matrix")
    if sender_covariance_matrix.shape[0] != sender_covariance_matrix.shape[1]:
        raise ValueError("sender covariance must be square")
    if receiver_projector.shape != sender_covariance_matrix.shape:
        raise ValueError("receiver projector must match sender covariance")
    covariance = sender_covariance_matrix.detach().double()
    projector = receiver_projector.detach().double()
    if not bool(torch.isfinite(covariance).all()) or not bool(
        torch.isfinite(projector).all()
    ):
        raise ValueError("seam audit matrices must be finite")
    denominator = torch.trace(covariance)
    if not bool(denominator > 0):
        raise ValueError("sender covariance must have positive trace")
    return torch.trace(projector.matmul(covariance)).div(denominator)


def validate_sender_null(
    sender_samples: torch.Tensor,
    null_samples: torch.Tensor,
    *,
    centered: bool,
    tolerance: float = 1e-12,
) -> SeamNullValidation:
    """Reject a null whose sender covariance is unchanged.

    D-EP-2's current row-shuffle proposal is intentionally rejected here.  A
    future non-invariant null can pass this preliminary guard, but this guard
    alone never mints the SEAM-AUDIT gate.
    """

    if sender_samples.shape != null_samples.shape:
        raise ValueError("sender and null samples must have identical shapes")
    if not isinstance(tolerance, float) or tolerance < 0.0:
        raise ValueError("tolerance must be a non-negative float")
    covariance = sender_covariance(sender_samples, centered=centered)
    null_covariance = sender_covariance(null_samples, centered=centered)
    maximum_difference = float((covariance - null_covariance).abs().max().item())
    validation = SeamNullValidation(
        maximum_covariance_difference=maximum_difference,
        tolerance=tolerance,
        changes_registered_statistic=maximum_difference > tolerance,
    )
    if not validation.changes_registered_statistic:
        raise SeamNullInvariantError(
            "proposed sender null leaves C_s unchanged and cannot satisfy the "
            "registered >=3x seam-audit comparison"
        )
    return validation
