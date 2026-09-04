from __future__ import annotations

import pytest
import torch

from analysis.weft1_seam_audit import (
    SeamNullInvariantError,
    captured_sender_fraction,
    sender_covariance,
    validate_sender_null,
)


@pytest.mark.parametrize("centered", (False, True))
def test_dep2_row_shuffle_is_mathematically_invalid_sender_null(
    centered: bool,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(902_2026)
    sender = torch.randn(37, 11, generator=generator, dtype=torch.float64)
    shuffled = sender[torch.randperm(sender.shape[0], generator=generator)]

    torch.testing.assert_close(
        sender_covariance(sender, centered=centered),
        sender_covariance(shuffled, centered=centered),
        rtol=0.0,
        atol=2e-15,
    )
    with pytest.raises(SeamNullInvariantError, match="leaves C_s unchanged"):
        validate_sender_null(
            sender,
            shuffled,
            centered=centered,
            tolerance=2e-15,
        )


def test_dep2_captured_fraction_is_unchanged_by_sender_row_shuffle() -> None:
    sender = torch.tensor(
        [[1.0, 2.0, 0.0], [3.0, -1.0, 2.0], [-2.0, 4.0, 1.0]],
        dtype=torch.float64,
    )
    projector = torch.diag(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64))
    original = captured_sender_fraction(
        sender_covariance(sender, centered=False), projector
    )
    shuffled = captured_sender_fraction(
        sender_covariance(sender[[2, 0, 1]], centered=False), projector
    )

    torch.testing.assert_close(original, shuffled, rtol=0.0, atol=1e-15)


def test_dep2_noninvariant_null_can_clear_only_the_preliminary_guard() -> None:
    sender = torch.eye(4, dtype=torch.float64)
    altered = sender.clone()
    altered[:, 0] *= 2.0

    receipt = validate_sender_null(
        sender,
        altered,
        centered=False,
        tolerance=1e-12,
    )
    assert receipt.changes_registered_statistic
    assert receipt.maximum_covariance_difference > receipt.tolerance
