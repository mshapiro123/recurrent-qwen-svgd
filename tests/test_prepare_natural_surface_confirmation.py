from __future__ import annotations

from eval.prepare_natural_surface_confirmation import freeze_confirmation_split


def _row(family: str, depth: int, index: int) -> dict:
    paired = f"paired_d{depth:02d}_{index:03d}"
    return {
        "id": f"{family}_{paired}",
        "paired_instance_id": paired,
        "verbal_surface_family": family,
        "depth": depth,
    }


def test_confirmation_split_is_balanced_and_disjoint_from_selection() -> None:
    relay = [_row("relay", depth, index) for depth in (1, 2) for index in range(4)]
    pointer = [_row("pointer", depth, index) for depth in (1, 2) for index in range(4)]

    confirmation, manifest = freeze_confirmation_split(
        relay,
        pointer,
        selection_rows_per_family_depth=2,
        confirmation_rows_per_family_depth=2,
        max_depth=2,
    )

    assert len(confirmation) == 8
    assert manifest["selection"]["rows"] == 8
    assert manifest["confirmation"]["rows"] == 8
    assert manifest["paired_overlap"] == []
    assert manifest["confirmation"]["by_family_depth"]["relay_d1"] == 2
    assert all(row["paired_instance_id"].endswith(("002", "003")) for row in confirmation)


def test_confirmation_split_rejects_insufficient_rows() -> None:
    relay = [_row("relay", 1, index) for index in range(3)]
    pointer = [_row("pointer", 1, index) for index in range(3)]

    try:
        freeze_confirmation_split(
            relay,
            pointer,
            selection_rows_per_family_depth=2,
            confirmation_rows_per_family_depth=2,
            max_depth=1,
        )
    except ValueError as exc:
        assert "Insufficient rows" in str(exc)
    else:
        raise AssertionError("Confirmation split accepted an undersized source")
