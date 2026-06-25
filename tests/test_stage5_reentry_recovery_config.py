from __future__ import annotations

from colab.reentry_recovery_config import (
    int_dict_max_key,
    mode_rows_from_counts,
    target_loop_rows_from_counts,
)


def test_target_loop_rows_preserves_counts_and_sorts_numerically() -> None:
    assert target_loop_rows_from_counts({"3": "8", "1": 24, "2": 16}) == "1=24,2=16,3=8"


def test_target_loop_rows_skips_invalid_and_nonpositive_entries() -> None:
    counts = {
        "1": 12,
        "0": 99,
        "-1": 99,
        "2": 0,
        "3": -4,
        "4": "bad",
        "five": 5,
        "6": "7",
    }

    assert target_loop_rows_from_counts(counts) == "1=12,6=7"


def test_target_loop_rows_blocks_fake_presence_only_ladder() -> None:
    counts = {"1": 48, "2": 16, "4": 8}

    assert target_loop_rows_from_counts(counts) != "1=1,2=1,4=1"
    assert target_loop_rows_from_counts(counts) == "1=48,2=16,4=8"


def test_mode_rows_preserves_mode_counts() -> None:
    assert mode_rows_from_counts({"wide": "4", "direct": 12, "deep_narrow": 8}) == (
        "deep_narrow=8,direct=12,wide=4"
    )


def test_int_dict_max_key_ignores_invalid_and_nonpositive_keys() -> None:
    assert int_dict_max_key({"0": 100, "-1": 100, "2": 1, "4": 1, "bad": 99}, default=3) == 4
    assert int_dict_max_key({"bad": 99, "0": 100}, default=3) == 3
