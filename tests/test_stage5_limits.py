from __future__ import annotations

import pytest

from colab.stage5_limits import limit_args, limit_label, parse_optional_limit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 17),
        ("20", 20),
        (20, 20),
        ("full", None),
        ("none", None),
        ("all", None),
        ("unlimited", None),
        ("0", None),
        ("-1", None),
        ("", None),
    ],
)
def test_parse_optional_limit(value: str | int | None, expected: int | None) -> None:
    assert parse_optional_limit(value, default=17) == expected


def test_limit_args_omits_full_split_limit() -> None:
    assert limit_args(None) == []
    assert limit_args(50) == ["--limit", "50"]


def test_limit_label_names_full_split() -> None:
    assert limit_label(None) == "full"
    assert limit_label(400) == "400"
