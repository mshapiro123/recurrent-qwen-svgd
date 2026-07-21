from __future__ import annotations

import pytest

from colab.stage5_frontier_metrics import bar_crossing_frontier


CURVES = {
    "support4": {
        "1": 1.0,
        "2": 0.9609375,
        "3": 0.96875,
        "4": 0.96875,
        "5": 0.890625,
        "6": 0.6484375,
        "7": 0.328125,
        "8": 0.09375,
    },
    "support6": {
        "1": 1.0,
        "2": 1.0,
        "3": 0.9765625,
        "4": 1.0,
        "5": 0.9921875,
        "6": 0.9609375,
        "7": 0.90625,
        "8": 0.828125,
        "9": 0.7109375,
        "10": 0.5390625,
    },
    "support8": {
        "1": 1.0,
        "2": 0.9921875,
        "3": 0.984375,
        "4": 0.9765625,
        "5": 0.9921875,
        "6": 0.984375,
        "7": 0.96875,
        "8": 0.953125,
        "9": 0.90625,
        "10": 0.8828125,
        "11": 0.7578125,
        "12": 0.6796875,
        "13": 0.4453125,
        "14": 0.2421875,
    },
    "support12": {
        "1": 1.0,
        "2": 0.9921875,
        "3": 0.9921875,
        "4": 1.0,
        "5": 0.9921875,
        "6": 0.9921875,
        "7": 0.9921875,
        "8": 0.9765625,
        "9": 0.9765625,
        "10": 0.9765625,
        "11": 0.96875,
        "12": 0.9765625,
        "13": 0.9609375,
        "14": 0.9140625,
        "15": 0.8828125,
        "16": 0.859375,
        "17": 0.8046875,
        "18": 0.703125,
        "19": 0.640625,
        "20": 0.4609375,
        "21": 0.34375,
        "22": 0.109375,
    },
}


@pytest.mark.parametrize(
    ("name", "bar", "expected"),
    [
        ("support4", 0.60, 6.151219512195122),
        ("support4", 0.71, 5.7458064516129035),
        ("support4", 0.80, 5.374193548387097),
        ("support6", 0.60, 9.645454545454545),
        ("support6", 0.71, 9.005454545454546),
        ("support6", 0.80, 8.24),
        ("support8", 0.60, 12.34),
        ("support8", 0.71, 11.612),
        ("support8", 0.80, 10.6625),
        ("support12", 0.60, 19.22608695652174),
        ("support12", 0.71, 17.93230769230769),
        ("support12", 0.80, 17.046153846153846),
    ],
)
def test_registered_curves_have_expected_frontiers(name: str, bar: float, expected: float) -> None:
    assert bar_crossing_frontier(CURVES[name], bar=bar) == pytest.approx(expected)


def test_ratio_variation_remains_bounded_at_each_bar() -> None:
    supports = {"support4": 4, "support6": 6, "support8": 8, "support12": 12}
    for bar in (0.60, 0.71, 0.80):
        ratios = [bar_crossing_frontier(CURVES[name], bar=bar) / support for name, support in supports.items()]
        assert max(ratios) - min(ratios) < 0.09
