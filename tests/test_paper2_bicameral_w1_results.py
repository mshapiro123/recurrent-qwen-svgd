from analysis.analyze_paper2_bicameral_w1_results import aggregate


def _seed(seed: int) -> dict:
    cells = []
    for family in ("a", "b", "c", "d", "g"):
        cells.extend(
            [
                {"arm": f"l0{family}", "seed": seed, "mean": 0.2, "ci_low": 0.1},
                {"arm": f"l5_{family}", "seed": seed, "mean": 0.0, "ci_low": -0.1},
            ]
        )
    cells.append({"arm": "l4", "seed": seed, "mean": 0.0, "ci_low": -0.1})
    return {"cells": cells}


def _w3() -> dict:
    seed = {
        "dm5": {
            "cells": [
                {"rank": 4, "mean_cosine": 0.8},
                {"rank": 8, "mean_cosine": 0.9},
            ]
        },
        "x6": {
            "common_mode_energy_fraction": 0.7,
            "directions": [
                {
                    "residual_energy_fraction": 0.3,
                    "battery_association": {"eta_squared": 0.9},
                    "cluster_association": {"eta_squared": 0.8},
                }
            ],
        },
    }
    return {"kind": "paper2_bicameral_w3_desk_wave_v1", "seeds": {"0": seed, "1": seed}}


def test_aggregate_applies_registered_tie_break() -> None:
    result = aggregate([_seed(0), _seed(1)], _w3())
    assert result["phase_a"]["decision"]["winner"] == "l0d"
    assert result["w3_desk"]["dm5_best_rank_cells"]["0"]["rank"] == 8
    assert result["optimizer_steps"] == 0
