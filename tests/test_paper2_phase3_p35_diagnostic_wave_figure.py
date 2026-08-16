from __future__ import annotations

from pathlib import Path

from analysis.build_paper2_phase3_p35_diagnostic_wave_figure import build


def test_diagnostic_wave_figure_separates_registered_and_clamped_depth(tmp_path: Path) -> None:
    amplitude = {
        "conditions": {
            f"seed_{seed}_ceiling_{str(ceiling).replace('.', 'p')}": {"net_rows": seed + index}
            for seed in (0, 1)
            for index, ceiling in enumerate((0.02, 0.05, 0.08, 0.11))
        },
        "selected_ceiling_under_preregistered_rule": 0.08,
    }
    depth = {
        "cells": {
            f"seed_{seed}_k_{k}": {"pooled": {"correct": 500 + seed + k}}
            for seed in (0, 1) for k in range(1, 7)
        },
        "marginal_improvement": {
            f"seed_{seed}_k_{k}_minus_k_{k-1}": {"pooled": {"net_rows": k - seed}}
            for seed in (0, 1) for k in range(2, 7)
        },
    }
    output = tmp_path / "wave"
    build(amplitude, depth, output)
    assert output.with_suffix(".png").stat().st_size > 1000
    assert output.with_suffix(".svg").stat().st_size > 1000
