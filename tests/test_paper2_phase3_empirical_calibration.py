from __future__ import annotations

from pathlib import Path

import torch

from eval.eval_paper2_phase3_empirical_calibration import (
    conservative_design,
    load_seed_trajectory,
)


def _write_rows(root: Path, seed: int) -> None:
    base = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, True],
            [True, False],
        ]
    )
    for look in range(1, 21):
        augmented = base.clone()
        # Persistent error windows model positively autocorrelated checkpoints.
        if 3 + seed <= look <= 12 + seed:
            augmented[0, 0] = False
        if 7 <= look <= 17:
            augmented[0, 1] = True
        if 10 <= look <= 19:
            augmented[2, 0] = True
        torch.save(
            {
                "base_correct_by_horizon": base,
                "bridge_correct_by_horizon": augmented,
            },
            root / f"rows_fixed_evaluation_step_{look * 1000:05d}.pt",
        )


def test_load_seed_trajectory_uses_positive_step_dev_looks(tmp_path: Path) -> None:
    _write_rows(tmp_path, 0)
    trajectory = load_seed_trajectory(tmp_path)
    assert trajectory["steps"] == list(range(1_000, 20_001, 1_000))
    assert trajectory["paired_cells"] == 8
    assert trajectory["estimate"]["looks"] == 20
    assert trajectory["confirm_scoring_spent"] is False


def test_conservative_design_uses_seedwise_envelope() -> None:
    rows = [
        {
            "estimate": {
                "paired_discordant_probability": 0.1,
                "adjacent_checkpoint_autocorrelation": 0.7,
            }
        },
        {
            "estimate": {
                "paired_discordant_probability": 0.2,
                "adjacent_checkpoint_autocorrelation": 0.5,
            }
        },
    ]
    design = conservative_design(rows)
    assert design["paired_discordant_probability"] == 0.2
    assert design["adjacent_checkpoint_autocorrelation"] == 0.7
    assert "no cross-seed pseudo-replication" in design["selection_rule"]
