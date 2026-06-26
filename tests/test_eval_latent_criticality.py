from __future__ import annotations

import torch

from eval.eval_latent_criticality import (
    auc_score,
    eval_feature_selector,
    fit_feature_selector,
    harmed_rescued_summary,
    spectral_signatures,
)


def test_spectral_signatures_detect_single_mode_concentration() -> None:
    mask = torch.ones(1, 12, dtype=torch.long)
    spread = torch.randn(1, 12, 8)
    single = torch.zeros(1, 12, 8)
    single[0, :, 0] = torch.linspace(-1, 1, 12)

    spread_sig = spectral_signatures(spread, mask)
    single_sig = spectral_signatures(single, mask)

    assert single_sig["participation_ratio"] < spread_sig["participation_ratio"]
    assert single_sig["effective_rank"] < spread_sig["effective_rank"]
    assert single_sig["dragon_king_gap"] > spread_sig["dragon_king_gap"]


def test_auc_score_and_feature_selector_capture_depth_signal() -> None:
    rows = [
        {"benchmark": "b", "id": "a", "depth": 1, "loop_hit": False, "is_correct_depth": False, "score": 0.1},
        {"benchmark": "b", "id": "a", "depth": 2, "loop_hit": True, "is_correct_depth": True, "score": 0.9},
        {"benchmark": "b", "id": "b", "depth": 1, "loop_hit": True, "is_correct_depth": True, "score": 0.8},
        {"benchmark": "b", "id": "b", "depth": 2, "loop_hit": False, "is_correct_depth": False, "score": 0.2},
    ]

    assert auc_score(rows, "score") == 1.0
    selector = fit_feature_selector(rows, "score")

    assert selector is not None
    assert selector["direction"] == "max"
    evaluated = eval_feature_selector(rows, selector)
    assert evaluated["correct"] == 2
    assert evaluated["total"] == 2


def test_harmed_rescued_summary_reports_opposite_transition_signs() -> None:
    rows = [
        {"benchmark": "b", "id": "h", "depth": 1, "loop_hit": True, "is_correct_depth": True, "state_rms": 1.0},
        {"benchmark": "b", "id": "h", "depth": 2, "loop_hit": False, "is_correct_depth": False, "state_rms": 2.0},
        {"benchmark": "b", "id": "r", "depth": 1, "loop_hit": False, "is_correct_depth": False, "state_rms": 2.0},
        {"benchmark": "b", "id": "r", "depth": 2, "loop_hit": True, "is_correct_depth": True, "state_rms": 1.0},
    ]

    summary = harmed_rescued_summary(rows)

    assert summary["harmed_examples"] == 1
    assert summary["rescued_examples"] == 1
    assert summary["features"]["state_rms"]["opposite_sign_depth2_minus_depth1"] is True
