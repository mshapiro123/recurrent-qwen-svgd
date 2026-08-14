from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

from analysis.build_paper2_phase3_p34_a2_autopsy import (
    exact_one_sided_sign_power,
    maximum_miss_streak,
    projected_log_weight_update,
    reconstruct_window_rungs,
    shares_for_weights,
)


def test_reconstruct_window_rungs_applies_share_then_task_transitions() -> None:
    summary = {
        "share_contract_events": [
            {"step": 100, "controller": None},
            {
                "step": 200,
                "controller": {"rung_before": 1, "rung_after": 0},
            },
            {"step": 300, "controller": None},
            {"step": 400, "controller": None},
            {"step": 500, "controller": None},
        ],
        "history": [
            {"step": 200, "controller": {"rung_before": 0, "rung_after": 0}},
            {"step": 400, "controller": {"rung_before": 0, "rung_after": 1}},
        ],
    }
    assert reconstruct_window_rungs(summary) == [1, 1, 0, 0, 1]


def test_projected_log_weight_update_is_clipped_and_kl_normalized() -> None:
    weights = {"kl": 1.0, "aim": 1.0, "ce": 1.0, "gate": 1.0, "preserve": 1.0}
    observed = {"kl": 0.05, "aim": 0.80, "ce": 0.05, "gate": 0.05, "preserve": 0.05}
    target = {"kl": 0.40, "aim": 0.20, "ce": 0.15, "gate": 0.05, "preserve": 0.20}
    updated, delta = projected_log_weight_update(
        weights=weights,
        observed_shares=observed,
        target_shares=target,
        gain=1.0,
        max_abs_log_update=0.25,
    )
    assert updated["kl"] == 1.0
    assert all(abs(value) <= 0.25 + 1e-12 for value in delta.values())
    assert delta["kl"] == 0.25
    assert delta["aim"] == -0.25


def test_shares_for_weights_normalizes_positive_masses() -> None:
    shares = shares_for_weights({"kl": 2.0, "aim": 1.0}, {"kl": 1.0, "aim": 2.0})
    assert shares == {"kl": 0.5, "aim": 0.5}


def test_maximum_miss_streak_resets_on_pass() -> None:
    rows = [
        {"failed_contracts": ["kl"]},
        {"failed_contracts": ["kl"]},
        {"failed_contracts": []},
        {"failed_contracts": ["gate"]},
        {"failed_contracts": ["gate"]},
        {"failed_contracts": ["gate"]},
    ]
    assert maximum_miss_streak(rows) == 3


def test_sign_test_power_increases_with_effect() -> None:
    low = exact_one_sided_sign_power(rows=1000, discordance=0.10, delta=0.005)
    high = exact_one_sided_sign_power(rows=1000, discordance=0.10, delta=0.03)
    assert 0.0 <= low < high <= 1.0
    assert math.isfinite(low) and math.isfinite(high)


def test_a2_draft_is_non_executable_and_receipted() -> None:
    root = Path(__file__).resolve().parents[1]
    draft = json.loads(
        (root / "training/paper2_phase3_p34_amendment_a2.draft.json").read_text(
            encoding="utf-8"
        )
    )
    assert draft["locked_before_resumed_training"] is False
    assert draft["training_authorized"] is False
    assert draft["mark_ratified"] is False
    assert draft["exact_preoptimizer_preflight_required"] is True
    assert draft["exact_preoptimizer_preflight_status"] == "not_run"
    assert draft["slot_seed_0"]["continuation_authorized"] is False
    for receipt_name in ("strategy_response", "amendment_document", "cpu_autopsy"):
        receipt = draft[receipt_name]
        payload = (root / receipt["path"]).read_bytes()
        assert len(payload) == receipt["bytes"]
        assert hashlib.sha256(payload).hexdigest() == receipt["sha256"]


def test_a2_rung_targets_are_probability_vectors() -> None:
    root = Path(__file__).resolve().parents[1]
    draft = json.loads(
        (root / "training/paper2_phase3_p34_amendment_a2.draft.json").read_text(
            encoding="utf-8"
        )
    )
    for target in draft["rung_targets"].values():
        assert set(target) == {"kl", "aim", "ce", "gate", "preserve"}
        assert math.isclose(sum(target.values()), 1.0, abs_tol=1e-12)
        assert target["kl"] >= 0.35
        assert target["aim"] >= 0.15
        assert target["ce"] >= 0.10
        assert target["gate"] >= 0.03
        assert target["preserve"] <= 0.25
