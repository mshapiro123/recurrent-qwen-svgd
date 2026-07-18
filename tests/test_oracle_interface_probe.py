from __future__ import annotations

import json
from pathlib import Path

from training.oracle_interface_probe_spec import (
    LOCKED_CONTROL_GROUPS,
    LOCKED_CONTROL_ROWS,
    LOCKED_CONTROL_TRANSITIONS,
    preregistration_payload,
    score_oracle_interface_probe,
    summarize_oracle_arm,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = (
    ROOT
    / "outputs"
    / "stage5"
    / "stage5_phase_g_multitarget_control_20260718"
    / "data"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_locked_source_data_matches_preregistered_denominators() -> None:
    train_rows = _read_jsonl(SOURCE_DATA / "train.jsonl")
    control_rows = _read_jsonl(SOURCE_DATA / "posterior_control.jsonl")

    assert len(train_rows) == 1899
    assert len(control_rows) == LOCKED_CONTROL_ROWS
    assert (
        len({str(row["base_problem_id"]) for row in control_rows})
        == LOCKED_CONTROL_GROUPS
    )
    assert (
        sum(int(row["depth"]) for row in control_rows)
        == LOCKED_CONTROL_TRANSITIONS
    )
    assert all(
        len({int(row["depth"]) for row in control_rows if row["base_problem_id"] == group})
        == 1
        for group in {row["base_problem_id"] for row in control_rows}
    )


def test_preregistration_forbids_automatic_successor() -> None:
    payload = preregistration_payload()

    assert payload["terminal_probe"]
    assert payload["heldout"] == {
        "rows": 106,
        "groups": 32,
        "transitions": 305,
    }
    assert payload["objective"].endswith("no KL or stochastic latent")
    assert "coverage" in payload["deferred"]


def _rows(
    *,
    nondefault_rate: float,
    overall_rate: float,
    legality_rate: float,
) -> tuple[list[dict], list[dict]]:
    nondefault_total = 100
    default_total = LOCKED_CONTROL_TRANSITIONS - nondefault_total
    nondefault_correct = round(nondefault_total * nondefault_rate)
    overall_correct = round(LOCKED_CONTROL_TRANSITIONS * overall_rate)
    default_correct = max(0, min(default_total, overall_correct - nondefault_correct))
    legal = round(LOCKED_CONTROL_TRANSITIONS * legality_rate)
    transitions = []
    for index in range(LOCKED_CONTROL_TRANSITIONS):
        is_default = index >= nondefault_total
        local = index - nondefault_total if is_default else index
        controlled = local < (default_correct if is_default else nondefault_correct)
        transitions.append(
            {
                "id": f"row_{index % LOCKED_CONTROL_ROWS}",
                "base_problem_id": f"group_{index % 32}",
                "depth": 1 + index % 4,
                "loop_index": 1 + index % (1 + index % 4),
                "command_is_default": is_default,
                "controlled": controlled,
                "legal": index < legal,
                "target_margin": 1.0 if controlled else -1.0,
            }
        )
    terminals = [
        {
            "id": f"row_{index}",
            "base_problem_id": f"group_{index % 32}",
            "valid": index < 80,
        }
        for index in range(LOCKED_CONTROL_ROWS)
    ]
    return transitions, terminals


def _arm(
    route: str,
    *,
    nondefault_rate: float,
    overall_rate: float,
    legality_rate: float = 0.98,
) -> dict:
    transitions, terminals = _rows(
        nondefault_rate=nondefault_rate,
        overall_rate=overall_rate,
        legality_rate=legality_rate,
    )
    return summarize_oracle_arm(
        transitions,
        terminals,
        route=route,
        identity_exact=True,
        frozen_lineage_unchanged=True,
    )


def test_nondefault_gate_prevents_free_default_control_from_passing() -> None:
    arm = _arm("additive", nondefault_rate=0.80, overall_rate=0.91)

    assert not arm["passed"]
    assert not arm["checks"]["nondefault_branch_control"]["passed"]
    assert arm["checks"]["overall_transition_control"]["passed"]


def test_film_only_pass_localizes_the_interface() -> None:
    additive = _arm("additive", nondefault_rate=0.80, overall_rate=0.88)
    film = _arm("film", nondefault_rate=0.90, overall_rate=0.93)

    result = score_oracle_interface_probe([additive, film])

    assert result["measured_reading"] == "FILM_CONTROLS_ADDITIVE_DOES_NOT"
    assert not result["automatic_successor_authorized"]


def test_both_fail_closes_reentry_conditioning() -> None:
    additive = _arm("additive", nondefault_rate=0.70, overall_rate=0.80)
    film = _arm("film", nondefault_rate=0.75, overall_rate=0.82)

    result = score_oracle_interface_probe([film, additive])

    assert result["measured_reading"] == "BOTH_FAIL"


def test_both_pass_localizes_a0_to_training_objective() -> None:
    additive = _arm("additive", nondefault_rate=0.90, overall_rate=0.93)
    film = _arm("film", nondefault_rate=0.90, overall_rate=0.93)

    result = score_oracle_interface_probe([additive, film])

    assert result["measured_reading"] == "BOTH_PASS"


def test_additive_only_is_explicitly_handled_without_automatic_successor() -> None:
    additive = _arm("additive", nondefault_rate=0.90, overall_rate=0.93)
    film = _arm("film", nondefault_rate=0.75, overall_rate=0.82)

    result = score_oracle_interface_probe([additive, film])

    assert result["measured_reading"] == "ADDITIVE_ONLY_UNEXPECTED"
    assert not result["automatic_successor_authorized"]
