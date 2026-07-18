from __future__ import annotations

import pytest

from training.internal_think_token_spec import (
    INTERNAL_CONTROL_TOKENS,
    natural_trace_survey,
    phase_t0_spec,
    validate_tokenizer_preflight,
)


def test_phase_t0_is_prep_only_and_keeps_all_training_gates_closed() -> None:
    spec = phase_t0_spec()

    assert spec["status"] == "preparation_only_not_authorized_for_training"
    assert len(spec["tokens"]["values"]) == 3
    assert spec["identity_contract"]["control_tokens_never_emitted"] is True
    assert spec["logging_contract"]["per_trajectory_unaveraged_control_readout"] is True
    assert spec["phase_t1"]["training_mix"]["rehearsal_fraction"] == 0.30
    assert spec["phase_t3"]["authorized"] is False


def test_tokenizer_preflight_rejects_collisions_and_wrong_added_count() -> None:
    receipt = validate_tokenizer_preflight(
        existing_vocabulary={"A", "B"},
        added_token_count=3,
    )
    assert receipt["tokens"] == list(INTERNAL_CONTROL_TOKENS)

    with pytest.raises(AssertionError, match="collision"):
        validate_tokenizer_preflight(
            existing_vocabulary={"A", INTERNAL_CONTROL_TOKENS[0]},
            added_token_count=3,
        )
    with pytest.raises(AssertionError, match="exactly 3"):
        validate_tokenizer_preflight(
            existing_vocabulary={"A", "B"},
            added_token_count=2,
        )


def test_trace_survey_separates_verifiable_math_from_fable_agent_traces() -> None:
    rows = {row["id"]: row for row in natural_trace_survey()["datasets"]}

    assert rows["open-r1/OpenR1-Math-220k"]["license"] == "apache-2.0"
    assert rows["open-r1/OpenR1-Math-220k"]["phase_t2_priority"] == "primary_candidate"
    assert rows["Glint-Research/Fable-5-traces"]["license"] == "agpl-3.0"
    assert (
        rows["Glint-Research/Fable-5-traces"]["phase_t2_priority"]
        == "exclude_from_primary_halting_curriculum"
    )
