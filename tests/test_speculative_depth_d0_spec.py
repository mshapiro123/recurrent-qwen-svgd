from __future__ import annotations

import pytest

from training.speculative_depth_d0_spec import (
    d0_draft,
    unresolved_paths,
    validate_locked_d0,
)


def test_d0_draft_is_fail_closed_and_has_no_launcher() -> None:
    spec = d0_draft()
    assert spec["status"] == "draft_not_locked"
    assert spec["training_authorized"] is False
    assert spec["launch_target_exists"] is False
    assert spec["substrate_family"] == "Qwen"
    assert spec["dependency"]["t1_lite_verdict_required"] is True
    assert spec["dependency"]["automatic_launch_from_t1"] is False
    assert len(unresolved_paths(spec)) >= 10


def test_d0_draft_cannot_validate_as_locked() -> None:
    with pytest.raises(AssertionError, match="not locked"):
        validate_locked_d0(d0_draft())
