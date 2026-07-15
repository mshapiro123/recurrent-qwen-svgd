import pytest

from training.continuation_policy import (
    GuardrailFloor,
    assert_launch_guardrail_floors,
    assert_training_lineage,
)


def test_launch_floor_assertion_reports_every_metric() -> None:
    receipt = {"synthetic": {"min_accuracy": 0.95}, "natural": {"accuracy": 0.94}}
    result = assert_launch_guardrail_floors(
        receipt,
        [
            GuardrailFloor("synthetic.min_accuracy", 0.93),
            GuardrailFloor("natural.accuracy", 0.90),
        ],
    )
    assert result["status"] == "green"
    assert [row["metric"] for row in result["checks"]] == [
        "synthetic.min_accuracy",
        "natural.accuracy",
    ]


def test_launch_floor_assertion_fails_before_continuation() -> None:
    with pytest.raises(RuntimeError, match="synthetic.min_accuracy=0.8125.*floor=0.93"):
        assert_launch_guardrail_floors(
            {"synthetic": {"min_accuracy": 0.8125}},
            [GuardrailFloor("synthetic.min_accuracy", 0.93)],
        )


def test_full_block_training_requires_non_promotable_disposable_branch() -> None:
    result = assert_training_lineage(
        regime="disposable_measurement",
        full_block_trainable=True,
        checkpoint_promotable=False,
        successor_source_allowed=False,
    )
    assert result["status"] == "allowed"
    assert result["keeper_successor"] is False

    with pytest.raises(RuntimeError, match="full-block training"):
        assert_training_lineage(
            regime="frozen_asset",
            full_block_trainable=True,
            checkpoint_promotable=False,
            successor_source_allowed=False,
        )
    with pytest.raises(RuntimeError, match="never be promoted"):
        assert_training_lineage(
            regime="disposable_measurement",
            full_block_trainable=True,
            checkpoint_promotable=True,
            successor_source_allowed=False,
        )


def test_frozen_asset_allows_detachable_adapter_only() -> None:
    result = assert_training_lineage(
        regime="frozen_asset",
        full_block_trainable=False,
        checkpoint_promotable=False,
        successor_source_allowed=False,
        detachable_adapter=True,
    )
    assert result["status"] == "allowed"
