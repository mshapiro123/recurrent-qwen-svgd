from __future__ import annotations

import pytest

from colab.run_stage4_opus_finetune import validate_dataset_source


def test_validate_dataset_source_allows_approved_opus_sft() -> None:
    payload = validate_dataset_source(
        dataset_id="lordx64/reasoning-distill-opus-4-7-max-sft",
        allow_unapproved=False,
    )

    assert payload["approved"] is True
    assert payload["allow_unapproved"] is False


def test_validate_dataset_source_blocks_unapproved_trace_source_by_default() -> None:
    with pytest.raises(ValueError, match="restricted to approved recovery datasets"):
        validate_dataset_source(
            dataset_id="Jackrong/Claude-opus-4.7-TraceInversion-5000x",
            allow_unapproved=False,
        )


def test_validate_dataset_source_allows_explicit_nondefault_override() -> None:
    payload = validate_dataset_source(
        dataset_id="Jackrong/Claude-opus-4.7-TraceInversion-5000x",
        allow_unapproved=True,
    )

    assert payload["approved"] is False
    assert payload["allow_unapproved"] is True
