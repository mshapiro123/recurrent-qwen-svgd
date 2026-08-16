from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hf_release.convert_checkpoint import split_release_state


ROOT = Path(__file__).resolve().parents[1]


def test_split_release_state_excludes_only_manifest_listed_tensors() -> None:
    source = {
        "base_model.model.layers.6.weight": torch.zeros(3),
        "bridge.bridge_gate": torch.zeros(1),
        "bridge.proj.weight": torch.zeros(2, 4),
        "bridge.proj.bias": torch.zeros(2),
    }
    release, excluded = split_release_state(
        source,
        {
            "excluded_checkpoint_keys": ["bridge.proj.weight", "bridge.proj.bias"],
            "expected_excluded_parameters": 10,
        },
    )

    assert set(release) == {
        "base_model.model.layers.6.weight",
        "bridge.bridge_gate",
    }
    assert set(excluded) == {"bridge.proj.weight", "bridge.proj.bias"}


def test_split_release_state_fails_on_exclusion_count_drift() -> None:
    with pytest.raises(RuntimeError, match="Excluded checkpoint parameter mismatch"):
        split_release_state(
            {"bridge.proj.weight": torch.zeros(2, 4)},
            {
                "excluded_checkpoint_keys": ["bridge.proj.weight"],
                "expected_excluded_parameters": 9,
            },
        )


def test_release_manifest_receipts_historical_legacy_projection() -> None:
    manifest = json.loads((ROOT / "hf_release/release_manifest.json").read_text(encoding="utf-8"))
    repos = manifest["repos"]
    for name in (
        "recurrent-qwen2.5-0.5b-full-block",
        "recurrent-qwen2.5-0.5b-natural-keeper",
    ):
        assert repos[name]["excluded_checkpoint_keys"] == [
            "bridge.proj.weight",
            "bridge.proj.bias",
        ]
        assert repos[name]["expected_excluded_parameters"] == 1_606_528
        assert repos[name]["expected_delta_parameters"] == 180_556_929

    adapter = repos["recurrent-qwen2.5-0.5b-r16-adapter"]
    assert adapter["excluded_checkpoint_keys"] == []
    assert adapter["expected_excluded_parameters"] == 0
