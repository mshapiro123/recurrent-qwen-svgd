import json
from pathlib import Path

from eval.eval_paper2_bicameral_w1_generation import (
    GENERATION_CONFIG,
    freeze_generation_manifest,
)


def test_generation_config_is_frozen_to_registered_values() -> None:
    assert GENERATION_CONFIG["gamma"] == 0.05
    assert GENERATION_CONFIG["batch_size"] == 8
    assert GENERATION_CONFIG["decoder"] == "greedy_incremental_cache_v1"
    assert GENERATION_CONFIG["optimizer_steps"] == 0


def test_freeze_generation_manifest_requires_exact_population(tmp_path: Path) -> None:
    rows = []
    for battery, count in (("gsm8k", 369), ("mbpp", 67), ("tier1", 25)):
        rows.extend({"item_id": f"{battery}:{i}", "battery": battery} for i in range(count))
    rows.append({"item_id": "mmlu:0", "battery": "mmlu"})
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    config = tmp_path / "config.json"
    receipt = freeze_generation_manifest(source, manifest, config)
    assert receipt["rows"] == 461
    assert receipt["status"] == "frozen_before_scoring"
    assert receipt["optimizer_steps"] == 0
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 461
