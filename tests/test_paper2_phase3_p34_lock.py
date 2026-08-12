from __future__ import annotations

from pathlib import Path

from training.paper2_phase3_p34_lock import (
    build_task_panel,
    largest_remainder_quotas,
    panel_identity_sha256,
)


def test_prepare_contract_requires_paired_base_score_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "prepare_paper2_phase3_p34_lock.py"
    ).read_text(encoding="utf-8")
    assert "panel base-score inputs must be supplied together" in source
    assert "panel base-score coverage is incomplete or duplicated" in source


def _rows() -> list[dict[str, object]]:
    counts = {
        "mmlu": 272,
        "tier1": 28,
        "arc_easy": 270,
        "gsm8k": 684,
        "mbpp": 125,
        "arc_challenge": 140,
    }
    rows = []
    for battery, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "battery": battery,
                    "item_id": f"{battery}-{index}",
                    "document_id": f"{battery}-doc-{index}",
                    "content_sha256": f"sha-{battery}-{index}",
                    "partition": "dev",
                }
            )
    return rows


def test_largest_remainder_quotas_close_exactly() -> None:
    assert largest_remainder_quotas(
        {"mmlu": 272, "tier1": 28, "arc_easy": 270}, total=512
    ) == {"mmlu": 244, "tier1": 25, "arc_easy": 243}
    assert largest_remainder_quotas(
        {"gsm8k": 684, "mbpp": 125, "arc_challenge": 140}, total=512
    ) == {"gsm8k": 369, "mbpp": 67, "arc_challenge": 76}


def test_panel_is_deterministic_balanced_and_score_blind() -> None:
    first, receipt = build_task_panel(_rows())
    second, repeated = build_task_panel(reversed(_rows()))
    assert first == second
    assert receipt == repeated
    assert receipt["rows"] == 1024
    assert receipt["group_counts"] == {"floor": 512, "target": 512}
    assert receipt["unique_documents"] == 1024
    assert panel_identity_sha256(first) == receipt["panel_sha256"]
    assert receipt["scores_computed"] is False
    assert receipt["optimizer_steps"] == 0
