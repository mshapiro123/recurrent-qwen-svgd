from __future__ import annotations

import json
from pathlib import Path

from colab.capacity_localization import (
    QWEN25_05B_RECURRENT_LORA_PARAMS_PER_RANK,
    add_baseline_deltas,
    extract_capacity_row,
    lora_trainable_params_for_rank,
    parse_int_csv,
    trainable_parameter_ledger,
    write_capacity_localization_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RANK32_SUMMARY = ROOT / "outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json"


def test_capacity_rank_parser_defaults_and_validates() -> None:
    assert parse_int_csv("", default=[64]) == [64]
    assert parse_int_csv("64, 128", default=[32]) == [64, 128]

    try:
        parse_int_csv("64,0", default=[64])
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("rank 0 should fail")


def test_lora_trainable_parameter_ledger_is_rank_scaled_not_stored_size() -> None:
    assert lora_trainable_params_for_rank(64) == 64 * QWEN25_05B_RECURRENT_LORA_PARAMS_PER_RANK
    rank64 = trainable_parameter_ledger(64)
    rank128 = trainable_parameter_ledger(128)

    assert rank64["stored_size_changes_with_rank"] is False
    assert rank64["per_loop_compute_changes_with_rank"] is False
    assert rank128["lora_trainable_params_estimate"] == 2 * rank64["lora_trainable_params_estimate"]
    assert rank64["stored_model_params_estimate"] == rank128["stored_model_params_estimate"]


def test_extract_rank32_capacity_row_from_current_recovery_summary() -> None:
    row = extract_capacity_row(ROOT, RANK32_SUMMARY)

    assert row["lora"]["rank"] == 32
    assert row["lora"]["alpha"] == 64
    assert row["loop_correct"] == {"1": 182, "2": 163, "3": 159}
    assert row["oracle_correct"] == 234
    assert row["rescued_vs_loop1"] == 52
    assert row["harmed_vs_loop1"] == 71
    assert row["post_reentry_health_status"] == "reentry_health_sane"
    assert round(row["tail_ratio_vs_entry"]["loop8"], 3) == 2.830
    assert row["loops_to_benefit_vs_loop1"] is None


def test_add_baseline_deltas_compares_to_rank32() -> None:
    baseline = extract_capacity_row(ROOT, RANK32_SUMMARY)
    candidate = json.loads(json.dumps(baseline))
    candidate["lora"]["rank"] = 64
    candidate["loop_correct"]["2"] = baseline["loop_correct"]["2"] + 5
    candidate["oracle_correct"] = baseline["oracle_correct"] + 7
    candidate["rescued_vs_loop1"] = baseline["rescued_vs_loop1"] + 3

    rows = add_baseline_deltas([baseline, candidate], baseline_rank=32)

    rank64 = next(row for row in rows if row["lora"]["rank"] == 64)
    assert rank64["delta_vs_rank32"]["loop2_correct"] == 5
    assert rank64["delta_vs_rank32"]["oracle_correct"] == 7
    assert rank64["delta_vs_rank32"]["rescued_vs_loop1"] == 3


def test_write_capacity_summary_uses_separate_capacity_pointer(tmp_path: Path) -> None:
    src_root = tmp_path / "repo"
    src_root.mkdir()
    target_summary = src_root / "outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json"
    target_summary.parent.mkdir(parents=True)
    target_summary.write_text(RANK32_SUMMARY.read_text(encoding="utf-8"), encoding="utf-8")

    child_dir = src_root / "outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft"
    child_dir.mkdir(parents=True)
    source_child_yaml = ROOT / "outputs/stage5/stage5_reentry_recovery_20260627_190155_curriculum_sft/phase1_curriculum_sft.yaml"
    (child_dir / "phase1_curriculum_sft.yaml").write_text(source_child_yaml.read_text(encoding="utf-8"), encoding="utf-8")

    summary_path = write_capacity_localization_summary(
        root=src_root,
        run_id="capacity_test",
        output_dir=src_root / "outputs/stage5/capacity_test",
        baseline_summaries=["outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json"],
        result_summaries=[],
        target_ranks=[64],
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "stage5_capacity_localization_sweep"
    assert payload["rows"][0]["lora"]["rank"] == 32
    assert not (src_root / "config/stage5_current_source_summary.txt").exists()
