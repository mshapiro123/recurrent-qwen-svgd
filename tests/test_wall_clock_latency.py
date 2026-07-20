import json
from pathlib import Path

from eval.eval_wall_clock_latency import (
    build_markdown_table,
    build_observation_schedule,
    observation_key,
    summarize_records,
    update_claim_ledger,
)


def _row(depth: int, index: int) -> dict:
    return {"id": f"d{depth:02d}_{index:03d}", "depth": depth}


def test_observation_schedule_interleaves_depths_and_adds_locked_stability_rows():
    rows = [_row(depth, index) for depth in range(1, 15) for index in range(40)]
    schedule = build_observation_schedule(rows, stability_depths=(4, 8, 12), stability_rows=32, repeats=3)

    assert [item["row"]["depth"] for item in schedule[:14]] == list(range(1, 15))
    assert sum(item["phase"] == "full" for item in schedule) == len(rows)
    assert sum(item["phase"] == "stability" for item in schedule) == 3 * 3 * 32
    assert len({observation_key(item) for item in schedule}) == len(schedule)


def test_summary_and_markdown_report_decode_medians_and_tokens():
    records = []
    for arm in ("A", "E", "C", "B", "D"):
        for depth in (1, 2, 4, 8, 11, 14):
            for offset in (0.0, 2.0, 4.0):
                records.append(
                    {
                        "arm": arm,
                        "phase": "full",
                        "repeat": 0,
                        "depth": depth,
                        "total_ms": 20.0 + depth + offset,
                        "prefill_ms": 5.0 + offset,
                        "decode_ms": 15.0 + depth,
                        "model_total_ms": 20.0 + depth,
                        "tokenization_ms": 1.0,
                        "generated_tokens": 1 if arm in {"A", "E", "B", "D"} else depth + 2,
                    }
                )
    summary = summarize_records(records)
    assert summary["arms"]["A"]["by_depth"]["4"]["decode_ms"]["median"] == 19.0
    assert summary["arms"]["C"]["overall"]["generated_tokens"]["median"] == 8.0

    table = build_markdown_table(summary, selected_depths=(1, 2, 4, 8, 11, 14))
    assert "| Depth | Arm A" in table
    assert "| 4 | 19.00" in table
    assert "Median generated tokens" in table
    assert "batch size 1" in table


def test_ledger_update_is_idempotent(tmp_path: Path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"claims": [{"id": "existing", "status": "supported"}]}), encoding="utf-8")
    evidence = "outputs/stage5/stage5_wall_clock_latency_20260719/summary.json"

    update_claim_ledger(ledger, evidence_path=evidence)
    update_claim_ledger(ledger, evidence_path=evidence)

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    claims = [claim for claim in payload["claims"] if claim["id"] == "wall_clock_latency_descriptive"]
    assert len(claims) == 1
    assert claims[0]["status"] == "descriptive"
    assert claims[0]["scope"] == "single hardware configuration, batch size 1, registered evaluation paths"
    assert claims[0]["evidence"][0]["path"] == evidence


def test_bootstrap_exposes_wall_clock_latency_target():
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    assert '"wall_clock_latency_descriptive"' in bootstrap
    assert "STAGE5_WALL_CLOCK_LATENCY_CELL_VERSION" in bootstrap
    assert "tests/test_wall_clock_latency.py" in bootstrap


def test_wall_clock_launcher_fetches_non_main_pin_before_reset():
    root = Path(__file__).resolve().parents[1]
    source = (root / "colab/STAGE5_WALL_CLOCK_LATENCY_CELL.py").read_text(
        encoding="utf-8"
    )

    assert "STAGE5_WALL_CLOCK_FETCH_REF" in source
    assert 'run(["git", "fetch", "origin", FETCH_REF])' in source
    assert 'run(["git", "reset", "--hard", SYNC_REF])' in source


def test_wall_clock_runner_can_repair_only_the_mixed_hardware_dense_cohort():
    root = Path(__file__).resolve().parents[1]
    source = (root / "colab/run_stage5_wall_clock_latency.py").read_text(encoding="utf-8")

    assert "STAGE5_WALL_CLOCK_FORCE_ARMS" in source
    assert 'FORCE_ARMS != {"B", "C", "D"}' in source
    assert "invalid_mixed_hardware" in source
    assert "archived_forced_latency_arm" in source
    assert "forced_cohort_hardware" in source
    assert "--query-gpu=name,driver_version,memory.total" in source
    assert "archived_incomplete_checkpoint_restore" in source
    assert "source checkpoint hash mismatch" in source


def test_dense_latency_uses_registered_generate_not_a_manual_cache_loop():
    root = Path(__file__).resolve().parents[1]
    source = (root / "eval/eval_wall_clock_latency.py").read_text(encoding="utf-8")

    assert "_timed_registered_dense_generate" in source
    assert "model.generate(" in source
    assert "_assert_dense_equivalence" not in source
    assert "prepare_inputs_for_generation" not in source
    assert "_update_model_kwargs_for_generation" not in source
