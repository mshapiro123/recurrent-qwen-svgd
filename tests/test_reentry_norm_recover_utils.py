from __future__ import annotations

import json
from pathlib import Path

from colab.reentry_norm_recover_utils import (
    build_summary_payload,
    final_stage2_complete,
    raw_stage2_complete,
    recoverable_stage2,
    summary_markdown,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_raw_stage2(run_dir: Path) -> None:
    for mode in ("none", "entry_rms"):
        write_json(
            run_dir / "reentry_norm" / f"reentry_drift_{mode}.json",
            {
                "aggregate": {
                    "mean_exit_over_entry_rms": 1.0,
                    "loop_summary": {"8": {"output_over_entry_rms": 1.0}},
                }
            },
        )
        write_jsonl(run_dir / "reentry_norm" / f"reentry_drift_{mode}.jsonl", [{"mode": mode}])
        write_json(
            run_dir / "reentry_norm" / f"effective_pathways_{mode}.json",
            {
                "aggregate": {
                    "mean_initial_pairwise_distance": 0.1,
                    "mean_final_pairwise_distance": 0.2,
                    "mean_effective_pathways": {"2": 1.0},
                }
            },
        )
        write_jsonl(run_dir / "reentry_norm" / f"effective_pathways_{mode}.jsonl", [{"mode": mode}])
        write_jsonl(
            run_dir / "reentry_norm" / f"candidate_conversion_{mode}.jsonl",
            [
                {
                    "reentry_rescale_mode": mode,
                    "max_loops": 4,
                    "particle_init_noise": 0.05,
                    "task": "task-a",
                    "hit": True,
                    "candidate": "answer",
                },
                {
                    "reentry_rescale_mode": mode,
                    "max_loops": 4,
                    "particle_init_noise": 0.05,
                    "task": "task-a",
                    "hit": False,
                    "candidate": "other",
                },
            ],
        )


def test_raw_complete_stage2_is_recoverable_without_summary(tmp_path) -> None:
    run_dir = tmp_path / "stage5_reentry_norm_partial"
    write_raw_stage2(run_dir)

    assert raw_stage2_complete(run_dir)
    assert recoverable_stage2(run_dir)
    assert not final_stage2_complete(run_dir)


def test_build_summary_payload_from_raw_stage2_outputs(tmp_path) -> None:
    run_dir = tmp_path / "stage5_reentry_norm_partial"
    write_raw_stage2(run_dir)

    summary = build_summary_payload(run_dir, cell_version="recover-test")

    assert summary["kind"] == "stage5_reentry_norm_eval_only"
    assert summary["run_id"] == "stage5_reentry_norm_partial"
    assert summary["cell_version"] == "recover-test"
    assert summary["recovered_summary"] is True
    assert summary["candidate_conversion"]["none"]["by_mode"]["none"]["candidate_hits"] == 1
    assert summary["candidate_conversion"]["entry_rms"]["by_mode"]["entry_rms"]["total_candidates"] == 2
    assert "Recovered summary" in summary_markdown(summary)
