from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from colab.review_stage5_recovery_curriculum import (
    is_gate_ready_trace_collection,
    resolve_trace_collection_summary,
    trace_collection_candidates,
)


def write_summary(path: Path, *, run_id: str, gate_ready: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "kind": "stage5_capability_ladder_trace_collection",
        "status": "trace_curriculum_gate_ready" if gate_ready else "trace_curriculum_needs_review",
        "curriculum": {
            "counts": {
                "typed_records": 24,
                "positive_sft_rows": 24,
                "mode_counts": {"direct": 12, "deep_narrow": 12},
                "target_loop_counts": {"1": 12, "2": 12},
            }
        },
        "gate": {"go": gate_ready},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_resolve_trace_collection_summary_picks_latest_gate_ready_local_summary(tmp_path: Path) -> None:
    older = write_summary(
        tmp_path / "outputs" / "stage5" / "older" / "summary.json",
        run_id="older",
    )
    newer_bad = write_summary(
        tmp_path / "outputs" / "stage5" / "newer_bad" / "summary.json",
        run_id="newer_bad",
        gate_ready=False,
    )
    newer_good = write_summary(
        tmp_path / "outputs" / "stage5" / "newer_good" / "summary.json",
        run_id="newer_good",
    )
    now = time.time()
    os.utime(older, (now - 30, now - 30))
    os.utime(newer_bad, (now - 20, now - 20))
    os.utime(newer_good, (now - 10, now - 10))

    resolved = resolve_trace_collection_summary(root=tmp_path, extra_roots=())

    assert resolved == newer_good


def test_resolve_trace_collection_summary_explicit_override_wins(tmp_path: Path) -> None:
    latest = write_summary(
        tmp_path / "outputs" / "stage5" / "latest" / "summary.json",
        run_id="latest",
    )
    explicit = write_summary(
        tmp_path / "outputs" / "stage5" / "explicit" / "summary.json",
        run_id="explicit",
    )
    now = time.time()
    os.utime(latest, (now, now))
    os.utime(explicit, (now - 60, now - 60))

    resolved = resolve_trace_collection_summary(
        explicit="outputs/stage5/explicit/summary.json",
        root=tmp_path,
        extra_roots=(),
    )

    assert resolved == explicit


def test_explicit_non_gate_ready_summary_is_rejected(tmp_path: Path) -> None:
    explicit = write_summary(
        tmp_path / "outputs" / "stage5" / "explicit_bad" / "summary.json",
        run_id="explicit_bad",
        gate_ready=False,
    )

    assert not is_gate_ready_trace_collection(explicit)
    with pytest.raises(RuntimeError, match="No gate-ready"):
        resolve_trace_collection_summary(
            explicit="outputs/stage5/explicit_bad/summary.json",
            root=tmp_path,
            extra_roots=(),
        )


def test_trace_collection_candidates_appends_default_after_scan(tmp_path: Path) -> None:
    found = write_summary(tmp_path / "outputs" / "stage5" / "found" / "summary.json", run_id="found")

    candidates = trace_collection_candidates(root=tmp_path, extra_roots=())

    assert candidates[0] == found
    assert candidates[-1] == (
        tmp_path
        / "outputs"
        / "stage5"
        / "stage5_capability_ladder_trace_collection_20260623_194537"
        / "summary.json"
    )
