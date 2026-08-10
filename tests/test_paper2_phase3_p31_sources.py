from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from eval.prepare_paper2_phase3_p31_sources import (
    arc_rows,
    gsm8k_final_answer,
    mbpp_rows,
    stable_slice,
)


def test_gsm8k_reader_uses_registered_final_delimiter() -> None:
    assert gsm8k_final_answer("work\n#### 1,234") == "1234"


def test_stable_slice_is_seeded_and_order_independent() -> None:
    rows = [{"item_id": f"row-{index}"} for index in range(20)]
    first = stable_slice(rows, size=5, seed=20260809)
    second = stable_slice(list(reversed(rows)), size=5, seed=20260809)
    assert first == second


def test_arc_and_mbpp_training_rows_keep_real_verifier_contracts() -> None:
    arc = arc_rows(
        [
            {
                "id": "arc-1",
                "question": "q",
                "choices": {"label": ["A", "B"], "text": ["x", "y"]},
                "answerKey": "B",
            }
        ],
        battery="arc_challenge",
        native_split="train",
    )[0]
    mbpp = mbpp_rows(
        [
            {
                "task_id": 1,
                "prompt": "write f",
                "code": "def f(): return 1",
                "test_imports": [],
                "test_list": ["assert f() == 1"],
            }
        ],
        native_split="train",
    )[0]
    assert arc["programmatic_verifier_available"] is True
    assert mbpp["programmatic_verifier_available"] is True
    assert mbpp["tests"] == ["assert f() == 1"]


def test_tier1_manifest_uses_canonical_git_blob_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "training/paper2_phase3_p31_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    tier1 = manifest["sources"]["tier1"]
    blob = subprocess.check_output(["git", "show", f"HEAD:{tier1['path']}"], cwd=root)
    assert len(blob) == tier1["bytes"]
    assert hashlib.sha256(blob).hexdigest() == tier1["sha256"]
