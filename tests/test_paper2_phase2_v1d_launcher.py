from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v1d_bootstrap_target_is_separate_and_safety_marked() -> None:
    bootstrap = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "paper2_phase2_v1d",
        "STAGE5_PAPER2_PHASE2_V1D_CELL.py",
        "paper2_phase2_v1d_v1",
        "c 0.15 p99 RMS cap matched V1c cohorts no optimizer no training",
        "constants file hash carried in receipt E1 remains locked",
    ):
        assert marker in bootstrap


def test_v1d_runner_locks_cap_cohorts_and_statistical_reading() -> None:
    runner = (ROOT / "colab/run_stage5_paper2_phase2_v1d.py").read_text(
        encoding="utf-8"
    )
    assert '"--state_rms_cap"' in runner
    assert "v1c/summary.json" in runner
    assert "position_key_sha256" in runner
    assert "PRESERVATION_POINT_FLOOR = 0.997" in runner
    assert "PRESERVATION_CI_LOWER_FLOOR = 0.990" in runner
    assert "wilson_lower_bound" in runner
    assert "torch.optim" not in runner
    assert "training/train_" not in runner


def test_dc2_constants_are_single_source_and_confirmed_by_v1d() -> None:
    payload = json.loads(
        (ROOT / "training/paper2_phase2_dc2_constants.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["tube_c"] == 0.15
    assert 0.55 < payload["p99_state_rms_cap"] < 0.551
    assert payload["status"] == "confirmed_by_v1d"
    assert payload["source_receipt_sha256"] == (
        "b8ec5e81649d7a7917d98a0f988cd39c64be16ea51a34b150b02ef07df6d86ca"
    )
    assert payload["preservation_gate"] == "pass"
    source = ROOT / payload["source"]
    canonical_bytes = source.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical_bytes).hexdigest() == payload["source_receipt_sha256"]
