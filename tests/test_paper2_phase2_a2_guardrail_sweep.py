from __future__ import annotations

import json
from pathlib import Path

from training.run_paper2_phase2_a2_guardrail_sweep import run, sha256_lf_text


ROOT = Path(__file__).resolve().parents[1]


def test_lf_text_hash_is_independent_of_checkout_newlines(tmp_path: Path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "value": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
    assert sha256_lf_text(lf) == sha256_lf_text(crlf)


def test_guardrail_sweep_validates_locked_step237_sources(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = run(root=ROOT, output=output)
    assert result["status"] == "complete_lock_valid"
    assert result["training_authorized_by_this_job"] is False
    assert result["gpu_used"] is False
    assert result["generator_contract"]["next_attempt"] == 238
    assert result["inventory_validation"]["valid"]
    assert result["source_receipt_hash_contract"]["mode"] == "utf8_lf_normalized_sha256"
    assert len(result["source_checks"]) == 4
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete_lock_valid"
