from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "training/paper2_phase3_opening_contract.json"


def test_governing_artifacts_are_byte_verified() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for artifact in contract["governing_artifacts"].values():
        path = ROOT / artifact["path"]
        payload = path.read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]


def test_opening_contract_authorizes_build_but_not_p33_training() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "p31_p32_build_authorized_p33_training_not_locked"
    assert contract["authorization"]["per_position_gate_build"] is True
    assert contract["authorization"]["p31_infrastructure_build"] is True
    assert contract["authorization"]["p32_schema_and_preflight_build"] is True
    assert contract["authorization"]["p33_training"] is False
    assert contract["architecture"]["phase2_trainable_parameters_historical"] == 1_184_917
    assert contract["architecture"]["phase3_trainable_parameters"] == 1_185_973
    source_manifest = ROOT / contract["p31"]["source_manifest"]["path"]
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == contract["p31"][
        "source_manifest"
    ]["sha256"]
    assert contract["p31"]["planning_assumption"] == {
        "rows": 512,
        "one_sided_alpha": 0.00005,
        "binding": False,
    }
    assert contract["p32"]["concurrence_relaxation_permitted"] is False
    migration_sources = ROOT / contract["architecture"]["migration_sources"]["path"]
    assert hashlib.sha256(migration_sources.read_bytes()).hexdigest() == contract[
        "architecture"
    ]["migration_sources"]["sha256"]
