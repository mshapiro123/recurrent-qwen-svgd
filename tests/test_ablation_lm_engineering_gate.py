from __future__ import annotations

from datetime import date, timedelta
import json

import pytest

from scripts.run_ablation_lm_engineering_gate import RECEIPT, _load_gate_contract


def _receipt_payload() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def _write_receipt(tmp_path, payload: dict[str, object]):
    path = tmp_path / "quarantine.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quarantine_review_date_is_live_only_before_due_date(tmp_path) -> None:
    payload = _receipt_payload()
    expected_due = payload["review_due_on"]
    expected_nodes = {row["node_id"] for row in payload["expected_failures"]}
    receipt_path = _write_receipt(tmp_path, payload)

    expected, expected_passed = _load_gate_contract(
        receipt_path,
        today=date.fromisoformat(expected_due) - timedelta(days=1),
    )

    assert expected == expected_nodes
    assert expected_passed == payload["last_observed_full_suite"]["passed"]
    with pytest.raises(RuntimeError, match=f"review is stale as of {expected_due}"):
        _load_gate_contract(receipt_path, today=date.fromisoformat(expected_due))
    with pytest.raises(RuntimeError, match=f"review is stale as of {expected_due}"):
        _load_gate_contract(
            receipt_path,
            today=date.fromisoformat(expected_due) + timedelta(days=1),
        )


@pytest.mark.parametrize("value", [None, "2026-9-2", "09/02/2026", "tomorrow"])
def test_quarantine_review_date_requires_canonical_iso(tmp_path, value) -> None:
    payload = _receipt_payload()
    payload["review_due_on"] = value
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(RuntimeError, match="review_due_on"):
        _load_gate_contract(receipt_path, today=date(2026, 8, 26))


def test_quarantine_today_injection_requires_exact_date(tmp_path) -> None:
    receipt_path = _write_receipt(tmp_path, _receipt_payload())

    with pytest.raises(TypeError, match="exact datetime.date"):
        _load_gate_contract(receipt_path, today="2026-08-26")  # type: ignore[arg-type]
