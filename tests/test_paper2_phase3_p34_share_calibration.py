from __future__ import annotations

from training.paper2_phase3_p32 import GateLabel
from training.prepare_paper2_phase3_p34_share_calibration import (
    P34_SHARE_NEGATIVES_PER_STRATUM,
    P34_SHARE_POSITIVES_PER_STRATUM,
    select_share_calibration_rows,
)


def test_share_calibration_selection_is_fixed_balanced_and_audit_free() -> None:
    rows = []
    for stratum in ("code", "general"):
        for label, count in (
            (int(GateLabel.POSITIVE), 80),
            (int(GateLabel.NEGATIVE), 140),
        ):
            for index in range(count):
                rows.append(
                    {
                        "record_id": f"{stratum}-{label}-{index}",
                        "stratum": stratum,
                        "gate_label": label,
                        "training_eligible": True,
                        "audit_holdout": False,
                        "source": "old" if index % 2 else "new",
                    }
                )
    rows.append(
        {
            "record_id": "excluded-audit",
            "stratum": "code",
            "gate_label": int(GateLabel.POSITIVE),
            "training_eligible": True,
            "audit_holdout": True,
            "source": "old",
        }
    )
    first, receipt = select_share_calibration_rows(rows)
    second, _ = select_share_calibration_rows(reversed(rows))
    assert [row["record_id"] for row in first] == [row["record_id"] for row in second]
    assert receipt["rows"] == 256
    for stratum in ("code", "general"):
        selected = [row for row in first if row["stratum"] == stratum]
        assert sum(row["gate_label"] == int(GateLabel.POSITIVE) for row in selected) == P34_SHARE_POSITIVES_PER_STRATUM
        assert sum(row["gate_label"] == int(GateLabel.NEGATIVE) for row in selected) == P34_SHARE_NEGATIVES_PER_STRATUM
    assert all(not row["audit_holdout"] for row in first)
