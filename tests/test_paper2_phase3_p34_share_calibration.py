from __future__ import annotations

from pathlib import Path

import torch

from eval.eval_paper2_phase3_p34_share_calibration import extend_main_bundle_with_slot
from training.paper2_phase3_p34 import P34_LOSS_NAMES, P34_SLOT_LOSS_NAMES, SlotSupervisionLift

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


def test_share_calibration_runner_binds_both_seed_endpoints() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "colab/run_stage5_paper2_phase3_p34_share_calibration.py"
    ).read_text(encoding="utf-8")
    assert 'P34_SHARE_SEED' in source
    assert '1: "3ca1cdf8dd16bf4f435e81a675d7514778144c5c881af52a70171659f7734b4f"' in source
    assert '1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067"' in source
    assert '1: "2ed3296f510a6c3a66c451051ecbe2284de03b35dde4052827174a66a10c1d4a"' in source
    assert '["--main_only"] if SEED == 1' in source
    assert "p34_share_compact_batch.pt" in source
    assert 'old = json.loads(old_summary.read_text(encoding="utf-8"))' in source
    assert '"--compact_batch"' in source
    assert '"--direction_cache"' not in source


def test_share_compaction_is_cpu_only_and_training_disabled() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "colab/run_stage5_paper2_phase3_p34_share_compaction.py"
    ).read_text(encoding="utf-8")
    assert 'device="cpu"' in source
    assert '"optimizer_steps": 0' in source
    assert '"training_authorized": False' in source
    assert "p34_share_compact_batch.pt" in source


def test_zero_init_slot_bundle_reuses_main_gradients_exactly() -> None:
    main_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    main_bundle = {
        "names": P34_LOSS_NAMES,
        "parameter_ids": [id(main_parameter)],
        "gradients": {
            name: {id(main_parameter): torch.tensor([float(index + 1)])}
            for index, name in enumerate(P34_LOSS_NAMES)
        },
        "parameter_groups": {id(main_parameter): ("bridge", 0.5)},
    }
    lift = SlotSupervisionLift(latent_dim=1, hidden_size=1)
    slot_loss = (2.0 * lift.lift.weight).sum()
    extended = extend_main_bundle_with_slot(
        main_bundle=main_bundle,
        slot_lift=lift,
        slot_loss=slot_loss,
    )
    lift_id = id(lift.lift.weight)
    assert extended["names"] == P34_SLOT_LOSS_NAMES
    assert extended["parameter_ids"] == [id(main_parameter), lift_id]
    assert extended["gradients"]["kl"] == main_bundle["gradients"]["kl"]
    assert torch.equal(extended["gradients"]["slot"][lift_id], torch.tensor([[2.0]]))
    assert extended["parameter_groups"][lift_id] == ("heads", 1.0)
