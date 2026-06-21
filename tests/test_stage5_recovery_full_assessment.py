from __future__ import annotations


def test_selected_checkpoint_from_distill_autopilot(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovery_full_assessment as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "kind": "stage5_balanced_recovery_autopilot",
        "status": "distill_gate_passed",
        "distill": {
            "status": "proxy_lift",
            "best_arm": {
                "checkpoint": "outputs/stage5/distill_child/phase1/phase1_step_50.pt",
            },
        },
    }

    gate, checkpoint, gate_payload = module.selected_checkpoint(payload)

    assert gate == "distill"
    assert checkpoint == tmp_path / "outputs" / "stage5" / "distill_child" / "phase1" / "phase1_step_50.pt"
    assert gate_payload["status"] == "proxy_lift"


def test_selected_checkpoint_from_arc_mix_autopilot(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovery_full_assessment as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "kind": "stage5_balanced_recovery_autopilot",
        "status": "arc_mix_gate_passed",
        "arc_mix": {
            "status": "proxy_matches_base",
            "best_arm": {
                "best_checkpoint": {
                    "checkpoint": "outputs/stage5/arc_mix_child/phase1/phase1_step_100.pt",
                },
            },
        },
    }

    gate, checkpoint, _ = module.selected_checkpoint(payload)

    assert gate == "arc_mix"
    assert checkpoint == tmp_path / "outputs" / "stage5" / "arc_mix_child" / "phase1" / "phase1_step_100.pt"


def test_selected_checkpoint_rejects_failed_autopilot() -> None:
    import pytest
    import colab.run_stage5_recovery_full_assessment as module

    with pytest.raises(ValueError, match="did not pass"):
        module.selected_checkpoint(
            {
                "kind": "stage5_balanced_recovery_autopilot",
                "status": "no_recovery_gate_lift",
            }
        )


def test_infer_stage5_run_id() -> None:
    import colab.run_stage5_recovery_full_assessment as module

    assert module.infer_stage5_run_id("outputs/stage5/run/phase1/phase1_step_50.pt") == "run"
