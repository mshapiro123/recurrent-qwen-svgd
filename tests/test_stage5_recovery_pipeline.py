from __future__ import annotations


def test_autopilot_passed_statuses() -> None:
    import colab.run_stage5_recovery_pipeline as module

    assert module.autopilot_passed({"status": "distill_gate_passed"})
    assert module.autopilot_passed({"status": "arc_mix_gate_passed"})
    assert not module.autopilot_passed({"status": "no_recovery_gate_lift"})


def test_build_summary_waits_when_autopilot_missing() -> None:
    import colab.run_stage5_recovery_pipeline as module

    payload = module.build_summary(autopilot_payload=None, full_payload=None)

    assert payload["status"] == "autopilot_missing"


def test_build_summary_stops_after_failed_recovery_gate() -> None:
    import colab.run_stage5_recovery_pipeline as module

    payload = module.build_summary(
        autopilot_payload={"status": "no_recovery_gate_lift", "next_step": "revise"},
        full_payload=None,
    )

    assert payload["status"] == "recovery_gate_not_passed"
    assert payload["next_step"] == "revise"


def test_build_summary_reports_full_assessment_status() -> None:
    import colab.run_stage5_recovery_pipeline as module

    payload = module.build_summary(
        autopilot_payload={"status": "distill_gate_passed"},
        full_payload={"status": "balanced_nonnegative", "next_step": "ship it"},
    )

    assert payload["status"] == "full_assessment_balanced_nonnegative"
    assert payload["next_step"] == "ship it"


def test_failure_summary_records_failed_stage() -> None:
    import colab.run_stage5_recovery_pipeline as module

    payload = module.failure_summary(stage="autopilot", error="boom")

    assert payload["status"] == "pipeline_failed"
    assert payload["failed_stage"] == "autopilot"
    assert payload["error"] == "boom"
