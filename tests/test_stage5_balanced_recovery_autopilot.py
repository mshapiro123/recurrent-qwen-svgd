from __future__ import annotations


def test_should_run_arc_mix_after_no_proxy_lift() -> None:
    import colab.run_stage5_balanced_recovery_autopilot as module

    assert module.should_run_arc_mix({"status": "no_proxy_lift"})
    assert not module.should_run_arc_mix({"status": "proxy_lift"})
    assert not module.should_run_arc_mix({"status": "proxy_matches_base"})


def test_build_summary_prefers_distill_pass() -> None:
    import colab.run_stage5_balanced_recovery_autopilot as module

    payload = module.build_summary(
        distill_run_id="distill",
        distill_payload={"status": "proxy_lift", "passed": True},
        arc_mix_run_id=None,
        arc_mix_payload=None,
    )

    assert payload["status"] == "distill_gate_passed"
    assert payload["arc_mix_summary"] is None


def test_build_summary_uses_arc_mix_pass_after_distill_miss() -> None:
    import colab.run_stage5_balanced_recovery_autopilot as module

    payload = module.build_summary(
        distill_run_id="distill",
        distill_payload={"status": "no_proxy_lift", "passed": False},
        arc_mix_run_id="arc_mix",
        arc_mix_payload={"status": "proxy_matches_base", "passed": True},
    )

    assert payload["status"] == "arc_mix_gate_passed"
    assert payload["arc_mix_summary"] == "outputs/stage5/arc_mix/summary.json"


def test_stageable_result_paths_includes_nested_distill_children(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_balanced_recovery_autopilot as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    payload = {
        "run_id": "parent_gate",
        "arms": [
            {
                "child_run_id": "nested_ladder",
                "checkpoint": "outputs/stage5/nested_ladder/phase1/phase1_step_50.pt",
            }
        ],
    }

    paths = module.stageable_result_paths(payload)

    assert tmp_path / "outputs" / "stage5" / "parent_gate" in paths
    assert tmp_path / "outputs" / "stage5" / "nested_ladder" in paths
