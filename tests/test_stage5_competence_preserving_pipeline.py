from __future__ import annotations


def test_child_env_defaults_to_distilled_easy_weighted_arc_mix(monkeypatch) -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    monkeypatch.delenv("STAGE5_ARC_MIX_ARMS", raising=False)
    monkeypatch.delenv("STAGE5_ARC_MIX_ARC_EASY_REPEAT", raising=False)
    monkeypatch.delenv("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", raising=False)

    env = module.child_env()

    assert (
        env["STAGE5_ARC_MIX_SOURCE_SUMMARY"].replace("\\", "/")
        == "outputs/stage5/stage5_recovery_full_assessment_current/summary.json"
    )
    assert env["STAGE5_ARC_MIX_ARMS"] == "arc_mix_response_w005_lr2e6,arc_mix_response_w01_lr2e6"
    assert env["STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT"] == "2"
    assert env["STAGE5_ARC_MIX_ARC_EASY_REPEAT"] == "4"
    assert env["STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY"].replace("\\", "/").endswith("_arc_mix/summary.json")


def test_arc_mix_passed_accepts_lift_or_base_match() -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    assert module.arc_mix_passed({"status": "proxy_lift"}) is True
    assert module.arc_mix_passed({"status": "proxy_matches_base"}) is True
    assert module.arc_mix_passed({"status": "no_proxy_lift"}) is False
    assert module.arc_mix_passed(None) is False


def test_build_summary_waits_for_full_assessment_after_arc_mix_passes() -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    payload = module.build_summary(
        source_payload={"status": "needs_competence_recovery"},
        arc_payload={"status": "proxy_lift"},
        full_payload=None,
    )

    assert payload["status"] == "full_assessment_missing"
    assert payload["arc_mix_status"] == "proxy_lift"


def test_build_summary_reports_full_assessment_status() -> None:
    import colab.run_stage5_competence_preserving_pipeline as module

    payload = module.build_summary(
        source_payload={"status": "needs_competence_recovery"},
        arc_payload={"status": "proxy_lift"},
        full_payload={"status": "balanced_nonnegative", "next_step": "ship it"},
    )

    assert payload["status"] == "full_assessment_balanced_nonnegative"
    assert payload["next_step"] == "ship it"
