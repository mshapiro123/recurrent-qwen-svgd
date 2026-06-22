from __future__ import annotations

import json
import os


def test_repair_profile_selects_direct_heavy_arc_easy_mix() -> None:
    from colab.run_stage5_routing_repair import repair_profile

    profile = repair_profile("needs_direct_halting_repair")

    assert profile["repair_mode"] == "direct_halting"
    assert profile["STAGE5_ARC_MIX_ARC_EASY_REPEAT"] == "8"
    assert profile["STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT"] == "1"
    assert profile["STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP"] == "1"
    assert profile["STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP"] == "2"
    assert profile["STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE"] == "direct"
    assert profile["STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE"] == "deep_narrow_probe"
    assert profile["STAGE5_ARC_MIX_EVAL_CONFIG"] == "ARC-Easy"
    assert profile["STAGE5_ARC_MIX_MIN_MARGIN_DELTA"] == "0.0"


def test_direct_repair_profile_audit_requires_distillation() -> None:
    from colab.run_stage5_routing_repair import profile_objective_audit, repair_profile

    audit = profile_objective_audit(repair_profile("needs_direct_halting_repair"))

    assert audit["repair_mode"] == "direct_halting"
    assert audit["requires_distillation"] is True
    assert audit["distillation_ok"] is True
    assert audit["arms"]
    assert all(arm["distillation"]["enabled"] for arm in audit["arms"])


def test_profile_objective_audit_rejects_no_distill_direct_repair() -> None:
    import pytest

    from colab.run_stage5_routing_repair import profile_objective_audit, repair_profile

    profile = repair_profile("needs_direct_halting_repair")
    profile["STAGE5_ARC_MIX_ARMS"] = "arc_mix_nodistill_lr3e6"

    with pytest.raises(ValueError, match="response-distillation ARC-mix arms"):
        profile_objective_audit(profile)


def test_profile_objective_audit_rejects_empty_direct_repair_arms() -> None:
    import pytest

    from colab.run_stage5_routing_repair import profile_objective_audit, repair_profile

    profile = repair_profile("needs_direct_halting_repair")
    profile["STAGE5_ARC_MIX_ARMS"] = ""

    with pytest.raises(ValueError, match="response-distillation ARC-mix arms"):
        profile_objective_audit(profile)


def test_repair_profile_selects_deep_challenge_mix() -> None:
    from colab.run_stage5_routing_repair import repair_profile

    profile = repair_profile("needs_deep_narrow_recovery")

    assert profile["repair_mode"] == "deep_narrow"
    assert profile["STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT"] == "5"
    assert profile["STAGE5_ARC_MIX_ARC_EASY_REPEAT"] == "2"
    assert profile["STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP"] == "1"
    assert profile["STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP"] == "3"
    assert profile["STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE"] == "direct_anchor"
    assert profile["STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE"] == "deep_narrow"
    assert profile["STAGE5_ARC_MIX_EVAL_CONFIG"] == "ARC-Challenge"


def test_deep_repair_profile_audit_requires_distillation() -> None:
    from colab.run_stage5_routing_repair import profile_objective_audit, repair_profile

    audit = profile_objective_audit(repair_profile("needs_deep_narrow_recovery"))

    assert audit["repair_mode"] == "deep_narrow"
    assert audit["requires_distillation"] is True
    assert audit["distillation_ok"] is True
    assert len(audit["arms"]) == 2


def test_benchmark_summary_from_assessment_uses_payload_path(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_routing_repair as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    summary = tmp_path / "outputs" / "stage5" / "routing" / "benchmark_run" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    payload = {"benchmark_summary": summary.relative_to(tmp_path).as_posix()}

    assert module.benchmark_summary_from_assessment(payload, tmp_path / "routing" / "summary.json") == summary


def test_build_child_env_points_arc_mix_at_benchmark_summary(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_routing_repair as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_ID", "repair")
    source = tmp_path / "outputs" / "stage5" / "routing" / "benchmark_run" / "summary.json"
    profile = module.repair_profile("needs_direct_halting_repair")

    env = module.build_child_env(source_summary=source, profile=profile)

    assert env["STAGE5_ARC_MIX_RUN_ID"] == "repair_direct_halting_arc_mix"
    assert env["STAGE5_ARC_MIX_SOURCE_SUMMARY"] == source.relative_to(tmp_path).as_posix()
    assert env["STAGE5_ARC_MIX_PUSH"] == "0"


def test_child_best_checkpoint_and_passed_extract_child_gate() -> None:
    import colab.run_stage5_routing_repair as module

    payload = {
        "status": "proxy_lift",
        "best_arm": {
            "best_checkpoint": {
                "checkpoint": "outputs/stage5/child/phase1/phase1_step_150.pt",
            }
        },
    }

    assert module.child_passed(payload) is True
    assert module.child_best_checkpoint(payload) == {
        "checkpoint": "outputs/stage5/child/phase1/phase1_step_150.pt",
    }


def test_resolve_source_summary_finds_latest_routing_summary(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_routing_repair as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", module.Path(""))
    old = tmp_path / "outputs" / "stage5" / "stage5_routing_diagnostic_old" / "summary.json"
    new = tmp_path / "outputs" / "stage5" / "stage5_routing_diagnostic_new" / "summary.json"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text(json.dumps({"status": "needs_direct_halting_repair"}), encoding="utf-8")
    new.write_text(json.dumps({"status": "needs_deep_narrow_recovery"}), encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    assert module.resolve_source_summary() == new
