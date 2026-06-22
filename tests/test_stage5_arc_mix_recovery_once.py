from __future__ import annotations

import subprocess
import sys

import pytest


def test_arc_mix_recovery_once_defaults_to_low_credit_single_arm(monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    monkeypatch.delenv("STAGE5_ARC_MIX_ARMS", raising=False)
    monkeypatch.delenv("STAGE5_ARC_MIX_ARC_EASY_REPEAT", raising=False)
    monkeypatch.delenv("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", raising=False)
    monkeypatch.delenv("STAGE5_ARC_MIX_OPUS_LIMIT", raising=False)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", module.DEFAULT_SOURCE_SUMMARY)
    monkeypatch.setattr(module, "RUN_ID", "arc_mix_once")

    env = module.child_env()

    assert env["STAGE5_ARC_MIX_RUN_ID"] == "arc_mix_once"
    assert env["STAGE5_ARC_MIX_SOURCE_SUMMARY"] == module.DEFAULT_SOURCE_SUMMARY
    assert env["STAGE5_ARC_MIX_ARMS"] == "arc_mix_response_w01_lr2e6"
    assert env["STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT"] == "2"
    assert env["STAGE5_ARC_MIX_ARC_EASY_REPEAT"] == "4"
    assert env["STAGE5_ARC_MIX_ARC_EVAL_LIMIT"] == "128"
    assert env["STAGE5_ARC_MIX_OPUS_LIMIT"] == "3000"
    assert env["STAGE5_ARC_MIX_MIN_MARGIN_DELTA"] == "-0.05"
    assert env["STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT"] == "16"
    assert env["STAGE5_ARC_MIX_PUSH"] == "1"


def test_arc_mix_recovery_once_preserves_explicit_arm_and_push(monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    monkeypatch.setenv("STAGE5_ARC_MIX_ARMS", "arc_mix_response_w01_lr2e6")
    monkeypatch.setenv("STAGE5_ARC_MIX_PUSH", "0")
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "outputs/stage5/custom/summary.json")
    monkeypatch.setattr(module, "RUN_ID", "custom_run")

    env = module.child_env()

    assert env["STAGE5_ARC_MIX_RUN_ID"] == "custom_run"
    assert env["STAGE5_ARC_MIX_SOURCE_SUMMARY"] == "outputs/stage5/custom/summary.json"
    assert env["STAGE5_ARC_MIX_ARMS"] == "arc_mix_response_w01_lr2e6"
    assert env["STAGE5_ARC_MIX_PUSH"] == "0"


def test_arc_mix_recovery_once_preflight_resolves_selected_checkpoint(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    checkpoint = tmp_path / "outputs" / "stage5" / "recovery" / "phase1" / "phase1_step_150.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    source = tmp_path / "outputs" / "stage5" / "assessment" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
        {
          "kind": "stage5_recovery_full_assessment",
          "status": "needs_competence_recovery",
          "selected_checkpoint": "outputs/stage5/recovery/phase1/phase1_step_150.pt"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr("colab.run_stage5_balanced_arc_mix_gate.ROOT", tmp_path)

    status, resolved = module.preflight_source_summary(source)

    assert status == "needs_competence_recovery"
    assert resolved == checkpoint


def test_arc_mix_recovery_once_preflight_only_skips_cuda_and_child_run(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    source = tmp_path / "outputs" / "stage5" / "assessment" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    called = False

    def fake_preflight(path):
        assert path == source
        return "needs_competence_recovery", tmp_path / "checkpoint.pt"

    def fake_go_no_go(path):
        assert path == source
        return {
            "decision": {"go": True, "status": "go_bounded_proxy", "spend_class": "single_arc_mix_proxy"},
            "checkpoint_preflight": {"available": True},
        }

    def fake_cuda():
        raise AssertionError("cuda should not be checked in preflight-only mode")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal called
        called = True
        raise AssertionError("child run should not launch in preflight-only mode")

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "outputs/stage5/assessment/summary.json")
    monkeypatch.setattr(module, "PREFLIGHT_ONLY", True)
    monkeypatch.setattr(module, "require_go_no_go", fake_go_no_go)
    monkeypatch.setattr(module, "preflight_source_summary", fake_preflight)
    monkeypatch.setattr(module, "require_cuda_runtime", fake_cuda)
    monkeypatch.setattr(module, "run", fake_run)

    assert module.run_recovery_gate() == 0
    assert called is False


def test_arc_mix_recovery_once_prints_next_plan_after_child_run(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    source = tmp_path / "outputs" / "stage5" / "assessment" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    output_summary = tmp_path / "outputs" / "stage5" / "arc_once" / "summary.json"
    output_summary.parent.mkdir(parents=True)
    output_summary.write_text(
        '{"kind": "stage5_balanced_arc_mix_gate", "decision": "stop_and_revise_objective"}',
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_go_no_go(path):
        assert path == source
        return {
            "decision": {"go": True, "status": "go_bounded_proxy", "spend_class": "single_arc_mix_proxy"},
            "checkpoint_preflight": {"available": True},
        }

    def fake_preflight(path):
        assert path == source
        return "needs_competence_recovery", tmp_path / "checkpoint.pt"

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "outputs/stage5/assessment/summary.json")
    monkeypatch.setattr(module, "RUN_ID", "arc_once")
    monkeypatch.setattr(module, "PREFLIGHT_ONLY", False)
    monkeypatch.setattr(module, "require_go_no_go", fake_go_no_go)
    monkeypatch.setattr(module, "preflight_source_summary", fake_preflight)
    monkeypatch.setattr(module, "require_cuda_runtime", lambda: None)
    monkeypatch.setattr(module, "run", fake_run)

    assert module.run_recovery_gate() == 0

    assert [sys.executable, "colab/run_stage5_balanced_arc_mix_gate.py"] in calls
    assert [
        sys.executable,
        "colab/review_stage5_arc_mix_result.py",
        "--summary",
        "outputs/stage5/arc_once/summary.json",
        "--no-write",
    ] in calls


def test_arc_mix_recovery_once_skips_next_plan_when_summary_missing(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "run", fake_run)

    module.print_next_plan(tmp_path / "outputs" / "stage5" / "missing" / "summary.json")

    assert calls == []


def test_arc_mix_recovery_once_go_no_go_blocks_unapproved_spend(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    source = tmp_path / "outputs" / "stage5" / "assessment" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")

    def fake_go_no_go(path):
        assert path == source
        raise RuntimeError("A100 go/no-go blocked ARC-mix recovery")

    def fake_preflight(path):
        raise AssertionError("source preflight should not run after go/no-go blocks")

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "outputs/stage5/assessment/summary.json")
    monkeypatch.setattr(module, "require_go_no_go", fake_go_no_go)
    monkeypatch.setattr(module, "preflight_source_summary", fake_preflight)

    with pytest.raises(RuntimeError, match="go/no-go blocked"):
        module.run_recovery_gate()


def test_arc_mix_recovery_once_refuses_cpu_by_default(monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    monkeypatch.setattr(module, "ALLOW_CPU", False)
    monkeypatch.setattr(module, "cuda_runtime_status", lambda: (False, "no cuda"))

    with pytest.raises(RuntimeError, match="Refusing to run ARC-mix recovery without CUDA"):
        module.require_cuda_runtime()


def test_arc_mix_recovery_once_allows_explicit_cpu(monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    called = False

    def fake_status() -> tuple[bool, str]:
        nonlocal called
        called = True
        return False, "no cuda"

    monkeypatch.setattr(module, "ALLOW_CPU", True)
    monkeypatch.setattr(module, "cuda_runtime_status", fake_status)

    module.require_cuda_runtime()

    assert called is False


def test_arc_mix_recovery_once_disconnects_on_early_failure(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_arc_mix_recovery_once as module

    disconnected = False

    def fake_disconnect() -> None:
        nonlocal disconnected
        disconnected = True

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "missing/summary.json")
    monkeypatch.setattr(module, "disconnect_if_requested", fake_disconnect)

    with pytest.raises(FileNotFoundError):
        module.main()

    assert disconnected is True
