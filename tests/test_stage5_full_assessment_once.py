from __future__ import annotations

import pytest


def test_full_assessment_once_defaults_to_latest_proxy_summary(monkeypatch) -> None:
    import colab.run_stage5_full_assessment_once as module

    monkeypatch.delenv("STAGE5_FULL_ASSESS_SOURCE_SUMMARY", raising=False)
    monkeypatch.delenv("STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY", raising=False)
    monkeypatch.setattr(
        module,
        "SOURCE_SUMMARY",
        module.DEFAULT_SOURCE_SUMMARY,
    )
    monkeypatch.setattr(module, "RUN_ID", "full_once")

    env = module.child_env()

    assert env["STAGE5_RECOVERY_FULL_ASSESS_RUN_ID"] == "full_once"
    assert env["STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY"] == module.DEFAULT_SOURCE_SUMMARY
    assert env["STAGE5_RECOVERY_FULL_ASSESS_PUSH"] == "1"


def test_full_assessment_once_child_env_preserves_explicit_push(monkeypatch) -> None:
    import colab.run_stage5_full_assessment_once as module

    monkeypatch.setenv("STAGE5_RECOVERY_FULL_ASSESS_PUSH", "0")
    monkeypatch.setattr(module, "SOURCE_SUMMARY", "outputs/stage5/custom/summary.json")
    monkeypatch.setattr(module, "RUN_ID", "custom_run")

    env = module.child_env()

    assert env["STAGE5_RECOVERY_FULL_ASSESS_RUN_ID"] == "custom_run"
    assert env["STAGE5_RECOVERY_FULL_ASSESS_SOURCE_SUMMARY"] == "outputs/stage5/custom/summary.json"
    assert env["STAGE5_RECOVERY_FULL_ASSESS_PUSH"] == "0"


def test_full_assessment_once_does_not_disconnect_by_default(monkeypatch) -> None:
    import colab.run_stage5_full_assessment_once as module

    called = False

    def fake_import(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal called
        called = True
        raise AssertionError("runtime should not be imported")

    monkeypatch.setattr(module, "AUTO_DISCONNECT", False)
    monkeypatch.setattr("builtins.__import__", fake_import)

    module.disconnect_if_requested()

    assert called is False


def test_full_assessment_once_run_tolerates_missing_optional_command(monkeypatch) -> None:
    import colab.run_stage5_full_assessment_once as module

    def missing_popen(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("missing")

    monkeypatch.setattr(module.subprocess, "Popen", missing_popen)

    proc = module.run(["not-a-command"], check=False)

    assert proc.returncode == 127
    assert "command not found" in proc.stdout


def test_full_assessment_once_refuses_cpu_by_default(monkeypatch) -> None:
    import colab.run_stage5_full_assessment_once as module

    monkeypatch.setattr(module, "ALLOW_CPU", False)
    monkeypatch.setattr(module, "cuda_runtime_status", lambda: (False, "no cuda"))

    with pytest.raises(RuntimeError, match="Refusing to run full ARC assessment without CUDA"):
        module.require_cuda_runtime()


def test_full_assessment_once_allows_explicit_cpu(monkeypatch) -> None:
    import colab.run_stage5_full_assessment_once as module

    called = False

    def fake_status() -> tuple[bool, str]:
        nonlocal called
        called = True
        return False, "no cuda"

    monkeypatch.setattr(module, "ALLOW_CPU", True)
    monkeypatch.setattr(module, "cuda_runtime_status", fake_status)

    module.require_cuda_runtime()

    assert called is False


def test_full_assessment_once_disconnects_on_early_failure(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_full_assessment_once as module

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
