from __future__ import annotations

from colab.run_stage5_colab_continue import (
    auto_pull_enabled,
    committable_stage5_files,
    continuation_profile,
    default_env,
    default_max_actions,
    fast_forward_pull_latest,
    focused_test_paths,
    is_safe_output_artifact,
    mask_command,
    post_action_commands,
    run,
    stage5_output_paths,
)


def test_colab_continue_focuses_gate1_and_next_action_tests() -> None:
    paths = focused_test_paths()

    assert "tests/test_stage5_next_plan.py" in paths
    assert "tests/test_stage5_next_action.py" in paths
    assert "tests/test_arc_agi_training_signal.py" in paths
    assert "tests/test_stage5_arc_agi_candidate_distill_gate.py" in paths
    assert "tests/test_stage5_curriculum_particle_autopilot.py" in paths
    assert "tests/test_stage5_gate1_assessment.py" in paths
    assert "tests/test_stage5_gate2_assessment.py" in paths
    assert "tests/test_stage5_recipe_control_assessment.py" in paths
    assert "tests/test_stage5_recipe_selector_conversion.py" in paths
    assert "tests/test_stage5_release_gate.py" in paths
    assert "tests/test_stage5_selector_replication.py" in paths
    assert "tests/test_stage5_benchmark_assessment.py" in paths
    assert "tests/test_stage5_claim_packet.py" in paths
    assert "tests/test_stage5_arc_agi_baseline_registry.py" in paths
    assert "tests/test_stage5_arc_agi_sota_comparison.py" in paths
    assert "tests/test_stage5_progress_ledger.py" in paths
    assert "tests/test_stage5_benchmark_suite.py" in paths
    assert "tests/test_stage5_routing_repair.py" in paths
    assert "tests/test_lora.py" in paths
    assert "tests/test_stage5_dense_sft_control.py" in paths
    assert "tests/test_stage5_curriculum_sft.py" in paths
    assert "tests/test_curriculum_sft_gate.py" in paths
    assert "tests/test_curriculum_pipeline_from_artifacts.py" in paths
    assert "tests/test_curriculum_pipeline_fixture.py" in paths
    assert "tests/test_curriculum_generation_jobs.py" in paths
    assert "tests/test_collect_curriculum_job_outputs.py" in paths
    assert "tests/test_assemble_curriculum_records.py" in paths
    assert "tests/test_curriculum_jsonl.py" in paths
    assert "tests/test_stage5_notebooks.py" in paths


def test_colab_continue_defaults_to_credit_saving_single_action_loop(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS", raising=False)
    monkeypatch.delenv("STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE", raising=False)

    env = default_env()

    assert env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] == "1"
    assert env["STAGE5_A100_BUDGET_PROFILE"] == "credit_saver"
    assert continuation_profile() == "credit_saver"
    assert env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "1"
    assert env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] == "0"
    assert auto_pull_enabled()


def test_colab_continue_auto_pull_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_ARC_AGI_COLAB_CONTINUE_AUTO_PULL", "0")

    assert not auto_pull_enabled()


def test_colab_continue_gate_profile_runs_three_action_loop(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS", raising=False)
    monkeypatch.setenv("STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE", "gate")

    assert continuation_profile() == "gate"
    assert default_max_actions() == "3"
    assert default_env()["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "3"


def test_colab_continue_same_recipe_profile_runs_longer_ladder(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS", raising=False)
    monkeypatch.setenv("STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE", "same_recipe")

    assert continuation_profile() == "same_recipe"
    assert default_max_actions() == "6"
    assert default_env()["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "6"


def test_colab_continue_claim_profile_runs_release_ladder(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS", raising=False)
    monkeypatch.setenv("STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE", "claim")

    assert continuation_profile() == "claim"
    assert default_max_actions() == "10"
    assert default_env()["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "10"


def test_colab_continue_max_actions_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_ARC_AGI_COLAB_CONTINUE_PROFILE", "same_recipe")
    monkeypatch.setenv("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS", "1")

    assert default_env()["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "1"


def test_colab_continue_commits_stage5_and_hf_export_outputs() -> None:
    assert stage5_output_paths() == ["outputs/stage5", "outputs/hf_exports"]


def test_fast_forward_pull_latest_skips_dirty_worktree(monkeypatch) -> None:
    import colab.run_stage5_colab_continue as module

    calls: list[list[str]] = []

    def fake_run(cmd, *, check=True, env=None):
        calls.append(cmd)
        if cmd == ["git", "status", "--porcelain"]:
            return module.subprocess.CompletedProcess(cmd, 0, " M file.py\n", None)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(module, "run", fake_run)

    assert fast_forward_pull_latest() == {"status": "dirty_worktree"}
    assert calls == [["git", "status", "--porcelain"]]


def test_fast_forward_pull_latest_fetches_and_pulls_clean_worktree(monkeypatch) -> None:
    import colab.run_stage5_colab_continue as module

    calls: list[list[str]] = []

    def fake_run(cmd, *, check=True, env=None):
        calls.append(cmd)
        return module.subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "run", fake_run)

    assert fast_forward_pull_latest() == {"status": "ok"}
    assert calls == [
        ["git", "status", "--porcelain"],
        ["git", "fetch", "origin", "main"],
        ["git", "pull", "--ff-only", "origin", "main"],
    ]


def test_colab_continue_only_commits_safe_text_artifacts(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_colab_continue as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    safe = tmp_path / "outputs" / "stage5" / "run" / "summary.json"
    safe.parent.mkdir(parents=True)
    safe.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "outputs" / "stage5" / "run" / "phase1_step_100.pt"
    checkpoint.write_bytes(b"checkpoint")
    hf_checkpoint = tmp_path / "outputs" / "hf_exports" / "run" / "recurrent_adapter_checkpoint.pt"
    hf_checkpoint.parent.mkdir(parents=True)
    hf_checkpoint.write_bytes(b"checkpoint")
    card = tmp_path / "outputs" / "hf_exports" / "run" / "README.md"
    card.write_text("# card\n", encoding="utf-8")

    assert is_safe_output_artifact(safe)
    assert not is_safe_output_artifact(checkpoint)
    assert not is_safe_output_artifact(hf_checkpoint)
    assert sorted(committable_stage5_files()) == [
        "outputs/hf_exports/run/README.md",
        "outputs/stage5/run/summary.json",
    ]


def test_colab_continue_skips_oversized_text_artifacts(tmp_path) -> None:
    large = tmp_path / "large.jsonl"
    large.write_text("x" * 11, encoding="utf-8")

    assert not is_safe_output_artifact(large, max_bytes=10)


def test_colab_continue_summarizes_after_release_gate() -> None:
    scripts = [command[1] for command in post_action_commands()]

    assert scripts == [
        "colab/assess_stage5_release_gate.py",
        "colab/summarize_stage5_progress.py",
    ]


def test_mask_command_redacts_tokens(monkeypatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "secret-token")

    assert "secret-token" not in mask_command(["git", "clone", "https://secret-token@example/repo.git"])
    assert "****" in mask_command(["git", "clone", "https://secret-token@example/repo.git"])


def test_run_returns_nonzero_for_missing_optional_command() -> None:
    proc = run(["definitely_missing_stage5_colab_command"], check=False)

    assert proc.returncode == 127
    assert "command not found" in proc.stdout
