from __future__ import annotations

from colab.run_stage5_colab_continue import default_env, focused_test_paths, mask_command, stage5_output_paths


def test_colab_continue_focuses_gate1_and_next_action_tests() -> None:
    paths = focused_test_paths()

    assert "tests/test_stage5_next_plan.py" in paths
    assert "tests/test_stage5_next_action.py" in paths
    assert "tests/test_stage5_gate1_assessment.py" in paths
    assert "tests/test_stage5_gate2_assessment.py" in paths
    assert "tests/test_stage5_recipe_control_assessment.py" in paths
    assert "tests/test_stage5_release_gate.py" in paths
    assert "tests/test_stage5_benchmark_assessment.py" in paths
    assert "tests/test_stage5_claim_packet.py" in paths
    assert "tests/test_stage5_progress_ledger.py" in paths
    assert "tests/test_stage5_benchmark_suite.py" in paths
    assert "tests/test_lora.py" in paths
    assert "tests/test_stage5_dense_sft_control.py" in paths


def test_colab_continue_defaults_to_bounded_two_action_loop(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS", raising=False)

    env = default_env()

    assert env["STAGE5_ARC_AGI_NEXT_ACTION_EXECUTE"] == "1"
    assert env["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "2"
    assert env["STAGE5_ARC_AGI_NEXT_ACTION_ALLOW_REPEAT"] == "0"


def test_colab_continue_max_actions_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_ARC_AGI_COLAB_CONTINUE_MAX_ACTIONS", "1")

    assert default_env()["STAGE5_ARC_AGI_NEXT_ACTION_MAX_ACTIONS"] == "1"


def test_colab_continue_commits_stage5_and_hf_export_outputs() -> None:
    assert stage5_output_paths() == ["outputs/stage5", "outputs/hf_exports"]


def test_mask_command_redacts_tokens(monkeypatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "secret-token")

    assert "secret-token" not in mask_command(["git", "clone", "https://secret-token@example/repo.git"])
    assert "****" in mask_command(["git", "clone", "https://secret-token@example/repo.git"])
