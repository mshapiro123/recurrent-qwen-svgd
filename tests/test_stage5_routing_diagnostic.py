from __future__ import annotations

from pathlib import Path


def test_build_benchmark_env_runs_small_arc_easy_and_challenge(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_routing_diagnostic as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_ID", "routing")

    env = module.build_benchmark_env(
        checkpoint=tmp_path / "outputs" / "stage5" / "run" / "phase1.pt",
        source_summary=tmp_path / "outputs" / "stage5" / "run" / "summary.json",
        easy_limit="32",
        challenge_limit="48",
    )

    assert env["STAGE5_BENCHMARKS"] == "arc_easy,arc_challenge"
    assert env["STAGE5_BENCHMARK_ARC_EASY_LIMIT"] == "32"
    assert env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] == "48"
    assert env["STAGE5_BENCHMARK_RECURRENT_MODE"] == "phase1"
    assert env["STAGE5_BENCHMARK_NUM_TRAJECTORIES"] == "1"
    assert env["STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS"] == "1"
    assert env["STAGE5_BENCHMARK_PUSH"] == "0"


def test_assess_flags_direct_overloop_or_margin_drift() -> None:
    from colab.run_stage5_routing_diagnostic import assess

    payload = {
        "routing_diagnostics": {
            "arc_easy": {
                "label": {
                    "paired_examples": 2,
                    "delta": -1,
                    "routing_buckets": {
                        "base_confident_direct_proxy": {
                            "n": 2,
                            "delta": 0,
                            "wins": 0,
                            "losses": 0,
                            "mean_margin_delta": -0.2,
                            "mean_candidate_expected_loops": 2.8,
                        }
                    },
                }
            }
        }
    }

    result = assess(payload)

    assert result["status"] == "needs_direct_halting_repair"
    assert "base-logit distillation" in result["next_action"]


def test_assess_passes_when_direct_and_deep_nonnegative() -> None:
    from colab.run_stage5_routing_diagnostic import assess

    payload = {
        "routing_diagnostics": {
            "arc_challenge": {
                "label": {
                    "paired_examples": 2,
                    "delta": 1,
                    "routing_buckets": {
                        "base_confident_direct_proxy": {
                            "n": 1,
                            "delta": 0,
                            "wins": 0,
                            "losses": 0,
                            "mean_margin_delta": 0.01,
                            "mean_candidate_expected_loops": 1.4,
                        },
                        "deep_numeric_proxy": {
                            "n": 1,
                            "delta": 1,
                            "wins": 1,
                            "losses": 0,
                            "mean_margin_delta": 0.2,
                            "mean_candidate_expected_loops": 3.0,
                        },
                    },
                }
            }
        }
    }

    result = assess(payload)

    assert result["status"] == "routing_diagnostic_pass"


def test_default_benchmark_run_id_keeps_full_alias() -> None:
    import colab.run_stage5_routing_diagnostic as module

    run_id = module.default_benchmark_run_id("all", "64")

    assert "easyfull" in run_id
    assert "challenge64" in run_id
