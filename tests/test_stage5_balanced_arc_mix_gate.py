from __future__ import annotations

from pathlib import Path

from colab.run_stage5_balanced_arc_mix_gate import (
    arm_config,
    build_summary,
    checkpoint_run_id,
)


def _arm(name: str, *, start: int, base: int, best: int) -> dict:
    return {
        "arm": name,
        "base_arc": {"mean": {"correct": base, "total": 128, "accuracy": base / 128}},
        "phase1_start": {
            "checkpoint": "outputs/stage5/source/phase1/phase1_step_150.pt",
            "val": {},
            "arc": {"mean": {"correct": start, "total": 128, "accuracy": start / 128}},
        },
        "checkpoints": [],
        "best_checkpoint": {
            "checkpoint": f"outputs/stage5/child/{name}/phase1/phase1_step_50.pt",
            "arc": {"mean": {"correct": best, "total": 128, "accuracy": best / 128}},
        },
    }


def test_checkpoint_run_id_from_stage5_path() -> None:
    assert (
        checkpoint_run_id("outputs/stage5/stage5_run/phase1/phase1_step_150.pt")
        == "stage5_run"
    )


def test_arm_config_defaults_to_arc_mix_no_distill() -> None:
    cfg = arm_config("arc_mix_nodistill_lr3e6")

    assert cfg.distill_enabled == "0"
    assert cfg.learning_rate == "3e-6"
    assert cfg.steps == "150"


def test_build_summary_passes_when_arc_mix_lifts_proxy(tmp_path) -> None:
    payload = build_summary(
        source_summary=tmp_path / "balanced.json",
        source_payload={"status": "needs_competence_recovery"},
        resume_checkpoint=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
        data_summary={"mixed_rows": 10},
        arms=[_arm("arc_mix", start=71, base=72, best=73)],
    )

    assert payload["status"] == "proxy_lift"
    assert payload["passed"] is True
    assert payload["best_arm"]["arm"] == "arc_mix"


def test_build_summary_fails_without_proxy_lift(tmp_path) -> None:
    payload = build_summary(
        source_summary=tmp_path / "balanced.json",
        source_payload={"status": "needs_competence_recovery"},
        resume_checkpoint=Path("outputs/stage5/source/phase1/phase1_step_150.pt"),
        data_summary={"mixed_rows": 10},
        arms=[_arm("arc_mix", start=71, base=72, best=70)],
    )

    assert payload["status"] == "no_proxy_lift"
    assert payload["passed"] is False
