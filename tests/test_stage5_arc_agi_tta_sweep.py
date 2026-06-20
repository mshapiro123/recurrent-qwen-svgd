from __future__ import annotations

from pathlib import Path

from colab.run_stage5_arc_agi_tta_sweep import (
    ModelArm,
    TTA_VARIANTS,
    arm_variant_label,
    compute_deltas,
    eval_arm,
    paired_comparison_specs,
    requested_model_arms,
    requested_tta_variants,
)


def _row(arm: str, variant: str, *, best: int, selected: int, model_exact: int) -> dict[str, object]:
    return {
        "arm": arm,
        "tta_variant": variant,
        "first_exact": selected,
        "selected_exact": selected,
        "best_of_k_exact": best,
        "tasks_solved_best_of_k": best,
        "model_exact_count": model_exact,
    }


def test_requested_tta_variants_defaults_to_none_and_all(monkeypatch) -> None:
    monkeypatch.delenv("STAGE5_ARC_AGI_TTA_VARIANTS", raising=False)

    variants = requested_tta_variants()

    assert [variant.name for variant in variants] == ["none", "all"]
    assert variants[1] == TTA_VARIANTS["all"]


def test_requested_tta_variants_validates_names(monkeypatch) -> None:
    monkeypatch.setenv("STAGE5_ARC_AGI_TTA_VARIANTS", "none,rotations")
    assert [variant.name for variant in requested_tta_variants()] == ["none", "rotations"]

    monkeypatch.setenv("STAGE5_ARC_AGI_TTA_VARIANTS", "bogus")
    try:
        requested_tta_variants()
    except ValueError as exc:
        assert "Unknown TTA variants" in str(exc)
    else:
        raise AssertionError("unknown TTA variant should fail")


def test_requested_model_arms_validates_names(monkeypatch) -> None:
    start = Path("start.pt")
    recovered = Path("recovered.pt")
    monkeypatch.setenv("STAGE5_ARC_AGI_TTA_MODELS", "base,recovered")

    arms = requested_model_arms(start, recovered)

    assert arms == [ModelArm("base", "base"), ModelArm("recovered", "phase1", recovered)]


def test_compute_deltas_tracks_tta_and_recurrent_lift() -> None:
    rows = [
        _row("base", "none", best=3, selected=2, model_exact=3),
        _row("base", "all", best=5, selected=3, model_exact=6),
        _row("recovered", "none", best=2, selected=1, model_exact=2),
        _row("recovered", "all", best=4, selected=3, model_exact=5),
    ]

    deltas = compute_deltas(rows)

    assert deltas["base:all_vs_none"]["best_of_k_exact_delta"] == 2
    assert deltas["recovered:all_vs_none"]["selected_exact_delta"] == 2
    assert deltas["recovered:vs_base_at_all"]["best_of_k_exact_delta"] == -1
    assert deltas["recovered:vs_base_at_none"]["model_exact_count_delta"] == -1


def test_arm_variant_label_matches_eval_output_prefix() -> None:
    assert arm_variant_label("recovered", "all") == "recovered__tta_all"


def test_paired_comparison_specs_cover_tta_and_base_gap() -> None:
    rows = [
        _row("base", "none", best=3, selected=2, model_exact=3),
        _row("phase1_start", "none", best=2, selected=1, model_exact=2),
        _row("recovered", "none", best=2, selected=1, model_exact=2),
        _row("base", "all", best=5, selected=3, model_exact=6),
        _row("phase1_start", "all", best=3, selected=2, model_exact=3),
        _row("recovered", "all", best=4, selected=3, model_exact=5),
    ]

    assert paired_comparison_specs(rows) == [
        ("phase1_start__vs_base__tta_none", "base__tta_none", "phase1_start__tta_none"),
        ("recovered__vs_base__tta_none", "base__tta_none", "recovered__tta_none"),
        ("base__tta_all_vs_none", "base__tta_none", "base__tta_all"),
        ("phase1_start__tta_all_vs_none", "phase1_start__tta_none", "phase1_start__tta_all"),
        ("phase1_start__vs_base__tta_all", "base__tta_all", "phase1_start__tta_all"),
        ("recovered__tta_all_vs_none", "recovered__tta_none", "recovered__tta_all"),
        ("recovered__vs_base__tta_all", "base__tta_all", "recovered__tta_all"),
    ]


def test_eval_arm_omits_limit_for_full_split(monkeypatch, tmp_path) -> None:
    import colab.run_stage5_arc_agi_tta_sweep as module

    commands: list[list[str]] = []
    summary_json = module.RUN_DIR / "base__tta_none_summary.json"
    summary_json.write_text(
        json_payload(
            {
                "summary": {
                    "first_exact": 0,
                    "selected_exact": 0,
                    "best_of_k_exact": 0,
                    "examples_with_targets": 0,
                    "tasks_solved_best_of_k": 0,
                    "tasks_with_targets": 0,
                    "valid_candidate_rate": 0.0,
                },
                "candidate_source_summary": {},
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd: list[str], **kwargs) -> None:
        commands.append(cmd)

    monkeypatch.setattr(module, "LIMIT", None)
    monkeypatch.setattr(module, "RESUME", False)
    monkeypatch.setattr(module, "run", fake_run)

    eval_arm(ModelArm("base", "base"), TTA_VARIANTS["none"], tmp_path)

    assert commands
    assert "--limit" not in commands[0]


def json_payload(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload)
