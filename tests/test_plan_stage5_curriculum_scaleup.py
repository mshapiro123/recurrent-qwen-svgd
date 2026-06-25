from __future__ import annotations

from pathlib import Path

from colab.plan_stage5_curriculum_scaleup import build_plan, estimate_count_per_combo
from colab.reentry_recovery_config import assess_trace_curriculum_for_reentry_recovery


def tiny_trace_summary() -> dict:
    return {
        "kind": "stage5_capability_ladder_trace_collection",
        "status": "trace_curriculum_gate_ready",
        "gate": {"go": True},
        "curriculum": {
            "counts": {
                "typed_records": 63,
                "positive_sft_rows": 63,
                "mode_counts": {"direct": 26, "deep_narrow": 37},
                "target_loop_counts": {"1": 26, "2": 28, "3": 9},
            }
        },
    }


def test_estimate_count_per_combo_is_conservative_for_current_claim_deficit() -> None:
    assert (
        estimate_count_per_combo(
            positive_deficit=1937,
            seed_models=2,
            domains=2,
            difficulties=2,
            target_steps=4,
            overgenerate_factor=2.0,
        )
        == 122
    )


def test_curriculum_scaleup_plan_keeps_gpu_on_reentry_and_reports_deficits(tmp_path: Path) -> None:
    assessment = assess_trace_curriculum_for_reentry_recovery(
        tiny_trace_summary(),
        claim_min_positive_rows=2000,
        claim_min_mode_rows="direct=1000,deep_narrow=1000",
    )

    plan = build_plan(
        trace_summary=tmp_path / "summary.json",
        assessment=assessment,
        work_dir="data/curriculum/claim_direct_deep_001",
        provider_backend="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        model_map="data/curriculum/claim_direct_deep_001/model_map.json",
        overgenerate_factor=2.0,
        claim_min_positive_rows=2000,
        claim_min_mode_rows="direct=1000,deep_narrow=1000",
        claim_min_target_loop_rows="1=1000,2=500,3=500",
    )

    assert plan["positive_deficit"] == 1937
    assert plan["mode_deficits"] == {"deep_narrow": 963, "direct": 974}
    assert plan["generation_assumptions"]["estimated_count_per_combo"] == 122
    assert plan["actions"][0]["name"] == "keep_gpu_on_phase0"
    assert plan["actions"][0]["command"] == "STAGE5_CURRENT_A100_TARGET=reentry_repair_smoke"
    assert plan["actions"][1]["command"] == "STAGE5_CURRENT_A100_TARGET=claim_curriculum_scaleup_cpu"
    assert "training/run_curriculum_pipeline_from_artifacts.py" in plan["actions"][2]["command"]
    assert "--min_positive_rows 2000" in plan["actions"][4]["command"]
    assert "--min_mode_rows deep_narrow=1000,direct=1000" in plan["actions"][4]["command"]
    assert "--min_target_loop_rows 1=1000,2=500,3=500" in plan["actions"][4]["command"]
    assert "does not unlock Stage 4" in plan["phase_order_warning"]
