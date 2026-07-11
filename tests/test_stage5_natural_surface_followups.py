from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "colab" / "run_stage5_natural_surface_followups.py"
    spec = importlib.util.spec_from_file_location("natural_surface_followups", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_followup_plan_covers_corrected_names_syntax_and_probes() -> None:
    runner = load_runner()

    plan = runner.build_followup_plan(
        checkpoint_labels=["frozen_n24", "step_2000"],
        eval_variants=runner.DEFAULT_EVAL_VARIANTS,
        probe_families=runner.DEFAULT_PROBE_FAMILIES,
    )

    assert {row["variant"] for row in plan["active"]} >= {
        "corrected_relay_unseen_single_token_d1_12",
        "corrected_pointer_unseen_single_token_d1_12",
        "robust_relay_fronted_d1_12",
        "robust_pointer_fronted_d1_12",
        "robust_baton_fronted_d1_12",
        "robust_baton_passive_d1_12",
    }
    assert {row["family"] for row in plan["probes"]} == {
        "paired_relay",
        "paired_pointer",
        "baton_default",
    }
    assert len(plan["active"]) == 2 * len(runner.DEFAULT_EVAL_VARIANTS)
    assert len(plan["probes"]) == 2 * len(runner.DEFAULT_PROBE_FAMILIES)


def test_followup_plan_skips_completed_units() -> None:
    runner = load_runner()
    payload = {
        "active_evals": {"step_2000": {"robust_relay_fronted_d1_12": {"status": "finished"}}},
        "probes": {"step_2000": {"paired_relay": {"status": "finished"}}},
    }

    plan = runner.build_followup_plan(
        checkpoint_labels=["step_2000"],
        eval_variants=["robust_relay_fronted_d1_12", "robust_pointer_fronted_d1_12"],
        probe_families=["paired_relay", "paired_pointer"],
        payload=payload,
    )

    assert plan["active"] == [{"checkpoint": "step_2000", "variant": "robust_pointer_fronted_d1_12"}]
    assert plan["probes"] == [{"checkpoint": "step_2000", "family": "paired_pointer"}]


def test_bootstrap_exposes_followups_2_4_target() -> None:
    bootstrap = (ROOT / "colab" / "CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")

    assert '"natural_surface_followups_2_4"' in bootstrap
    assert "colab/STAGE5_NATURAL_SURFACE_FOLLOWUPS_CELL.py" in bootstrap
    assert "colab/run_stage5_natural_surface_followups.py" in bootstrap
