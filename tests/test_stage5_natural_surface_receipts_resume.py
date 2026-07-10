from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "colab" / "run_stage5_natural_surface_receipts_resume.py"
    spec = importlib.util.spec_from_file_location("natural_receipts_resume", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_active_eval_requires_committed_artifacts(tmp_path: Path) -> None:
    runner = load_runner()
    summary = tmp_path / "active_summary.json"
    log = tmp_path / "active_eval.log"
    record = {
        "active_summary": str(summary),
        "eval_log": str(log),
    }

    assert not runner.active_eval_record_complete(record, root=tmp_path)
    summary.write_text("{}", encoding="utf-8")
    assert not runner.active_eval_record_complete(record, root=tmp_path)
    log.write_text("ok", encoding="utf-8")
    assert runner.active_eval_record_complete(record, root=tmp_path)


def test_resume_plan_skips_completed_and_keeps_missing() -> None:
    runner = load_runner()
    payload = {
        "evals": {
            "step_6000": {
                "robust_baton_default_d1_12": {"active_summary": "done.json", "eval_log": "done.log"},
                "robust_relay_unseen_names_d1_12": {"active_summary": "done2.json", "eval_log": "done2.log"},
            }
        },
        "same_reader": {},
    }

    plan = runner.resume_plan(
        payload,
        checkpoint_labels=["step_6000"],
        variant_names=[
            "robust_baton_default_d1_12",
            "robust_relay_unseen_names_d1_12",
            "robust_pointer_unseen_names_d1_12",
        ],
        active_complete=lambda record: bool(record),
        same_reader_families=["relay_original", "pointer_original"],
    )

    assert plan["active_skipped"] == 2
    assert plan["active_pending"] == [
        {"checkpoint": "step_6000", "variant": "robust_pointer_unseen_names_d1_12"}
    ]
    assert plan["same_reader_pending"] == [
        {"checkpoint": "step_6000", "family": "relay_original"},
        {"checkpoint": "step_6000", "family": "pointer_original"},
    ]


def test_bootstrap_exposes_receipts_resume_target() -> None:
    bootstrap = (ROOT / "colab" / "CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    launcher = (ROOT / "colab" / "STAGE5_NATURAL_SURFACE_RECEIPTS_RESUME_CELL.py").read_text(encoding="utf-8")

    assert '"natural_surface_receipts_resume"' in bootstrap
    assert "STAGE5_NATURAL_RECEIPTS_RESUME_RUN_ID" in bootstrap
    assert "colab/run_stage5_natural_surface_receipts_resume.py" in launcher
    assert "stage5_natural_surface_receipts_20260709_210151" in launcher
