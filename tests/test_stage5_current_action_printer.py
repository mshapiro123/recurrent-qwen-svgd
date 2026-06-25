from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_current_action_printer_names_master_sequence_status() -> None:
    script = (ROOT / "colab/print_current_stage5_action.py").read_text(encoding="utf-8")

    assert "Phase 0: loop-closure re-entry" in script
    assert 'PREFERRED_STATUS_TARGET = "master_sequence_status"' in script
    assert "colab/plan_stage5_next_run.py" in script
    assert "stage5_current_source_summary.txt" in script
