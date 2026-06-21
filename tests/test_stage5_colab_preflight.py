from __future__ import annotations


def test_infer_stage5_run_id() -> None:
    import colab.check_stage5_colab_preflight as module

    assert module.infer_stage5_run_id("outputs/stage5/run_a/phase1/phase1_step_1.pt") == "run_a"
    assert module.infer_stage5_run_id("outputs/other/run_a/file.pt") is None


def test_child_summary_path(tmp_path, monkeypatch) -> None:
    import colab.check_stage5_colab_preflight as module

    monkeypatch.setattr(module, "ROOT", tmp_path)

    assert module.child_summary("run", "distill") == tmp_path / "outputs" / "stage5" / "run_distill" / "summary.json"


def test_summary_status_reports_missing(tmp_path) -> None:
    import colab.check_stage5_colab_preflight as module

    assert module.summary_status(tmp_path / "missing.json") == "missing"
