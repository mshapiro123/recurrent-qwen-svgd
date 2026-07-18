from __future__ import annotations

from colab.run_stage5_phase_g_multitarget_prepare import prepare_data


def test_multitarget_prepare_writes_disjoint_train_and_control_manifests(tmp_path) -> None:
    summary = prepare_data(
        tmp_path,
        train_rows_per_depth=2,
        control_rows_per_depth=2,
        n_symbols=12,
        max_depth=2,
    )

    assert summary["status"] == "prepared"
    assert summary["train"]["validation"]["status"] == "passed"
    assert summary["control"]["validation"]["status"] == "passed"
    assert summary["train"]["validation"]["all_reachable_targets_covered"] is True
    assert summary["control"]["validation"]["groups_with_multiple_targets"] == 4
    assert summary["train"]["base_question_sha256"] != summary["control"]["base_question_sha256"]
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "posterior_control.jsonl").exists()
    assert (tmp_path / "preregistration.json").exists()
