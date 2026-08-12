from __future__ import annotations

import inspect

from training import run_paper2_phase3_p33_i1 as runner


def test_i1_runner_pins_authority_endpoints_and_schedule() -> None:
    assert runner.EXPECTED_STRATEGY_SHA256 == (
        "bf60468965c5feb117cb2a3dd110f6746a5a56d2fddfb4fdca4aefad0b0aea3f"
    )
    assert runner.EXPECTED_P33_FINAL_SHA256 == {
        0: "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
        1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067",
    }
    assert runner.EXPECTED_OPTIMIZER_MARKED == 114_688
    assert runner.P33_TOTAL_STEPS == 1000
    assert runner.P33_LOOKS == 20
    assert runner.P33_LOOK_INTERVAL == 50


def test_i1_runner_uses_canonical_reader_and_never_scores_tasks() -> None:
    assert "canonical_top1" in inspect.getsource(runner._top1)
    source = inspect.getsource(runner.run)
    assert '"task_level_capability_scoring": False' in source
    assert '"confirm_scored": False' in source
    assert '"eval_e_scored": False' in source
    assert "p33_i1_total" in source
    assert "calibrate_aim_weight" in source
