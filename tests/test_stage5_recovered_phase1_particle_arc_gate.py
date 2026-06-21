from __future__ import annotations

import json


def test_parse_repulsion_arms_names_decimal_scales() -> None:
    from colab.run_stage5_recovered_phase1_particle_arc_gate import parse_repulsion_arms

    arms = parse_repulsion_arms("0, 0.5,2")

    assert [arm.name for arm in arms] == ["rep0", "rep0p5", "rep2"]
    assert [arm.repulsion_scale for arm in arms] == ["0", "0.5", "2"]


def test_compact_result_extracts_arc_comparisons(tmp_path, monkeypatch) -> None:
    import colab.run_stage5_recovered_phase1_particle_arc_gate as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    summary = tmp_path / "outputs" / "stage5" / "run" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "checkpoint": "ckpt.pt",
                "recurrent_num_trajectories": 4,
                "comparisons": {
                    "arc_challenge": {
                        "label": {
                            "mean": {
                                "correct_delta_recurrent_vs_base": 1,
                                "base": {"correct": 2, "total": 3},
                                "recurrent": {"correct": 3, "total": 3},
                            }
                        }
                    }
                },
                "paired_comparisons": {
                    "arc_challenge": {
                        "label": {
                            "mean": {
                                "wins": 1,
                                "losses": 0,
                                "ties": 2,
                                "sign_test_p_value": 1.0,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    compact = module.compact_result(summary)

    assert compact["checkpoint"] == "ckpt.pt"
    assert compact["recurrent_num_trajectories"] == 4
    assert compact["comparisons"]["mean"]["correct_delta_recurrent_vs_base"] == 1
    assert compact["paired_comparisons"]["mean"]["wins"] == 1
