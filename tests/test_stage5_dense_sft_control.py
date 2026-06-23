from __future__ import annotations

import json
import subprocess

import colab.run_stage5_arc_agi_dense_sft as module
from colab.run_stage5_arc_agi_dense_sft import paired_comparisons, paired_selected_line


def _payload(*, selected: list[bool], best: list[bool] | None = None) -> dict[str, object]:
    best = selected if best is None else best
    examples = []
    for idx, (selected_hit, best_hit) in enumerate(zip(selected, best)):
        examples.append(
            {
                "task_id": f"task_{idx}",
                "test_index": 0,
                "has_target": True,
                "selected_exact": selected_hit,
                "best_of_k_exact": best_hit,
                "first_exact": selected_hit,
                "difficulty_bucket": "hard" if idx % 2 == 0 else "easy",
            }
        )
    return {
        "summary": {
            "selected_exact": sum(1 for value in selected if value),
            "best_of_k_exact": sum(1 for value in best if value),
            "first_exact": sum(1 for value in selected if value),
            "examples_with_targets": len(selected),
        },
        "examples": examples,
    }


def test_dense_sft_paired_comparisons_write_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "RUN_DIR", tmp_path)
    base = _payload(selected=[True, False, False])
    dense = _payload(selected=[True, True, False])
    phase1 = _payload(selected=[False, False, False])

    comparisons = paired_comparisons(base=base, dense_tuned=dense, phase1_start=phase1)

    assert comparisons["dense_tuned_vs_base"]["metrics"]["selected_exact"]["delta_exact"] == 1
    assert comparisons["dense_tuned_vs_phase1_start"]["metrics"]["selected_exact"]["delta_exact"] == 2
    assert (tmp_path / "dense_tuned_vs_base_paired_comparison.json").exists()
    assert (tmp_path / "dense_tuned_vs_base_paired_comparison.md").exists()
    saved = json.loads((tmp_path / "dense_tuned_vs_base_paired_comparison.json").read_text(encoding="utf-8"))
    assert saved["candidate_label"] == "dense_tuned"


def test_paired_selected_line_reports_sign_test_shape() -> None:
    comparison = {
        "metrics": {
            "selected_exact": {
                "delta_exact": 2,
                "wins": 2,
                "losses": 0,
                "ties": 8,
                "sign_test_p_value": 0.5,
            }
        }
    }

    assert paired_selected_line("dense_tuned_vs_base", comparison) == (
        "- dense_tuned_vs_base: selected delta `2` (2/0/8 W/L/T, p `0.5`)"
    )


def test_dense_sft_commit_uses_skip_ci(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "stage5" / "dense"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(cmd, *, check=True, log_name=None):
        commands.append([str(item) for item in cmd])
        if [str(item) for item in cmd] == ["git", "diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(cmd, 1, "", None)
        return subprocess.CompletedProcess(cmd, 0, "", None)

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUN_DIR", run_dir)
    monkeypatch.setattr(module, "run", fake_run)

    module.git_commit_results()

    commit_commands = [cmd for cmd in commands if cmd[:2] == ["git", "commit"]]
    assert commit_commands
    assert "[skip ci]" in " ".join(commit_commands[0])
