from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from training.abductive_injective_task import PhaseGFrozenEvalConfig

from colab.run_stage5_inverse_rendered_width_gate import (
    assess_deterministic_validity,
    prepare_inverse_rendered_split,
)


def test_inverse_rendered_split_is_balanced_and_manifested() -> None:
    rows, receipt = prepare_inverse_rendered_split(
        PhaseGFrozenEvalConfig(rows_per_stratum=4, seed=7194203),
        split="calibration",
    )

    assert len(rows) == 12
    assert receipt["validation"]["status"] == "passed"
    assert receipt["manifest"]["stratum_counts"] == {"unique": 4, "small": 4, "large": 4}
    assert all(row["table_direction"] == "inverse_relation_given" for row in rows)


def test_deterministic_validity_gate_requires_pooled_and_every_depth() -> None:
    passing = {
        "overall": {"rows": 384, "greedy_chain_valid": 300},
        "by_depth": {
            "1": {"rows": 96, "greedy_chain_valid": 80},
            "2": {"rows": 96, "greedy_chain_valid": 75},
            "3": {"rows": 96, "greedy_chain_valid": 70},
            "4": {"rows": 96, "greedy_chain_valid": 75},
        },
    }
    failing = {
        **passing,
        "by_depth": {
            **passing["by_depth"],
            "4": {"rows": 96, "greedy_chain_valid": 57},
        },
    }

    assert assess_deterministic_validity(passing)["pass"] is True
    assert assess_deterministic_validity(failing)["pass"] is False
    assert assess_deterministic_validity(failing)["by_depth"]["4"]["required_correct"] == 58


def test_inverse_rendered_width_target_is_wired() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = Path("colab/STAGE5_INVERSE_RENDERED_WIDTH_GATE_CELL.py").read_text(encoding="utf-8")
    runner = Path("colab/run_stage5_inverse_rendered_width_gate.py").read_text(encoding="utf-8")

    assert '"inverse_rendered_width_gate"' in bootstrap
    assert "STAGE5_INVERSE_RENDERED_WIDTH_GATE_CELL_VERSION" in cell
    assert "data/phase_g_alpha_inverse_rendered" in cell
    assert "accepted_returncodes={0, 2}" in cell
    assert 'os.environ.get("STAGE5_BOOTSTRAP_REF", "main")' in cell
    assert "Pinned checkout verified" in cell
    assert runner.index("sys.path.insert(0, str(REPO_ROOT))") < runner.index("from colab.")
    assert "--backup_output_jsonl" in runner
    assert "--resume_source_jsonl" in runner


def test_inverse_rendered_runner_resolves_repo_packages_when_run_by_path(tmp_path: Path) -> None:
    env = {**os.environ, "STAGE5_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "colab/run_stage5_inverse_rendered_width_gate.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stdout
    assert "FileNotFoundError" in result.stdout
