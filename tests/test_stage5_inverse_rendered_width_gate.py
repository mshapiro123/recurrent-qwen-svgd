from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from training.abductive_injective_task import PhaseGFrozenEvalConfig

from colab.run_stage5_inverse_rendered_width_gate import (
    SOURCE_CAP3_SHA256,
    assess_deterministic_validity,
    prepare_inverse_rendered_split,
    resolve_keeper_source,
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


def test_original_staircase_keeper_source_remains_supported() -> None:
    source = {
        "kind": "stage5_inverse_table_rebase_caps",
        "stages": [
            {
                "cap": 3,
                "checkpoint": "cap3.pt",
                "checkpoint_drive_backup": "drive/cap3.pt",
                "checkpoint_sha256": SOURCE_CAP3_SHA256,
                "gate": {"correct": 46, "total": 64, "passed": True},
                "synthetic_guardrail": {"passed": False, "active_diagonal_min": 0.8125},
            }
        ],
    }

    keeper = resolve_keeper_source(source)

    assert keeper["checkpoint_sha256"] == SOURCE_CAP3_SHA256
    assert keeper["checkpoint_candidates"] == ["drive/cap3.pt", "cap3.pt"]
    assert keeper["retention"]["passed"] is False


def test_green_rehearsal_keeper_source_is_accepted_with_its_exact_hash() -> None:
    source = {
        "kind": "stage5_inverse_table_cap3_rehearsal",
        "status": "rehearsal_green_cap4_authorized",
        "cap4_authorized": True,
        "checkpoint": "rehearsal.pt",
        "checkpoint_drive_backup": "drive/rehearsal.pt",
        "checkpoint_sha256": "a" * 64,
        "task_gate": {"passed": True},
        "synthetic_retention": {"passed": True, "active_diagonal_min": 0.96875},
        "natural_canary": {"passed": True, "verdict": {"status": "green"}},
    }

    keeper = resolve_keeper_source(source)

    assert keeper["checkpoint_sha256"] == "a" * 64
    assert keeper["checkpoint_candidates"] == ["drive/rehearsal.pt", "rehearsal.pt"]
    assert keeper["retention"]["passed"] is True
    assert keeper["source_kind"] == "stage5_inverse_table_cap3_rehearsal"


def test_failed_rehearsal_keeper_source_is_rejected_before_restore() -> None:
    source = {
        "kind": "stage5_inverse_table_cap3_rehearsal",
        "status": "rehearsal_failed_review_required",
        "cap4_authorized": False,
        "checkpoint": "rehearsal.pt",
        "checkpoint_sha256": "b" * 64,
        "task_gate": {"passed": True},
        "synthetic_retention": {"passed": False},
        "natural_canary": {"passed": True},
    }

    try:
        resolve_keeper_source(source)
    except RuntimeError as exc:
        assert "not fully green" in str(exc)
    else:
        raise AssertionError("A failed rehearsal source was accepted as a deterministic keeper")


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
    assert '"inverse_rendered_width_gate_rehearsal"' in bootstrap


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
