"""Run the read-only Phase G injective curriculum construction autopsy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_abductive_coverage import uniform_expected_coverage  # noqa: E402
from training.abductive_injective_task import (  # noqa: E402
    AbductiveInjectiveConfig,
    build_rows,
    row_manifest,
    with_inverse_table_prompt,
    write_jsonl,
)


EXPECTED_TRAIN_SHA = "4ab6377a15d64cf5e07c8855ed05f432feed75e512e196cbd53f648dc9fcb4a5"
EXPECTED_TEST_SHA = "4dd29d9fb7b4170390234646c7c1773377eea56145f6ae659e38f3ae443f2068"
BASELINE_SHA = "0d6cf119bd66290a2c85686bf58fdc6f9363109c8fdae0ea625f32d13409a1a6"
RECOVERY_SHA = "fc98feb5d5bd450f7ecc4f6d43ce36fd436418d7ad2cd69df38a089d5ec453d1"
BASELINE_SUMMARY = ROOT / "outputs/stage5/stage5_phase_g_experiment1_fixed_boundary_20260712/summary.json"
RECOVERY_SUMMARY = ROOT / "outputs/stage5/stage5_phase_g_injective_curriculum_recovery_20260712/summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def restore_checkpoint(candidates: list[str], destination: Path, *, expected_sha: str) -> Path:
    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.exists():
            continue
        actual = sha256_file(candidate)
        if actual != expected_sha:
            raise RuntimeError(f"Checkpoint SHA mismatch for {candidate}: {actual} != {expected_sha}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if candidate.resolve() != destination.resolve():
            shutil.copy2(candidate, destination)
        if sha256_file(destination) != expected_sha:
            raise RuntimeError(f"Copied checkpoint SHA mismatch for {destination}")
        print(f"restored_checkpoint={candidate} sha256={actual}", flush=True)
        return destination
    raise FileNotFoundError(f"Could not restore checkpoint with SHA {expected_sha} from {candidates}")


def construction_audit() -> dict[str, Any]:
    wrapper_source = (ROOT / "models/recurrent_wrapper.py").read_text(encoding="utf-8")
    runner_source = (ROOT / "colab/run_stage5_phase_g_experiment1.py").read_text(encoding="utf-8")
    evaluator_source = (ROOT / "eval/eval_abductive_coverage.py").read_text(encoding="utf-8")
    bootstrap_source = (ROOT / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    checks = {
        "per_loop_labels_mask_active_only": 'loss = (ce_tensor * active).sum()' in wrapper_source,
        "labels_beyond_available_loops_masked": "labels_for_loop = torch.full_like(labels_flat, -100)" in wrapper_source,
        "training_max_loops_was_eight": '"max_loops": 8' in runner_source,
        "evaluation_requests_row_depth": (
            "loop_counts = list(range(1, depth + 1))" in evaluator_source
            and "scores = scores_by_loop[depth]" in evaluator_source
        ),
        "recovery_ramped_compute": (
            '"STAGE5_PHASE_G_EXP1_CURRICULUM_RAMP_COMPUTE": "1"' in bootstrap_source
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Construction audit static check failed: {checks}")
    return {
        "checks": checks,
        "conclusion": (
            "The first arm used fixed eight-loop compute, but loops beyond row depth had no active labels; "
            "evaluation read loop d. There was no trained hold objective or end-reader mismatch."
        ),
    }


def sampling_recalibration(source: dict[str, Any]) -> dict[str, Any]:
    sampling = source["injective_smoke"]["overall"]["sampling"]
    rows: dict[str, Any] = {}
    for key, value in sampling.items():
        count = int(key)
        observed = float(value["mean_coverage"])
        uniform = uniform_expected_coverage(n_symbols=20, samples=count)
        rows[key] = {
            "K": count,
            "observed_mean_coverage": observed,
            "uniform_expected_coverage": uniform,
            "coverage_minus_uniform": observed - uniform,
        }
    return {
        "n_symbols": 20,
        "formula": "1 - (1 - 1/N)^K",
        "sampling": rows,
        "k20_beats_uniform": rows["20"]["coverage_minus_uniform"] > 0.0,
    }


def seen_training_indices(*, dataset_size: int, seed: int, steps: int) -> list[int]:
    """Reproduce the single-process shuffled DataLoader order used in training."""

    loader = DataLoader(
        list(range(int(dataset_size))),
        batch_size=1,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    return [int(batch.item()) for batch, _step in zip(loader, range(int(steps)))]


def balanced_seen_rows(
    rows: list[dict[str, Any]],
    *,
    seen_indices: list[int],
    rows_per_depth: int,
) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for index in seen_indices:
        row = rows[int(index)]
        depth = int(row["depth"])
        if counts.get(depth, 0) >= int(rows_per_depth):
            continue
        selected.append(row)
        counts[depth] = counts.get(depth, 0) + 1
    expected = {depth: int(rows_per_depth) for depth in range(1, 9)}
    if counts != expected:
        raise RuntimeError(f"Could not construct balanced seen-row audit set: {counts} != {expected}")
    return selected


def generate_locked_data(data_dir: Path) -> dict[str, Any]:
    train_config = AbductiveInjectiveConfig(n_symbols=20, max_depth=8, rows_per_depth=256, seed=1_104_729)
    test_config = AbductiveInjectiveConfig(n_symbols=20, max_depth=8, rows_per_depth=128, seed=1_104_729)
    train_rows = build_rows(train_config, split="train", mode="injective")
    test_rows = build_rows(test_config, split="test", mode="injective")
    manifests = {"train": row_manifest(train_rows), "test": row_manifest(test_rows)}
    if manifests["train"]["row_sha256"] != EXPECTED_TRAIN_SHA:
        raise RuntimeError("Regenerated train rows do not match the locked manifest")
    if manifests["test"]["row_sha256"] != EXPECTED_TEST_SHA:
        raise RuntimeError("Regenerated test rows do not match the locked manifest")
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "train_injective.jsonl", train_rows)
    write_jsonl(data_dir / "test_injective.jsonl", test_rows)
    baseline_seen_indices = seen_training_indices(dataset_size=len(train_rows), seed=81_001, steps=1_000)
    train_seen_common = balanced_seen_rows(
        train_rows,
        seen_indices=baseline_seen_indices,
        rows_per_depth=16,
    )
    write_jsonl(data_dir / "train_injective_seen_by_both.jsonl", train_seen_common)
    inverse_train = [with_inverse_table_prompt(row) for row in train_rows]
    inverse_test = [with_inverse_table_prompt(row) for row in test_rows]
    write_jsonl(data_dir / "train_injective_inverse_given.jsonl", inverse_train)
    write_jsonl(data_dir / "test_injective_inverse_given.jsonl", inverse_test)
    return {
        "canonical": manifests,
        "training_exposure_selection": {
            "seed": 81_001,
            "baseline_steps": 1_000,
            "selection": "16_per_depth_from_exact_seeded_dataloader_prefix_seen_by_both_checkpoints",
            "manifest": row_manifest(train_seen_common),
        },
        "inverse_control": {
            "train": row_manifest(inverse_train),
            "test": row_manifest(inverse_test),
            "status": "prepared_not_trained",
        },
    }


def publish(run_dir: Path) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or "restored" in path.parts or "data" in path.parts:
            continue
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(
        ["git", "commit", "-m", f"Record Phase G curriculum autopsy {run_dir.name} [skip ci]"],
        cwd=ROOT,
        check=True,
    )
    push = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if push.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def main() -> int:
    run_id = os.environ.get("STAGE5_PHASE_G_AUTOPSY_RUN_ID") or time.strftime(
        "stage5_phase_g_curriculum_autopsy_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    data_receipt = generate_locked_data(run_dir / "data")
    baseline_source = read_json(BASELINE_SUMMARY)
    recovery_source = read_json(RECOVERY_SUMMARY)
    checkpoints = {
        "fixed_boundary_step1000": restore_checkpoint(
            [
                os.environ.get("STAGE5_PHASE_G_AUTOPSY_BASELINE_CHECKPOINT", ""),
                str(ROOT / baseline_source["injective_train"]["checkpoint"]),
                str(baseline_source["injective_train"]["checkpoint_drive_backup"]),
            ],
            run_dir / "restored/fixed_boundary_step1000.pt",
            expected_sha=BASELINE_SHA,
        ),
        "curriculum_recovery_step2000": restore_checkpoint(
            [
                os.environ.get("STAGE5_PHASE_G_AUTOPSY_RECOVERY_CHECKPOINT", ""),
                str(ROOT / recovery_source["injective_train"]["checkpoint"]),
                str(recovery_source["injective_train"]["checkpoint_drive_backup"]),
            ],
            run_dir / "restored/curriculum_recovery_step2000.pt",
            expected_sha=RECOVERY_SHA,
        ),
    }
    summaries: dict[str, Any] = {}
    for label, checkpoint in checkpoints.items():
        output_dir = run_dir / "eval" / label
        run(
            [
                sys.executable,
                "eval/eval_abductive_curriculum_autopsy.py",
                "--train_jsonl",
                path_for_cli(run_dir / "data/train_injective_seen_by_both.jsonl"),
                "--test_jsonl",
                path_for_cli(run_dir / "data/test_injective.jsonl"),
                "--checkpoint",
                path_for_cli(checkpoint),
                "--output_dir",
                path_for_cli(output_dir),
                "--rows_per_depth",
                os.environ.get("STAGE5_PHASE_G_AUTOPSY_ROWS_PER_DEPTH", "16"),
                "--max_loops",
                "8",
                "--state_query_examples",
                os.environ.get("STAGE5_PHASE_G_AUTOPSY_STATE_QUERY_EXAMPLES", "8"),
                "--bridge_projection_mode",
                "split",
                "--dtype",
                os.environ.get("STAGE5_PHASE_G_AUTOPSY_DTYPE", "bfloat16"),
                "--adapter_dtype",
                "float32",
                "--device",
                os.environ.get("DEVICE", "cuda"),
            ]
        )
        summaries[label] = read_json(output_dir / "summary.json")
        publish(run_dir)

    summary = {
        "kind": "stage5_phase_g_curriculum_autopsy",
        "run_id": run_id,
        "status": "finished_read_only_pause_for_review",
        "construction_audit": construction_audit(),
        "prediction_confusion_prior_from_landed_smokes": {
            "fixed_boundary_step1000": {
                "other_valid_name": 83,
                "correct_start": 22,
                "other_orbit_intermediate": 19,
                "one_step_preimage": 4,
            },
            "curriculum_recovery_step2000": {
                "other_valid_name": 82,
                "correct_start": 26,
                "other_orbit_intermediate": 14,
                "one_step_preimage": 6,
            },
            "conclusion": "The saved diagonal predictions falsify the invert-once-then-hold account.",
        },
        "answer_head_sampling_recalibration": sampling_recalibration(baseline_source),
        "data_receipt": data_receipt,
        "checkpoint_sha256": {
            "fixed_boundary_step1000": BASELINE_SHA,
            "curriculum_recovery_step2000": RECOVERY_SHA,
        },
        "checkpoint_autopsies": summaries,
        "next_training": "disabled_pending_strategy_review",
        "inverse_table_control": "prepared_but_not_launched",
    }
    write_json(run_dir / "summary.json", summary)
    lines = [
        f"# Phase G Curriculum Autopsy - {run_id}",
        "",
        "- Status: `finished_read_only_pause_for_review`",
        "- No training was run.",
        "- Fixed-eight compute did not create an end-read or supervised hold objective.",
        "- Train/held-out loop matrices, above-diagonal behavior, and state-query ranks are in the checkpoint summaries.",
        "- The inverse-table control data is prepared but deliberately not trained.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    publish(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
