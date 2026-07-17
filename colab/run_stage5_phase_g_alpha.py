"""Run the frozen-substrate Phase G-alpha KL sweep and coverage gate."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_phase_g_alpha import (  # noqa: E402
    sample_temperature_predictions,
)
from eval.phase_g_branching import (  # noqa: E402
    exact_branching_coverage,
    solve_global_temperature,
)
from training.branching_relations_task import (  # noqa: E402
    BranchingRelationsConfig,
    build_rows,
    row_manifest,
    validate_rows,
)
from colab.stage5_path_utils import (  # noqa: E402
    repo_relative_text as _repo_relative_text,
    resolve_repo_path as _resolve_repo_path,
)


KEEPER_SHA256 = "0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f"
KEEPER_DRIVE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-backups/"
    "natural_surface_backup_20260709_180835/checkpoints/"
    "stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812/"
    "unfrozen_recurrent_step_2000.pt"
)
DETERMINISTIC_TEST_ROWS = (
    ROOT
    / "outputs/stage5/stage5_part1_closeout_pivot_20260715/"
    "branching_screen/natural_step2000_N20_verbal/rows.jsonl"
)
EXPECTED_TEST_MANIFEST = {
    "rows": 512,
    "row_id_sha256": "a75171fd3f8a22b632dfb525fd8c1b44136b095d0db6ec36433751b521c60758",
    "row_sha256": "eb80ef24637aee511a3e35607e87ae2530842ce11c551e6fa90ecda4d4115ef8",
}


def run(
    command: list[str],
    *,
    allow_blocked: bool = False,
) -> int:
    print("$", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code and not (allow_blocked and return_code == 2):
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve a checkpoint path reported as either repo-relative or absolute."""

    return _resolve_repo_path(ROOT, raw)


def repo_relative_text(raw: str | Path) -> str:
    """Return a stable repo-relative path without assuming input path form."""

    return _repo_relative_text(ROOT, raw)


def publish(run_dir: Path, message: str) -> None:
    subprocess.run(
        ["git", "pull", "--rebase", "--autostash", "origin", "main"],
        cwd=ROOT,
        check=False,
    )
    paths = [run_dir / "summary.json"]
    paths.extend(run_dir.glob("**/summary.json"))
    paths.extend(run_dir.glob("**/rng_manifest.jsonl"))
    for path in sorted(set(paths)):
        if path.exists():
            subprocess.run(
                ["git", "add", "-f", path.relative_to(ROOT).as_posix()],
                cwd=ROOT,
                check=True,
            )
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", "origin", "main"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def sync_receipts_to_drive(run_dir: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    for source in run_dir.rglob("*"):
        if not source.is_file() or source.suffix in {".pt", ".pth", ".safetensors"}:
            continue
        destination = drive_dir / source.relative_to(run_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def restore_keeper(run_dir: Path) -> Path:
    local = run_dir / "restored" / "natural_step2000.pt"
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        if not KEEPER_DRIVE.exists():
            raise FileNotFoundError(f"Missing locked keeper: {KEEPER_DRIVE}")
        shutil.copy2(KEEPER_DRIVE, local)
    import hashlib

    hash_state = hashlib.sha256()
    with local.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_state.update(chunk)
    digest = hash_state.hexdigest()
    if digest != KEEPER_SHA256:
        raise RuntimeError(f"Keeper SHA mismatch: {digest}")
    return local


def prepare_data(run_dir: Path) -> dict[str, Any]:
    data_dir = run_dir / "data"
    train_rows = build_rows(
        BranchingRelationsConfig(rows_per_depth=512, max_depth=4),
        split="phase_g_train",
        rendering="verbal",
        n_symbols=20,
    )
    calibration_rows = build_rows(
        BranchingRelationsConfig(rows_per_depth=128, max_depth=4),
        split="calibration",
        rendering="verbal",
        n_symbols=20,
    )
    test_rows = build_rows(
        BranchingRelationsConfig(rows_per_depth=128, max_depth=4),
        split="test",
        rendering="verbal",
        n_symbols=20,
    )
    for name, rows in (
        ("train", train_rows),
        ("calibration", calibration_rows),
        ("test", test_rows),
    ):
        validation = validate_rows(rows)
        if validation["status"] != "passed":
            raise RuntimeError(f"Invalid Phase G {name} rows: {validation['errors'][:5]}")
        write_jsonl(data_dir / f"{name}.jsonl", rows)
    if row_manifest(test_rows) != EXPECTED_TEST_MANIFEST:
        raise RuntimeError("Regenerated Phase G test rows differ from the frozen screen")
    if [row["id"] for row in test_rows] != [
        row["id"] for row in read_jsonl(DETERMINISTIC_TEST_ROWS)
    ]:
        raise RuntimeError("Frozen Phase G test IDs differ from the deterministic receipt")
    return {
        "train": row_manifest(train_rows),
        "calibration": row_manifest(calibration_rows),
        "test": row_manifest(test_rows),
    }


def deterministic_screen(
    *,
    run_dir: Path,
    keeper: Path,
    split_name: str,
) -> Path:
    output_dir = run_dir / "deterministic" / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "rows.jsonl").exists() and (output_dir / "summary.json").exists():
        print(f"resume_deterministic_screen={output_dir}", flush=True)
        return output_dir / "rows.jsonl"
    data = (
        run_dir / "data" / f"{split_name}.jsonl"
        if split_name != "test"
        else run_dir / "data/test.jsonl"
    )
    run(
        [
            sys.executable,
            "eval/eval_branching_relations.py",
            "--data_jsonl",
            str(data.relative_to(ROOT)),
            "--checkpoint",
            str(keeper),
            "--output_jsonl",
            str((output_dir / "rows.jsonl").relative_to(ROOT)),
            "--output_summary",
            str((output_dir / "summary.json").relative_to(ROOT)),
            "--bridge_projection_mode",
            "split",
            "--dtype",
            "bfloat16",
            "--adapter_dtype",
            "float32",
            "--device",
            "cuda",
        ]
    )
    return output_dir / "rows.jsonl"


def lock_margin(calibration_rows_path: Path, output_path: Path) -> dict[str, Any]:
    rows = read_jsonl(calibration_rows_path)
    temperature = solve_global_temperature(
        [dict(row["scores"]) for row in rows],
        target_mean_entropy=0.1432,
    )
    first: list[float] = []
    second: list[float] = []
    for index, row in enumerate(rows):
        predictions_a = sample_temperature_predictions(
            row["scores"],
            temperature=temperature["temperature"],
            k_max=20,
            seed=20260717 + index,
        )
        predictions_b = sample_temperature_predictions(
            row["scores"],
            temperature=temperature["temperature"],
            k_max=20,
            seed=20270717 + index,
        )
        first.append(exact_branching_coverage(predictions_a, row["reachable_symbols"])["coverage"])
        second.append(exact_branching_coverage(predictions_b, row["reachable_symbols"])["coverage"])
    differences = [a - b for a, b in zip(first, second)]
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    empirical_mde_80 = (1.959964 + 0.841621) * math.sqrt(variance / len(differences))
    locked = max(0.05, math.ceil(empirical_mde_80 / 0.005) * 0.005)
    payload = {
        "kind": "phase_g_alpha_powered_margin_lock",
        "status": "locked_before_guided_training",
        "paired_rows": len(rows),
        "alpha": 0.05,
        "power": 0.8,
        "null_comparator": "two independent entropy-matched answer-head K20 samples",
        "target_mean_entropy": 0.1432,
        "temperature_match": temperature,
        "null_paired_sd": math.sqrt(variance),
        "empirical_mde_80": empirical_mde_80,
        "program_floor": 0.05,
        "locked_absolute_mean_coverage_margin": locked,
    }
    write_json(output_path, payload)
    return payload


def evaluate(
    *,
    data: Path,
    deterministic_rows: Path,
    keeper: Path,
    output_dir: Path,
    guidance_checkpoint: Path | None,
    sample_counts: str,
    coverage_margin: float,
    include_temperature: bool,
    include_iso: bool,
    include_teacher: bool,
    resume_cache_path: Path,
) -> dict[str, Any]:
    if (output_dir / "summary.json").exists():
        print(f"resume_completed_phase_g_eval={output_dir}", flush=True)
        return read_json(output_dir / "summary.json")
    command = [
        sys.executable,
        "eval/eval_phase_g_alpha.py",
        "--data_jsonl",
        str(data.relative_to(ROOT)),
        "--deterministic_rows_jsonl",
        str(deterministic_rows.relative_to(ROOT)),
        "--keeper",
        str(keeper),
        "--expected_keeper_sha256",
        KEEPER_SHA256,
        "--output_dir",
        str(output_dir.relative_to(ROOT)),
        "--resume_cache_path",
        str(resume_cache_path),
        "--sample_counts",
        sample_counts,
        "--coverage_margin",
        str(coverage_margin),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
    ]
    if guidance_checkpoint:
        command.extend(["--guidance_checkpoint", str(guidance_checkpoint)])
    command.append("--include_temperature" if include_temperature else "--no-include_temperature")
    command.append("--include_iso_compute" if include_iso else "--no-include_iso_compute")
    command.append(
        "--include_posterior_teacher"
        if include_teacher
        else "--no-include_posterior_teacher"
    )
    run(command)
    return read_json(output_dir / "summary.json")


def main() -> int:
    run_id = os.environ.get(
        "STAGE5_PHASE_G_ALPHA_RUN_ID",
        "stage5_phase_g_alpha_guided_width_20260717",
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifact_dir = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    if drive_artifact_dir.exists():
        shutil.copytree(drive_artifact_dir, run_dir, dirs_exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = int(os.environ.get("STAGE5_PHASE_G_ALPHA_STEPS", "1000"))
    kl_coefficients = [
        float(value)
        for value in os.environ.get(
            "STAGE5_PHASE_G_ALPHA_KL_SWEEP",
            "0.0001,0.001,0.01",
        ).split(",")
    ]
    summary: dict[str, Any] = {
        "kind": "stage5_phase_g_alpha_session",
        "run_id": run_id,
        "status": "started",
        "keeper_sha256": KEEPER_SHA256,
        "kl_coefficients": kl_coefficients,
        "steps_per_arm": steps,
    }
    write_json(run_dir / "summary.json", summary)

    keeper = restore_keeper(run_dir)
    summary["data_manifests"] = prepare_data(run_dir)
    calibration_deterministic = deterministic_screen(
        run_dir=run_dir,
        keeper=keeper,
        split_name="calibration",
    )
    margin = lock_margin(calibration_deterministic, run_dir / "margin_lock" / "summary.json")
    summary["margin_lock"] = margin
    summary["status"] = "margin_locked"
    write_json(run_dir / "summary.json", summary)
    publish(run_dir, f"Lock Phase G-alpha margin {run_id} [skip ci]")
    sync_receipts_to_drive(run_dir, drive_artifact_dir)

    preflight = evaluate(
        data=run_dir / "data/test.jsonl",
        deterministic_rows=DETERMINISTIC_TEST_ROWS,
        keeper=keeper,
        output_dir=run_dir / "preflight_k1",
        guidance_checkpoint=None,
        sample_counts="1",
        coverage_margin=margin["locked_absolute_mean_coverage_margin"],
        include_temperature=False,
        include_iso=False,
        include_teacher=False,
        resume_cache_path=drive_artifact_dir / "preflight_k1" / "row_cache.jsonl",
    )
    summary["k1_preflight"] = preflight["k1_parity_gate"]
    if not preflight["k1_parity_gate"]["passed"]:
        summary["status"] = "blocked_k1_parity"
        write_json(run_dir / "summary.json", summary)
        publish(run_dir, f"Block Phase G-alpha K1 parity {run_id} [skip ci]")
        sync_receipts_to_drive(run_dir, drive_artifact_dir)
        return 2
    summary["status"] = "k1_parity_passed"
    write_json(run_dir / "summary.json", summary)
    publish(run_dir, f"Pass Phase G-alpha K1 parity {run_id} [skip ci]")
    sync_receipts_to_drive(run_dir, drive_artifact_dir)

    arms: list[dict[str, Any]] = []
    drive_root = Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / run_id
    drive_root.mkdir(parents=True, exist_ok=True)
    for arm_index, coefficient in enumerate(kl_coefficients):
        label = f"kl_{coefficient:g}".replace(".", "p")
        train_dir = run_dir / "train" / label
        expected_raw = train_dir / f"phase_g_raw_step_{steps}.pt"
        expected_ema = train_dir / f"phase_g_ema_step_{steps}.pt"
        drive_raw = drive_root / f"{label}_raw.pt"
        drive_ema = drive_root / f"{label}_ema.pt"
        train_summary_path = train_dir / "summary.json"
        local_complete = (
            train_summary_path.exists()
            and expected_raw.exists()
            and expected_ema.exists()
        )
        drive_complete = (
            train_summary_path.exists()
            and drive_raw.exists()
            and drive_ema.exists()
        )
        if local_complete:
            print(f"resume_completed_local_phase_g_training={label}", flush=True)
        elif drive_complete:
            train_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_raw, expected_raw)
            shutil.copy2(drive_ema, expected_ema)
            print(f"resume_completed_drive_phase_g_training={label}", flush=True)
        else:
            run(
                [
                    sys.executable,
                    "training/train_phase_g_alpha.py",
                    "--train_jsonl",
                    str((run_dir / "data/train.jsonl").relative_to(ROOT)),
                    "--keeper",
                    str(keeper),
                    "--expected_keeper_sha256",
                    KEEPER_SHA256,
                    "--output_dir",
                    str(train_dir.relative_to(ROOT)),
                    "--steps",
                    str(steps),
                    "--kl_coefficient",
                    str(coefficient),
                    "--seed",
                    str(20260717 + arm_index * 1000),
                    "--device",
                    "cuda",
                    "--dtype",
                    "bfloat16",
                ]
            )
        train_summary = read_json(train_summary_path)
        checkpoint_paths = {
            "raw": resolve_repo_path(train_summary["raw_checkpoint"]),
            "ema": resolve_repo_path(train_summary["ema_checkpoint"]),
        }
        for kind, checkpoint in checkpoint_paths.items():
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Completed Phase G-alpha arm is missing {kind} checkpoint: {checkpoint}"
                )
            destination = drive_root / f"{label}_{kind}.pt"
            shutil.copy2(checkpoint, destination)
        arm = {
            "label": label,
            "kl_coefficient": coefficient,
            "train_summary": str((train_dir / "summary.json").relative_to(ROOT)),
            "checkpoints": {
                kind: repo_relative_text(path) for kind, path in checkpoint_paths.items()
            },
            "drive_checkpoints": {
                kind: str(drive_root / f"{label}_{kind}.pt")
                for kind in checkpoint_paths
            },
            "calibration": {},
        }
        for kind, checkpoint in checkpoint_paths.items():
            arm["calibration"][kind] = evaluate(
                data=run_dir / "data/calibration.jsonl",
                deterministic_rows=calibration_deterministic,
                keeper=keeper,
                output_dir=run_dir / "calibration" / label / kind,
                guidance_checkpoint=checkpoint,
                sample_counts="1,2,4,8,20",
                coverage_margin=margin["locked_absolute_mean_coverage_margin"],
                include_temperature=True,
                include_iso=False,
                include_teacher=True,
                resume_cache_path=(
                    drive_artifact_dir / "calibration" / label / kind / "row_cache.jsonl"
                ),
            )
        arms.append(arm)
        summary["arms"] = arms
        summary["status"] = f"calibrated_{label}"
        write_json(run_dir / "summary.json", summary)
        publish(run_dir, f"Record Phase G-alpha arm {label} {run_id} [skip ci]")
        sync_receipts_to_drive(run_dir, drive_artifact_dir)

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for arm in arms:
        for weight_kind, result in arm["calibration"].items():
            if result["k1_parity_gate"]["passed"]:
                score = result["summaries"]["prior"]["20"]["overall"]["mean_coverage"]
                candidates.append((float(score), weight_kind, arm))
    if not candidates:
        summary["status"] = "blocked_all_calibration_k1_parity"
        write_json(run_dir / "summary.json", summary)
        publish(run_dir, f"Block Phase G-alpha calibration {run_id} [skip ci]")
        sync_receipts_to_drive(run_dir, drive_artifact_dir)
        return 2
    _, selected_kind, selected_arm = max(candidates, key=lambda item: item[0])
    selected_checkpoint = ROOT / selected_arm["checkpoints"][selected_kind]
    summary["selected_arm"] = {
        "label": selected_arm["label"],
        "weight_kind": selected_kind,
        "checkpoint": str(selected_checkpoint.relative_to(ROOT)),
        "selection_rule": "highest calibration K20 prior mean coverage among K1-parity arms",
    }
    summary["status"] = "test_evaluation_started"
    write_json(run_dir / "summary.json", summary)
    publish(run_dir, f"Select Phase G-alpha arm {run_id} [skip ci]")
    sync_receipts_to_drive(run_dir, drive_artifact_dir)

    test_result = evaluate(
        data=run_dir / "data/test.jsonl",
        deterministic_rows=DETERMINISTIC_TEST_ROWS,
        keeper=keeper,
        output_dir=run_dir / "test" / selected_arm["label"] / selected_kind,
        guidance_checkpoint=selected_checkpoint,
        sample_counts="1,2,4,8,20",
        coverage_margin=margin["locked_absolute_mean_coverage_margin"],
        include_temperature=True,
        include_iso=True,
        include_teacher=True,
        resume_cache_path=(
            drive_artifact_dir
            / "test"
            / selected_arm["label"]
            / selected_kind
            / "row_cache.jsonl"
        ),
    )
    summary["test"] = test_result
    summary["status"] = "finished"
    summary["verdict"] = test_result["verdict"]
    write_json(run_dir / "summary.json", summary)
    publish(run_dir, f"Finish Phase G-alpha {run_id} [skip ci]")
    sync_receipts_to_drive(run_dir, drive_artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
