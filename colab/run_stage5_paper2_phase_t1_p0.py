"""Build, run, and publish the preregistered Paper Two T1 P0 pilot."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.internal_think_token_t1 import (
    augment_control_row,
    build_pilot_mixture_rows,
    pilot_grid,
    select_pilot_cell,
)
from training.internal_think_token_t1_spec import phase_t1_draft
from training.synthetic_depth_task import (
    SyntheticDepthConfig,
    write_synthetic_depth_dataset,
)


RUN_ID = os.environ.get(
    "STAGE5_PAPER2_T1_P0_RUN_ID",
    "stage5_paper2_internal_token_t1_p0_letter_v2_20260724",
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
DRIVE_ROOT = Path(
    os.environ.get(
        "STAGE5_PAPER2_T1_P0_DRIVE_ROOT",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/{RUN_ID}",
    )
)
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = os.environ.get("STAGE5_PAPER2_T1_P0_DTYPE", "bfloat16")
DATA_SCHEMA_VERSION = "p0_phase_a_letter_symbols_v2"
TRAIN_DATASET_SEED = 2026072301
PILOT_DATASET_SEED = 2026072399


def path_for_cli(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    result = subprocess.CompletedProcess(command, process.wait(), "".join(lines), None)
    if result.returncode:
        print("P0_FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(result.stdout.splitlines()[-240:]), flush=True)
        print("P0_FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def publish(message: str) -> None:
    subprocess.run(
        ["git", "pull", "--rebase", "--autostash", "origin", "main"],
        cwd=ROOT,
        check=False,
    )
    for path in sorted(RUN_DIR.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".md", ".log"}:
            continue
        subprocess.run(
            ["git", "add", "-f", path.relative_to(ROOT).as_posix()],
            cwd=ROOT,
            check=True,
        )
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    if subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode:
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


def prepare_data() -> dict[str, Any]:
    data_dir = RUN_DIR / "data"
    source_dir = data_dir / "source_depth1_8"
    pilot_dir = data_dir / "pilot_eval_depth1_8"
    if not (source_dir / "summary.json").exists():
        write_synthetic_depth_dataset(
            output_dir=source_dir,
            config=SyntheticDepthConfig(
                n_symbols=16,
                max_depth=8,
                rows_per_depth=256,
                seed=TRAIN_DATASET_SEED,
                num_choices=4,
                max_target_loops=8,
                value_prefix="letter:",
            ),
        )
    if not (pilot_dir / "summary.json").exists():
        write_synthetic_depth_dataset(
            output_dir=pilot_dir,
            config=SyntheticDepthConfig(
                n_symbols=16,
                max_depth=8,
                rows_per_depth=32,
                seed=PILOT_DATASET_SEED,
                num_choices=4,
                max_target_loops=8,
                value_prefix="letter:",
            ),
        )

    mixture_path = data_dir / "p0_train_mixture.jsonl"
    mixture_manifest_path = data_dir / "p0_train_mixture_manifest.json"
    if not mixture_path.exists():
        source_rows = read_jsonl(source_dir / "train_chain_symbol_sft.jsonl")
        mixture, manifest = build_pilot_mixture_rows(source_rows, seed=9999)
        write_jsonl(mixture_path, mixture)
        write_json(mixture_manifest_path, manifest)
    mixture_manifest = read_json(mixture_manifest_path)
    if mixture_manifest["control_rows"] != 1400 or mixture_manifest["rehearsal_rows"] != 600:
        raise AssertionError(f"P0 mixture drifted: {mixture_manifest}")

    pilot_path = data_dir / "p0_pilot_eval_256.jsonl"
    if not pilot_path.exists():
        pilot_source = read_jsonl(pilot_dir / "test_chain_symbol_sft.jsonl")
        pilot_rows = [augment_control_row(row) for row in pilot_source]
        if len(pilot_rows) != 256:
            raise AssertionError(f"Expected 256 dedicated pilot rows, got {len(pilot_rows)}")
        write_jsonl(pilot_path, pilot_rows)
    return {
        "train_jsonl": path_for_cli(mixture_path),
        "train_sha256": sha256_file(mixture_path),
        "train_manifest": mixture_manifest,
        "pilot_jsonl": path_for_cli(pilot_path),
        "pilot_sha256": sha256_file(pilot_path),
        "pilot_rows": 256,
        "pilot_rows_per_depth": 32,
        "train_dataset_seed": TRAIN_DATASET_SEED,
        "pilot_dataset_seed": PILOT_DATASET_SEED,
        "pilot_disjoint_from_registered_sets": True,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "value_prefix": "letter:",
        "symbol_surface": "A_through_P_matching_phase_a",
    }


def requested_cells() -> list[Any]:
    raw = os.environ.get("STAGE5_PAPER2_T1_P0_CELLS", "").strip()
    cells = pilot_grid()
    if not raw:
        return cells
    wanted = {value.strip() for value in raw.split(",") if value.strip()}
    unknown = wanted - {cell.cell_id for cell in cells}
    if unknown:
        raise ValueError(f"Unknown P0 cell IDs: {sorted(unknown)}")
    return [cell for cell in cells if cell.cell_id in wanted]


def backup_cell(cell_id: str) -> dict[str, Any]:
    cell_dir = RUN_DIR / "cells" / cell_id
    drive_dir = DRIVE_ROOT / "cells" / cell_id
    drive_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for checkpoint in sorted(cell_dir.glob("p0_compact_step_*.pt")):
        destination = drive_dir / checkpoint.name
        shutil.copy2(checkpoint, destination)
        checkpoints.append(
            {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )
    summary = cell_dir / "summary.json"
    if summary.exists():
        shutil.copy2(summary, drive_dir / summary.name)
    return {"drive_dir": str(drive_dir), "checkpoints": checkpoints}


def restore_finished_cell_from_drive(cell_id: str) -> bool:
    """Recover a completed cell that crashed before its Git receipt was published."""

    cell_dir = RUN_DIR / "cells" / cell_id
    local_summary = cell_dir / "summary.json"
    if local_summary.exists() and read_json(local_summary).get("status") == "finished":
        return False

    drive_dir = DRIVE_ROOT / "cells" / cell_id
    drive_summary = drive_dir / "summary.json"
    if not drive_summary.exists() or read_json(drive_summary).get("status") != "finished":
        return False

    cell_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(drive_dir.iterdir()):
        if source.is_file():
            shutil.copy2(source, cell_dir / source.name)
    print(f"restored_finished_p0_cell_from_drive={cell_id}", flush=True)
    return True


def write_summary(data: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    all_cell_ids = {cell.cell_id for cell in pilot_grid()}
    complete_ids = {str(row["cell"]["cell_id"]) for row in results}
    complete = complete_ids == all_cell_ids
    selection = select_pilot_cell(results) if complete else {
        "status": "incomplete_grid_no_selection",
        "selected_cell_id": None,
        "missing_cells": sorted(all_cell_ids - complete_ids),
    }
    payload = {
        "kind": "stage5_paper2_internal_token_t1_p0",
        "run_id": RUN_ID,
        "status": "finished" if complete else "partial_cells_finished",
        "registered_t1_training": False,
        "citable": False,
        "phase_t1_remains_unlocked": True,
        "draft_spec": phase_t1_draft(),
        "data": data,
        "cells": results,
        "selection": selection,
        "next_action": (
            "fill_lambda_ratio_and_lock_preregistration"
            if selection.get("status") == "selected"
            else "reassess_openly_before_lock_no_silent_extension"
            if complete
            else "run_missing_p0_cells"
        ),
    }
    write_json(RUN_DIR / "summary.json", payload)
    lines = [
        f"# Paper Two T1 P0 Pilot - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        "- Registered T1 training: `False`",
        "- Citable: `False`",
        f"- Selection: `{selection.get('status')}`",
        f"- Selected cell: `{selection.get('selected_cell_id')}`",
        "",
        "| Cell | Lambda | Stop/continue ratio | Stop recall | Continue recall | Answer accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["step_1500"]
        cell = result["cell"]
        lines.append(
            f"| {cell['cell_id']} | {cell['control_loss_lambda']} | "
            f"{cell['stop_to_continue_ratio']} | {metrics['stop_recall']:.3f} | "
            f"{metrics['continue_recall']:.3f} | {metrics['answer_accuracy']:.3f} |"
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    return payload


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    write_json(RUN_DIR / "draft_preregistration.json", phase_t1_draft())
    write_json(RUN_DIR / "data_manifest.json", data)

    for cell in requested_cells():
        cell_dir = RUN_DIR / "cells" / cell.cell_id
        cell_summary = cell_dir / "summary.json"
        restore_finished_cell_from_drive(cell.cell_id)
        if not cell_summary.exists() or read_json(cell_summary).get("status") != "finished":
            cell_config = RUN_DIR / "configs" / f"{cell.cell_id}.json"
            write_json(cell_config, cell.to_dict())
            result = run(
                [
                    sys.executable,
                    "training/run_internal_think_token_p0_cell.py",
                    "--cell_json",
                    path_for_cli(cell_config),
                    "--train_jsonl",
                    data["train_jsonl"],
                    "--pilot_jsonl",
                    data["pilot_jsonl"],
                    "--output_dir",
                    path_for_cli(cell_dir),
                    "--dtype",
                    DTYPE,
                    "--device",
                    DEVICE,
                    "--eval_batch_size",
                    os.environ.get("STAGE5_PAPER2_T1_P0_EVAL_BATCH_SIZE", "4"),
                ]
            )
            (cell_dir / "train.log").write_text(result.stdout, encoding="utf-8")
        receipt = read_json(cell_summary)
        receipt["drive_backup"] = backup_cell(cell.cell_id)
        write_json(cell_summary, receipt)
        completed_results = [
            read_json(path)
            for path in sorted((RUN_DIR / "cells").glob("*/summary.json"))
            if read_json(path).get("status") == "finished"
        ]
        write_summary(data, completed_results)
        publish(f"Record Paper Two T1 P0 cell {cell.cell_id} {RUN_ID} [skip ci]")

    results = [
        read_json(path)
        for path in sorted((RUN_DIR / "cells").glob("*/summary.json"))
        if read_json(path).get("status") == "finished"
    ]
    summary = write_summary(data, results)
    publish(f"Finish Paper Two T1 P0 pilot {RUN_ID} [skip ci]")
    if summary["status"] != "finished":
        return 2
    if summary["selection"]["status"] != "selected":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
