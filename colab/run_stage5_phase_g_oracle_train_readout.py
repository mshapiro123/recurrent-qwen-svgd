"""Run the read-only WP1 train-row diagnostic on existing oracle checkpoints."""

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

from colab import run_stage5_phase_g_alpha as alpha  # noqa: E402
from colab.run_stage5_phase_g_multitarget_control import publish_receipts  # noqa: E402
from training.oracle_train_readout_spec import (  # noqa: E402
    MATCHED_GROUPS,
    MATCHED_ROWS,
    MATCHED_TRANSITIONS,
    READOUT_SEED,
    combined_readout,
    preregistration_payload,
    row_id_sha256,
    select_matched_training_rows,
    validate_matched_training_rows,
)


SOURCE_RUN_ID = "stage5_phase_g_oracle_interface_probe_20260718"
SOURCE_DIR = ROOT / "outputs" / "stage5" / SOURCE_RUN_ID
SOURCE_SUMMARY = SOURCE_DIR / "summary.json"
TRAIN_JSONL = (
    ROOT
    / "outputs/stage5/stage5_phase_g_multitarget_control_20260718/data/train.jsonl"
)
CONDITIONER_DRIVE = (
    Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / SOURCE_RUN_ID
)
EXPECTED_CONDITIONER_SHA = {
    "additive": "4d5f2cb78f8bab14c6449b4cea8d971f59ad76a661a995ae7e62e883e125235c",
    "film": "d551f1136cf1582e9a6e95be43be952a5f4da1ed5649c45ed29ecd4115e051de",
}


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


def run(command: list[str]) -> None:
    printable = "$ " + " ".join(map(str, command))
    print(printable, flush=True)
    alpha.append_runtime_transcript(printable + "\n")
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
        alpha.append_runtime_transcript(line)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def restore_conditioner(route: str, run_dir: Path) -> Path:
    source = CONDITIONER_DRIVE / f"{route}_ema_step_1500.pt"
    destination = run_dir / "restored" / f"oracle_{route}_ema_step_1500.pt"
    if not destination.exists():
        if not source.exists():
            raise FileNotFoundError(f"Missing existing oracle checkpoint: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    observed = sha256_file(destination)
    if observed != EXPECTED_CONDITIONER_SHA[route]:
        raise AssertionError(
            f"{route} conditioner SHA mismatch: {observed} != "
            f"{EXPECTED_CONDITIONER_SHA[route]}"
        )
    return destination


def evaluate(
    *,
    route: str,
    cohort: str,
    data_jsonl: Path,
    keeper: Path,
    conditioner: Path,
    run_dir: Path,
    drive_artifacts: Path,
    expected_rows: int,
    expected_groups: int,
    expected_transitions: int,
    dtype: str,
) -> dict[str, Any]:
    output_dir = run_dir / "eval" / cohort / route
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        run(
            [
                sys.executable,
                "eval/eval_oracle_interface_probe.py",
                "--data_jsonl",
                str(data_jsonl.relative_to(ROOT)),
                "--keeper",
                str(keeper),
                "--expected_keeper_sha256",
                alpha.KEEPER_SHA256,
                "--conditioner_checkpoint",
                str(conditioner),
                "--expected_conditioner_sha256",
                EXPECTED_CONDITIONER_SHA[route],
                "--route",
                route,
                "--output_dir",
                str(output_dir.relative_to(ROOT)),
                "--resume_cache_path",
                str(drive_artifacts / "eval_cache" / cohort / route / "rows.jsonl"),
                "--expected_rows",
                str(expected_rows),
                "--expected_groups",
                str(expected_groups),
                "--expected_transitions",
                str(expected_transitions),
                "--evaluation_kind",
                "posthoc_train_readout",
                "--dtype",
                dtype,
                "--device",
                "cuda",
            ]
        )
    summary = read_json(summary_path)
    if summary.get("evaluation_kind") != "posthoc_train_readout":
        raise AssertionError("WP1 accidentally used the registered held-out gate mode")
    if summary.get("registered_heldout_verdict_mutable") is not False:
        raise AssertionError("WP1 must not mutate the registered held-out verdict")
    return summary


def write_receipt(run_dir: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Paper Two WP1 Oracle Train-Row Readout",
        "",
        "Post-hoc diagnostic only. The registered held-out verdict remains `BOTH_FAIL`.",
        "",
        "| Route | Matched non-default control | Matched reading | Full non-default control | Full reading | Held-out |",
        "|---|---:|---|---:|---|---:|",
    ]
    for route in ("additive", "film"):
        arm = result["arms"][route]
        lines.append(
            f"| {route} | {arm['matched_nondefault_control']:.4f} | "
            f"`{arm['matched_interpretation']}` | {arm['full_nondefault_control']:.4f} | "
            f"`{arm['full_interpretation']}` | {arm['heldout_nondefault_control']:.4f} |"
        )
    lines.extend(
        [
            "",
            "No parameter was trained or mutated. No successor is automatically authorized.",
            "",
        ]
    )
    (run_dir / "receipt.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    run_id = os.environ.get(
        "STAGE5_ORACLE_TRAIN_READOUT_RUN_ID",
        "stage5_phase_g_oracle_train_readout_20260722",
    )
    dtype = os.environ.get("STAGE5_ORACLE_TRAIN_READOUT_DTYPE", "bfloat16")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    if drive_artifacts.exists():
        shutil.copytree(drive_artifacts, run_dir, dirs_exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    alpha.configure_runtime_transcript(run_dir / "runtime.log")

    source = read_json(SOURCE_SUMMARY)
    if source.get("gate", {}).get("measured_reading") != "BOTH_FAIL":
        raise AssertionError("WP1 requires the canonical terminal BOTH_FAIL source")
    if source.get("keeper_sha256") != alpha.KEEPER_SHA256:
        raise AssertionError("WP1 source keeper SHA mismatch")

    preregistration = preregistration_payload()
    alpha.write_json(run_dir / "preregistration.json", preregistration)
    train_rows = read_jsonl(TRAIN_JSONL)
    if len(train_rows) != 1899:
        raise AssertionError("WP1 requires the committed 1,899 training variants")
    matched_rows = select_matched_training_rows(train_rows, seed=READOUT_SEED)
    manifest = validate_matched_training_rows(matched_rows)
    manifest["seed"] = READOUT_SEED
    manifest["row_id_sha256"] = row_id_sha256(matched_rows)
    matched_path = run_dir / "data" / "matched_train_106.jsonl"
    alpha.write_jsonl(matched_path, matched_rows)
    alpha.write_json(run_dir / "data" / "matched_manifest.json", manifest)

    summary: dict[str, Any] = {
        "kind": "phase_g_oracle_train_readout_session",
        "status": "started",
        "run_id": run_id,
        "posthoc_diagnostic_only": True,
        "registered_heldout_verdict": "BOTH_FAIL",
        "registered_heldout_verdict_changed": False,
        "automatic_successor_authorized": False,
        "matched_manifest": manifest,
    }
    alpha.write_json(run_dir / "summary.json", summary)
    publish_receipts(
        run_dir,
        f"Preregister Paper Two WP1 oracle train readout {run_id} [skip ci]",
    )

    keeper = alpha.restore_keeper(run_dir)
    conditioners = {
        route: restore_conditioner(route, run_dir) for route in ("additive", "film")
    }
    immutable_before = {
        "keeper": sha256_file(keeper),
        **{route: sha256_file(path) for route, path in conditioners.items()},
    }

    matched: dict[str, dict[str, Any]] = {}
    full: dict[str, dict[str, Any]] = {}
    for route in ("additive", "film"):
        matched[route] = evaluate(
            route=route,
            cohort="matched106",
            data_jsonl=matched_path,
            keeper=keeper,
            conditioner=conditioners[route],
            run_dir=run_dir,
            drive_artifacts=drive_artifacts,
            expected_rows=MATCHED_ROWS,
            expected_groups=MATCHED_GROUPS,
            expected_transitions=MATCHED_TRANSITIONS,
            dtype=dtype,
        )
        summary["matched"] = matched
        summary["status"] = f"matched_{route}_complete"
        alpha.write_json(run_dir / "summary.json", summary)
        alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
        publish_receipts(
            run_dir,
            f"Record Paper Two WP1 matched readout {route} {run_id} [skip ci]",
        )

    for route in ("additive", "film"):
        full[route] = evaluate(
            route=route,
            cohort="full1899",
            data_jsonl=TRAIN_JSONL,
            keeper=keeper,
            conditioner=conditioners[route],
            run_dir=run_dir,
            drive_artifacts=drive_artifacts,
            expected_rows=1899,
            expected_groups=512,
            expected_transitions=5617,
            dtype=dtype,
        )
        summary["full"] = full
        summary["status"] = f"full_{route}_complete"
        alpha.write_json(run_dir / "summary.json", summary)
        alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
        publish_receipts(
            run_dir,
            f"Record Paper Two WP1 full readout {route} {run_id} [skip ci]",
        )

    immutable_after = {
        "keeper": sha256_file(keeper),
        **{route: sha256_file(path) for route, path in conditioners.items()},
    }
    if immutable_after != immutable_before:
        raise AssertionError("WP1 read-only evaluation mutated a checkpoint")
    result = combined_readout(matched_summaries=matched, full_summaries=full)
    result["immutable_sha256_before"] = immutable_before
    result["immutable_sha256_after"] = immutable_after
    result["matched_manifest"] = manifest
    alpha.write_json(run_dir / "summary.json", result)
    write_receipt(run_dir, result)
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(
        run_dir,
        f"Finish Paper Two WP1 oracle train readout {run_id} [skip ci]",
    )
    return 0


def guarded_main() -> int:
    run_id = os.environ.get(
        "STAGE5_ORACLE_TRAIN_READOUT_RUN_ID",
        "stage5_phase_g_oracle_train_readout_20260722",
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    try:
        return main()
    except BaseException as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        alpha.record_runtime_failure(run_dir, drive_artifacts, exc)
        raise
    finally:
        alpha.configure_runtime_transcript(None)


if __name__ == "__main__":
    raise SystemExit(guarded_main())
