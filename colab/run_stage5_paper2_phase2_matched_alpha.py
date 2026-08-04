"""Stage, run, decide, and publish the locked Phase-2 matched-alpha pilots."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

from training.run_paper2_phase2_matched_alpha import sha256_file, sha256_lf_file


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_matched_alpha_20260804"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
ARBITRATION_ID = "stage5_paper2_phase2_arbitration_build_20260804"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_STAGE0A = DRIVE_ROOT / STAGE0A_ID / "private/stage0a"
DRIVE_RUN = DRIVE_ROOT / RUN_ID
DRIVE_CANONICALIZER = (
    DRIVE_ROOT
    / ARBITRATION_ID
    / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
)
PROTOCOL = ROOT / "training/paper2_phase2_matched_alpha_preregistration.json"
CONSTANTS = ROOT / "training/paper2_phase2_dc2_constants.json"


def write_status(status: str, **details: object) -> None:
    path = DRIVE_RUN / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_matched_alpha_status",
                "status": status,
                "updated_at_unix": time.time(),
                "gpu_name": os.environ.get("STAGE5_MATCHED_ALPHA_GPU_NAME"),
                "gpu_vram_gib": os.environ.get("STAGE5_MATCHED_ALPHA_GPU_VRAM_GIB"),
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"matched_alpha_status status={status} details={details}", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    tail: deque[str] = deque(maxlen=400)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode:
        print("matched_alpha_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("matched_alpha_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def validate_locked_inputs() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    constants_lf = sha256_lf_file(CONSTANTS)
    canonicalizer_sha = sha256_file(DRIVE_CANONICALIZER)
    observed = {
        "constants_lf_sha256": constants_lf,
        "canonicalizer_sha256": canonicalizer_sha,
    }
    print(
        f"matched_alpha_locked_input_preflight observed={json.dumps(observed, sort_keys=True)}",
        flush=True,
    )
    if constants_lf != protocol["constants_lf_sha256"]:
        raise RuntimeError("V1d constants LF hash does not match the preregistration")
    if canonicalizer_sha != protocol["canonicalizer"]["sha256"]:
        raise RuntimeError("canonicalizer hash does not match the preregistration")


def stage_inputs() -> tuple[Path, Path, Path]:
    scratch_root = Path("/content/local-scratch")
    if not scratch_root.is_dir():
        scratch_root = Path("/content")
    local = scratch_root / "recurrent-qwen-svgd-stage" / RUN_ID
    stage0a = local / "stage0a"
    canonicalizer = local / DRIVE_CANONICALIZER.name
    cache = local / "matched_alpha_cache.pt"
    stage0a.mkdir(parents=True, exist_ok=True)
    for relative in (
        "sample_manifest.jsonl",
        "lattice",
        "model_cache/student_0p5b",
        "model_cache/teacher_14b",
    ):
        source = DRIVE_STAGE0A / relative
        destination = stage0a / relative
        print(f"matched_alpha_stage source={source} destination={destination}", flush=True)
        if source.is_dir():
            destination.parent.mkdir(parents=True, exist_ok=True)
            run(["rsync", "-a", "--info=progress2", f"{source}/", f"{destination}/"])
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    shutil.copy2(DRIVE_CANONICALIZER, canonicalizer)
    return stage0a, canonicalizer, cache


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 matched alpha pilots [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def training_command(
    *, stage0a: Path, canonicalizer: Path, cache: Path, extra_alpha: float | None = None
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "training.run_paper2_phase2_matched_alpha",
        "--stage0a_summary",
        str(STAGE0A_SUMMARY),
        "--stage0a_private",
        str(stage0a),
        "--canonicalizer",
        str(canonicalizer),
        "--cache",
        str(cache),
        "--output_dir",
        str(RUN_DIR),
        "--private_dir",
        str(DRIVE_RUN / "private/pilots"),
        "--device",
        "cuda",
    ]
    if extra_alpha is not None:
        command.extend(["--extra_alpha", str(extra_alpha)])
    return command


def decide() -> dict[str, object]:
    decision_path = RUN_DIR / "decision.json"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_matched_alpha_decision",
            "--summary",
            str(RUN_DIR / "summary.json"),
            "--output",
            str(decision_path),
        ]
    )
    return json.loads(decision_path.read_text(encoding="utf-8"))


def main() -> int:
    required = [STAGE0A_SUMMARY, DRIVE_STAGE0A / "sample_manifest.jsonl", DRIVE_CANONICALIZER]
    for path in required:
        print(f"matched_alpha_preflight path={path} exists={path.exists()}", flush=True)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing matched-alpha inputs: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    validate_locked_inputs()
    write_status("staging_inputs")
    stage0a, canonicalizer, cache = stage_inputs()
    write_status("training_initial_grid")
    run(training_command(stage0a=stage0a, canonicalizer=canonicalizer, cache=cache))
    decision = decide()
    if decision["status"] == "refinement_required":
        extra = float(decision["refinement_required"])
        write_status("training_registered_refinement", alpha=extra)
        run(
            training_command(
                stage0a=stage0a, canonicalizer=canonicalizer, cache=cache, extra_alpha=extra
            )
        )
        decision = decide()
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.glob("*.json"):
        shutil.copy2(path, receipt_dir / path.name)
    write_status(
        "publishing",
        decision_status=decision["status"],
        selected_alpha=decision.get("selected_alpha"),
    )
    commit = publish()
    write_status(
        "complete",
        publish_commit=commit,
        decision_status=decision["status"],
        selected_alpha=decision.get("selected_alpha"),
    )
    print(json.dumps({"decision": decision, "publish_commit": commit}, indent=2, sort_keys=True))
    return 0 if decision["status"] == "selected_dev_configuration" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0, 2):
            try:
                write_status(
                    "failed",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                    traceback=traceback.format_exc(),
                )
            except Exception as status_error:
                print(f"status_write_failed={status_error!r}", flush=True)
        raise
