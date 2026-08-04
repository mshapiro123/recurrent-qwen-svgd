"""Stage, run, and publish the read-only matched-alpha terminal audit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT_ID = "stage5_paper2_phase2_matched_alpha_20260804"
AUDIT_ID = "stage5_paper2_phase2_matched_alpha_audit_20260804"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
ARBITRATION_ID = "stage5_paper2_phase2_arbitration_build_20260804"
RUN_DIR = ROOT / "outputs/stage5" / AUDIT_ID
PILOT_SUMMARY = ROOT / "outputs/stage5" / PILOT_ID / "summary.json"
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"
PROTOCOL = ROOT / "training/paper2_phase2_matched_alpha_preregistration.json"
CONSTANTS = ROOT / "training/paper2_phase2_dc2_constants.json"
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_STAGE0A = DRIVE_ROOT / STAGE0A_ID / "private/stage0a"
DRIVE_PILOT = DRIVE_ROOT / PILOT_ID
DRIVE_AUDIT = DRIVE_ROOT / AUDIT_ID
DRIVE_CANONICALIZER = (
    DRIVE_ROOT
    / ARBITRATION_ID
    / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> None:
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
    if returncode not in allowed:
        print("matched_alpha_audit_child_failure_tail_begin", flush=True)
        print("\n".join(tail), flush=True)
        print("matched_alpha_audit_child_failure_tail_end", flush=True)
        raise subprocess.CalledProcessError(returncode, command)


def write_status(status: str, **details: object) -> None:
    path = DRIVE_AUDIT / "receipts/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_matched_alpha_audit_status",
                "status": status,
                "updated_at_unix": time.time(),
                **details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(f"matched_alpha_audit_status status={status} details={details}", flush=True)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        run(["rsync", "-a", "--info=progress2", f"{source}/", f"{destination}/"])
    else:
        shutil.copy2(source, destination)


def _stage_head(summary_source: Path, stage0a: Path) -> None:
    relative_summary = summary_source.relative_to(DRIVE_STAGE0A)
    destination_summary = stage0a / relative_summary
    _copy(summary_source, destination_summary)
    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    marker = "/private/stage0a/"
    normalized = str(summary["lm_head"]["path"]).replace("\\", "/")
    if marker not in normalized:
        raise RuntimeError(f"unrecognized Stage 0A LM-head path: {normalized}")
    relative_head = Path(normalized.split(marker, 1)[1])
    _copy(DRIVE_STAGE0A / relative_head, stage0a / relative_head)


def stage_inputs() -> tuple[Path, Path, Path]:
    scratch = Path("/content/local-scratch")
    if not scratch.is_dir():
        scratch = Path("/content")
    local = scratch / "recurrent-qwen-svgd-stage" / AUDIT_ID
    stage0a = local / "stage0a"
    canonicalizer = local / DRIVE_CANONICALIZER.name
    cache = local / "matched_alpha_cache.pt"
    legacy_cache = (
        scratch
        / "recurrent-qwen-svgd-stage"
        / PILOT_ID
        / "matched_alpha_cache.pt"
    )
    stage0a.mkdir(parents=True, exist_ok=True)
    _copy(DRIVE_CANONICALIZER, canonicalizer)
    if not cache.is_file() and legacy_cache.is_file():
        print(f"matched_alpha_audit_reuse_legacy_cache={legacy_cache}", flush=True)
        try:
            os.link(legacy_cache, cache)
        except OSError:
            shutil.copy2(legacy_cache, cache)
    if cache.is_file():
        print(f"matched_alpha_audit_cache_ready={cache}", flush=True)
        _stage_head(DRIVE_STAGE0A / "model_cache/student_0p5b/summary.json", stage0a)
        _stage_head(DRIVE_STAGE0A / "model_cache/teacher_14b/summary.json", stage0a)
    else:
        print("matched_alpha_audit_cache_missing=rebuild_from_immutable_stage0a", flush=True)
        for relative in (
            "sample_manifest.jsonl",
            "lattice",
            "model_cache/student_0p5b",
            "model_cache/teacher_14b",
        ):
            _copy(DRIVE_STAGE0A / relative, stage0a / relative)
    return stage0a, canonicalizer, cache


def write_markdown(summary: dict[str, object]) -> None:
    arms = summary["arms"]
    lines = [
        "# Phase-2 Matched-Alpha Read-Only Audit Receipt",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This receipt evaluates the six terminal pilot checkpoints without constructing an optimizer or updating model parameters.",
        "",
        "| Arm | Abort step | Exact retention | Exact acceptance delta | Demand above permission | Huber linear regime |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        exact = arm["exact_abort_evaluation"]
        geometry = arm["demanded_permitted_and_huber"]
        lines.append(
            "| {arm} | {step} | {retention:.6f} | {delta:.6f} | {demand:.2%} | {huber:.2%} |".format(
                arm=arm["arm"],
                step=arm["abort_step"],
                retention=exact["retention"],
                delta=exact["acceptance_delta"],
                demand=geometry["fraction_demand_exceeds_permission"],
                huber=geometry["huber_linear_regime_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- DEV-only; no frozen E1 evaluation partition was touched.",
            "- This audit does not select alpha or authorize E1.",
            "- Per-training-step trust magnitudes were not stored. Scheduled evaluation rent is reported only as a proxy.",
            "- The old 0.997 retention rule is reported as endpoint qualification, not relabeled as a catastrophe tripwire.",
            "",
        ]
    )
    (RUN_DIR / "receipt.md").write_text("\n".join(lines), encoding="utf-8")


def publish() -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 matched alpha read-only audit [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    required = [
        PILOT_SUMMARY,
        STAGE0A_SUMMARY,
        PROTOCOL,
        CONSTANTS,
        DRIVE_CANONICALIZER,
        DRIVE_STAGE0A / "model_cache/student_0p5b/summary.json",
        DRIVE_STAGE0A / "model_cache/teacher_14b/summary.json",
    ]
    pilot = json.loads(PILOT_SUMMARY.read_text(encoding="utf-8"))
    required.extend(Path(arm["checkpoint"]["path"]) for arm in pilot["arms"])
    for path in required:
        print(f"matched_alpha_audit_preflight path={path} exists={path.exists()}", flush=True)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing read-only audit inputs: {missing}")
    constants = json.loads(CONSTANTS.read_text(encoding="utf-8"))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_status("staging_inputs", model_optimizer_constructed=False, model_parameter_updates=0)
    stage0a, canonicalizer, cache = stage_inputs()
    write_status("auditing_exact_abort_checkpoints")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_matched_alpha_audit",
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--stage0a_private",
            str(stage0a),
            "--canonicalizer",
            str(canonicalizer),
            "--cache",
            str(cache),
            "--pilot_summary",
            str(PILOT_SUMMARY),
            "--protocol",
            str(PROTOCOL),
            "--output_dir",
            str(RUN_DIR),
            "--private_dir",
            str(DRIVE_AUDIT / "private/exact_abort_rows"),
            "--rms_cap",
            str(constants["p99_state_rms_cap"]),
            "--device",
            "cuda",
            "--batch_size",
            os.environ.get("STAGE5_MATCHED_ALPHA_AUDIT_BATCH_SIZE", "32"),
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    write_markdown(summary)
    receipts = DRIVE_AUDIT / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.glob("*"):
        if path.is_file():
            shutil.copy2(path, receipts / path.name)
    write_status("publishing")
    commit = publish()
    write_status("complete", publish_commit=commit)
    print(json.dumps({"status": summary["status"], "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
            try:
                write_status(
                    "failed",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                    traceback=traceback.format_exc(),
                )
            except Exception as status_error:
                print(f"matched_alpha_audit_status_write_failed={status_error!r}", flush=True)
        raise
