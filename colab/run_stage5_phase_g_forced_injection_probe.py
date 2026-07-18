"""Run the one authorized eval-only Phase G forced-injection causal probe."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab import run_stage5_phase_g_alpha as alpha  # noqa: E402
from colab.run_stage5_phase_g_multitarget_control import publish_receipts  # noqa: E402
from training.phase_g_forced_injection_spec import LOCKED_INJECTION_FACTORS  # noqa: E402


SOURCE_RUN_ID = "stage5_phase_g_multitarget_control_20260718"
SOURCE_DIR = ROOT / "outputs" / "stage5" / SOURCE_RUN_ID
SOURCE_SUMMARY = SOURCE_DIR / "summary.json"
DRIVE_CHECKPOINT_DIR = (
    Path("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints") / SOURCE_RUN_ID
)
ARMS = (
    {
        "label": "kl_0p001",
        "kl_coefficient": 0.001,
        "drive_checkpoint": DRIVE_CHECKPOINT_DIR / "kl_0p001_ema.pt",
    },
    {
        "label": "kl_0p0001_confirmation",
        "kl_coefficient": 0.0001,
        "drive_checkpoint": DRIVE_CHECKPOINT_DIR / "kl_0p0001_confirmation_ema.pt",
    },
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, allow_blocked: bool = False) -> int:
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
    return_code = process.wait()
    if return_code and not (allow_blocked and return_code == 2):
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def validate_source() -> dict[str, Any]:
    if not SOURCE_SUMMARY.exists():
        raise FileNotFoundError(f"Missing closed A0 source summary: {SOURCE_SUMMARY}")
    source = read_json(SOURCE_SUMMARY)
    if source.get("status") != "blocked_posterior_control_after_confirmation":
        raise AssertionError("Forced-injection probe requires the closed two-arm A0 result")
    source_arms = {str(arm["label"]): arm for arm in source.get("arms", [])}
    if set(source_arms) != {arm["label"] for arm in ARMS}:
        raise AssertionError("Closed A0 source does not contain exactly both preserved KL arms")
    if any(source_arms[arm["label"]]["gate"]["status"] != "blocked" for arm in ARMS):
        raise AssertionError("Forced-injection probe requires both A0 arms to be blocked")
    control = source["data"]["control"]
    if int(control["rows"]) != 106:
        raise AssertionError("Forced-injection probe requires the frozen 106-row control set")
    if int(control["validation"]["base_problem_groups"]) != 32:
        raise AssertionError("Forced-injection probe requires the frozen 32 groups")
    return source


def restore_guidance_checkpoint(
    *,
    run_dir: Path,
    label: str,
    coefficient: float,
    drive_checkpoint: Path,
) -> tuple[Path, str]:
    if not drive_checkpoint.exists():
        raise FileNotFoundError(
            "Missing preserved A0 EMA checkpoint; this eval-only probe must not retrain it: "
            f"{drive_checkpoint}"
        )
    local = run_dir / "restored" / f"{label}_ema.pt"
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        shutil.copy2(drive_checkpoint, local)
    digest = sha256_file(local)
    if digest != sha256_file(drive_checkpoint):
        raise AssertionError(f"Restored {label} checkpoint differs from its Drive source")
    checkpoint = torch.load(local, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 1000:
        raise AssertionError(f"{label} is not the preserved step-1000 A0 checkpoint")
    if str(checkpoint.get("phase")) != "phase_g_ema":
        raise AssertionError(f"{label} is not an EMA Phase G checkpoint")
    config = dict(checkpoint.get("config") or {})
    if float(config.get("kl_coefficient", -1.0)) != coefficient:
        raise AssertionError(f"{label} KL coefficient does not match its locked arm")
    if config.get("keeper_sha256") != alpha.KEEPER_SHA256:
        raise AssertionError(f"{label} was not trained from the locked keeper")
    state = dict(checkpoint.get("trainable_state_dict") or {})
    if not state or any(not str(name).startswith("phase_g_") for name in state):
        raise AssertionError(f"{label} checkpoint contains non-Phase-G trainable tensors")
    print(
        f"restored_guidance_checkpoint label={label} sha256={digest} "
        f"kl={coefficient:g} step=1000",
        flush=True,
    )
    return local, digest


def main() -> int:
    run_id = os.environ.get(
        "STAGE5_PHASE_G_FORCED_INJECTION_RUN_ID",
        "stage5_phase_g_forced_injection_probe_20260718",
    )
    factors = tuple(
        float(value)
        for value in os.environ.get(
            "STAGE5_PHASE_G_FORCED_INJECTION_FACTORS",
            "1,3,10,30,100",
        ).split(",")
    )
    if factors != LOCKED_INJECTION_FACTORS:
        raise AssertionError("Forced-injection factors are preregistered and cannot be changed")
    run_dir = ROOT / "outputs" / "stage5" / run_id
    drive_artifacts = (
        Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / run_id
    )
    if drive_artifacts.exists():
        shutil.copytree(drive_artifacts, run_dir, dirs_exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    alpha.configure_runtime_transcript(run_dir / "runtime.log")
    source = validate_source()
    keeper = alpha.restore_keeper(run_dir)
    summary: dict[str, Any] = {
        "kind": "stage5_phase_g_forced_injection_probe",
        "status": "started",
        "run_id": run_id,
        "source_summary": str(SOURCE_SUMMARY.relative_to(ROOT)),
        "source_status": source["status"],
        "keeper_sha256": alpha.KEEPER_SHA256,
        "factors": list(factors),
        "training_performed": False,
        "coverage_performed": False,
        "locked_readings": {
            "CHANNEL-EXISTS": "switching >=16/32 at any factor with K1 validity >0.50",
            "NO-CHANNEL": (
                "switching <8/32 at every factor, or >=16 only with validity <0.50"
            ),
            "AMBIGUOUS": "all intermediate outcomes; closed by default",
        },
        "arms": [],
    }
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="startup",
        status="started",
        run_id=run_id,
    )

    arm_summaries: list[Path] = []
    for arm in ARMS:
        label = str(arm["label"])
        checkpoint, checkpoint_sha = restore_guidance_checkpoint(
            run_dir=run_dir,
            label=label,
            coefficient=float(arm["kl_coefficient"]),
            drive_checkpoint=Path(arm["drive_checkpoint"]),
        )
        source_eval = SOURCE_DIR / "posterior_control_eval" / label
        baseline = source_eval / "posterior_teacher_K1.jsonl"
        rng_manifest = source_eval / "rng_manifest.jsonl"
        if not baseline.exists() or not rng_manifest.exists():
            raise FileNotFoundError(f"Missing published A0 factor-1 receipts for {label}")
        arm_output = run_dir / "arms" / label
        resume_cache = drive_artifacts / "resume_cache" / label
        run(
            [
                sys.executable,
                "eval/eval_phase_g_forced_injection.py",
                "--data_jsonl",
                str((SOURCE_DIR / "data" / "posterior_control.jsonl").relative_to(ROOT)),
                "--keeper",
                str(keeper),
                "--expected_keeper_sha256",
                alpha.KEEPER_SHA256,
                "--guidance_checkpoint",
                str(checkpoint),
                "--expected_guidance_sha256",
                checkpoint_sha,
                "--baseline_posterior_jsonl",
                str(baseline.relative_to(ROOT)),
                "--rng_manifest_jsonl",
                str(rng_manifest.relative_to(ROOT)),
                "--output_dir",
                str(arm_output.relative_to(ROOT)),
                "--resume_cache_dir",
                str(resume_cache),
                "--arm_label",
                label,
                "--factors",
                ",".join(f"{factor:g}" for factor in factors),
                "--device",
                "cuda",
                "--dtype",
                os.environ.get("STAGE5_PHASE_G_FORCED_INJECTION_DTYPE", "bfloat16"),
            ]
        )
        arm_summary = arm_output / "summary.json"
        arm_summaries.append(arm_summary)
        summary["arms"].append(
            {
                "label": label,
                "kl_coefficient": arm["kl_coefficient"],
                "guidance_checkpoint_sha256": checkpoint_sha,
                "summary": str(arm_summary.relative_to(ROOT)),
            }
        )
        alpha.write_json(run_dir / "summary.json", summary)
        alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
        publish_receipts(
            run_dir,
            f"Record Phase G forced-injection arm {label} {run_id} [skip ci]",
        )

    gate_path = run_dir / "gate.json"
    gate_md = run_dir / "gate.md"
    gate_result = run(
        [
            sys.executable,
            "eval/score_phase_g_forced_injection.py",
            "--arm_summaries",
            *[str(path.relative_to(ROOT)) for path in arm_summaries],
            "--output_json",
            str(gate_path.relative_to(ROOT)),
            "--output_md",
            str(gate_md.relative_to(ROOT)),
            "--run_summary",
            str((run_dir / "summary.json").relative_to(ROOT)),
            "--runtime_status_json",
            str((run_dir / "runtime_status.json").relative_to(ROOT)),
        ],
        allow_blocked=True,
    )
    gate = read_json(gate_path)
    summary["gate"] = gate
    summary["status"] = gate["status"]
    summary["training_performed"] = False
    summary["coverage_performed"] = False
    alpha.write_json(run_dir / "summary.json", summary)
    alpha.write_runtime_status(
        run_dir,
        drive_artifacts,
        stage="forced_injection_gate",
        status=summary["status"],
        measured_verdict=gate["measured_verdict"],
        authorization=gate["authorization"],
    )
    alpha.sync_receipts_to_drive(run_dir, drive_artifacts)
    publish_receipts(run_dir, f"Record Phase G forced-injection probe {run_id} [skip ci]")
    return gate_result


def guarded_main() -> int:
    run_id = os.environ.get(
        "STAGE5_PHASE_G_FORCED_INJECTION_RUN_ID",
        "stage5_phase_g_forced_injection_probe_20260718",
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
