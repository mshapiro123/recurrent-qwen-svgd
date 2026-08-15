"""Stage, preflight, run, and score the ratified P3.5 landing arms."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from colab.run_stage5_paper2_phase3_p34_a2 import (
    DRIVE_STAGE5,
    I1_ID,
    MIGRATED_SHA,
    MIGRATION_ID,
    NEW_ID,
    OLD_ID,
    P33_ID,
    P33_SHA,
    resolve_preflight,
    rsync,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p35_20260815"
P34_ID = "stage5_paper2_phase3_p34_a2_20260814"
PREREQUISITE_ID = "stage5_paper2_phase3_p35_prerequisites_20260815"
LOCK_PATH = ROOT / "training/paper2_phase3_p35_preregistration.draft.json"
I1_SHA = {
    0: "01c804bc69d35a01730fff236cf5a8d974899d2e4de7e15b92a227b2a9ce5d88",
    1: "2ed3296f510a6c3a66c451051ecbe2284de03b35dde4052827174a66a10c1d4a",
}
REGISTERED_ARMS = (("stabilized", 0), ("stabilized", 1), ("probe_reader", 0))


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> None:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)


def scratch_root() -> Path:
    for root in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")):
        if root.exists() and shutil.disk_usage(root).free >= 80 * 1024**3:
            return root / "recurrent-qwen-svgd-stage" / RUN_ID
    raise RuntimeError("P3.5 requires at least 80 GiB local scratch")


def arm_label(arm: str, seed: int) -> str:
    return f"arm_s_seed_{seed}" if arm == "stabilized" else "arm_r_seed_0"


def stage_common(scratch: Path) -> dict[str, Path]:
    old = scratch / "old"
    new = scratch / "new"
    preflight = resolve_preflight(scratch)
    rsync(
        DRIVE_STAGE5 / OLD_ID / "private/stage0a/sample_manifest.jsonl",
        old / "sample_manifest.jsonl",
    )
    rsync(DRIVE_STAGE5 / OLD_ID / "private/stage0a/lattice", old / "lattice")
    rsync(
        DRIVE_STAGE5 / OLD_ID / "private/stage0a/model_cache/student_0p5b",
        old / "model_cache/student_0p5b",
    )
    rsync(
        DRIVE_STAGE5 / NEW_ID / "private/full/sample_manifest.jsonl",
        new / "sample_manifest.jsonl",
    )
    rsync(DRIVE_STAGE5 / NEW_ID / "private/full/lattice", new / "lattice")
    rsync(
        DRIVE_STAGE5 / NEW_ID / "private/full/model_cache/student_0p5b",
        new / "model_cache/student_0p5b",
    )
    direction_cache = scratch / "agreement_oracle_directions_v2.pt"
    rsync(
        DRIVE_STAGE5
        / PREREQUISITE_ID
        / "private/serving_oracle/agreement_oracle_directions_v2.pt",
        direction_cache,
    )
    return {
        "old": old,
        "new": new,
        "preflight": preflight,
        "direction_cache": direction_cache,
    }


def stage_chain(scratch: Path, *, seed: int, expected_p34: str) -> dict[str, Path]:
    paths = {
        "migrated": scratch / f"seed_{seed}_migrated.pt",
        "p33": scratch / f"seed_{seed}_p33_step_1000.pt",
        "i1": scratch / f"seed_{seed}_i1.pt",
        "p34": scratch / f"seed_{seed}_p34_step_4000.pt",
    }
    rsync(
        DRIVE_STAGE5
        / MIGRATION_ID
        / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt",
        paths["migrated"],
    )
    rsync(
        DRIVE_STAGE5 / P33_ID / f"private/seed_{seed}/checkpoint_step_1000.pt",
        paths["p33"],
    )
    rsync(DRIVE_STAGE5 / I1_ID / f"private/seed_{seed}/resume.pt", paths["i1"])
    rsync(
        DRIVE_STAGE5 / P34_ID / f"private/main_seed_{seed}/checkpoint_step_4000.pt",
        paths["p34"],
    )
    expected = {
        "migrated": MIGRATED_SHA[seed],
        "p33": P33_SHA[seed],
        "i1": I1_SHA[seed],
        "p34": expected_p34,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != expected[name]:
            raise RuntimeError(
                f"P3.5 staged {name} SHA mismatch: expected={expected[name]} observed={observed}"
            )
    return paths


def runner_command(
    *,
    arm: str,
    seed: int,
    common: dict[str, Path],
    chain: dict[str, Path],
    output_dir: Path,
    private_dir: Path,
    preflight_only: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "training.run_paper2_phase3_p35",
        "--seed",
        str(seed),
        "--arm",
        arm,
        "--old_summary",
        str(ROOT / "outputs/stage5" / OLD_ID / "summary.json"),
        "--old_private",
        str(common["old"]),
        "--new_summary",
        str(DRIVE_STAGE5 / NEW_ID / "receipts/full_cache_summary.json"),
        "--new_private",
        str(common["new"]),
        "--staged_labels",
        str(common["preflight"] / "private/p33_prep/p33_staged_labels.jsonl"),
        "--positive_audit",
        str(common["preflight"] / "private/p33_prep/p33_audit_slice.jsonl"),
        "--negative_audit",
        str(common["preflight"] / "private/p33_prep/p33_negative_audit_slice.jsonl"),
        "--retention_panel",
        str(common["preflight"] / "private/p33_prep/p33_retention_panel.jsonl"),
        "--direction_cache",
        str(common["direction_cache"]),
        "--dev_panel",
        str(
            ROOT
            / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
        ),
        "--base_scores",
        str(
            ROOT
            / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
        ),
        "--migrated",
        str(chain["migrated"]),
        "--migrated_sha256",
        MIGRATED_SHA[seed],
        "--p33",
        str(chain["p33"]),
        "--p33_sha256",
        P33_SHA[seed],
        "--i1",
        str(chain["i1"]),
        "--i1_sha256",
        I1_SHA[seed],
        "--p34",
        str(chain["p34"]),
        "--lock",
        str(LOCK_PATH),
        "--output_dir",
        str(output_dir),
        "--private_dir",
        str(private_dir),
        "--device",
        "cuda",
    ]
    if preflight_only:
        command.append("--preflight_only")
    return command


def stage_primary_ema_reads(
    *, run_summary: dict[str, object], private_dir: Path, score_dir: Path, run_dir: Path
) -> None:
    score_dir.mkdir(parents=True, exist_ok=True)
    for entry in run_summary["history"]:
        step = int(entry["step"])
        source_rows = private_dir / f"ema_task_rows_step_{step}.jsonl"
        source_summary = run_dir / f"ema_task_summary_step_{step}.json"
        label = f"step_{step}_ema_ceiling_0.02"
        shutil.copy2(source_rows, score_dir / f"{label}.jsonl")
        shutil.copy2(source_summary, score_dir / f"{label}.json")


def run_score_bundle(
    *, seed: int, chain: dict[str, Path], run_dir: Path, private_dir: Path, score_dir: Path
) -> dict[str, object]:
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    stage_primary_ema_reads(
        run_summary=run_summary,
        private_dir=private_dir,
        score_dir=score_dir,
        run_dir=run_dir,
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_p35_score_bundle",
            "--run_summary",
            str(run_dir / "summary.json"),
            "--panel",
            str(
                ROOT
                / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
            ),
            "--base_scores",
            str(
                ROOT
                / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
            ),
            "--output_dir",
            str(score_dir),
            "--migrated",
            str(chain["migrated"]),
            "--migrated_sha256",
            MIGRATED_SHA[seed],
            "--p33",
            str(chain["p33"]),
            "--p33_sha256",
            P33_SHA[seed],
            "--i1",
            str(chain["i1"]),
            "--i1_sha256",
            I1_SHA[seed],
            "--p34",
            str(chain["p34"]),
            "--p34_sha256",
            str(run_summary["source"]["sha256"]),
        ]
    )
    return json.loads((score_dir / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument("--arm", choices=("stabilized", "probe_reader"))
    parser.add_argument("--seed", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.mode == "train" and (args.arm, args.seed) not in REGISTERED_ARMS:
        raise ValueError("P3.5 training requires one registered arm/seed pair")
    if args.mode == "preflight" and (args.arm is not None or args.seed is not None):
        raise ValueError("P3.5 preflight always checks all three registered arms")

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock_sha = sha256_file(LOCK_PATH)
    drive_run = DRIVE_STAGE5 / RUN_ID
    scratch = scratch_root()
    common = stage_common(scratch)

    def execute(arm: str, seed: int, *, preflight_only: bool) -> dict[str, object]:
        label = arm_label(arm, seed)
        phase = "preflight" if preflight_only else "train"
        run_dir = ROOT / "outputs/stage5" / RUN_ID / phase / label
        private_dir = drive_run / "private" / phase / label if preflight_only else drive_run / "private" / label
        receipts = drive_run / "receipts" / phase / label
        status_path = receipts / "status.json"

        def status(value: str, **details: object) -> None:
            write_json(
                status_path,
                {
                    "kind": "paper2_phase3_p35_colab_status_v1",
                    "arm": arm,
                    "seed": seed,
                    "phase": phase,
                    "status": value,
                    "updated_at_unix": time.time(),
                    **details,
                },
            )
            print(
                f"p35_status phase={phase} arm={arm} seed={seed} status={value} details={details}",
                flush=True,
            )

        try:
            status("staging")
            expected_p34 = lock["initialization"][f"seed_{seed}"]["sha256"]
            chain = stage_chain(scratch, seed=seed, expected_p34=expected_p34)
            if not preflight_only:
                gate_path = drive_run / "receipts/preflight/summary.json"
                if not gate_path.is_file():
                    raise RuntimeError("P3.5 all-arm preflight receipt is missing")
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                if gate.get("status") != "complete_all_preflights_passed":
                    raise RuntimeError("P3.5 all-arm preflight gate did not pass")
                if gate.get("lock_sha256") != lock_sha:
                    raise RuntimeError("P3.5 preflight and training lock SHAs differ")
            status("exact_preflight" if preflight_only else "training")
            run(
                runner_command(
                    arm=arm,
                    seed=seed,
                    common=common,
                    chain=chain,
                    output_dir=run_dir,
                    private_dir=private_dir,
                    preflight_only=preflight_only,
                ),
                allowed=(0, 2),
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            write_json(receipts / "summary.json", summary)
            result: dict[str, object] = {
                "arm": arm,
                "seed": seed,
                "run_status": summary["status"],
                "run_summary_path": str(receipts / "summary.json"),
                "run_summary_sha256": sha256_file(receipts / "summary.json"),
                "source_sha256": expected_p34,
            }
            if not preflight_only and summary["status"] == "complete":
                status("score_bundle")
                score_dir = drive_run / "private/score_bundle" / label
                score = run_score_bundle(
                    seed=seed,
                    chain=chain,
                    run_dir=run_dir,
                    private_dir=private_dir,
                    score_dir=score_dir,
                )
                result.update(
                    {
                        "score_bundle_path": str(score_dir / "summary.json"),
                        "score_bundle_sha256": sha256_file(score_dir / "summary.json"),
                        "registered_primary": score["registered_primary"],
                    }
                )
            write_json(receipts / "wave_summary.json", result)
            status("complete", **result)
            return result
        except Exception as error:
            status(
                "failed",
                exception_type=type(error).__name__,
                exception=str(error),
                traceback=traceback.format_exc(),
            )
            raise

    if args.mode == "preflight":
        results = [
            execute(arm, seed, preflight_only=True) for arm, seed in REGISTERED_ARMS
        ]
        if any(result["run_status"] != "complete_preflight_only" for result in results):
            raise RuntimeError("P3.5 one or more registered preflights failed")
        consolidated = {
            "kind": "paper2_phase3_p35_all_arm_preflight_v1",
            "status": "complete_all_preflights_passed",
            "lock_sha256": lock_sha,
            "arms": results,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        write_json(drive_run / "receipts/preflight/summary.json", consolidated)
        print(json.dumps(consolidated, indent=2, sort_keys=True))
        return 0

    result = execute(str(args.arm), int(args.seed), preflight_only=False)
    return 0 if result["run_status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
