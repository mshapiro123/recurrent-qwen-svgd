"""Execute the gated TM-0 CPU pipeline over a verified downloaded cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from training.paper2_tm0 import atomic_json, sha256_file


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest_a", type=Path, required=True)
    parser.add_argument("--manifest_b", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--teacher_7b_scores", type=Path, required=True)
    parser.add_argument("--teacher_14b_scores", type=Path, required=True)
    parser.add_argument("--w1_seed0", type=Path, required=True)
    parser.add_argument("--w1_seed1", type=Path, required=True)
    parser.add_argument("--w2p_summary", type=Path, required=True)
    parser.add_argument("--failed_loop_archive_receipt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "tm0_cpu_pipeline_status.json"
    common = {
        "kind": "paper2_tm0_cpu_pipeline_status_v1",
        "panel_sha256": sha256_file(args.panel),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "injection_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(status_path, {**common, "status": "RUNNING_CKA"})
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "analysis.analyze_paper2_tm0_tm1_cka",
            "--cache_root",
            str(args.cache_root),
            "--manifest_a",
            str(args.manifest_a),
            "--manifest_b",
            str(args.manifest_b),
            "--output_dir",
            str(args.output_dir),
        ]
    )
    cka_path = args.output_dir / "tm1_cka_calibration.json"
    cka = json.loads(cka_path.read_text(encoding="utf-8"))
    if cka["status"] != "PASS_STABLE_LAYER_SELECTION":
        atomic_json(
            status_path,
            {
                **common,
                "status": "STOPPED_UNSTABLE_LAYER_SELECTION",
                "cka_sha256": sha256_file(cka_path),
            },
        )
        return 2
    atomic_json(status_path, {**common, "status": "RUNNING_STITCH"})
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "analysis.analyze_paper2_tm0_tm1_stitch",
            "--cache_root",
            str(args.cache_root),
            "--panel",
            str(args.panel),
            "--cka_summary",
            str(cka_path),
            "--output_dir",
            str(args.output_dir),
        ]
    )
    stitch_path = args.output_dir / "tm1_stitch_summary.json"
    stitch = json.loads(stitch_path.read_text(encoding="utf-8"))
    r1_path = args.output_dir / "tm0_r1_receipt.json"
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "analysis.analyze_paper2_tm0_r1",
            "--phase_d_summary",
            str(args.w2p_summary),
            "--output",
            str(r1_path),
        ]
    )
    atomic_json(status_path, {**common, "status": "RUNNING_TM2G_J"})
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "analysis.analyze_paper2_tm0_tm2g_jet",
            "--cache_root",
            str(args.cache_root),
            "--panel",
            str(args.panel),
            "--cka_summary",
            str(cka_path),
            "--base_scores",
            str(args.base_scores),
            "--teacher_7b_scores",
            str(args.teacher_7b_scores),
            "--teacher_14b_scores",
            str(args.teacher_14b_scores),
            "--failed_loop_archive_receipt",
            str(args.failed_loop_archive_receipt),
            "--output_dir",
            str(args.output_dir),
        ]
    )
    jet_path = args.output_dir / "tm2g_jet_summary.json"
    jet = json.loads(jet_path.read_text(encoding="utf-8"))
    if stitch["gate_key"] == "STITCH-DEAD":
        atomic_json(
            status_path,
            {
                **common,
                "status": "COMPLETE_STITCH_DEAD_JET_COMPLETE",
                "cka_sha256": sha256_file(cka_path),
                "stitch_sha256": sha256_file(stitch_path),
                "r1_sha256": sha256_file(r1_path),
                "tm2g_jet_sha256": sha256_file(jet_path),
                "decision_keys": {
                    "tm1": "STITCH-DEAD",
                    "tm2g_jet": jet["decision_key"],
                },
            },
        )
        return 0
    atomic_json(status_path, {**common, "status": "RUNNING_TM2_TM2G"})
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "analysis.analyze_paper2_tm0_tm2",
            "--cache_root",
            str(args.cache_root),
            "--panel",
            str(args.panel),
            "--stitch_prefit",
            str(args.output_dir / "tm1_stitch_prefit.json"),
            "--stitch_states",
            str(args.output_dir / "tm1_stitch_states.pt"),
            "--base_scores",
            str(args.base_scores),
            "--teacher_7b_scores",
            str(args.teacher_7b_scores),
            "--teacher_14b_scores",
            str(args.teacher_14b_scores),
            "--w1_seed0",
            str(args.w1_seed0),
            "--w1_seed1",
            str(args.w1_seed1),
            "--output_dir",
            str(args.output_dir),
            "--tm2g_mode",
            "disabled",
        ]
    )
    tm2_path = args.output_dir / "tm2_tm2g_summary.json"
    tm2 = json.loads(tm2_path.read_text(encoding="utf-8"))
    atomic_json(
        status_path,
        {
            **common,
            "status": "COMPLETE",
            "cka_sha256": sha256_file(cka_path),
            "stitch_sha256": sha256_file(stitch_path),
            "r1_sha256": sha256_file(r1_path),
            "tm2_sha256": sha256_file(tm2_path),
            "tm2g_jet_sha256": sha256_file(jet_path),
            "decision_keys": {
                "tm1": stitch["gate_key"],
                "tm2": tm2["decision_keys"]["tm2"],
                "tm2g_jet": jet["decision_key"],
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
