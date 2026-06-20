"""Run Stage 1 heldout SVGD replication inside one Colab runtime.

This script assumes the repo has already been cloned, dependencies installed,
and the authenticated GitHub remote configured by the bootstrap cell.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = "outputs/qwen_0_5b_phase1_recreated_beta008_150/phase1_step_150.pt"
PHASE2 = "outputs/qwen_0_5b_phase2_svgd_recreated_smoke25/phase2_step_25.pt"
PROJ = "outputs/calibration/recreated_within_group_pca_projection.pt"
SEEDS = "5,6,7,8,9"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def require_artifacts() -> None:
    required = [
        PHASE1,
        PHASE2,
        PROJ,
        "outputs/heldout_task_splits/fold0_heldout.jsonl",
        "outputs/heldout_task_splits/fold1_heldout.jsonl",
    ]
    for path in required:
        assert (ROOT / path).exists(), f"missing required artifact: {path}"


def read_rows(path: str) -> list[dict]:
    rows = []
    for line in (ROOT / path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def metrics(paths: list[str]) -> dict[str, int]:
    rows = []
    for path in paths:
        rows.extend(read_rows(path))
    grouped: dict[tuple[object, object], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row.get("seed"), row.get("task")), []).append(row)
    return {
        "best_hits": sum(any(item.get("hit") for item in items) for items in grouped.values()),
        "total_tasks": len(grouped),
        "candidate_hits": sum(1 for row in rows if row.get("hit")),
        "total_candidates": len(rows),
    }


def main() -> int:
    require_artifacts()

    common = [
        sys.executable,
        "eval/eval_best_of_k_jsonl.py",
        "--skip_phase1",
        "--compact",
        "--seeds",
        SEEDS,
        "--phase1_checkpoint",
        PHASE1,
        "--phase2_checkpoint",
        PHASE2,
        "--phase2_num_trajectories",
        "4",
        "--phase2_particle_update_mode",
        "svgd",
        "--particle_init_noise",
        "0.05",
        "--particle_noise_every_step",
        "--particle_noise_steps",
        "16",
        "--svgd_repulsion_max_norm",
        "none",
        "--temperature",
        "0.0",
        "--max_new_tokens",
        "140",
        "--dtype",
        "bfloat16",
        "--adapter_dtype",
        "float32",
        "--device",
        "cuda",
    ]

    jobs = [
        (
            "extended_fold0_random32_rep05_seeds5_9",
            "outputs/heldout_task_splits/fold0_heldout.jsonl",
            "outputs/diagnostics/extended_fold0_random32_rep05_seeds5_9.jsonl",
            [
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_dim",
                "32",
                "--svgd_projection_seed",
                "123",
                "--svgd_repulsion_scale",
                "0.5",
            ],
        ),
        (
            "extended_fold0_wg_dim8_rep2_seeds5_9",
            "outputs/heldout_task_splits/fold0_heldout.jsonl",
            "outputs/diagnostics/extended_fold0_wg_dim8_rep2_seeds5_9.jsonl",
            [
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                PROJ,
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "2",
            ],
        ),
        (
            "extended_fold1_random32_rep05_seeds5_9",
            "outputs/heldout_task_splits/fold1_heldout.jsonl",
            "outputs/diagnostics/extended_fold1_random32_rep05_seeds5_9.jsonl",
            [
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_dim",
                "32",
                "--svgd_projection_seed",
                "123",
                "--svgd_repulsion_scale",
                "0.5",
            ],
        ),
        (
            "extended_fold1_wg_dim8_rep2_seeds5_9",
            "outputs/heldout_task_splits/fold1_heldout.jsonl",
            "outputs/diagnostics/extended_fold1_wg_dim8_rep2_seeds5_9.jsonl",
            [
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                PROJ,
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "2",
            ],
        ),
    ]

    for label, tasks, out, extra in jobs:
        print("\n\n====", label, "====")
        cmd = common + ["--tasks_jsonl", tasks, "--output_jsonl", out] + extra
        log_path = ROOT / "outputs" / "diagnostics" / f"{label}.log"
        proc = run(cmd, check=False)
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"{label} failed; see {log_path}")

    summary_sets = {
        "fold0_seeds0_4_random32": ["outputs/diagnostics/recreated_fold0_random32_rep05.jsonl"],
        "fold0_seeds0_4_wg_dim8": ["outputs/diagnostics/recreated_fold0_wg_dim8_rep2.jsonl"],
        "fold1_seeds0_4_random32": ["outputs/diagnostics/recreated_fold1_random32_rep05.jsonl"],
        "fold1_seeds0_4_wg_dim8": ["outputs/diagnostics/recreated_fold1_wg_dim8_rep2.jsonl"],
        "fold0_seeds5_9_random32": ["outputs/diagnostics/extended_fold0_random32_rep05_seeds5_9.jsonl"],
        "fold0_seeds5_9_wg_dim8": ["outputs/diagnostics/extended_fold0_wg_dim8_rep2_seeds5_9.jsonl"],
        "fold1_seeds5_9_random32": ["outputs/diagnostics/extended_fold1_random32_rep05_seeds5_9.jsonl"],
        "fold1_seeds5_9_wg_dim8": ["outputs/diagnostics/extended_fold1_wg_dim8_rep2_seeds5_9.jsonl"],
        "heldout_seeds0_9_random32": [
            "outputs/diagnostics/recreated_fold0_random32_rep05.jsonl",
            "outputs/diagnostics/recreated_fold1_random32_rep05.jsonl",
            "outputs/diagnostics/extended_fold0_random32_rep05_seeds5_9.jsonl",
            "outputs/diagnostics/extended_fold1_random32_rep05_seeds5_9.jsonl",
        ],
        "heldout_seeds0_9_wg_dim8": [
            "outputs/diagnostics/recreated_fold0_wg_dim8_rep2.jsonl",
            "outputs/diagnostics/recreated_fold1_wg_dim8_rep2.jsonl",
            "outputs/diagnostics/extended_fold0_wg_dim8_rep2_seeds5_9.jsonl",
            "outputs/diagnostics/extended_fold1_wg_dim8_rep2_seeds5_9.jsonl",
        ],
    }

    lines = []
    for name, paths in summary_sets.items():
        line = f"{name}: {metrics(paths)}"
        print(line)
        lines.append(line)

    rand = metrics(summary_sets["heldout_seeds0_9_random32"])
    wg = metrics(summary_sets["heldout_seeds0_9_wg_dim8"])
    delta = {
        "best_hits": wg["best_hits"] - rand["best_hits"],
        "candidate_hits": wg["candidate_hits"] - rand["candidate_hits"],
    }
    lines.append(f"heldout_delta_wg_minus_random32: {delta}")
    print(lines[-1])

    summary_path = ROOT / "outputs" / "diagnostics" / "extended_heldout_seeds0_9_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", summary_path)

    run(["git", "status", "-sb"])
    run(["git", "add", "-f", "outputs/diagnostics/extended_*", "outputs/diagnostics/recreated_*", "outputs/heldout_task_splits/*.jsonl"])
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No staged changes to commit.")
    else:
        run(["git", "commit", "-m", "Extend heldout SVGD diagnostics seeds 5-9"])
        run(["git", "push", "origin", "main"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
