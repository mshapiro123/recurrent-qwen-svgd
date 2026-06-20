"""Run a small ARC-Challenge MCQ likelihood smoke test inside one Colab runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = "data/arc_challenge_validation_smoke32.jsonl"
PHASE1 = "outputs/qwen_0_5b_phase1_recreated_beta008_150/phase1_step_150.pt"
PHASE2 = "outputs/qwen_0_5b_phase2_svgd_recreated_smoke25/phase2_step_25.pt"
PROJ = "outputs/calibration/recreated_within_group_pca_projection.pt"
OUTDIR = ROOT / "outputs" / "benchmarks"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_file(label: str, path: Path) -> list[str]:
    rows = read_rows(path)
    by_aggregate: dict[str, list[dict]] = {}
    for row in rows:
        by_aggregate.setdefault(str(row["aggregate"]), []).append(row)
    lines = [f"{label}: {path.relative_to(ROOT)}"]
    for aggregate_name, aggregate_rows in sorted(by_aggregate.items()):
        correct = sum(1 for row in aggregate_rows if row["hit"])
        total = len(aggregate_rows)
        lines.append(f"  aggregate={aggregate_name} correct={correct}/{total} accuracy={correct / max(total, 1):.4f}")
    return lines


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for path in [PHASE1, PHASE2, PROJ]:
        assert (ROOT / path).exists(), f"missing required artifact: {path}"

    run(
        [
            sys.executable,
            "eval/prepare_arc_mcq.py",
            "--config",
            "ARC-Challenge",
            "--split",
            "validation",
            "--limit",
            "32",
            "--seed",
            "0",
            "--output_jsonl",
            DATA,
        ]
    )
    assert (ROOT / DATA).exists(), f"missing prepared data: {DATA}"

    common = [
        sys.executable,
        "eval/eval_mcq.py",
        "--data_jsonl",
        DATA,
        "--prompt_style",
        "with_options",
        "--score_target",
        "label",
        "--dtype",
        "bfloat16",
        "--adapter_dtype",
        "float32",
        "--device",
        "cuda",
        "--seed",
        "0",
    ]

    jobs = [
        (
            "base/label",
            OUTDIR / "arc_challenge_smoke32_base_label.jsonl",
            common + ["--mode", "base", "--aggregate", "mean"],
        ),
        (
            "phase1/label",
            OUTDIR / "arc_challenge_smoke32_phase1_label.jsonl",
            common
            + [
                "--mode",
                "phase1",
                "--checkpoint",
                PHASE1,
                "--max_loops",
                "4",
                "--num_trajectories",
                "1",
                "--aggregate",
                "mean",
            ],
        ),
        (
            "phase2_svgd/label",
            OUTDIR / "arc_challenge_smoke32_phase2_svgd_label.jsonl",
            common
            + [
                "--mode",
                "phase2",
                "--checkpoint",
                PHASE2,
                "--max_loops",
                "4",
                "--num_trajectories",
                "4",
                "--aggregates",
                "mean,max,vote",
                "--particle_update_mode",
                "svgd",
                "--particle_init_noise",
                "0.05",
                "--svgd_kernel_geometry",
                "euclidean",
                "--svgd_kernel_projection_path",
                PROJ,
                "--svgd_kernel_projection_dim",
                "8",
                "--svgd_repulsion_scale",
                "2",
                "--svgd_repulsion_max_norm",
                "none",
            ],
        ),
    ]

    for label, output_path, cmd in jobs:
        print("\n\n====", label, "====")
        if output_path.exists():
            output_path.unlink()
        proc = run(cmd + ["--output_jsonl", str(output_path.relative_to(ROOT))], check=False)
        log_path = OUTDIR / f"{output_path.stem}.log"
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(f"{label} failed; see {log_path}")

    lines = []
    for label, output_path, _ in jobs:
        lines.extend(summarize_file(label, output_path))
    summary_path = OUTDIR / "arc_challenge_smoke32_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))

    run(["git", "status", "-sb"])
    run(["git", "add", "-f", "outputs/benchmarks/arc_challenge_smoke32*"])
    status = run(["git", "diff", "--cached", "--quiet"], check=False)
    if status.returncode == 0:
        print("No benchmark outputs changed.")
    else:
        run(["git", "commit", "-m", "Record ARC-Challenge smoke benchmark"])
        run(["git", "push", "origin", "main"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
