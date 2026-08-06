"""Run and publish the CPU-only A2 helped/harmed localization receipt."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase2_a2_localization_20260806"
A2_ID = "stage5_paper2_phase2_a2_step237_continuation_20260806"
STAGE0A_ID = "stage5_paper2_phase2_stage0a_20260803"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
PRIVATE_A2 = DRIVE_ROOT / A2_ID / "private/a2"
STAGE0A_MANIFEST = DRIVE_ROOT / STAGE0A_ID / "private/stage0a/sample_manifest.jsonl"
DRIVE_RECEIPTS = DRIVE_ROOT / RUN_ID / "receipts"
A2_SUMMARY = ROOT / "outputs/stage5" / A2_ID / "summary.json"
STAGE0A_SUMMARY = ROOT / "outputs/stage5" / STAGE0A_ID / "summary.json"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def markdown(summary: dict) -> str:
    lines = [
        "# Phase-2 A2 Helped/Harmed Localization Receipt",
        "",
        "CPU-only post-processing of banked DEV row tensors. No model inference or training.",
        "",
        "## Population units",
        "",
        "- Stage 0A anchors: 50,000",
        "- Stage 0A horizon samples: 200,000",
        "- A2 training anchors: 41,969",
        "- A2 evaluation anchors: 8,031",
        "",
        "The existing Stage 0A lattice does not contain approximately 190,000 anchors. "
        "Reaching that anchor count requires a new teacher/cache pass.",
        "",
        "## Cross-seed consistency",
        "",
        f"- Helped in both seeds: {summary['seed_consistency']['helped_both']:,}",
        f"- Harmed in both seeds: {summary['seed_consistency']['harmed_both']:,}",
        f"- Opposite sign across seeds: {summary['seed_consistency']['sign_flip']:,}",
        f"- Quality loss in both seeds: {summary['seed_consistency']['quality_loss_both']:,}",
        "",
        "## Structural mask decision",
        "",
    ]
    selected = summary["recommended_single_mask"]
    if selected is None:
        lines.append("No structural group cleared the pre-stated two-seed mask rule.")
    else:
        lines.append(f"Candidate for strategy preregistration: `{selected['label']}`.")
        for row in selected["per_seed"]:
            lines.append(
                "- Seed {seed}: EAL change {change:.6f}, 95% document-bootstrap CI "
                "[{lower:.6f}, {upper:.6f}], retained-correct change {retained:+d}.".format(
                    seed=row["seed"],
                    change=row["mean_eal_change_vs_full"],
                    lower=row["document_bootstrap_95_ci"][0],
                    upper=row["document_bootstrap_95_ci"][1],
                    retained=row["retained_correct_change"],
                )
            )
    lines.extend(
        [
            "",
            "This is a post-hoc DEV localization result. Any mask must be locked before the "
            "Option B curve and cannot be described as confirmatory evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    required = [PRIVATE_A2, STAGE0A_MANIFEST, A2_SUMMARY, STAGE0A_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing localization inputs: {missing}")
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase2_a2_localization",
            "--a2_summary",
            str(A2_SUMMARY),
            "--stage0a_summary",
            str(STAGE0A_SUMMARY),
            "--stage0a_manifest",
            str(STAGE0A_MANIFEST),
            "--private_a2",
            str(PRIVATE_A2),
            "--output_dir",
            str(RUN_DIR),
        ]
    )
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    (RUN_DIR / "receipt.md").write_text(markdown(summary), encoding="utf-8")
    DRIVE_RECEIPTS.mkdir(parents=True, exist_ok=True)
    for path in RUN_DIR.iterdir():
        if path.is_file():
            shutil.copy2(path, DRIVE_RECEIPTS / path.name)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", RUN_DIR.relative_to(ROOT).as_posix()])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record Phase-2 A2 localization [skip ci]"])
        run(["git", "push", "origin", "main"])
    print(json.dumps({
        "status": summary["status"],
        "population_units": summary["population_units"],
        "recommended_single_mask": summary["recommended_single_mask"],
    }, indent=2, sort_keys=True))
    print("Phase-2 A2 localization landed.", flush=True)


if __name__ == "__main__":
    main()
