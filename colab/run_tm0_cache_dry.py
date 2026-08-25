"""Run the registered TM-0 state-cache cost probe on a live Colab GPU."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path("/content/tm0_repo")
OUTPUT = Path("/content/tm0_dry")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/content/tm0_minimal.tar.gz", "r:gz") as archive:
        archive.extractall(ROOT, filter="data")
    run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "accelerate>=1.0"])
    for model_key in ("teacher_7b", "teacher_14b"):
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "eval.cache_paper2_tm0",
                "--panel",
                "/content/tm0_panel.jsonl",
                "--probe_manifest",
                "/content/tm0_cost_probe.jsonl",
                "--output_dir",
                str(OUTPUT / model_key),
                "--model_cache",
                "/content/model-cache",
                "--model_key",
                model_key,
                "--shard_rows",
                "16",
                "--dry_run",
            ],
            cwd=ROOT,
        )
    receipts = [
        json.loads((OUTPUT / key / f"{key}_cache_index.json").read_text())
        for key in ("teacher_7b", "teacher_14b")
    ]
    result = {
        "kind": "paper2_tm0_cache_dry_run_summary_v1",
        "gpu_forward_hours_projection": sum(
            receipt["projected_full_forward_seconds"] for receipt in receipts
        )
        / 3600.0,
        "cap_gpu_hours": 1.5,
        "passes": receipts,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "injection_performed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    (OUTPUT / "dry_run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
