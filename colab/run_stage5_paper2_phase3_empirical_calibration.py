"""Run the read-only Phase 3 empirical DEV sequential-floor calibration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_empirical_calibration_20260810"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_RUN = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5") / RUN_ID
RECEIPT_DIR = DRIVE_RUN / "receipts"
OPTION_B_PRIVATE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_phase2_option_b_20260807/private/option_b"
)
SEED_DIRS = [
    OPTION_B_PRIVATE / "seed_0_full_a2",
    OPTION_B_PRIVATE / "seed_1_full_a2",
]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_status(status: str, **details: object) -> None:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        RECEIPT_DIR / "status.json",
        {
            "kind": "paper2_phase3_empirical_calibration_status_v1",
            "status": status,
            "updated_at_unix": time.time(),
            "p33_training_authorized": False,
            "optimizer_steps": 0,
            **details,
        },
    )
    print(f"phase3_empirical_calibration_status status={status} details={details}", flush=True)


def main() -> int:
    missing = [str(path) for path in SEED_DIRS if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing Option B DEV trajectory directories: {missing}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_status("calibrating_from_saved_dev_rows")
    output = RUN_DIR / "summary.json"
    command = [
        sys.executable,
        "-u",
        "-m",
        "eval.eval_paper2_phase3_empirical_calibration",
    ]
    for directory in SEED_DIRS:
        command.extend(["--seed_row_dir", str(directory)])
    command.extend(["--output_summary", str(output)])
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["optimizer_steps"] != 0 or result["p33_training_authorized"]:
        raise RuntimeError("empirical calibration crossed the no-training boundary")
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, RECEIPT_DIR / output.name)
    write_status("complete", summary=str(RECEIPT_DIR / output.name))
    print(json.dumps({"status": result["status"], "drive": str(DRIVE_RUN)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise
