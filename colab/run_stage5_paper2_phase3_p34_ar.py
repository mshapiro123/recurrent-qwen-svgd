"""Stage and run the read-only P3.4 A_r pricing audit."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p34_prerequisites_20260812"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
DRIVE_RUN = DRIVE_STAGE5 / RUN_ID
ORACLE_RUN = DRIVE_STAGE5 / "stage5_paper2_phase3_oracle_forecast_20260810"
I1_RUN = DRIVE_STAGE5 / "stage5_paper2_phase3_p33_i1_20260812"
RECEIPTS = DRIVE_RUN / "receipts"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    status = RECEIPTS / "ar_status.json"
    try:
        write_json(status, {"status": "running", "updated_at_unix": time.time()})
        direction = ORACLE_RUN / "private/oracle_cache/agreement_oracle_directions.pt"
        features = [
            ORACLE_RUN / f"private/oracle_cache/agreement_features_seed_{seed}_loop_4.pt"
            for seed in (0, 1)
        ]
        checkpoints = [I1_RUN / f"private/seed_{seed}/resume.pt" for seed in (0, 1)]
        missing = [str(path) for path in [direction, *features, *checkpoints] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing P3.4 A_r inputs: {missing}")
        output = RECEIPTS / "ar_pricing_audit.json"
        command = [
            sys.executable,
            "-u",
            "-m",
            "eval.eval_paper2_phase3_p34_ar",
            "--direction_cache",
            str(direction),
        ]
        for path in features:
            command.extend(["--feature_cache", str(path)])
        for path in checkpoints:
            command.extend(["--checkpoint", str(path)])
        command.extend(["--output", str(output)])
        print("$", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads(output.read_text(encoding="utf-8"))
        if result["optimizer_steps"] != 0 or result["training_authorized"]:
            raise RuntimeError("P3.4 A_r crossed its read-only boundary")
        write_json(
            status,
            {
                "status": "complete",
                "updated_at_unix": time.time(),
                "receipt": str(output),
                "strategy_confirmation_required": True,
            },
        )
        print(json.dumps({"status": "complete", "receipt": str(output)}, indent=2))
        return 0
    except Exception as error:
        write_json(
            status,
            {
                "status": "failed",
                "updated_at_unix": time.time(),
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
