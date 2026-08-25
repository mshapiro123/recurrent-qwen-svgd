"""Run one durable TM-0 state-cache phase on Colab.

Each model is intentionally isolated so a completed cache can be downloaded
before the next billable phase starts.  The scientific cache implementation is
unchanged; this file only provides resumable execution and packaging.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path


ROOT = Path("/content/tm0_repo")
PANEL = Path("/content/tm0_panel.jsonl")
PROBE = Path("/content/tm0_cost_probe.jsonl")
CONFIG = Path("/content/tm0_phase.json")
OUTPUT = Path("/content/tm0_phase_output")
STATUS = Path("/content/tm0_phase_status.json")
EXPECTED_PANEL_SHA = "e108b0a92fdc69b9cb27274ac420908b65303213307f9d8dfc1f4ba73d58b5ca"
MODEL_KEYS = {"student", "teacher_7b", "teacher_14b"}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(status: str, **details: object) -> None:
    payload = {
        "kind": "paper2_tm0_cache_phase_status_v1",
        "status": status,
        "updated_at_unix": time.time(),
        **details,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)
    print(f"tm0_phase_status status={status} details={details}", flush=True)


def main() -> int:
    if sha256_file(PANEL) != EXPECTED_PANEL_SHA:
        raise RuntimeError("TM-0 frozen panel SHA mismatch")
    config = json.loads(CONFIG.read_text())
    model_key = str(config.get("model_key"))
    if model_key not in MODEL_KEYS:
        raise RuntimeError(f"Invalid TM-0 model phase: {model_key}")

    target = OUTPUT / model_key
    write_status("running", model_key=model_key, panel_sha256=EXPECTED_PANEL_SHA)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.cache_paper2_tm0",
            "--panel",
            str(PANEL),
            "--probe_manifest",
            str(PROBE),
            "--output_dir",
            str(target),
            "--model_cache",
            "/content/model-cache",
            "--model_key",
            model_key,
            "--shard_rows",
            "64",
        ],
        cwd=ROOT,
        check=True,
    )
    index = target / f"{model_key}_cache_index.json"
    receipt = json.loads(index.read_text())
    if receipt.get("rows") != 6144 or receipt.get("dry_run"):
        raise RuntimeError(f"TM-0 cache receipt is incomplete: {model_key}")

    bundle = Path(f"/content/tm0_{model_key}_bundle.tar")
    with tarfile.open(bundle, "w") as archive:
        archive.add(target, arcname=model_key)
    write_status(
        "complete",
        model_key=model_key,
        index_sha256=sha256_file(index),
        bundle_path=str(bundle),
        bundle_sha256=sha256_file(bundle),
        bundle_bytes=bundle.stat().st_size,
        optimizer_steps=0,
        confirm_scored=False,
        eval_e_scored=False,
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        write_status(
            "failed",
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    raise SystemExit(exit_code)
