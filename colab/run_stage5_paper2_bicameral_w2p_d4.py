"""Run the W2-prime prompt-only D4 cache with immutable local inputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any


KIND = "paper2_bicameral_w2p_d4_runner_v1"
AUTHORITY_SHA256 = "f89b45ef100fa46536dd93a3ef936aa8c9cfa1fc624b401b4bfc0d2b50bc2aa4"
INPUTS = {
    "step1_manifest.jsonl": (158238, "06b2ab04bde4eb0a66bfb2db21600ef31637940d6d7c84d916e358bade4c7bea"),
    "p31_partitioned_rows.jsonl": (7917039, "5e32eb1905b05076a59b2c5b315ccf9319c04eda18af450565128fd34c18ffa5"),
    "seed_0_k2_initializers.pt": (28650, "9da58e15c518d3b1f35aa3459fb443418dc894163004d69d976ec4afd5dde018"),
    "seed_1_k2_initializers.pt": (28650, "2c1671fd2f103af1d408959b6b1cf3a44332bcc05acd836030e34f1c51a954a2"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    input_root = Path(os.environ.get("W2P_D4_INPUT_ROOT", "/content/w2p_d4_inputs"))
    output_root = Path(os.environ.get("W2P_D4_OUTPUT_ROOT", "/content/w2p_d4_outputs"))
    model_cache = Path(os.environ.get("W2P_D4_MODEL_CACHE", "/content/hf_cache"))
    status_path = output_root / "status.json"
    output_root.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "kind": KIND,
        "status": "preflight",
        "authority_sha256": AUTHORITY_SHA256,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(status_path, receipt)
    started = time.perf_counter()
    try:
        observed = {}
        for name, (expected_bytes, expected_sha) in INPUTS.items():
            path = input_root / name
            if not path.is_file():
                raise FileNotFoundError(f"missing immutable D4 input: {path}")
            observed[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if observed[name] != {"bytes": expected_bytes, "sha256": expected_sha}:
                raise RuntimeError(f"D4 input identity mismatch for {name}: {observed[name]}")
        receipt.update(status="running_forward_only", inputs=observed)
        atomic_json(status_path, receipt)
        command = [
            sys.executable,
            "-u",
            "-m",
            "eval.cache_paper2_bicameral_w2p_d4",
            "--manifest",
            str(input_root / "step1_manifest.jsonl"),
            "--reference_rows",
            str(input_root / "p31_partitioned_rows.jsonl"),
            "--initializer_seed_0",
            str(input_root / "seed_0_k2_initializers.pt"),
            "--initializer_seed_1",
            str(input_root / "seed_1_k2_initializers.pt"),
            "--output_dir",
            str(output_root / "cache"),
            "--model_cache",
            str(model_cache),
            "--wall_seconds_cap",
            "840",
        ]
        subprocess.run(command, cwd=root, check=True)
        summary = output_root / "cache" / "w2p_d4_summary.json"
        if not summary.is_file():
            raise RuntimeError("D4 cache command returned without its summary")
        archive = output_root / "paper2_bicameral_w2p_d4_bundle.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted((output_root / "cache").iterdir()):
                handle.add(path, arcname=path.name)
        receipt.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            summary={"bytes": summary.stat().st_size, "sha256": sha256_file(summary)},
            archive={"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256_file(archive)},
        )
        atomic_json(status_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        receipt.update(
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        atomic_json(status_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
