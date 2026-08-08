"""Build and publish the score-blind, Option-B-compatible E1 EVAL-D cache."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_paper2_phase2_stage0a import write_json  # noqa: E402
from training.paper2_phase2_e1_eval_d import (  # noqa: E402
    RUN_ID,
    build_score_blind_config,
)
from training.paper2_phase2_option_b import load_locked_registration  # noqa: E402
from training.paper2_phase2_stage0a import sha256_file  # noqa: E402


# Safety marker: EVAL-D infrastructure only no endpoint checkpoint no outcome score
# Safety marker: 8000 anchors 4000 general 4000 code seed 20260808
# Safety marker: base student forward materializes cache tensors only no quality score
# Safety marker: no EAL no retention no acceptance no optimizer no training
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DRIVE_ROOT = Path(
    f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}"
)
PREWINDOW = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_phase2_prewindow_20260731"
)
STAGE0A = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_phase2_stage0a_20260803"
)
DC1_PREFLIGHT = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_preflight_20260729"
)
DC1_STAGE_A = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_dc1_stage_a_20260730"
)
OPTION_B_CACHE = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_phase2_option_b_teacher_cache_20260806"
)
ARBITRATION = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/"
    "stage5_paper2_phase2_arbitration_build_20260804"
)


def run(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
        tail = tail[-300:]
    code = process.wait()
    if code not in allowed:
        print("\nE1 cache child-process tail:\n" + "\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(code, command)
    return code


def status_event(status: str, **details: Any) -> None:
    payload = {
        "kind": "paper2_phase2_e1_eval_d_cache_status",
        "run_id": RUN_ID,
        "status": status,
        "scores_exposed": False,
        "read_once_scoring_spent": False,
        "training_started": False,
        "optimizer_steps": 0,
        **details,
    }
    print("e1_eval_d_status:", json.dumps(payload, sort_keys=True), flush=True)
    write_json(DRIVE_ROOT / "receipts/status.json", payload)


def select_scratch() -> Path:
    candidates = [Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content")]
    usable = []
    for path in candidates:
        if not path.is_dir() or not os.access(path, os.W_OK):
            continue
        free = shutil.disk_usage(path).free
        if free >= 80 * 1024**3:
            usable.append(("scratch" in str(path), free, path))
    if not usable:
        raise RuntimeError("E1 cache generation requires at least 80 GiB free local disk")
    usable.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = usable[0][2] / RUN_ID
    selected.mkdir(parents=True, exist_ok=True)
    print(
        f"e1_eval_d_scratch path={selected} free_gib={shutil.disk_usage(selected).free / 1024**3:.1f}",
        flush=True,
    )
    return selected


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Freeze score-blind E1 EVAL-D cache [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    scratch = select_scratch()
    data = PREWINDOW / "private/eval_de/eval_d/eval_d.jsonl"
    legacy = PREWINDOW / "receipts/eval_de_freeze_summary.json"
    canonicalizer = (
        ARBITRATION
        / "private/canonicalizer/learned_mixture_rrr_seed_20260814.pt"
    )
    dev_manifest = STAGE0A / "private/stage0a/sample_manifest.jsonl"
    exclusions = [
        PREWINDOW / "private/eval_de/eval_e/eval_e.jsonl",
        DC1_PREFLIGHT / "private/dev_c/dev_c.jsonl",
        DC1_STAGE_A / "private/eval_c/eval_c.jsonl",
        OPTION_B_CACHE / "private/new_documents_target.jsonl",
    ]
    required = [data, legacy, canonicalizer, dev_manifest, *exclusions]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E1 cache prerequisites are missing: {missing}")

    registration = load_locked_registration()
    config = build_score_blind_config(registration=registration, data_path=data)
    config_path = RUN_DIR / "config/e1_eval_d_cache_config.json"
    write_json(config_path, config)
    private = DRIVE_ROOT / "private/e1_eval_d"
    lattice_summary = RUN_DIR / "cache/e1_eval_d_lattice_summary.json"
    freeze = RUN_DIR / "receipts/e1_eval_d_freeze_summary.json"
    readiness = RUN_DIR / "receipts/e1_readiness.json"
    private_cache = DRIVE_ROOT / "private/e1_eval_d_option_b_cache.pt"
    admission = DRIVE_ROOT / "private/e1_eval_d_anchor_admission.jsonl"

    status_event(
        "building_score_blind_cache",
        data_sha256=sha256_file(data),
        selection_seed=config["seed"],
        anchor_count=config["anchor_count"],
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.cache_paper2_phase2_stage0a",
            "--data_jsonl",
            str(data),
            "--private_dir",
            str(private),
            "--output_summary",
            str(lattice_summary),
            "--staging_dir",
            str(scratch / "staging"),
            "--config_json",
            str(config_path),
            "--score_blind",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--attn_implementation",
            "sdpa",
        ]
    )
    finalize = [
        sys.executable,
        "-u",
        "-m",
        "eval.finalize_paper2_phase2_e1_eval_d_cache",
        "--stage0a_summary",
        str(lattice_summary),
        "--stage0a_private",
        str(private),
        "--canonicalizer",
        str(canonicalizer),
        "--local_cache",
        str(scratch / "e1_eval_d_option_b_cache.pt"),
        "--private_cache",
        str(private_cache),
        "--data_jsonl",
        str(data),
        "--legacy_freeze_summary",
        str(legacy),
        "--dev_sample_manifest",
        str(dev_manifest),
        "--admission_ledger",
        str(admission),
        "--output",
        str(freeze),
    ]
    for path in exclusions:
        finalize.extend(["--exclude_jsonl", str(path)])
    run(finalize)
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "eval.check_paper2_phase2_e1_readiness",
            "--eval_d_freeze",
            str(freeze),
            "--output",
            str(readiness),
        ]
    )
    ready = json.loads(readiness.read_text(encoding="utf-8"))
    if not ready.get("ready_to_lock"):
        raise RuntimeError(f"E1 cache landed but readiness is blocked: {ready['blockers']}")

    receipt_dir = DRIVE_ROOT / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(freeze, receipt_dir / "e1_eval_d_freeze_summary.json")
    shutil.copy2(readiness, receipt_dir / "e1_readiness.json")
    commit = publish([config_path, lattice_summary, freeze, readiness])
    status_event(
        "complete_frozen_unscored_ready_to_lock",
        freeze_sha256=sha256_file(freeze),
        readiness_sha256=sha256_file(readiness),
        private_cache_sha256=sha256_file(private_cache),
        publish_commit=commit,
    )
    print(
        json.dumps(
            {
                "status": "complete_frozen_unscored_ready_to_lock",
                "publish_commit": commit,
                "freeze_receipt": str(freeze),
                "freeze_sha256": sha256_file(freeze),
                "readiness_receipt": str(readiness),
                "readiness_sha256": sha256_file(readiness),
                "read_once_scoring_spent": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
