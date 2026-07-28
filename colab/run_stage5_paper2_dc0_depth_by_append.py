"""Run and publish the forward-only DC0 depth-by-append diagnostic."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from training.speculative_depth_d0_corpus import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_dc0_20260728"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
D0_DRIVE = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d0_20260726")
D1_DRIVE = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d1_causal_allocation_audit_20260727")
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")
CHECKPOINT_SHA = "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf"


def run(command: list[str], allowed: tuple[int, ...] = (0,)) -> int:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode not in allowed:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def resolve_sha(candidates: list[Path], expected: str, destination: Path) -> Path:
    diagnostics = []
    for source in candidates:
        observed = sha256_file(source) if source.exists() else None
        diagnostics.append({"path": str(source), "sha256": observed})
        if observed == expected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_file(destination) != expected:
                shutil.copy2(source, destination)
            print("dc0_restore:", json.dumps(diagnostics), flush=True)
            return destination
    raise FileNotFoundError(f"DC0 required artifact not found: {diagnostics}")


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record DC0 depth-by-append diagnostic [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    eval_b_public = RUN_DIR / "eval_b/summary.json"
    if not eval_b_public.exists():
        raise FileNotFoundError("DC0 requires the landed EVAL-B aggregate receipt")
    eval_b = json.loads(eval_b_public.read_text(encoding="utf-8"))
    if eval_b.get("status") != "complete_unspent" or eval_b.get("read_once_scoring_spent") is not False:
        raise RuntimeError("EVAL-B is absent or already spent")
    data = DRIVE_RUN / "private/eval_b/eval_b.jsonl"
    cache_summary = DRIVE_RUN / "private/eval_b/teacher_cache_summary.json"
    if not data.exists() or sha256_file(data) != eval_b["data"]["jsonl_sha256"]:
        raise RuntimeError("EVAL-B private data hash mismatch")
    if not cache_summary.exists():
        raise FileNotFoundError("EVAL-B private teacher cache summary is missing")
    checkpoint = resolve_sha(
        [
            D0_DRIVE / "private/training/d0_ema_step_4000.pt",
            D0_DRIVE / "private/train/d0_ema_step_4000.pt",
            D0_DRIVE / "checkpoints/d0_ema_step_4000.pt",
        ],
        CHECKPOINT_SHA,
        RUN_DIR / "runtime/d0_ema_step_4000.pt",
    )
    audit_candidates = [
        D1_DRIVE / "private/evaluation_feature_cache.pt",
        D1_DRIVE / "private/evaluation/evaluation_feature_cache.pt",
    ]
    audit = next((path for path in audit_candidates if path.exists()), None)
    if audit is None:
        raise FileNotFoundError(f"missing banked D1 feature cache: {audit_candidates}")
    output_dir = RUN_DIR / "dc0"
    private_dir = DRIVE_RUN / "private/dc0"
    code = run(
        [
            sys.executable,
            "-u",
            "eval/eval_paper2_dc0_depth_by_append.py",
            "--data_jsonl",
            str(data),
            "--teacher_cache_summary",
            str(cache_summary),
            "--checkpoint",
            str(checkpoint),
            "--expected_checkpoint_sha256",
            CHECKPOINT_SHA,
            "--audit_feature_cache",
            str(audit),
            "--composite_preflight_summary",
            str(
                ROOT
                / "outputs/stage5/stage5_coconut_composite_rg1_rg11_20260725/summary.json"
            ),
            "--output_dir",
            str(output_dir),
            "--private_dir",
            str(private_dir),
            "--device",
            os.environ.get("DEVICE", "cuda"),
            "--dtype",
            os.environ.get("STAGE5_DC0_DTYPE", "bfloat16"),
            "--attn_implementation",
            os.environ.get("STAGE5_DC0_ATTN", "sdpa"),
            "--append_batch_size",
            os.environ.get("STAGE5_DC0_APPEND_BATCH_SIZE", "8"),
        ],
        allowed=(0, 2),
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    eval_b["status"] = "spent_dc0_complete" if code == 0 else "spent_dc0_blocked"
    eval_b["read_once_scoring_spent"] = True
    eval_b["read_log"].append(
        {
            "purpose": "registered DC0 in-place and depth-by-append comparison",
            "interpretive_scoring": True,
            "dc0_status": summary["status"],
        }
    )
    eval_b_public.write_text(json.dumps(eval_b, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = [eval_b_public, output_dir / "summary.json"]
    for optional in ("summary.md", "dc0_first_transition.png", "dc0_first_transition.svg"):
        path = output_dir / optional
        if path.exists():
            paths.append(path)
    receipt_dir = DRIVE_RUN / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, receipt_dir / path.name)
    commit = publish(paths)
    print(json.dumps({"status": summary["status"], "publish_commit": commit}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
