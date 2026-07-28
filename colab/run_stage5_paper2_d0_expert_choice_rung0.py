"""Restore and publish the CPU-only D0 expert-choice Rung 0 receipt."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speculative_depth_d0_corpus import sha256_file

RUN_ID = "stage5_paper2_d0_expert_choice_rung0_20260728"
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
D0 = ROOT / "outputs/stage5/stage5_paper2_d0_20260726"
D1 = ROOT / "outputs/stage5/stage5_paper2_d1_causal_allocation_audit_20260727"
DRIVE_D0 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d0_20260726")
DRIVE_D1 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_d1_causal_allocation_audit_20260727")
DRIVE_RUN = Path(f"/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/{RUN_ID}")


def run(command: list[str]) -> None:
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
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_private_artifact(
    *,
    label: str,
    candidates: list[Path],
    search_root: Path,
    filename: str,
) -> Path:
    expanded = list(candidates)
    if search_root.exists():
        expanded.extend(sorted(search_root.rglob(filename)))
    seen: set[str] = set()
    diagnostics = []
    for path in expanded:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append(
            {
                "path": key,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )
        if path.is_file():
            print(f"resolved_{label}=" + json.dumps(diagnostics), flush=True)
            return path
    raise FileNotFoundError(f"missing {label}: {json.dumps(diagnostics)}")


def publish(paths: list[Path]) -> str:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "add", "-f", "--", *[path.relative_to(ROOT).as_posix() for path in paths]])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        run(["git", "commit", "-m", "Record D0 expert-choice Rung 0 [skip ci]"])
        run(["git", "push", "origin", "main"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    audit_summary = read_json(D1 / "summary.json")
    recorded_feature = Path(
        str(audit_summary["private_artifacts"]["evaluation_feature_cache"])
    )
    feature_candidates = [
        recorded_feature,
        DRIVE_D1 / "private/evaluation_feature_cache.pt",
        DRIVE_D1 / "private/evaluation/evaluation_feature_cache.pt",
    ]
    feature = resolve_private_artifact(
        label="d1_feature_cache",
        candidates=feature_candidates,
        search_root=DRIVE_D1,
        filename="evaluation_feature_cache.pt",
    )
    floor = resolve_private_artifact(
        label="pre_d0_floor_rows",
        candidates=[DRIVE_D0 / "private/floor/floor_rows.json"],
        search_root=DRIVE_D0,
        filename="floor_rows.json",
    )
    floor_summary = read_json(D0 / "floor/summary.json")
    expected_floor_sha = str(floor_summary["private_rows_sha256"])
    observed_floor_sha = sha256_file(floor)
    print(
        f"pre_d0_floor_sha observed={observed_floor_sha} expected={expected_floor_sha}",
        flush=True,
    )
    if observed_floor_sha != expected_floor_sha:
        raise RuntimeError("pre-D0 private floor rows hash does not match the banked receipt")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "eval.rescore_d0_expert_choice",
            "--feature_cache",
            str(feature),
            "--audit_summary",
            str(D1 / "summary.json"),
            "--floor_private_rows",
            str(floor),
            "--output_summary",
            str(RUN_DIR / "summary.json"),
        ]
    )
    paths = [RUN_DIR / name for name in ("summary.json", "summary.md", "summary.png", "summary.svg")]
    DRIVE_RUN.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        shutil.copy2(path, DRIVE_RUN / path.name)
    commit = publish(paths)
    print(json.dumps({"status": "complete", "publish_commit": commit}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        traceback.print_exc()
        try:
            if not Path("/content/drive/MyDrive").is_dir():
                raise RuntimeError("Drive is not mounted; failure receipt not written")
            DRIVE_RUN.mkdir(parents=True, exist_ok=True)
            failure = {
                "kind": "paper2_d0_expert_choice_rung0_failure",
                "status": "errored",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            (DRIVE_RUN / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"failure_receipt={DRIVE_RUN / 'failure.json'}", flush=True)
        except Exception:
            traceback.print_exc()
        raise
