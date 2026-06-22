"""Publish a green curriculum SFT gate as the current Stage 5 source.

Generated curriculum data can be large and is normally restored from Drive.
The gate JSON is small enough to force-add to git, which lets a fresh A100
runtime discover the next guarded action through
``config/stage5_current_source_summary.txt`` while the actual
``positive_sft.jsonl`` remains in the Drive curriculum backup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLISHED_DIR = Path("outputs/stage5/programmatic_direct_deep_curriculum_gate")


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def validate_gate_payload(payload: dict[str, Any], *, gate_json: Path) -> None:
    if payload.get("kind") != "curriculum_sft_gate":
        raise ValueError(f"{gate_json} is not a curriculum_sft_gate payload.")
    if payload.get("go") is not True:
        raise ValueError(f"{gate_json} is not green; refusing to publish no-go gate.")
    if payload.get("status") != "go_train_recurrent_sft":
        raise ValueError(f"{gate_json} status is not go_train_recurrent_sft.")


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> Path:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")
    return pointer


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}")
    return proc


def publish_gate(
    *,
    gate_json: Path,
    gate_md: Path | None,
    published_dir: Path,
) -> dict[str, str]:
    payload = read_json(gate_json)
    validate_gate_payload(payload, gate_json=gate_json)

    published_dir.mkdir(parents=True, exist_ok=True)
    published_gate = published_dir / "curriculum_sft_gate.json"
    shutil.copy2(gate_json, published_gate)
    published_md = published_dir / "curriculum_sft_gate.md"
    if gate_md and gate_md.exists():
        shutil.copy2(gate_md, published_md)

    pointer = update_current_source_summary(published_gate)
    return {
        "published_gate": path_for_cli(published_gate),
        "published_md": path_for_cli(published_md) if published_md.exists() else "",
        "pointer": path_for_cli(pointer),
    }


def git_commit_and_push(paths: list[str], *, commit_message: str, push: bool) -> bool:
    run(["git", "add", "-f", *paths])
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No staged curriculum gate pointer changes to commit.", flush=True)
        return False
    run(["git", "commit", "-m", commit_message])
    if push:
        run(["git", "push", "origin", "main"])
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate_json", required=True)
    parser.add_argument("--gate_md")
    parser.add_argument("--published_dir", default=str(DEFAULT_PUBLISHED_DIR))
    parser.add_argument("--commit_message", default="Publish programmatic curriculum SFT gate")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--no_commit", action="store_true")
    args = parser.parse_args(argv)

    result = publish_gate(
        gate_json=resolve_path(args.gate_json),
        gate_md=resolve_path(args.gate_md) if args.gate_md else None,
        published_dir=resolve_path(args.published_dir),
    )
    print(f"published_gate={result['published_gate']}")
    print(f"current_source_pointer={result['pointer']}")
    if not args.no_commit:
        paths = [result["published_gate"], result["pointer"]]
        if result.get("published_md"):
            paths.append(result["published_md"])
        committed = git_commit_and_push(paths, commit_message=args.commit_message, push=args.push)
        print(f"committed={committed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
