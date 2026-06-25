"""Colab launcher: recover/publish completed Stage 2 re-entry norm artifacts.

Use this only when the long Stage 2 eval-only re-entry normalization cell
finished or mostly finished but did not land in Git. It does not rerun GPU
evaluation. It searches Drive for the newest ``stage5_reentry_norm_*`` run,
copies it into the repo, rebuilds the assessment if needed, and publishes the
artifact.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, runtime, userdata


STAGE5_REENTRY_NORM_RECOVER_CELL_VERSION = "stage5_reentry_norm_recover_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
LEGACY_DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd")

REQUIRED_FILES = [
    "summary.json",
    "summary.md",
    "reentry_norm/reentry_drift_none.json",
    "reentry_norm/reentry_drift_none.jsonl",
    "reentry_norm/effective_pathways_none.json",
    "reentry_norm/effective_pathways_none.jsonl",
    "reentry_norm/candidate_conversion_none.jsonl",
    "reentry_norm/reentry_drift_entry_rms.json",
    "reentry_norm/reentry_drift_entry_rms.jsonl",
    "reentry_norm/effective_pathways_entry_rms.json",
    "reentry_norm/effective_pathways_entry_rms.jsonl",
    "reentry_norm/candidate_conversion_entry_rms.jsonl",
]


def secret(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = redact(" ".join(map(str, cmd)))
    print(f"$ {printable}", flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(redact(proc.stdout), flush=True)
    if check:
        proc.check_returncode()
    return proc


def normalize_rel_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def ensure_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def has_valid_json(path: Path) -> bool:
    try:
        read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return True


def has_valid_jsonl(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            return False
        for line in rows:
            json.loads(line)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def stage2_missing_reasons(path: Path) -> list[str]:
    from colab.reentry_norm_recover_utils import (
        FINAL_REQUIRED_FILES,
        RAW_REQUIRED_FILES,
        final_stage2_complete,
        missing_reasons,
    )

    if final_stage2_complete(path):
        return []
    raw_reasons = missing_reasons(path, RAW_REQUIRED_FILES)
    if not raw_reasons:
        return []
    final_reasons = missing_reasons(path, FINAL_REQUIRED_FILES)
    return final_reasons or raw_reasons


def stage2_complete(path: Path) -> bool:
    from colab.reentry_norm_recover_utils import recoverable_stage2

    return recoverable_stage2(path)


def partial_artifact_report(paths: list[Path], *, limit: int = 8) -> str:
    if not paths:
        return "No stage5_reentry_norm_* candidate directories were visible on Drive."
    lines = ["Visible incomplete stage5_reentry_norm_* candidates:"]
    for path in sorted(paths, key=lambda candidate: candidate.stat().st_mtime if candidate.exists() else 0, reverse=True)[:limit]:
        if not path.is_dir():
            lines.append(f"  - {path}: not a directory")
            continue
        reasons = stage2_missing_reasons(path)
        if not reasons:
            lines.append(f"  - {path}: complete")
            continue
        lines.append(f"  - {path}: incomplete")
        for reason in reasons[:8]:
            lines.append(f"      * {reason}")
        if len(reasons) > 8:
            lines.append(f"      * ... {len(reasons) - 8} more")
    return "\n".join(lines)


def candidate_roots() -> list[Path]:
    explicit = os.environ.get("STAGE5_REENTRY_NORM_RECOVER_SOURCE", "").strip()
    roots: list[Path] = []
    if explicit:
        raw = Path(explicit)
        roots.append(raw if raw.is_absolute() else DRIVE_ARTIFACT_ROOT / normalize_rel_path(explicit))
    for root in (
        DRIVE_ARTIFACT_ROOT / "outputs" / "stage5",
        DRIVE_ARTIFACT_ROOT,
        LEGACY_DRIVE_ROOT / "outputs" / "stage5",
        LEGACY_DRIVE_ROOT,
    ):
        if root.exists():
            roots.extend(sorted(root.glob("stage5_reentry_norm_*")))
            roots.extend(sorted(root.glob("outputs/stage5/stage5_reentry_norm_*")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in roots:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def latest_complete_stage2() -> Path:
    drive.mount("/content/drive", force_remount=False)
    roots = candidate_roots()
    complete = [path for path in roots if path.is_dir() and stage2_complete(path)]
    if not complete:
        searched = "\n".join(f"  - {path}" for path in roots[:40])
        raise FileNotFoundError(
            "No complete stage5_reentry_norm_* artifact found on Drive. "
            "The Stage 2 run may still be active or may not have reached final backup.\n"
            f"{partial_artifact_report(roots)}\n"
            f"Searched:\n{searched}"
        )
    return sorted(complete, key=lambda path: path.stat().st_mtime)[-1]


def copy_to_repo(source: Path) -> Path:
    target = ROOT / "outputs" / "stage5" / source.name
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    print(f"recovered_stage2={source} -> {target}", flush=True)
    return target


def ensure_summary(out_dir: Path) -> None:
    from colab.reentry_norm_recover_utils import build_summary_payload, summary_markdown

    summary_path = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    if has_valid_json(summary_path):
        summary = read_json(summary_path)
        if summary.get("kind") == "stage5_reentry_norm_eval_only" and summary_md.exists():
            print(f"summary already present: {summary_path}", flush=True)
            return
    summary = build_summary_payload(out_dir, cell_version=STAGE5_REENTRY_NORM_RECOVER_CELL_VERSION)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(summary_markdown(summary), encoding="utf-8")
    print(f"rebuilt_summary={summary_path}", flush=True)


def ensure_assessment(out_dir: Path) -> None:
    assessment_json = out_dir / "reentry_assessment.json"
    assessment_md = out_dir / "reentry_assessment.md"
    if has_valid_json(assessment_json) and assessment_md.exists():
        print(f"assessment already present: {assessment_json}", flush=True)
        return
    run(
        [
            sys.executable,
            "colab/assess_stage5_reentry.py",
            "--summary_json",
            str((out_dir / "summary.json").relative_to(ROOT)),
            "--output_json",
            str(assessment_json.relative_to(ROOT)),
            "--output_md",
            str(assessment_md.relative_to(ROOT)),
        ]
    )
    print(assessment_md.read_text(encoding="utf-8"), flush=True)


def publish(out_dir: Path) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from colab.stage5_publish_utils import publishable_artifact_paths, update_current_source_summary

    run(["git", "status", "-sb"])
    pointer = update_current_source_summary(ROOT, out_dir / "summary.json")
    publishable = publishable_artifact_paths(out_dir)
    if not publishable:
        print(f"No lightweight publishable artifacts found under {out_dir}.", flush=True)
        return
    publishable.append(pointer)
    for path in publishable:
        run(["git", "add", "-f", str(path.relative_to(ROOT))])
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No recovered Stage 2 changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", f"Recover Stage 5 re-entry norm {out_dir.name} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode != 0:
        print("Initial push failed; attempting one fast rebase and retry.", flush=True)
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


def main() -> None:
    print(f"cell_version={STAGE5_REENTRY_NORM_RECOVER_CELL_VERSION}", flush=True)
    ensure_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_reentry_assessment.py",
            "tests/test_review_stage5_reentry.py",
        ]
    )
    source = latest_complete_stage2()
    out_dir = copy_to_repo(source)
    ensure_summary(out_dir)
    ensure_assessment(out_dir)
    publish(out_dir)
    run([sys.executable, "colab/review_stage5_reentry.py", "--no_write"])
    if os.environ.get("STAGE5_REENTRY_NORM_RECOVER_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        print("Disconnecting Colab runtime after Stage 2 recovery publish.", flush=True)
        runtime.unassign()


main()
