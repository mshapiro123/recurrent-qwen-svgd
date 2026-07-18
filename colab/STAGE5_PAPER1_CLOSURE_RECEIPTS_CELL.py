"""CPU-safe Colab launcher for compiling and backing up Paper 1 closure receipts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive, userdata


STAGE5_PAPER1_CLOSURE_RECEIPTS_CELL_VERSION = "paper1_closure_receipts_v1"
# Safety marker: PAPER1_EXPERIMENTAL_CLOSURE_RECEIPTS_20260718
# Safety marker: tests/test_paper1_closure_receipts.py
# Safety marker: Bonferroni
# Safety marker: manuscript prose was not edited

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_DIR = Path(
    "/content/drive/MyDrive/recurrent-qwen-svgd-manuscript-closure/"
    "Paper1/receipts_20260718"
)
SYNC_REF = os.environ.get("STAGE5_BOOTSTRAP_REF", "main").strip() or "main"


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
assert GH_TOKEN, "Missing GH_TOKEN in Colab secrets."


def run(command: list[str | os.PathLike[str]], *, cwd: Path = ROOT) -> None:
    safe = " ".join(map(str, command)).replace(GH_TOKEN, "****")
    print("$", safe, flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


drive.mount("/content/drive", force_remount=False)
clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    run(["git", "remote", "set-url", "origin", clone_url])
    run(["git", "fetch", "origin", "main"])
    run(["git", "checkout", "main"])
    run(["git", "reset", "--hard", SYNC_REF])
else:
    run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "reset", "--hard", SYNC_REF])
run(["git", "config", "user.email", "colab-runner@local"])
run(["git", "config", "user.name", "Colab Runner"])

print(
    f"STAGE5_PAPER1_CLOSURE_RECEIPTS_CELL_VERSION="
    f"{STAGE5_PAPER1_CLOSURE_RECEIPTS_CELL_VERSION}",
    flush=True,
)
run([sys.executable, "-m", "pytest", "-q", "tests/test_paper1_closure_receipts.py"])
run([sys.executable, "colab/build_paper1_closure_receipts.py"])

files = [
    ROOT / "docs/PAPER1_EXPERIMENTAL_CLOSURE_RECEIPTS_20260718.json",
    ROOT / "docs/PAPER1_EXPERIMENTAL_CLOSURE_RECEIPTS_20260718.md",
    ROOT / "docs/STAGE5_STRATEGY_CLOSURE_ADDENDA_10_11_20260717.md",
    ROOT / "docs/part1_claim_evidence_ledger.json",
    ROOT / "docs/STAGE5_PEFT_SELECTOR_CLOSURE_HANDOFF_20260717.md",
]
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
manifest = {"kind": "paper1_closure_drive_manifest", "files": []}
for source in files:
    if not source.exists():
        raise FileNotFoundError(source)
    destination = DRIVE_DIR / source.name
    shutil.copy2(source, destination)
    manifest["files"].append(
        {
            "name": source.name,
            "source": str(source.relative_to(ROOT)),
            "drive_path": str(destination),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
    )
(DRIVE_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2), flush=True)
