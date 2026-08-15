"""Paper One Hugging Face release: convert, package, upload private, round-trip verify.

Runs in Colab because the three keeper checkpoints live in Google Drive, not on
any local machine. Nothing in this cell makes a repository public: the flip to
public is a manual step in each repo's Settings, taken only after the round-trip
gate reports green.
"""

import json, os, shutil, subprocess, sys
from pathlib import Path

from google.colab import drive, userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
WORK = Path("/content/paper_one_release")
CONVERTED = WORK / "converted"
REPOS = WORK / "repos"
RECEIPTS = WORK / "receipts"
NAMESPACE_OVERRIDE = ""  # leave blank to use the token's own namespace


def secret(*names):
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
assert GH_TOKEN, "Missing GH_TOKEN/GITHUB_TOKEN in Colab secrets."
assert HF_TOKEN, "Missing HF_TOKEN in Colab secrets. It must be a WRITE token."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def run(cmd, cwd=None, check=True, env=None):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****").replace(HF_TOKEN, "****")
    print("$", printable, flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc


# --- 1. Code: GitHub is the source of truth -------------------------------
clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
if ROOT.exists():
    try:
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        pull = run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT, check=False)
        if pull.returncode != 0:
            shutil.rmtree(ROOT)
            run(["git", "clone", clone_url, str(ROOT)])
    except Exception:
        shutil.rmtree(ROOT)
        run(["git", "clone", clone_url, str(ROOT)])
else:
    run(["git", "clone", clone_url, str(ROOT)])
run(["git", "log", "--oneline", "-3"], cwd=ROOT, check=False)
run([sys.executable, "-m", "pip", "-q", "install", "-U", "huggingface_hub", "safetensors"])

# --- 2. Checkpoints: Drive ------------------------------------------------
if not Path("/content/drive/MyDrive").exists():
    print("Mounting Google Drive for checkpoint visibility.", flush=True)
    drive.mount("/content/drive", force_remount=True)
else:
    print("Drive already mounted.", flush=True)

manifest = json.loads((ROOT / "hf_release/release_manifest.json").read_text(encoding="utf-8"))
repos = manifest["repos"]

missing = [name for name, spec in repos.items() if not Path(spec["drive_checkpoint"]).is_file()]
if missing:
    for name in missing:
        print(f"  MISSING {repos[name]['drive_checkpoint']}", flush=True)
    raise SystemExit(
        "PREFLIGHT_RED: keeper checkpoints are not visible in Drive. "
        "Reauthorize Drive or fix the backup paths; nothing was uploaded."
    )
print("PREFLIGHT_GREEN: all three keeper checkpoints are visible in Drive.", flush=True)

# --- 3. Card hygiene gate -------------------------------------------------
RECEIPTS.mkdir(parents=True, exist_ok=True)
run([sys.executable, "-m", "hf_release.card_hygiene",
     "--receipt", str(RECEIPTS / "card_hygiene.json")], cwd=ROOT)

# --- 4. Convert each keeper to a safetensors delta ------------------------
# convert_checkpoint verifies the source SHA-256 against the manifest and the
# parameter counts against the paper, and refuses on any mismatch.
for name, spec in repos.items():
    run([sys.executable, "-m", "hf_release.convert_checkpoint",
         "--repo", name,
         "--checkpoint", spec["drive_checkpoint"],
         "--output-dir", str(CONVERTED / name)], cwd=ROOT)

# --- 5. Assemble the three repository folders ----------------------------
# No --allow-missing-weights: this must fail if a delta did not land.
run([sys.executable, "-m", "hf_release.package_repos",
     "--converted-root", str(CONVERTED),
     "--output-root", str(REPOS)], cwd=ROOT)
for folder in sorted(REPOS.iterdir()):
    files = sorted(p.name for p in folder.iterdir() if p.is_file())
    print(f"  {folder.name}: {files}", flush=True)

# --- 6. Create private repos and upload ----------------------------------
upload_cmd = [sys.executable, "-m", "hf_release.upload_private", "--repos-root", str(REPOS)]
if NAMESPACE_OVERRIDE:
    upload_cmd += ["--namespace", NAMESPACE_OVERRIDE]
upload = run(upload_cmd, cwd=ROOT)
report = json.loads(upload.stdout[upload.stdout.index("{"):])
(RECEIPTS / "upload_private.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
namespace = report["namespace"]

# --- 7. Round-trip gate: download each private repo and run its frozen eval
# This is the only end-to-end check that the uploaded weights load and score as
# the paper says. It requires the repo to still be private, so it must run now.
results = {}
for name in repos:
    repo_id = f"{namespace}/{name}"
    receipt_path = RECEIPTS / f"roundtrip_{name}.json"
    proc = run([sys.executable, "-m", "hf_release.roundtrip_verify",
                "--repo-id", repo_id, "--receipt", str(receipt_path)],
               cwd=ROOT, check=False)
    status = "red"
    if receipt_path.is_file():
        status = json.loads(receipt_path.read_text(encoding="utf-8"))["status"]
    results[repo_id] = status

print("\n=== ROUND-TRIP GATE ===", flush=True)
for repo_id, status in results.items():
    print(f"  {status.upper():5s}  {repo_id}", flush=True)

if all(status == "green" for status in results.values()):
    print(
        "\nRELEASE_GREEN: all three private repos load and reproduce their frozen counts.\n"
        "Manual step, one repo at a time: open each repo -> Settings -> Change visibility\n"
        "-> Public. Then confirm the base-model link and the Apache 2.0 badge render on\n"
        "each model page, and within a day check the Hub paper page for arXiv:2608.11233.\n"
        f"Receipts: {RECEIPTS}",
        flush=True,
    )
else:
    print(
        "\nRELEASE_RED: at least one round-trip check failed. Leave every repo PRIVATE\n"
        "and inspect the receipts before going further.",
        flush=True,
    )
