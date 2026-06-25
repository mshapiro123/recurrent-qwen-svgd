# Stage 5 Master Sequence Status Cell

Cheap CPU readout for restarted Colab sessions. It fetches the latest repo,
prints the current Stage 5 source pointer, asks the planner and re-entry
reviewer for the next target, prints the Stage 4 curriculum readiness and
claim-sized scale-up plan, prints the queue excerpt, and disconnects by
default. It does not mount Drive, download models, train, or evaluate.

```python
import os, shutil, subprocess, sys
from pathlib import Path
from google.colab import runtime, userdata

STAGE5_MASTER_SEQUENCE_STATUS_CELL_VERSION = "master_sequence_status_v1"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT = os.environ.get("STAGE5_MASTER_SEQUENCE_STATUS_DISCONNECT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


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
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print("HF token loaded from Colab secrets.", flush=True)
else:
    print("HF token not found; status target will not download models.", flush=True)


def run(cmd, cwd=None, check=True, env=None):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc


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

print("MASTER_SEQUENCE_STATUS: cheap CPU readout; no model downloads or training.", flush=True)
run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
run([sys.executable, "colab/print_current_stage5_action.py"], cwd=ROOT, check=False)
run([sys.executable, "colab/review_stage5_reentry.py", "--no_write"], cwd=ROOT, check=False)
print("\nStage 4 Recovery Curriculum Readiness:", flush=True)
run([sys.executable, "colab/review_stage5_recovery_curriculum.py"], cwd=ROOT, check=False)
print("\nClaim-Sized Curriculum Scale-Up Plan:", flush=True)
run([sys.executable, "colab/plan_stage5_curriculum_scaleup.py"], cwd=ROOT, check=False)

sequence = ROOT / "colab" / "NEXT_COLAB_SEQUENCE.md"
if sequence.exists():
    print("\nNEXT_COLAB_SEQUENCE excerpt:", flush=True)
    text = sequence.read_text(encoding="utf-8")
    start = text.find("## Queue")
    excerpt = text[start:] if start >= 0 else text
    print(excerpt[:4500], flush=True)

if DISCONNECT:
    print("Disconnecting status runtime; set STAGE5_MASTER_SEQUENCE_STATUS_DISCONNECT=0 to keep it open.", flush=True)
    runtime.unassign()
```
