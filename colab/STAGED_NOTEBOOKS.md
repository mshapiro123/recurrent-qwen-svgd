# Single A100 Colab Runbook

The preferred workflow is **one Colab notebook attached to one A100 runtime**.
Do not open each stage as a separate notebook while an expensive runtime is
active. The staged notebooks remain in `colab/` for review, but the execution
path below stays inside the current runtime.

## Start or Resume Stage 1 In-Place

Paste this single cell into the already-attached A100 notebook. It clones or
updates the private GitHub repo, installs dependencies, validates optional HF
auth, then runs Stage 1 directly in the current runtime.

```python
import os, subprocess, sys
from pathlib import Path
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

GH_TOKEN = (
    os.environ.get("GH_TOKEN")
    or os.environ.get("GITHUB_TOKEN")
    or secret("GH_TOKEN")
    or secret("GITHUB_TOKEN")
)
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or secret("HF_TOKEN")
    or secret("HUGGINGFACE_HUB_TOKEN")
)
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

def run(cmd, cwd=None, check=True):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable)
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if check and proc.returncode:
        raise RuntimeError(f"failed: {printable}")
    return proc

if ROOT.exists():
    run(["git", "remote", "set-url", "origin", f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"], cwd=ROOT)
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT)
else:
    run(["git", "clone", f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git", str(ROOT)])

run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

if HF_TOKEN:
    from huggingface_hub import HfApi, login
    login(token=HF_TOKEN, add_to_git_credential=False)
    who = HfApi(token=HF_TOKEN).whoami()
    print("HF auth OK:", who.get("name") or who.get("email") or "authenticated user")

run([sys.executable, "colab/run_stage1_single_runtime.py"], cwd=ROOT)
```

## Optional Notebook Reference

Open `colab/00_single_a100_runbook.ipynb` only when you want a clean notebook
layout. Once it is open, keep all stage execution inside that one notebook.

The older split notebooks are kept only as references:

1. `01_stage1_svgd_seed_replication.ipynb`
2. `02_stage2_benchmark_harness.ipynb`
3. `03_stage3_hf_packaging.ipynb`
4. `04_stage4_modified_opus_finetune.ipynb`
5. `05_stage5_benchmarks.ipynb`
6. `06_stage6_writeup_and_release.ipynb`

## Stage Map

1. Finish heldout seed `5-9` replication for random32 vs recreated
   within-group PCA.
2. Build/validate the MCQ benchmark harness.
3. Export and push adapter/controller package to Hugging Face.
4. Fine-tune on modified Opus traces.
5. Run base vs recurrent benchmarks, then run
   `colab/run_stage5_arc_agi_candidate_gate.py` to separate model-only,
   symbolic-only, and hybrid candidate value before more particle tuning.
   If symbolic candidates help, run `colab/run_stage5_arc_agi_trace_sft_gate.py`
   to compare grid-only ARC SFT against symbolic-trace ARC SFT on trace-covered
   examples. Then run `colab/run_stage5_arc_agi_distill_sft_gate.py` to test
   whether frozen-base logit distillation preserves competence during ARC SFT.
6. Write the report, model card, and release notes.
