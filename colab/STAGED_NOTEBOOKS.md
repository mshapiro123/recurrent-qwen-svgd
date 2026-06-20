# Staged Colab Notebooks

Run this single cell in any Colab notebook to fetch the private repo and print
links to the staged notebooks:

```python
import os, subprocess
from pathlib import Path
from IPython.display import Markdown, display
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or secret("GH_TOKEN") or secret("GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."

def run(cmd, cwd=None):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable)
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode:
        raise RuntimeError(printable)

if ROOT.exists():
    run(["git", "remote", "set-url", "origin", f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"], cwd=ROOT)
    run(["git", "fetch", "origin", "main"], cwd=ROOT)
    run(["git", "checkout", "main"], cwd=ROOT)
    run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT)
else:
    run(["git", "clone", f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git", str(ROOT)])

base = "https://colab.research.google.com/github/mshapiro123/recurrent-qwen-svgd/blob/main/colab"
links = [
    ("Stage 1 - SVGD seed replication", "01_stage1_svgd_seed_replication.ipynb"),
    ("Stage 2 - Benchmark harness", "02_stage2_benchmark_harness.ipynb"),
    ("Stage 3 - Hugging Face packaging", "03_stage3_hf_packaging.ipynb"),
    ("Stage 4 - Modified Opus fine-tune", "04_stage4_modified_opus_finetune.ipynb"),
    ("Stage 5 - Benchmarks", "05_stage5_benchmarks.ipynb"),
    ("Stage 6 - Write-up and release", "06_stage6_writeup_and_release.ipynb"),
]
display(Markdown("\n".join(f"- [{name}]({base}/{path})" for name, path in links)))
```

## Stage Map

1. `01_stage1_svgd_seed_replication.ipynb`: finish heldout seed `5-9`
   replication for random32 vs recreated within-group PCA.
2. `02_stage2_benchmark_harness.ipynb`: build/validate MCQ benchmark harness.
3. `03_stage3_hf_packaging.ipynb`: export and push adapter/controller package.
4. `04_stage4_modified_opus_finetune.ipynb`: fine-tune on modified Opus traces.
5. `05_stage5_benchmarks.ipynb`: run base vs recurrent benchmarks.
6. `06_stage6_writeup_and_release.ipynb`: produce report, model card, and release notes.
