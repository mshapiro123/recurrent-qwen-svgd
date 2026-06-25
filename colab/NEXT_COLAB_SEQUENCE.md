# Next Colab Sequence

This is the short execution queue for the current program state. It follows
the master sequence:

1. Phase 0: re-entry and loop closure.
2. Phase 1: deterministic depth recovery.
3. Phase 2: breadth and multistability.
4. Phase 3: particles/SVGD and selector.

The maintained interface is one Colab notebook plus
`STAGE5_CURRENT_A100_TARGET`, not a pile of separate notebooks. Prefer L4/T4
for the current 0.5B diagnostics and smokes.

## Paste-Anywhere Launcher

Set `TARGET` to one of the targets below.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
TARGET = "reentry_repair_smoke"

gh = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
assert gh, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."

hf = userdata.get("HF_TOKEN") or userdata.get("HUGGINGFACE_HUB_TOKEN")
if hf:
    os.environ["HF_TOKEN"] = hf
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf

os.environ["STAGE5_CURRENT_A100_TARGET"] = TARGET

headers = {
    "Authorization": f"Bearer {gh}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

ref = gh_json(
    f"https://api.github.com/repos/{REPO}/git/refs/heads/main?cache_bust={time.time_ns()}"
)["object"]["sha"]
payload = gh_json(
    f"https://api.github.com/repos/{REPO}/contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    f"?ref={ref}&cache_bust={time.time_ns()}"
)

code = base64.b64decode(payload["content"]).decode("utf-8")
required = [
    "sha_resolved_nested_fetch_v3",
    TARGET,
    "STAGE5_CURRENT_A100_TARGET",
]
missing = [marker for marker in required if marker not in code]
assert not missing, f"Fetched stale or incomplete bootstrap: {missing}"
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", ref[:12], "target:", TARGET)
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Queue

### 0. Cheap status after any runtime restart

Target:

```python
TARGET = "master_sequence_status"
```

This does not mount Drive, download models, train, or evaluate. It fetches the
latest repo, prints the current source-summary pointer, asks the planner and
re-entry reviewer for the next target, prints the queue excerpt, and
disconnects. Use it when a runtime was restarted or when the notebook state is
unclear.

### 1. Finish or recover Stage 2

Target:

```python
TARGET = "reentry_norm_diagnostic"
```

Use this only if the current L4 run did not finish and cannot be recovered.
Current `main` runs the cheaper bounded version by default.

If the old long run produced raw files but died before publish/summary:

```python
TARGET = "reentry_norm_recover_only"
```

Run recover-only before rerunning the GPU cell. It can rebuild `summary.json`,
`summary.md`, and `reentry_assessment` from raw drift, effective-pathway, and
candidate-conversion files.

Review command after publish or recovery:

```bash
python colab/review_stage5_reentry.py --no_write
```

Gate: continue only if recommendation is `run_reentry_repair_smoke`.

### 2. Stage 3: trainable re-entry repair smoke

Target:

```python
TARGET = "reentry_repair_smoke"
```

Purpose: make bridge and re-entry adapter gradient-live and verify loop-1
preservation before any more recovery training.

Runtime: L4/T4 is sufficient.

Gate: continue only if recommendation is
`run_bounded_recovery_training_with_reentry_repair`.

Stop conditions:

- missing loop-1 preservation evidence;
- loop-1 regression;
- adapter gradients not live;
- adapter or bridge live but unmoved.

### 3. Stage 4: bounded deterministic recovery SFT

Target:

```python
TARGET = "reentry_recovery_training"
```

Purpose: recover deterministic recurrent competence after loop closure is live.
This uses:

- repaired Stage 3 checkpoint;
- `entry_rms` re-entry normalization during training and validation;
- re-entry adapter enabled;
- learned loop control;
- target-loop NLL supervision;
- strict target-loop row count gates.

Runtime: L4 may work for 0.5B; use G4/A100 only if L4 is too slow or unavailable.

Gate: finite validation, loop-depth gradient present, easy/direct behavior not
collapsed. Benchmark before returning to breadth.

### 4. Phase 1 depth benchmark/control arm

After Stage 4 produces a sane deterministic recurrent checkpoint, run the
paired depth assessment against:

- base Qwen 0.5B;
- recurrent checkpoint;
- standard Qwen same-curriculum LoRA control.

First benchmark base versus the repaired recurrent checkpoint:

```python
TARGET = "debiased_benchmark_suite"
```

Use the Stage 4 summary as the source summary if it is not already the current
pointer:

```python
os.environ["STAGE5_CURRENT_A100_SOURCE_SUMMARY"] = "outputs/stage5/<stage4_run>/summary.json"
```

Then run the matched dense recipe control, pointing the source override at the
benchmark-suite summary produced by the previous step:

```python
TARGET = "dense_mcq_trace_sft_control"
os.environ["STAGE5_CURRENT_A100_SOURCE_SUMMARY"] = "outputs/stage5/<debiased_benchmark_run>/summary.json"
```

For this target the override serves two roles: it identifies the recurrent
benchmark summary to compare against, and the dense-control runner follows that
summary's source chain back to the Stage 4 curriculum rows.

This is the decisive Phase 1 question: does recurrence convert depth-shaped
failures while preserving easy items, beyond what the data alone gives a dense
control?

Read this step in order:

1. recurrent versus base on debiased content/cyclic scoring;
2. dense LoRA control versus base on the same rows and scoring;
3. recurrent versus dense control, especially on hard/depth-shaped rows.

If recurrent improves over its pre-repair state but dense control matches or
beats it, the data recipe helped but the architecture has not yet earned the
claim. If recurrent beats dense control on hard rows without easy regression,
Phase 1 has a real architecture signal.

Do not run Phase 2/SVGD until this is answered.

### 5. Phase 2 breadth diagnostic

Only after deterministic depth is base-competitive:

- rerun effective-pathway diagnostics on the repaired/depth-trained checkpoint;
- split correct vs wrong candidates;
- use Leinster similarity-sensitive diversity;
- test multi-solution tasks, not single-answer arithmetic.

Gate: effective pathway count above one with correct-bearing diversity.

### 6. Phase 3 particles/SVGD and selector

Only after the Phase 2 gate:

- reintroduce particles/SVGD as a soft regularizer;
- train/measure method-anchored pathways;
- use selector conversion as the metric, not superficial diversity.

Gate for public claim: hard-stratum gains, easy preservation, held-out prompts,
and debiased scoring.
