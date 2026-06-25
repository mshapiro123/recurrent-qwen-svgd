# Current GPU Action

## Front Of Queue

The current paid-GPU action is:

```text
traced_sft_competence_preserving_pipeline
```

This resumes from the repaired Stage 4 recurrent checkpoint and runs a bounded
competence-preserving deterministic recovery mix. The goal is to recover
ARC-Challenge while preserving the ARC-Easy gains from the repaired recurrent
path. Do **not** run Phase 2/SVGD, particle-noise sweeps, GPQA Diamond, dense
control, or larger Qwen scales until this deterministic gate has a clean review
artifact.

The previous attempt failed before training because the selected Stage 4
checkpoint was not local and Drive was not mounted in the top-level Colab
process. Current `main` contains the fixes:

- top-level Drive mount before child subprocesses;
- parent-process checkpoint restore preflight;
- stale checkpoint/Drive failure routing to same-run-id resume;
- CPU-only competence pipeline review helper.

## Fresh Colab Launcher

Use this in a fresh or restarted Colab runtime. It fetches the tracked launcher,
clones or hard-resets the repo, mounts Drive, verifies current bootstrap
markers, and launches the current target.

```python
import base64, json, time, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
PATH = "colab/CURRENT_STAGE5_FRESH_LAUNCHER_CELL.py"

GH_TOKEN = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."

req = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/{PATH}?ref=main&cache_bust={int(time.time())}",
    headers={
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "Cache-Control": "no-cache",
    },
)

with urllib.request.urlopen(req) as response:
    payload = json.load(response)

code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched", PATH, "sha=", payload.get("sha"))
exec(compile(code, PATH, "exec"))
```

Expected early output:

```text
launcher_version: fresh_launcher_v1
ee304c7 ... or newer
checkpoint_restore_preflight=ok ...
```

If the commit is older than `ee304c7`, stop and rerun the launcher. If Drive
authorization is requested, approve it; if it fails, disconnect the GPU runtime
and repair Drive/auth locally or in a CPU runtime.

## Review After It Lands

After the Colab job pushes results, review with:

```bash
python colab/review_stage5_competence_pipeline.py \
  --source-summary outputs/stage5/stage5_competence_recovery_from_reentry_benchmark/summary.json
```

The review prints:

- wrapper status;
- ARC-mix child status and selected checkpoint;
- full-assessment child status if it ran;
- the planner-selected next action.

## Current Stop Conditions

Stop and review before spending more GPU if any of these happen:

- checkpoint restore preflight does not print `ok`;
- the competence pipeline fails before training;
- ARC-mix summary is missing or nonfinite;
- ARC-mix proxy does not pass;
- full assessment runs and still trails base under the balanced/debiased gate;
- the next action is dense control, particles, GPQA, or scale-up before the
  deterministic recurrent gate has passed.

## Research Sequence Reminder

Current sequence:

```text
re-entry repair passed
-> Stage 4 recovery health passed
-> competence-preserving deterministic recovery  <-- current
-> recurrent-vs-base balanced assessment (`debiased_benchmark_suite`)
-> same-curriculum dense Qwen control (`dense_mcq_trace_sft_control`)
-> breadth / particles / SVGD only if deterministic recurrence clears the gate
```

The umbrella program order is in
[`docs/PROGRAM_TRACK_MASTER_SEQUENCE.md`](../docs/PROGRAM_TRACK_MASTER_SEQUENCE.md).
The detailed experiment log is in
[`docs/EXPERIMENT_LOG.md`](../docs/EXPERIMENT_LOG.md).
