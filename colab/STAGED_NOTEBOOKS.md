# Single-Runtime Colab Runbook

The preferred workflow is **one Colab notebook attached to one runtime** plus
`STAGE5_CURRENT_A100_TARGET`. Do not hop between stage notebooks while a paid
runtime is active. The older split notebooks remain in `colab/` for provenance,
but the maintained execution path is the bootstrap target queue in
[`NEXT_COLAB_SEQUENCE.md`](NEXT_COLAB_SEQUENCE.md).

The maintained notebook is [`00_single_a100_runbook.ipynb`](00_single_a100_runbook.ipynb).
It defines one helper cell, then separate explicit cells for the current master
sequence targets. Leave `KEEP_RUNTIME_OPEN = False` to conserve credits after
each target, or set it to `True` when you intentionally want to run several
bounded L4/T4 cells back-to-back in one attached runtime.

The same notebook also exposes `claim_curriculum_scaleup_cpu` as a separate
CPU/API data-prep cell. That cell prepares the later claim-sized direct/deep
curriculum shard while the GPU queue remains on Phase 0/1; it is not a GPU gate
and does not replace `reentry_repair_smoke` or `reentry_recovery_training`.

For the shortest current instruction, use
[`CURRENT_A100_ACTION.md`](CURRENT_A100_ACTION.md). For the full phase order,
use [`../docs/PROGRAM_TRACK_MASTER_SEQUENCE.md`](../docs/PROGRAM_TRACK_MASTER_SEQUENCE.md).

## Current Target Queue

The program is in **Phase 0: loop-closure re-entry**. Stage 1 found a dead
bridge; Stage 2 found eval-only `entry_rms` re-entry normalization safe enough
for a tiny trainable repair smoke. The next targets are:

1. `reentry_repair_smoke`
   - Runtime: L4/T4 is enough.
   - Purpose: make bridge and re-entry adapter gradient-live, verify movement,
     preserve loop-1 behavior, and prove `bridge_gate` stayed active rather
     than merely moving the bridge projection.
   - Gate: continue only if `review_stage5_reentry.py --no_write` recommends
     `run_bounded_recovery_training_with_reentry_repair`.
2. `reentry_recovery_training`
   - Runtime: L4/T4 for 0.5B unless too slow.
   - Purpose: deterministic depth recovery after loop closure is live.
   - Gate: finite validation, target-loop supervision active, easy/direct
     behavior not collapsed, and post-recovery re-entry health sane.
3. `debiased_benchmark_suite`
   - Runtime: L4/T4 for 0.5B benchmark slices.
   - Purpose: compare base Qwen 0.5B and repaired recurrent Qwen on
     ARC-Easy, ARC-Challenge, and GPQA-lite with debiased MCQ scoring.
4. `dense_mcq_trace_sft_control`
   - Runtime: L4/T4 for short 0.5B control.
   - Purpose: train/evaluate standard dense Qwen LoRA on the same curriculum.
   - Gate: recurrent architecture earns a claim only if it beats this control
     on the relevant hard/depth-shaped rows without easy regression.
5. `claim_curriculum_scaleup_cpu`
   - Runtime: CPU or cheap non-GPU.
   - Purpose: build/resume the claim-sized direct/deep curriculum shard in
     parallel with Phase 0/1 GPU work.
   - Gate: provider calls stay disabled until concrete model ids, API secrets,
     and a tiny provider smoke are configured.
6. Phase 2 breadth diagnostics only after the Phase 1 benchmark/control gate.
7. Phase 3 particles/SVGD only after breadth is correct-bearing.

## Paste-Anywhere Launcher

Set `TARGET` to one of the current targets above.

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

resolved_ref = gh_json(
    f"https://api.github.com/repos/{REPO}/git/refs/heads/main?cache_bust={time.time_ns()}"
)["object"]["sha"]

payload = gh_json(
    f"https://api.github.com/repos/{REPO}/contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py"
    f"?ref={resolved_ref}&cache_bust={time.time_ns()}"
)

code = base64.b64decode(payload["content"]).decode("utf-8")
required = [
    "sha_resolved_nested_fetch_v3",
    TARGET,
    "STAGE5_CURRENT_A100_TARGET",
]
missing = [marker for marker in required if marker not in code]
assert not missing, f"Fetched stale or incomplete bootstrap: {missing}"
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12], "target:", TARGET)
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Cheap Status Target

When a runtime restarts or the notebook state is unclear, use:

```python
TARGET = "master_sequence_status"
```

This target does not mount Drive, train, evaluate, or download models. It prints
the current source-summary pointer, planner recommendation, re-entry review,
Stage 4 recovery review, and Phase 1 benchmark/control gate review.

## Historical Notebooks

The following notebooks are kept for provenance and old-run reproduction, not
for the current front-of-queue action:

1. `00_single_a100_runbook.ipynb`
2. `00_stage_launcher.ipynb`
3. `01_stage1_svgd_seed_replication.ipynb`
4. `02_stage2_benchmark_harness.ipynb`
5. `03_stage3_hf_packaging.ipynb`
6. `04_stage4_modified_opus_finetune.ipynb`
7. `05_stage5_benchmarks.ipynb`
8. `06_stage6_writeup_and_release.ipynb`
9. `07_stage5_full_arc_assessment.ipynb`
10. `08_stage5_safe_continue.ipynb`
11. `09_stage5_arc_mix_recovery_once.ipynb`
12. `10_stage5_direct_preservation_precheck.ipynb`
13. `11_stage5_direct_preservation_g4_auto.ipynb`

If a historical notebook conflicts with this runbook, trust the maintained
bootstrap target queue and `CURRENT_A100_ACTION.md`.
