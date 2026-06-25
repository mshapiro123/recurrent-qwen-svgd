# Current GPU Action

## Preferred Launch Path

Use one maintained Colab notebook and one target variable:

[`colab/00_single_a100_runbook.ipynb`](00_single_a100_runbook.ipynb)

or the paste-anywhere launcher below. The current queue is intentionally short:

```text
master_sequence_status
reentry_repair_smoke
master_sequence_status
reentry_recovery_training
debiased_benchmark_suite
dense_mcq_trace_sft_control
```

This follows the master sequence:

```text
Phase 0 re-entry repair -> Phase 1 deterministic depth recovery/control
-> Phase 2 breadth only after deterministic depth works
-> Phase 3 particles/SVGD only after correct-bearing breadth exists
```

The detailed Stage 3/4 contract is in
[`docs/STAGE5_REENTRY_STAGE3_STAGE4_RUNBOOK.md`](../docs/STAGE5_REENTRY_STAGE3_STAGE4_RUNBOOK.md).
The umbrella plan is in
[`docs/PROGRAM_TRACK_MASTER_SEQUENCE.md`](../docs/PROGRAM_TRACK_MASTER_SEQUENCE.md).
The target queue is in
[`colab/NEXT_COLAB_SEQUENCE.md`](NEXT_COLAB_SEQUENCE.md).

## Current Front-Of-Queue Action

The active blocker is re-entry architecture repair. Stage 1 showed the current
recovered recurrent checkpoint has a dead bridge: `bridge_gate=0.0`, bridge
delta RMS `0.0`, and zero bridge projection/bias/gate gradients. Stage 2 found
`entry_rms` loop re-entry normalization safe enough for a tiny repair smoke.
The Stage 3 assessment now requires the current full repair topology:
`entry_rms` re-entry rescaling, the trainable re-entry adapter enabled,
`bridge_gate_active=true`, live/moved adapter evidence, and live/moved bridge
evidence; bridge projection movement alone is not enough if the scalar gate
collapses back near zero, and a legacy bridge-only repair smoke no longer
unlocks Stage 4.

```text
latest reviewer state: stage2_norm / entry_rms_safe_for_smoke
current source summary: outputs/stage5/stage5_reentry_norm_20260625_013527/summary.json
next target: reentry_repair_smoke
```

Run Stage 3 on L4/T4. Do **not** run ARC-mix depth training, GPQA, scale-up,
Phase 2/SVGD, or particle-noise sweeps until the loop-closure path is
gradient-live and the repair smoke passes loop-1 preservation.

For scale probes, set `MODEL_NAME` and optionally
`STAGE5_RECURRENT_LAYER_SPLIT`. The default split is now `auto`, which maps to
the prior 0.5B `6,18` split while remaining valid for larger Qwen layer counts.
Treat `model_viability_probe` and `model_viability_queue` as standing
information-value runs only. A larger Qwen identity/loop-preservation pass does
not unlock Stage 4 or particles; it only tells us whether a later scale move is
technically viable if Phase 1 says 0.5B is capacity-limited.

The Stage 3 target now performs an immediate GPU-runtime preflight. If Colab is
not attached to L4/T4/A100/H100, it stops before repo sync, Drive restoration,
or checkpoint work.
It also requires the passed Stage 2 norm assessment to expose the source
checkpoint. If that lineage is missing, rerun/recover Stage 2 or set
`STAGE5_REENTRY_REPAIR_CHECKPOINT` explicitly for a deliberate artifact test.

## Paste-Anywhere Launcher

Change only `TARGET` as you move through the queue.

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

## Mandatory Readout Pauses

After Stage 3 publishes, run:

```bash
python colab/review_stage5_reentry.py --no_write
```

Continue to Stage 4 only if the recommendation is:

```text
run_bounded_recovery_training_with_reentry_repair
```

After Stage 4 publishes, run `master_sequence_status` and only then benchmark.
Stage 4 now also publishes `post_reentry_health_checks`; if that object is
missing or not `reentry_health_sane`, rerun or inspect Stage 4 instead of
benchmarking.
The intended order is:

```text
reentry_repair_smoke -> reentry_recovery_training
-> debiased_benchmark_suite -> dense_mcq_trace_sft_control
```

The benchmark/control step is the Phase 1 architecture test: recurrent versus
base, then recurrent versus a standard Qwen same-curriculum LoRA control.
The `Phase 1 Gate Review` section from `master_sequence_status` is the
authority for whether to run the dense control, stop for recovery, or review a
real architecture signal before Phase 2.

## Parallel CPU/API Data Work

The current trace collection is enough for a bounded Stage 4 recovery smoke,
but it is not claim-sized. The cheap status target now prints a
claim-sized curriculum scale-up plan. Current default deficits are:

```text
positive rows: 1937
direct rows:   974
deep_narrow:   963
```

This work should run on CPU or a cheap non-GPU runtime while the GPU queue
stays on Phase 0. It prepares data for later Phase 1 depth training; it does
not unlock Stage 4 by itself. Stage 4 still requires Stage 3 to publish a
repair assessment recommending:

```text
run_bounded_recovery_training_with_reentry_repair
```

CPU launch target:

```python
TARGET = "claim_curriculum_scaleup_cpu"
```

Leave provider calls disabled until the provider API secret and model map are
configured. The model map can be supplied as `STAGE5_CURRICULUM_MODEL_MAP_JSON`
or as individual `STAGE5_CURRICULUM_OPUS_MODEL`,
`STAGE5_CURRICULUM_GLM_MODEL`, and
`STAGE5_CURRICULUM_WEAK_REFERENCE_MODEL` values. Use
`STAGE5_CURRICULUM_PROVIDER_LIMIT=2` for the first paid provider smoke.
Each pass writes `curriculum_readiness.json` in the work directory, so after an
interrupted CPU/API run, inspect that file first for pending provider pairs and
the next safe action.

## Explicit Stops

Stop and review if any of these happen:

- Stage 3 bridge or re-entry adapter gradients are not live.
- Stage 3 did not use `entry_rms` or disabled the re-entry adapter.
- Stage 3 bridge projection moves but `bridge_gate` is inactive.
  Use the reviewer-provided bounded retry env, which switches to
  `bridge_proj,reentry,halt`.
- Stage 3 loop-1 preservation regresses.
- Stage 4 validation is not finite or lacks a target-loop gradient.
- Stage 4 post-recovery re-entry health is missing or not
  `reentry_health_sane`.
- The debiased benchmark shows deterministic recurrence still trails base on
  easy/direct rows.

Do not return to particles/SVGD until deterministic recurrence is base
competitive and the breadth diagnostic shows correct-bearing alternatives.
When Phase 2 eventually opens, the breadth launchers resolve the checkpoint
from the passed Phase 1 source-summary chain. Use explicit Phase 2 checkpoint
overrides only for intentional archaeology runs.
