# Current GPU Action

## Preferred Launch Path

Shortest path from any trusted Colab notebook:

[`colab/CURRENT_A100_BOOTSTRAP_CELL.md`](CURRENT_A100_BOOTSTRAP_CELL.md)
or
[`colab/CURRENT_A100_BOOTSTRAP_CELL.py`](CURRENT_A100_BOOTSTRAP_CELL.py).

The bootstrap defaults to `preflight`. To intentionally execute a maintained
target, set `STAGE5_CURRENT_A100_TARGET=<target>` before running it on the
appropriate runtime.
The generic guarded planner path remains available as
`STAGE5_CURRENT_A100_TARGET=safe_continue_execute`, but the explicit target
below is preferred for the current evidence state.

## Current Front-of-Queue Action

The current preferred path is **capability-ladder depth-label probing**. The
latest learned-depth recurrent checkpoint passed the broader balanced ARC
assessment:

```text
ARC-Easy content:       recurrent 319/512 vs base 298/512, delta +21
ARC-Easy cyclic:        recurrent 406/512 vs base 406/512, delta 0
ARC-Challenge content:  recurrent 108/299 vs base 98/299, delta +10
ARC-Challenge cyclic:   recurrent 177/299 vs base 177/299, delta 0
```

That is enough to stop re-running the ARC-mix recovery loop and test whether
Qwen model-scale gaps can provide useful depth labels before spending on more
recurrent SFT. The next GPU action is:

```text
STAGE5_CURRENT_A100_TARGET=capability_ladder_mcq_probe
```

This scores a bounded ARC-Train slice with Qwen 0.5B, 1.5B, and 3B, then builds
answer-only capability-ladder rows:

```text
0.5B correct                         -> target loop 1 / direct preservation
0.5B miss, 1.5B correct              -> target loop 2 / medium depth
0.5B + 1.5B miss, 3B correct         -> target loop 3 / deep narrow
```

The maintained bootstrap target now uses:

```text
STAGE5_CAPABILITY_LADDER_ARC_LIMIT=96
STAGE5_CAPABILITY_LADDER_SCORE_MODE=content_question_only
STAGE5_CAPABILITY_LADDER_MODEL_LADDER=qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3
STAGE5_CAPABILITY_LADDER_BACKUP_DRIVE=0
```

Success is not final training quality. Success is enough direct and deep rows
to justify trace enrichment and a bounded recurrent SFT run. If the ladder is
sparse, use the high-memory 7B target or increase the ARC slice before training.

For a high-memory G4/A100/H100 runtime, the stronger optional target is:

```text
STAGE5_CURRENT_A100_TARGET=capability_ladder_7b_trace_chain
```

That adds Qwen 7B as target loop 4 and immediately builds provider-neutral
trace-generation jobs from the scored rows, without provider/API spend by
default.

## Historical ARC-Mix Recovery Result

The earlier CE8 balanced ARC depth curve showed a useful hard-slice depth
signal but a serious easy/content calibration gap:

```text
ARC-Easy cyclic best:       depth 1, recurrent 206/256 vs base 202/256, delta +4
ARC-Easy content best:      depth 1, recurrent 131/256 vs base 146/256, delta -15
ARC-Challenge cyclic best:  depth 2-4, recurrent 153/256 vs base 154/256, delta -1
ARC-Challenge content best: depth 3-4, recurrent 92/256 vs base 87/256, delta +5
```

The next training action should therefore preserve depth 1 on
easy/direct/base-correct rows while routing hard rows toward depth 2-3. Keep
particles/SVGD off. Treat ARC-Easy content-only regression as a stop condition.
The reference artifact is:

```text
outputs/stage5/stage5_ce8_balanced_arc256_depth_curve_summary_20260623/summary.json
```

After the capability-ladder probe lands, the trace-job and traced-SFT targets
below become the next expected sequence if the ladder has enough direct and
deep rows. The older MCQ debias and debiased benchmark targets are retained as
historical/fallback cells.
For a high-memory overnight run before provider trace generation, prefer
`STAGE5_CURRENT_A100_TARGET=capability_ladder_7b_trace_chain`: it runs the
0.5B/1.5B/3B/7B capability-ladder probe and immediately builds trace jobs from
the local scored rows before disconnecting. By default it does not spend on a
teacher/provider API. To deliberately keep the same high-memory runtime moving
through teacher trace generation, trace collection, and bounded Phase 1 SFT,
set `STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_PROVIDER=1`, provide
`STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE` or
`STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON_INLINE`, and leave
`STAGE5_CAPABILITY_LADDER_7B_TRACE_CHAIN_RUN_SFT=1` enabled.
The bootstrap now auto-resumes from
[`config/stage5_current_source_summary.txt`](../config/stage5_current_source_summary.txt)
when that pointer exists and targets an available summary. To force a specific
summary for both preflight and safe-continue, set the single bootstrap override
`STAGE5_CURRENT_A100_SOURCE_SUMMARY=outputs/stage5/<run_id>/summary.json`.
No-argument planner/go/no-go runs also read the same pointer, so they follow
this run card instead of the newest file mtime in `outputs/`.

## Previous Paste-Anywhere ARC-Mix Offset Confirmation Cell

Use this from a live Colab notebook after pulling the latest `main`.

```python
import os
os.environ["STAGE5_CURRENT_A100_TARGET"] = "arc_mix_offset_then_depth_chain"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

If the repo is not already cloned, use the preferred bootstrap loader above;
set the same target before executing the fetched bootstrap. To force
measurement-only behavior, set `STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH=0`.
The chain skips upfront Drive mounting by default; if the checkpoint is not
already local, the restore path will request Drive only at the point it is
actually needed.

## Next Paste-Anywhere ARC-Mix Depth-Routing Probe Cell

Use this only after the offset-256 confirmation is non-negative or after you
deliberately choose to proceed with depth-routing training. It starts from the
current recovered ARC-mix checkpoint, keeps particles/SVGD off, uses ARC-Easy
train rows as direct target-loop-1 rows, uses ARC-Challenge train rows as
deep-narrow target-loop-3 rows, and enables learned loop-control supervision.

```python
import os
os.environ["STAGE5_CURRENT_A100_TARGET"] = "arc_mix_depth_routing_probe"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

The depth probe uses:

```text
STAGE5_ARC_MIX_OPUS_LIMIT=0
STAGE5_ARC_MIX_PROMPT_STYLE=question_only
STAGE5_ARC_MIX_SCORE_TARGET=option_text
STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP=1
STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP=3
STAGE5_ARC_MIX_USE_LEARNED_LOOP_CONTROL=1
STAGE5_ARC_MIX_EVAL_USE_LEARNED_LOOP_CONTROL=1
STAGE5_ARC_MIX_LOOP_CONTROL_CE_WEIGHT=0.05
STAGE5_ARC_MIX_HALT_TARGET_NLL_WEIGHT=0.03
```

Success is not "more loop depth everywhere." Success is preserving ARC-Easy
content calibration while retaining or improving the ARC-Challenge content lift
seen in the recovered checkpoint.

Use the safe-continue cell from a normal Drive-backed or blank Colab notebook:

[`colab/STAGE5_SAFE_CONTINUE_CELL.md`](STAGE5_SAFE_CONTINUE_CELL.md)
or the directly fetchable plain cell
[`colab/STAGE5_SAFE_CONTINUE_CELL.py`](STAGE5_SAFE_CONTINUE_CELL.py).

If the runtime has reset or Drive authorization is stale, first run the cheap
preflight cell on CPU or a low-cost runtime:

[`colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.md`](STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.md)
or
[`colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py`](STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py).

That cell mounts Drive, verifies the recovered deterministic Phase 1 checkpoint
is visible, runs the A100 go/no-go guard, runs the same next-action dry-run
wrapper that will execute on GPU, and disconnects. Only attach an A100/H100
after `checkpoint_preflight.available` is `True` and `next_action_guard.allowed`
is `True`.
The checkpoint preflight/restore path searches the project-scoped Drive roots
`recurrent-qwen-svgd-artifacts`, `recurrent-qwen-svgd`, and
`recurrent-qwen-svgd-fresh`, plus any explicit `DRIVE_BACKUP_DIR`,
`DRIVE_BACKUP_DIRS`, or `STAGE5_DRIVE_BACKUP_DIR`. It recognizes both
`<run_id>/run_dir/phase1/phase1_step_125.pt` style backups and preserved repo
paths such as `outputs/stage5/<run_id>/phase1/phase1_step_125.pt`.

Keep the runtime disconnected while editing cells. Use CPU for trace collection.
Use an A100/H100 only after the planner/go-no-go guard reports that the traced
curriculum gate is green and the next action is the bounded recurrent SFT.

## Previous Paste-Anywhere ARC-Challenge Cell

Use this in the live Colab notebook when the repo is not already freshly cloned.
It resolves `main`, fetches the maintained bootstrap cell, and selects the
current bounded ARC-Challenge MCQ debias target.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "arc_challenge_mcq_debias_confirm"

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "arc_challenge_mcq_debias_confirm" in code, "Fetched bootstrap without ARC-Challenge target."
assert "colab/apply_stage5_mcq_scoring_policy.py" in code, "Fetched stale bootstrap; rerun after GitHub refresh."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Next Paste-Anywhere Debiased Benchmark Cell

Use this after the ARC-Challenge debias cell has produced a `policy_summary:`
line or after `config/stage5_current_source_summary.txt` points at an active
`stage5_mcq_scoring_policy` artifact.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "debiased_benchmark_suite"

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "debiased_benchmark_suite" in code, "Fetched stale bootstrap without debiased benchmark target."
assert "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py" in code, "Fetched stale bootstrap launcher."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Current Paste-Anywhere Capability-Ladder Probe Cell

Use this now to test whether Qwen model scale gives a useful depth-label
ladder before training.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_mcq_probe"
os.environ.setdefault("STAGE5_CAPABILITY_LADDER_ARC_LIMIT", "96")
os.environ.setdefault("STAGE5_CAPABILITY_LADDER_SCORE_MODE", "content_question_only")
os.environ.setdefault("STAGE5_CAPABILITY_LADDER_MODEL_LADDER", "qwen_0_5b:1,qwen_1_5b:2,qwen_3b:3")

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "capability_ladder_mcq_probe" in code, "Fetched stale bootstrap without capability-ladder target."
assert "STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL.py" in code, "Fetched stale capability-ladder launcher."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Next Paste-Anywhere Capability-Ladder Trace Jobs Cell

Use this after the capability-ladder MCQ probe has landed. Prefer a CPU runtime;
this step restores scored rows from Drive if needed and builds strong-model
trace-generation jobs without model inference.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_trace_jobs_cpu"
os.environ.setdefault("STAGE5_CAPABILITY_LADDER_TRACE_MODELS", "opus-strong,glm-strong")

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "capability_ladder_trace_jobs_cpu" in code, "Fetched stale bootstrap without trace-jobs target."
assert "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL.py" in code, "Fetched stale trace-jobs launcher."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Next Paste-Anywhere Capability-Ladder Trace Responses Cell

Prefer the combined response+collection cell below when you want one unattended
CPU/network run from trace jobs to gated traced curriculum.

## Next Paste-Anywhere Local-HF Trace Response+Collection Cell

Use this on a high-memory GPU runtime when trace jobs are ready and you want a
no-provider-spend baseline. It runs `Qwen/Qwen2.5-7B-Instruct` locally against
the current trace jobs, accepts only answer-verified traces, pushes summaries to
GitHub, and disconnects. Default limit is 32 jobs.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_local_hf_trace_collect"

# Optional: increase after the first pilot is healthy.
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT"] = "64"
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME"] = "Qwen/Qwen2.5-7B-Instruct"

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "capability_ladder_local_hf_trace_collect" in code, "Fetched stale bootstrap without local-HF target."
assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_HF_MODEL_NAME" in code, "Fetched stale local-HF env."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Next Paste-Anywhere Capability-Ladder Trace Response+Collection Cell

Use this after trace jobs are ready. Prefer CPU. It will not call a provider
unless `STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER=1` is set. For
OpenAI-compatible providers, set either
`STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE` or
`STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON`.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_trace_response_collect_cpu"

# Required before actual provider/API spend:
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER"] = "1"
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKEND"] = "openai_compatible"
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_API_KEY_ENV"] = "OPENROUTER_API_KEY"
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE"] = "anthropic/claude-3.5-sonnet"
# Or map the two logical teacher jobs separately:
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON_INLINE"] = (
#     '{"opus-strong":"anthropic/claude-3.5-sonnet","glm-strong":"google/gemini-2.5-pro"}'
# )

# Optional bounded pilot:
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT"] = "16"

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "capability_ladder_trace_response_collect_cpu" in code, "Fetched stale bootstrap without combined target."
assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_COLLECT_CELL.py" in code, "Fetched stale response+collection launcher."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Separate Paste-Anywhere Capability-Ladder Trace Responses Cell

Use this after trace jobs are ready when you intentionally want response
generation and collection as separate steps. Prefer CPU. It will not call a provider
unless `STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER=1` is set. For
OpenAI-compatible providers, set either
`STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE` or
`STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON`.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_trace_responses_cpu"

# Required before actual provider/API spend:
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER"] = "1"
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_BACKEND"] = "openai_compatible"
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_API_KEY_ENV"] = "OPENAI_API_KEY"
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_OVERRIDE"] = "gpt-5-mini"
# Or set STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_MODEL_MAP_JSON_INLINE to a JSON object.

# Optional bounded pilot:
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_LIMIT"] = "16"

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "capability_ladder_trace_responses_cpu" in code, "Fetched stale bootstrap without trace-responses target."
assert "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL.py" in code, "Fetched stale trace-responses launcher."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Current Paste-Anywhere Capability-Ladder Trace Collection Cell

Use this after provider responses have been written. By default the collector
understands the trace-response summary written by the response target. It also
looks for `trace_responses.jsonl`, `capability_ladder_trace_responses.jsonl`, or
`responses.jsonl` beside the trace-job summary or in the Drive backup. You can
still set `STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_JSONL` explicitly.

Target breadcrumb: `STAGE5_CURRENT_A100_TARGET=capability_ladder_trace_collect_cpu`.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_trace_collect_cpu"
# Optional explicit response path:
# os.environ["STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_JSONL"] = "outputs/stage5/<trace_job_run>/trace_responses.jsonl"

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={time.time_ns()}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={time.time_ns()}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "capability_ladder_trace_collect_cpu" in code, "Fetched stale bootstrap without trace-collection target."
assert "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py" in code, "Fetched stale trace-collection launcher."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

## Current Paste-Anywhere Traced Capability-Ladder SFT Cell

Use this only after a gate-ready capability-ladder trace collection exists.
This is the depth-ladder training branch, not the immediate ARC-mix offset
confirmation.

Target breadcrumb: `STAGE5_CURRENT_A100_TARGET=traced_capability_ladder_sft`.

```python
import os
os.environ["STAGE5_CURRENT_A100_TARGET"] = "traced_capability_ladder_sft"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

Set:

```python
RUN_A100_ACTION = True
```

only when you intentionally want the guarded action to execute. The cell pulls
latest GitHub, authenticates GitHub/Hugging Face, mounts Drive when needed,
runs the go/no-go guard, executes one allowlisted action, backs up/commits safe
artifacts, and disconnects by default.

If the go/no-go output is `routing_checkpoint_missing_no_go`, do not keep the
GPU session alive. Disconnect and run the Drive/checkpoint preflight first.

## Fetch Doctor

If Colab appears to fetch stale GitHub contents, run this diagnostic-only cell
on CPU before launching anything. It resolves `main` to an immutable commit and
prints the exact bootstrap and programmatic launcher blobs that Colab sees.

```python
import json, os, time, urllib.request
from google.colab import userdata

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={int(time.time())}"
)
resolved_ref = ref_payload["object"]["sha"]

for path in [
    "colab/CURRENT_A100_BOOTSTRAP_CELL.py",
    "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py",
]:
    payload = gh_json(
        "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
        f"contents/{path}?ref={resolved_ref}&cache_bust={int(time.time())}"
    )
    print(path, "blob=", payload.get("sha"), "commit=", resolved_ref)
```

## Previous Curriculum Sequence

This sequence is retained for the direct/deep curriculum path after the MCQ
debias gate settles. It is not the current paid action while the ARC-Challenge
debias confirmation is pending.

### Step 1: CPU-Only Programmatic Curriculum Gate

Run this on a CPU Colab runtime. If the repo is already cloned in the notebook:
Set `STAGE5_CURRENT_A100_TARGET=programmatic_curriculum_cpu`. This target
refuses attached GPU runtimes by default.

```python
import os
os.environ["STAGE5_CURRENT_A100_TARGET"] = "programmatic_curriculum_cpu"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

If the repo is not already cloned, use this paste-anywhere loader instead:

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "programmatic_curriculum_cpu"

def colab_secret(*names):
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

token = colab_secret("GH_TOKEN", "GITHUB_TOKEN")
assert token, "Add GH_TOKEN or GITHUB_TOKEN to Colab secrets."

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Cache-Control": "no-cache",
}

def gh_json(url):
    req = urllib.request.Request(url, headers=headers)
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

ref_payload = gh_json(
    f"https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/git/ref/heads/main?cache_bust={int(time.time())}"
)
resolved_ref = ref_payload["object"]["sha"]
payload = gh_json(
    "https://api.github.com/repos/mshapiro123/recurrent-qwen-svgd/"
    f"contents/colab/CURRENT_A100_BOOTSTRAP_CELL.py?ref={resolved_ref}&cache_bust={int(time.time())}"
)
code = base64.b64decode(payload["content"]).decode("utf-8")
print("Fetched bootstrap sha:", payload.get("sha"), "commit:", resolved_ref[:12])
assert "RESOLVED_REF" in code, "Fetched stale bootstrap; rerun this cell."
assert "sha_resolved_nested_fetch_v3" in code, "Fetched stale bootstrap version; rerun this cell."
exec(compile(code, "colab/CURRENT_A100_BOOTSTRAP_CELL.py", "exec"))
```

This step generates the 2000-row direct/deep-narrow constructed curriculum,
exports only verified `positive_*` SFT rows, backs the artifact directory up to
Drive, publishes the small green gate summary to GitHub, and updates the current
source pointer.

### Step 2: Guarded A100 Curriculum SFT

Only after Step 1 succeeds and Drive is authorized, attach A100/H100 and run.
If the repo is already cloned:

```python
import os
os.environ["STAGE5_CURRENT_A100_TARGET"] = "safe_continue_execute"
exec(open("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read())
```

If the repo is not already cloned, rerun the paste-anywhere loader above with
`safe_continue_execute` instead of `programmatic_curriculum_cpu`.

The safe-continue cell must show a green curriculum input preflight before
running `colab/run_stage5_curriculum_sft.py`. If it reports missing curriculum
input artifacts, disconnect the GPU and rerun Step 1 or reauthorize Drive.

### Step 3: Routing Diagnostic Only After SFT

If SFT completes, the planner may recommend
`colab/run_stage5_routing_diagnostic.py`. Run that before any broader
ARC/GPQA benchmark expansion. The diagnostic checks whether the new
direct/deep curriculum actually moved base-confident direct rows toward
shallower loop use while preserving deep-narrow behavior. Only after that
passes should the planner advance to larger benchmark confirmation.

## Previous Routing Diagnostic

The current source of truth is the completed routing diagnostic:

```text
outputs/stage5/stage5_routing_diagnostic_20260622_041706/summary.json
```

Key result:

```text
status = needs_direct_halting_repair
next_action = Train Phase 1 direct-mode recovery with base-logit distillation and shallow halt supervision.
ARC-Easy direct delta = -2, mean direct loops = 2.58, mean direct margin delta = -2.49
ARC-Challenge direct delta = -3, mean direct loops = 2.62, mean direct margin delta = -2.02
ARC-Challenge conceptual delta = +2
```

This still explains why the next work is direct/deep deterministic recovery:
do **not** run GPQA, Phase 2/SVGD, wide-particle training, or scale-up yet. The
model is still harming base-confident direct rows and over-looping on them.
However, the preferred recovery ingredient is now the generated
direct/deep-narrow curriculum gate above, not another ad hoc particle or broad
benchmark run.

## Previous Direct-Repair Experiment

The older direct-mode repair path remains available if the curriculum SFT gate
fails or if we intentionally compare it as an ablation:

```bash
python colab/run_stage5_routing_repair.py
```

The selected profile for `needs_direct_halting_repair`:

```text
repair_mode=direct_halting
STAGE5_ARC_MIX_ARC_EASY_REPEAT=8
STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT=1
STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP=1
STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP=2
STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE=direct
STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE=deep_narrow_probe
STAGE5_ARC_MIX_EVAL_CONFIG=ARC-Easy
STAGE5_ARC_MIX_ARMS=arc_mix_response_w02_lr2e6
```

The proxy eval is ARC-Easy for this direct-halting repair. That is deliberate:
the source diagnostic showed the model over-looping and regressing on
base-confident direct rows, so the bounded repair must clear the direct/Easy
proxy before a larger ARC-Easy/ARC-Challenge confirmation benchmark.
The A100 go/no-go summary should show
`routing_repair_profile.expected_arc_eval_config = "ARC-Easy"` before you
allow the paid action to run.

The runner restores the recovered deterministic Phase 1 checkpoint from Drive
if needed, delegates to `colab/run_stage5_balanced_arc_mix_gate.py`, keeps
particles/SVGD off, and writes:

```text
outputs/stage5/<run_id>/repair_run/summary.json
outputs/stage5/<run_id>/summary.json
outputs/stage5/<run_id>/summary.md
```

## How To Interpret The Result

The repair summary wraps the child ARC-mix gate status:

| Status | Meaning | Next action |
|---|---|---|
| `repair_proxy_lift` | Direct repair lifted proxy accuracy and passed calibration. | Run the full balanced ARC confirmation. |
| `repair_proxy_matches_base` | Direct repair restored proxy to base without calibration warning. | Run the full balanced ARC confirmation. |
| `repair_proxy_lift_calibration_warning` | Accuracy lifted but base calibration degraded. | Stop and tighten preservation/distillation. |
| `repair_proxy_matches_base_calibration_warning` | Accuracy matched base but calibration degraded. | Stop and tighten preservation/distillation. |
| `repair_no_proxy_lift` | Repair did not improve the proxy. | Stop and revise direct-loop supervision. |

Do not proceed to width/particles until direct rows stop regressing. This is
the calibration floor for the wider depth/width curriculum.

## Optional Constructed-Curriculum Lever

If the routing repair still reports direct/deep calibration problems, the repo
also contains a bounded constructed-curriculum runner:

```bash
STAGE5_PROGRAMMATIC_SOURCE_SUMMARY=outputs/stage5/<source_run>/summary.json \
python colab/run_stage5_programmatic_depth_repair.py
```

That runner generates verified direct/deep-narrow arithmetic-chain rows on CPU,
exports only `positive_direct` and `positive_depth` traces, performs one short
Phase 1 continuation with base-logit distillation, and evaluates on a held-out
constructed split. Treat it as a calibration ingredient only. Any checkpoint it
produces still needs the ARC routing/benchmark gate before particles, SVGD,
or wider data should resume.
