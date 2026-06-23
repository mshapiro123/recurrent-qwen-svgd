# Current GPU Action

## Preferred Launch Path

Shortest path from any trusted Colab notebook:

[`colab/CURRENT_A100_BOOTSTRAP_CELL.md`](CURRENT_A100_BOOTSTRAP_CELL.md)
or
[`colab/CURRENT_A100_BOOTSTRAP_CELL.py`](CURRENT_A100_BOOTSTRAP_CELL.py).

The bootstrap defaults to `preflight`. To intentionally execute the guarded
paid action after preflight is green, set
`STAGE5_CURRENT_A100_TARGET=safe_continue_execute` before running it on an
A100/H100 runtime.
The current preferred path is **trace collection before more training**. The
next action is CPU-only unless the trace-collection gate has already passed.
Use `STAGE5_CURRENT_A100_TARGET=capability_ladder_trace_collect_cpu` on a CPU
runtime after provider responses have been written. This verifies final answers,
collects traced scored rows, builds the traced capability-ladder curriculum,
runs the SFT gate, pushes safe summaries, and disconnects. If the response
target updated
`config/stage5_current_source_summary.txt` to a
`stage5_capability_ladder_trace_responses` summary, the collector follows that
summary back to the trace-job metadata and uses its recorded response JSONL
directly.
If the committed pointer is stale or still references an unrelated Stage 5
summary, the collector searches local outputs and Drive for the newest usable
trace-response/job summary before failing. An explicit
`STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_SOURCE_SUMMARY` override remains strict
and must point to a trace-response or trace-job summary.
When that trace-collection summary reports `trace_curriculum_gate_ready`, the
use `STAGE5_CURRENT_A100_TARGET=traced_capability_ladder_sft` on the attached
GPU runtime. That target follows the latest gate-ready traced collection,
derives `STAGE5_CURRICULUM_WORK_DIR`, `STAGE5_CURRICULUM_SUMMARY_JSON`, row
gates, mode gates, max loops, Drive backup root, and bounded Phase 1 steps from
the summary, then runs `colab/run_stage5_curriculum_sft.py`. The older
`safe_continue_execute` target can still route through the planner, but the
direct traced-SFT target is the cleaner current path. The default floor is 16
answer-verified traced rows before it will spend GPU on this SFT path; smaller
collections remain CPU-side inspection/collect-more tasks unless explicitly
overridden for a tiny smoke run.
The older MCQ debias, debiased benchmark, capability-ladder probe, trace-job,
and trace-response targets are retained below as historical/fallback cells, but
they are no longer the current front-of-queue action unless the source pointer
is deliberately moved back to one of those stages.
The bootstrap now auto-resumes from
[`config/stage5_current_source_summary.txt`](../config/stage5_current_source_summary.txt)
when that pointer exists and targets an available summary. To force a specific
summary for both preflight and safe-continue, set the single bootstrap override
`STAGE5_CURRENT_A100_SOURCE_SUMMARY=outputs/stage5/<run_id>/summary.json`.
No-argument planner/go/no-go runs also read the same pointer, so they follow
this run card instead of the newest file mtime in `outputs/`.

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

## Next Paste-Anywhere Capability-Ladder Probe Cell

Use this after the debiased benchmark suite has landed and we want to test
whether Qwen model scale gives a useful depth-label ladder before training.

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

os.environ["STAGE5_CURRENT_A100_TARGET"] = "capability_ladder_mcq_probe"
os.environ.setdefault("STAGE5_CAPABILITY_LADDER_ARC_LIMIT", "48")
os.environ.setdefault("STAGE5_CAPABILITY_LADDER_SCORE_MODE", "content_question_only")

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
