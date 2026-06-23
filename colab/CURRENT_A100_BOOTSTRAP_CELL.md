# Current A100 Bootstrap Cell

Use this from a blank or Drive-backed Colab notebook when you want the shortest
GitHub-backed path. It fetches the maintained plain cell from the private repo,
checks safety markers, and executes it.

Default target is `preflight`, which mounts Drive, checks checkpoint visibility,
runs the A100 go/no-go guard, and disconnects. This is the cheap runtime path.

To generate and publish the cheap direct/deep curriculum gate on a CPU runtime:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] = "programmatic_curriculum_cpu"`
before running the bootstrap cell. This target refuses attached GPU runtimes by
default.

To run a dry safe-continue status check instead:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] = "safe_continue_dry_run"` before
running the bootstrap cell.

To confirm the MCQ option-label/position-bias result on ARC-Challenge before
spending on more training:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] =
"arc_challenge_mcq_debias_confirm"` before running the bootstrap cell. This
target runs the bounded cyclic-permutation MCQ diagnostic, pushes the summary,
and disconnects.

To run the next bounded benchmark pass with the active debiased MCQ scoring
policy:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] = "debiased_benchmark_suite"`
before running the bootstrap cell. This target defaults to ARC-Challenge plus
GPQA-lite with explicit limits, runs `label`, `content_question_only`, and
`cyclic_label_aggregated` scoring, assesses the cyclic aggregate, pushes
summaries, and disconnects.

To run the bounded capability-ladder MCQ scoring probe:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] =
"capability_ladder_mcq_probe"` before running the bootstrap cell. This target
scores a small ARC-Train slice with Qwen 0.5B/1.5B/3B, builds depth-labeled
capability-ladder rows, backs them up to Drive, pushes safe summaries, and
disconnects. It does not produce final reasoning traces; after it lands, use
the planner's CPU trace-job action before recurrent SFT.

To build capability-ladder trace-generation jobs without GPU:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] =
"capability_ladder_trace_jobs_cpu"` before running the bootstrap cell. This
target follows the current capability-ladder probe summary, restores private
scored rows from Drive if needed, builds provider-neutral strong-model trace
jobs, pushes safe summaries, and disconnects.

To run provider/API responses for capability-ladder trace jobs without GPU:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] =
"capability_ladder_trace_responses_cpu"` before running the bootstrap cell.
This target follows the current trace-job summary, requires explicit
`STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER=1` before provider spend,
writes `trace_responses.jsonl`, backs it up to Drive, pushes safe summaries, and
disconnects. Its summary becomes the preferred input to trace collection.

To collect completed trace responses into gated curriculum data without GPU:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] =
"capability_ladder_trace_collect_cpu"` before running the bootstrap cell. This
target follows either the current trace-response summary or the original
trace-job summary plus an explicit `STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_JSONL`,
verifies final answers, builds traced curriculum rows, runs the SFT gate, pushes
safe summaries, and disconnects.

To intentionally spend GPU on the guarded action after the preflight is green,
select an A100/H100 runtime and set:

Set `os.environ["STAGE5_CURRENT_A100_TARGET"] = "safe_continue_execute"` before
running the bootstrap cell. This target enables
`STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE=1`, so if no explicit source
summary is provided it first uses the newest gate-ready traced curriculum or
validated curriculum-SFT summary before falling back to the repository pointer.

To force a specific source summary, set
`os.environ["STAGE5_CURRENT_A100_SOURCE_SUMMARY"] =
"outputs/stage5/<run_id>/summary.json"`. If that variable is not set, the
bootstrap clears older target-specific source overrides so the fetched launcher
can follow `config/stage5_current_source_summary.txt`.

Then run the bootstrap cell:

```python
import base64, json, os, time, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"
BOOTSTRAP_VERSION = "sha_resolved_nested_fetch_v3"

# Safe default: verify Drive/checkpoint visibility on a CPU/cheap runtime.
# Other options:
#   "programmatic_curriculum_cpu" - generate/publish the direct/deep curriculum gate on CPU.
#   "safe_continue_dry_run" - fetch safe-continue but do not spend GPU.
#   "safe_continue_execute" - fetch safe-continue and opt in to the guarded paid action.
#   "arc_challenge_mcq_debias_confirm" - bounded no-training cyclic MCQ confirmation on ARC-Challenge.
#   "debiased_benchmark_suite" - bounded ARC-Challenge/GPQA-lite benchmark with debiased MCQ scoring.
#   "capability_ladder_mcq_probe" - bounded Qwen 0.5B/1.5B/3B ARC MCQ scoring for depth-label data.
#   "capability_ladder_trace_jobs_cpu" - CPU-only trace-job build from latest capability ladder probe.
#   "capability_ladder_trace_responses_cpu" - CPU/network provider responses for trace jobs.
#   "capability_ladder_trace_collect_cpu" - CPU-only trace-response collection into gated SFT data.
#   "direct_preservation_probe" - bounded max_loops=1 base-preservation training probe.
#   "depth_sweep_heldout" - L4/T4 held-out ARC tail loop-depth sweep for routing validation.
TARGET = os.environ.get("STAGE5_CURRENT_A100_TARGET", "preflight")
SOURCE_SUMMARY_OVERRIDE = os.environ.get("STAGE5_CURRENT_A100_SOURCE_SUMMARY", "").strip()

TARGETS = {
    "preflight": {
        "path": "colab/STAGE5_DRIVE_CHECKPOINT_PREFLIGHT_CELL.py",
        "markers": [
            "stage5_drive_checkpoint_preflight",
            "checkpoint_preflight",
            "STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "drive.mount",
            "runtime.unassign",
            "colab/check_stage5_a100_go_no_go.py",
            "colab/run_stage5_next_action.py",
            "next_action_guard",
            "stage5_current_source_summary.txt",
            "Using current source summary pointer",
        ],
        "env": {},
    },
    "safe_continue_dry_run": {
        "path": "colab/STAGE5_SAFE_CONTINUE_CELL.py",
        "markers": [
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "RUN_A100_ACTION",
            "colab/check_stage5_a100_go_no_go.py",
            "colab/run_stage5_next_action.py",
            "Skipping requirements install because no paid action will execute.",
        ],
        "env": {"STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "0"},
    },
    "programmatic_curriculum_cpu": {
        "path": "colab/STAGE5_PROGRAMMATIC_CURRICULUM_CELL.py",
        "markers": [
            "REFUSE_GPU_RUNTIME",
            "ALLOW_GPU_RUNTIME_FOR_CPU_WORK",
            "training/run_programmatic_curriculum_pipeline.py",
            "training/check_curriculum_sft_gate.py",
            "colab/publish_stage5_curriculum_gate.py",
            "REQUIRE_DRIVE_BACKUP_FOR_PUBLISH",
            "PUBLISH_GATE_TO_GITHUB",
            "stage5_current_source_summary",
            "PROGRAMMATIC_CURRICULUM_CELL_VERSION",
            "shutil.which(\"nvidia-smi\")",
            "FileNotFoundError",
            "OSError",
            "Refusing to run CPU-only programmatic curriculum generation",
        ],
        "env": {},
    },
    "safe_continue_execute": {
        "path": "colab/STAGE5_SAFE_CONTINUE_CELL.py",
        "markers": [
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION",
            "STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY",
            "STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE",
            "RUN_A100_ACTION",
            "mount_drive_for_paid_action",
            "tests/test_stage5_routing_repair.py",
            "tests/test_filter_mcq_sft_by_eval.py",
            "tests/test_mcq_debias.py",
            "colab/run_stage5_next_action.py",
        ],
        "env": {
            "STAGE5_SAFE_CONTINUE_RUN_A100_ACTION": "1",
            "STAGE5_SAFE_CONTINUE_PREFER_TRAINING_SOURCE": "1",
        },
    },
    "arc_challenge_mcq_debias_confirm": {
        "path": "colab/STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL.py",
        "markers": [
            "STAGE5_ARC_CHALLENGE_MCQ_DEBIAS_CELL_VERSION",
            "ARC-Challenge",
            "STAGE5_MCQ_DEBIAS_QUIET_EVAL",
            "STAGE5_MCQ_DEBIAS_RESUME_EXISTING",
            "STAGE5_MCQ_DEBIAS_PUSH",
            "colab/run_stage5_mcq_debias_diagnostic.py",
            "colab/assess_stage5_mcq_debias_pair.py",
            "colab/apply_stage5_mcq_scoring_policy.py",
            "tests/test_mcq_debias.py",
            "tests/test_stage5_next_plan.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "debiased_benchmark_suite": {
        "path": "colab/STAGE5_DEBIASED_BENCHMARK_SUITE_CELL.py",
        "markers": [
            "STAGE5_DEBIASED_BENCHMARK_SUITE_CELL_VERSION",
            "STAGE5_BENCHMARK_SCORE_TARGETS",
            "label,content_question_only,cyclic_label_aggregated",
            "cyclic_label_aggregated",
            "permutation_mean",
            "STAGE5_DEBIASED_BENCHMARKS",
            "arc_challenge,gpqa_lite",
            "colab/run_stage5_benchmark_suite.py",
            "colab/assess_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_stage5_benchmark_assessment.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_mcq_probe": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_MCQ_PROBE_CELL_VERSION",
            "capability_ladder_mcq_probe",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "STAGE5_CAPABILITY_LADDER_ARC_LIMIT",
            "content_question_only",
            "colab/run_stage5_capability_ladder_mcq_probe.py",
            "tests/test_stage5_capability_ladder_mcq_probe.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_trace_jobs_cpu": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_JOBS_CELL_VERSION",
            "capability_ladder_trace_jobs_cpu",
            "STAGE5_CAPABILITY_LADDER_TRACE_ALLOW_GPU",
            "colab/run_stage5_capability_ladder_trace_jobs.py",
            "training/build_capability_ladder_trace_jobs.py",
            "tests/test_capability_ladder_trace_jobs.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_trace_responses_cpu": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSES_CELL_VERSION",
            "capability_ladder_trace_responses_cpu",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_RUN_PROVIDER",
            "STAGE5_CAPABILITY_LADDER_TRACE_RESPONSE_ALLOW_GPU",
            "colab/run_stage5_capability_ladder_trace_responses.py",
            "training/run_curriculum_job_responses.py",
            "tests/test_stage5_capability_ladder_trace_responses_runner.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "capability_ladder_trace_collect_cpu": {
        "path": "colab/STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL.py",
        "markers": [
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_CELL_VERSION",
            "capability_ladder_trace_collect_cpu",
            "STAGE5_CAPABILITY_LADDER_TRACE_COLLECT_ALLOW_GPU",
            "colab/run_stage5_capability_ladder_trace_collect.py",
            "training/collect_capability_ladder_trace_outputs.py",
            "tests/test_stage5_capability_ladder_trace_collect_runner.py",
            "runtime.unassign",
        ],
        "env": {},
    },
    "direct_preservation_probe": {
        "path": "colab/STAGE5_DIRECT_PRESERVATION_PROBE_CELL.py",
        "markers": [
            "STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION",
            "direct_preservation_probe",
            "STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY",
            "STAGE5_DIRECT_PRESERVE_MAX_STEPS",
            "colab/run_stage5_direct_preservation_probe.py",
            "stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation",
            "runtime.unassign",
        ],
        "env": {},
    },
    "depth_sweep_heldout": {
        "path": "colab/STAGE5_DEPTH_SWEEP_BENCHMARK_CELL.py",
        "markers": [
            "STAGE5_DEPTH_SWEEP_BENCHMARK_CELL_VERSION",
            "STAGE5_DEPTH_SWEEP_DRIVE_BACKUP",
            "STAGE5_DEPTH_SWEEP_LOOPS",
            "STAGE5_BENCHMARK_ARC_EASY_OFFSET",
            "STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET",
            "eval/analyze_depth_sweep.py",
            "stage5_depth_sweep_arc_loop1234",
            "Drive backup disabled; using GitHub as primary artifact store.",
        ],
        "env": {
            "STAGE5_DEPTH_SWEEP_LOOPS": "1,2,3",
            "STAGE5_DEPTH_SWEEP_ARC_EASY_OFFSET": "256",
            "STAGE5_DEPTH_SWEEP_ARC_EASY_LIMIT": "full",
            "STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_OFFSET": "256",
            "STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_LIMIT": "full",
            "STAGE5_DEPTH_SWEEP_DISCONNECT": "0",
            "STAGE5_DEPTH_SWEEP_DRIVE_BACKUP": "0",
        },
    },
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

if TARGET not in TARGETS:
    raise AssertionError(f"Unknown TARGET={TARGET!r}; expected one of {sorted(TARGETS)}")

GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."


def github_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


ref_payload = github_json(
    f"https://api.github.com/repos/{REPO}/git/ref/heads/{REF}?cache_bust={int(time.time())}"
)
RESOLVED_REF = ((ref_payload.get("object") or {}).get("sha") or REF).strip()

selected = TARGETS[TARGET]
if SOURCE_SUMMARY_OVERRIDE:
    os.environ["STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
    os.environ["STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY"] = SOURCE_SUMMARY_OVERRIDE
else:
    # Avoid accidentally pinning a new session to an old target-specific source
    # summary. The safe-continue launcher will follow config/stage5_current_source_summary.txt.
    os.environ.pop("STAGE5_SAFE_CONTINUE_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DRIVE_PREFLIGHT_SOURCE_SUMMARY", None)
    os.environ.pop("STAGE5_DIRECT_PRESERVE_SOURCE_SUMMARY", None)
for key, value in selected["env"].items():
    os.environ[key] = value
if TARGET == "depth_sweep_heldout":
    if os.environ.get("STAGE5_DEPTH_SWEEP_RUN_ID_LOCKED", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
    }:
        os.environ["STAGE5_DEPTH_SWEEP_RUN_ID"] = time.strftime(
            "stage5_depth_sweep_arc_heldout_tail_loop123_%Y%m%d_%H%M%S"
        )
    else:
        os.environ.setdefault(
            "STAGE5_DEPTH_SWEEP_RUN_ID",
            time.strftime("stage5_depth_sweep_arc_heldout_tail_loop123_%Y%m%d_%H%M%S"),
        )
os.environ.setdefault("STAGE5_SAFE_CONTINUE_DISCONNECT", "1")

launcher_path = selected["path"]
url = f"https://api.github.com/repos/{REPO}/contents/{launcher_path}?ref={RESOLVED_REF}&cache_bust={int(time.time())}"
payload = github_json(url)

code = base64.b64decode(payload["content"]).decode("utf-8")
missing = [marker for marker in selected["markers"] if marker not in code]
assert not missing, f"Fetched launcher is missing expected safety markers: {missing}"

print(
    f"bootstrap_version={BOOTSTRAP_VERSION} resolved_ref={RESOLVED_REF} target={TARGET}",
    flush=True,
)
print(f"Fetched {launcher_path} from {REPO}@{REF} ({RESOLVED_REF[:12]}) sha={payload.get('sha')} target={TARGET}", flush=True)
exec(compile(code, launcher_path, "exec"))
```

