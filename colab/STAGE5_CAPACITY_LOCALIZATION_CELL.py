import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.colab import runtime, userdata


STAGE5_CAPACITY_LOCALIZATION_CELL_VERSION = "capacity_localization_v1"
# The child recovery launcher writes fixed_tail_damper_depth_readout; this
# parent summarizes it across capacity settings.
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")


def secret(*names: str) -> str:
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return str(value)
    return ""


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
os.environ["GH_TOKEN"] = GH_TOKEN
os.environ["GITHUB_TOKEN"] = GH_TOKEN
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = redact(" ".join(map(str, cmd)))
    print(f"$ {printable}", flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(list(map(str, cmd)), process.wait(), stdout, None)
    if check and proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout)
    return proc


def ensure_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def parse_ranks(value: str) -> list[int]:
    from colab.capacity_localization import parse_int_csv

    return parse_int_csv(value, default=[64])


def publish_summary(summary_path: Path) -> None:
    from colab.stage5_publish_utils import publishable_artifact_paths

    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=False)
    run_dir = summary_path.parent
    pointer = ROOT / "config" / "stage5_current_capacity_localization_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(summary_path.relative_to(ROOT).as_posix() + "\n", encoding="utf-8")

    for path in [*publishable_artifact_paths(run_dir), pointer]:
        run(["git", "add", "-f", path.relative_to(ROOT).as_posix()], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No Stage 5 capacity localization outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 capacity localization {summary_path.parent.name} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode != 0:
        print("Initial capacity localization push failed; attempting one fast rebase and retry.", flush=True)
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


def main() -> None:
    print(f"STAGE5_CAPACITY_LOCALIZATION_CELL_VERSION={STAGE5_CAPACITY_LOCALIZATION_CELL_VERSION}", flush=True)
    ensure_repo()
    run(["nvidia-smi"], cwd=Path("/content"))
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_capacity_localization.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_target_markers_exist_in_launcher_files",
        ]
    )

    from colab.capacity_localization import write_capacity_localization_summary

    ranks = parse_ranks(os.environ.get("STAGE5_CAPACITY_LOCALIZATION_RANKS", "64"))
    parent_run_id = os.environ.get("STAGE5_CAPACITY_LOCALIZATION_RUN_ID") or time.strftime(
        "stage5_capacity_localization_%Y%m%d_%H%M%S"
    )
    baseline_summaries = [
        item.strip()
        for item in os.environ.get(
            "STAGE5_CAPACITY_LOCALIZATION_BASELINE_SUMMARIES",
            "outputs/stage5/stage5_reentry_recovery_20260627_190155/summary.json",
        ).split(",")
        if item.strip()
    ]
    tail_damper_summary = os.environ.get(
        "STAGE5_CAPACITY_LOCALIZATION_TAIL_DAMPER_SOURCE_SUMMARY",
        "outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/summary.json",
    )
    trace_summary = os.environ.get(
        "STAGE5_CAPACITY_LOCALIZATION_TRACE_SOURCE_SUMMARY",
        "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json",
    )
    steps = os.environ.get("STAGE5_CAPACITY_LOCALIZATION_STEPS", "100")
    lr = os.environ.get("STAGE5_CAPACITY_LOCALIZATION_LR", "5e-6")

    print(
        json.dumps(
            {
                "capacity_localization_run_id": parent_run_id,
                "ranks": ranks,
                "baseline_summaries": baseline_summaries,
                "tail_damper_summary": tail_damper_summary,
                "trace_summary": trace_summary,
                "steps": steps,
                "lr": lr,
                "trainable_parameter_ledger": "rank-scaled LoRA trainables; stored model size unchanged",
            },
            indent=2,
        ),
        flush=True,
    )

    result_summaries: list[str] = []
    for rank in ranks:
        alpha = int(os.environ.get(f"STAGE5_CAPACITY_LOCALIZATION_ALPHA_RANK{rank}", str(2 * rank)))
        rank_run_id = f"{parent_run_id}_lora{rank}"
        env = os.environ.copy()
        env.update(
            {
                "STAGE5_REENTRY_RECOVERY_RUN_ID": rank_run_id,
                "STAGE5_REENTRY_RECOVERY_CHILD_RUN_ID": f"{rank_run_id}_curriculum_sft",
                "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_SOURCE_SUMMARY": tail_damper_summary,
                "STAGE5_REENTRY_RECOVERY_REENTRY_TAIL_DAMPER_STRENGTH": "1.0",
                "STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY": trace_summary,
                "STAGE5_REENTRY_RECOVERY_STEPS": steps,
                "STAGE5_REENTRY_RECOVERY_LR": lr,
                "STAGE5_REENTRY_RECOVERY_OPTIMIZER_MODULES": "bridge,reentry,halt,lora",
                "STAGE5_REENTRY_RECOVERY_LORA_RANK": str(rank),
                "STAGE5_REENTRY_RECOVERY_LORA_ALPHA": str(alpha),
                "STAGE5_REENTRY_RECOVERY_REENTRY_RESCALE_MODE": "entry_rms",
                "STAGE5_REENTRY_RECOVERY_REENTRY_ADAPTER_MODE": "spectral",
                "STAGE5_REENTRY_RECOVERY_REQUIRE_TARGET_LOOP_GRADIENT": "0",
                "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_STRENGTHS": "0,1.0",
                "STAGE5_REENTRY_RECOVERY_TAIL_DAMPER_READOUT_LIMIT": os.environ.get(
                    "STAGE5_CAPACITY_LOCALIZATION_TAIL_DAMPER_READOUT_LIMIT",
                    "512",
                ),
                "STAGE5_REENTRY_RECOVERY_DISCONNECT": "0",
                "STAGE5_REENTRY_RECOVERY_COMMIT_CHECKPOINTS": "0",
            }
        )
        if HF_TOKEN:
            env["HF_TOKEN"] = HF_TOKEN
            env["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
        env["GH_TOKEN"] = GH_TOKEN
        env["GITHUB_TOKEN"] = GH_TOKEN
        print(f"\n===== capacity localization rank={rank} alpha={alpha} =====", flush=True)
        run([sys.executable, "colab/STAGE5_REENTRY_RECOVERY_TRAINING_CELL.py"], env=env)
        summary_path = ROOT / "outputs" / "stage5" / rank_run_id / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        result_summaries.append(summary_path.relative_to(ROOT).as_posix())

    output_dir = ROOT / "outputs" / "stage5" / parent_run_id
    summary_path = write_capacity_localization_summary(
        root=ROOT,
        run_id=parent_run_id,
        output_dir=output_dir,
        baseline_summaries=baseline_summaries,
        result_summaries=result_summaries,
        target_ranks=ranks,
        baseline_rank=int(os.environ.get("STAGE5_CAPACITY_LOCALIZATION_BASELINE_RANK", "32")),
    )
    print(summary_path.with_suffix(".md").read_text(encoding="utf-8"), flush=True)
    publish_summary(summary_path)

    if os.environ.get("STAGE5_CAPACITY_LOCALIZATION_DISCONNECT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        print("Disconnecting Colab runtime after capacity localization.", flush=True)
        runtime.unassign()


try:
    main()
except Exception:
    print("Leaving Colab runtime connected: capacity localization errored", flush=True)
    raise
