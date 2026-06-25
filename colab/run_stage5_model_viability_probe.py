"""No-training recurrent viability probe for Qwen model scales.

This runner is intentionally bounded. It asks whether a Qwen model at a given
scale can pass the recurrent surgery gates before any SFT spend:

1. strict one-pass identity;
2. base versus untrained recurrent loop-1 MCQ preservation;
3. tiny loop-depth sweep for early hard/easy allocation signal.

The default model is Qwen 1.5B, but all model-specific values are environment
variables so the same runner can probe 3B or larger Qwen checkpoints.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_benchmark_suite import two_sided_sign_p_value  # noqa: E402


MODEL_NAME = os.environ.get("STAGE5_MODEL_PROBE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_LABEL = os.environ.get("STAGE5_MODEL_PROBE_MODEL_LABEL", "")
LAYER_SPLIT = os.environ.get("STAGE5_MODEL_PROBE_LAYER_SPLIT", "auto")
IDENTITY_DTYPE = os.environ.get("STAGE5_MODEL_PROBE_IDENTITY_DTYPE", "float32")
EVAL_DTYPE = os.environ.get("STAGE5_MODEL_PROBE_EVAL_DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("STAGE5_MODEL_PROBE_ADAPTER_DTYPE", "float32")
IDENTITY_ATTN = os.environ.get("STAGE5_MODEL_PROBE_IDENTITY_ATTN", "eager")
EVAL_ATTN = os.environ.get("STAGE5_MODEL_PROBE_EVAL_ATTN", "default")
DEVICE = os.environ.get("STAGE5_MODEL_PROBE_DEVICE", "cuda")
BENCHMARKS = os.environ.get("STAGE5_MODEL_PROBE_BENCHMARKS", "arc_easy,arc_challenge")
SCORE_TARGETS = os.environ.get("STAGE5_MODEL_PROBE_SCORE_TARGETS", "label,content_question_only")
AGGREGATES = os.environ.get("STAGE5_MODEL_PROBE_AGGREGATES", "mean")
LOOPS = os.environ.get("STAGE5_MODEL_PROBE_LOOPS", "1,2")
ARC_EASY_LIMIT = os.environ.get("STAGE5_MODEL_PROBE_ARC_EASY_LIMIT", "32")
ARC_CHALLENGE_LIMIT = os.environ.get("STAGE5_MODEL_PROBE_ARC_CHALLENGE_LIMIT", "32")
ARC_EASY_OFFSET = int(os.environ.get("STAGE5_MODEL_PROBE_ARC_EASY_OFFSET", "0"))
ARC_CHALLENGE_OFFSET = int(os.environ.get("STAGE5_MODEL_PROBE_ARC_CHALLENGE_OFFSET", "0"))
PUSH_RESULTS = os.environ.get("STAGE5_MODEL_PROBE_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}
RUN_ID = os.environ.get("STAGE5_MODEL_PROBE_RUN_ID") or time.strftime(
    f"stage5_model_viability_{re.sub(r'[^a-zA-Z0-9]+', '_', MODEL_NAME).strip('_').lower()}_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
PRIVATE_DATA_DIR = ROOT / "data" / "stage5_model_viability" / RUN_ID


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    parsed = [int(part) for part in parse_csv(value)]
    if not parsed:
        raise ValueError("Expected at least one loop in STAGE5_MODEL_PROBE_LOOPS")
    return parsed


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run(cmd: list[str], *, log_name: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    printable = " ".join(map(str, cmd))
    print("$", printable, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    (RUN_DIR / log_name).write_text(proc.stdout, encoding="utf-8")
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def prepare_arc(config: str, *, limit: str, offset: int, output: Path) -> Path:
    cmd = [
        sys.executable,
        "eval/prepare_arc_mcq.py",
        "--config",
        config,
        "--split",
        "validation",
        "--seed",
        "0",
        "--output_jsonl",
        str(output),
    ]
    if offset:
        cmd.extend(["--offset", str(offset)])
    if limit.strip().lower() not in {"", "none", "full", "all"}:
        cmd.extend(["--limit", str(int(limit))])
    run(cmd, log_name=f"prepare_{output.stem}.log")
    return output


def benchmark_data() -> dict[str, Path]:
    PRIVATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, Path] = {}
    for benchmark in parse_csv(BENCHMARKS):
        if benchmark == "arc_easy":
            data[benchmark] = prepare_arc(
                "ARC-Easy",
                limit=ARC_EASY_LIMIT,
                offset=ARC_EASY_OFFSET,
                output=PRIVATE_DATA_DIR / "arc_easy_validation.jsonl",
            )
        elif benchmark == "arc_challenge":
            data[benchmark] = prepare_arc(
                "ARC-Challenge",
                limit=ARC_CHALLENGE_LIMIT,
                offset=ARC_CHALLENGE_OFFSET,
                output=PRIVATE_DATA_DIR / "arc_challenge_validation.jsonl",
            )
        else:
            raise ValueError(f"Unsupported benchmark for viability probe: {benchmark}")
    return data


def prompt_style_for_score_target(score_target: str) -> str:
    if score_target == "content_question_only":
        return "question_only"
    return "with_options"


def eval_score_target(score_target: str) -> str:
    if score_target == "content_question_only":
        return "option_text"
    return score_target


def eval_mcq(
    *,
    data_jsonl: Path,
    benchmark: str,
    arm: str,
    score_target: str,
    max_loops: int | None = None,
) -> Path:
    output = RUN_DIR / (
        f"{benchmark}_{arm}_{score_target}.jsonl"
        if max_loops is None
        else f"{benchmark}_{arm}_loop{max_loops}_{score_target}.jsonl"
    )
    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        "eval/eval_mcq.py",
        "--model_name",
        MODEL_NAME,
        "--data_jsonl",
        path_for_cli(data_jsonl),
        "--mode",
        "base" if arm == "base" else "phase1",
        "--prompt_style",
        prompt_style_for_score_target(score_target),
        "--score_target",
        eval_score_target(score_target),
        "--aggregates",
        AGGREGATES,
        "--split",
        LAYER_SPLIT,
        "--dtype",
        EVAL_DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--attn_implementation",
        EVAL_ATTN,
        "--device",
        DEVICE,
        "--seed",
        "0",
        "--output_jsonl",
        path_for_cli(output),
    ]
    if arm != "base":
        assert max_loops is not None
        cmd.extend(
            [
                "--allow_untrained_recurrent",
                "--max_loops",
                str(max_loops),
                "--num_trajectories",
                "1",
                "--include_loop_diagnostics",
            ]
        )
    run(cmd, log_name=f"{output.stem}.log")
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_by_aggregate(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        rows.setdefault(str(row.get("aggregate") or "mean"), []).append(row)
    return rows


def summarize_pair(base_path: Path, recurrent_path: Path) -> dict[str, Any]:
    base_by_agg = rows_by_aggregate(base_path)
    rec_by_agg = rows_by_aggregate(recurrent_path)
    out: dict[str, Any] = {}
    for aggregate in sorted(set(base_by_agg) | set(rec_by_agg)):
        base_rows = {str(row["id"]): row for row in base_by_agg.get(aggregate, [])}
        rec_rows = {str(row["id"]): row for row in rec_by_agg.get(aggregate, [])}
        ids = sorted(set(base_rows) & set(rec_rows))
        wins = 0
        losses = 0
        base_correct = 0
        rec_correct = 0
        expected_loops: list[float] = []
        for row_id in ids:
            base_hit = bool(base_rows[row_id].get("hit"))
            rec_hit = bool(rec_rows[row_id].get("hit"))
            base_correct += int(base_hit)
            rec_correct += int(rec_hit)
            wins += int(rec_hit and not base_hit)
            losses += int(base_hit and not rec_hit)
            loop_diag = rec_rows[row_id].get("loop_diagnostics") or {}
            value = loop_diag.get("mean_expected_loops")
            if isinstance(value, (int, float)):
                expected_loops.append(float(value))
        out[aggregate] = {
            "paired_examples": len(ids),
            "base_correct": base_correct,
            "recurrent_correct": rec_correct,
            "delta": rec_correct - base_correct,
            "wins": wins,
            "losses": losses,
            "ties": len(ids) - wins - losses,
            "sign_test_p": two_sided_sign_p_value(wins, losses),
            "mean_expected_loops": sum(expected_loops) / len(expected_loops) if expected_loops else None,
        }
    return out


def run_identity_gate() -> dict[str, Any]:
    cmd = [
        sys.executable,
        "eval/eval_identity.py",
        "--model_name",
        MODEL_NAME,
        "--split",
        LAYER_SPLIT,
        "--dtype",
        IDENTITY_DTYPE,
        "--attn_implementation",
        IDENTITY_ATTN,
        "--device",
        DEVICE,
        "--threshold",
        "1e-3",
    ]
    proc = run(cmd, log_name="identity_gate.log", check=False)
    stdout = proc.stdout
    max_match = re.search(r"max_abs_diff=([0-9.eE+-]+)", stdout)
    mean_match = re.search(r"mean_abs_diff=([0-9.eE+-]+)", stdout)
    return {
        "returncode": proc.returncode,
        "passed": proc.returncode == 0 and "PASS" in stdout,
        "max_abs_diff": float(max_match.group(1)) if max_match else None,
        "mean_abs_diff": float(mean_match.group(1)) if mean_match else None,
        "dtype": IDENTITY_DTYPE,
        "attn_implementation": IDENTITY_ATTN,
    }


def write_summary(payload: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Model Viability Probe - {RUN_ID}",
        "",
        f"- Model: `{payload['model_name']}`",
        f"- Model label: `{payload['model_label']}`",
        f"- Layer split: `{payload['layer_split']}`",
        f"- Identity passed: `{payload['identity']['passed']}`",
        f"- Identity max abs diff: `{payload['identity']['max_abs_diff']}`",
        f"- Loops: `{payload['loops']}`",
        f"- Score targets: `{payload['score_targets']}`",
        "",
        "## Loop Sweep",
        "",
    ]
    for benchmark, score_targets in payload["comparisons"].items():
        lines.append(f"### {benchmark}")
        for score_target, loops in score_targets.items():
            lines.append(f"- score target `{score_target}`")
            for loop, aggregates in loops.items():
                for aggregate, row in aggregates.items():
                    lines.append(
                        f"  - loop `{loop}` aggregate `{aggregate}`: recurrent "
                        f"`{row['recurrent_correct']}/{row['paired_examples']}`, base "
                        f"`{row['base_correct']}/{row['paired_examples']}`, delta "
                        f"`{row['delta']}`, W/L/T `{row['wins']}/{row['losses']}/{row['ties']}`, "
                        f"p `{row['sign_test_p']}`, mean loops `{row['mean_expected_loops']}`"
                    )
        lines.append("")
    (RUN_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)


def update_pointer() -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(RUN_DIR / "summary.json") + "\n", encoding="utf-8")


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    update_pointer()
    run(["git", "status", "-sb"], log_name="git_status_before_commit.log", check=False)
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], log_name="git_add_run.log", check=False)
    run(["git", "add", "-f", "config/stage5_current_source_summary.txt"], log_name="git_add_pointer.log", check=False)
    status = run(["git", "diff", "--cached", "--quiet"], log_name="git_diff_cached.log", check=False)
    if status.returncode == 0:
        print("No model viability outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 model viability probe {RUN_ID}"], log_name="git_commit.log")
    push = run(["git", "push", "origin", "main"], log_name="git_push.log", check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "origin", "main"], log_name="git_pull_rebase.log")
    run(["git", "push", "origin", "main"], log_name="git_push_retry.log")


def main() -> int:
    started = time.time()
    data = benchmark_data()
    identity = run_identity_gate()
    loops = parse_int_csv(LOOPS)
    score_targets = parse_csv(SCORE_TARGETS)
    comparisons: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    outputs: dict[str, Any] = {"base": {}, "recurrent": {}}

    for benchmark, data_jsonl in data.items():
        for score_target in score_targets:
            base_path = eval_mcq(
                data_jsonl=data_jsonl,
                benchmark=benchmark,
                arm="base",
                score_target=score_target,
            )
            outputs["base"].setdefault(benchmark, {})[score_target] = path_for_cli(base_path)
            for loop in loops:
                recurrent_path = eval_mcq(
                    data_jsonl=data_jsonl,
                    benchmark=benchmark,
                    arm="recurrent",
                    score_target=score_target,
                    max_loops=loop,
                )
                outputs["recurrent"].setdefault(benchmark, {}).setdefault(score_target, {})[
                    str(loop)
                ] = path_for_cli(recurrent_path)
                comparisons.setdefault(benchmark, {}).setdefault(score_target, {})[
                    str(loop)
                ] = summarize_pair(base_path, recurrent_path)

    payload = {
        "kind": "stage5_model_viability_probe",
        "program_phase": "standing_scale_probe",
        "program_role": (
            "Information-only scale probe. It does not unlock Stage 4, "
            "Phase 2 breadth work, or particle/SVGD work; those remain gated "
            "by re-entry repair and deterministic depth recovery."
        ),
        "run_id": RUN_ID,
        "model_name": MODEL_NAME,
        "model_label": MODEL_LABEL or MODEL_NAME,
        "layer_split": LAYER_SPLIT,
        "identity": identity,
        "benchmarks": list(data),
        "score_targets": score_targets,
        "aggregates": parse_csv(AGGREGATES),
        "loops": loops,
        "limits": {
            "arc_easy": ARC_EASY_LIMIT,
            "arc_challenge": ARC_CHALLENGE_LIMIT,
            "arc_easy_offset": ARC_EASY_OFFSET,
            "arc_challenge_offset": ARC_CHALLENGE_OFFSET,
        },
        "outputs": outputs,
        "comparisons": comparisons,
        "elapsed_seconds": time.time() - started,
    }
    write_summary(payload)
    commit_results()
    return 0 if identity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
