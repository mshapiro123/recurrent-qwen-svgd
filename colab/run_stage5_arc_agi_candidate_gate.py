"""Run ARC-AGI candidate-source value gates.

This runner asks a deliberately narrow question before more particle/SVGD
training:

Can alternate candidates help exact-grid ARC-AGI scoring at all?

It compares model-only, symbolic-only, and hybrid candidate pools for base Qwen
and the deterministic recurrent Phase1 checkpoint. If symbolic or hybrid
candidates help but particles do not, the next training target should be
learning useful transformation/action proposals rather than amplifying generic
latent diversity.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from colab.stage5_model_metadata import model_metadata
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    from stage5_model_metadata import model_metadata


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_GATE_RUN_ID") or time.strftime("stage5_arc_agi_candidate_gate_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
PHASE1_CKPT = Path(os.environ.get("STAGE5_PHASE1_CKPT", str(BASE_RUN_DIR / "phase1" / "phase1_step_500.pt")))
if not PHASE1_CKPT.is_absolute():
    PHASE1_CKPT = ROOT / PHASE1_CKPT

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_ROOT = ROOT / "data" / "arc_agi"
ARC_AGI_1_REPO = os.environ.get("ARC_AGI_1_REPO", "https://github.com/fchollet/ARC-AGI.git")
ARC_AGI_2_REPO = os.environ.get("ARC_AGI_2_REPO", "https://github.com/arcprize/ARC-AGI-2.git")
ARC_VERSION = os.environ.get("STAGE5_ARC_AGI_VERSION", "1")
ARC_SPLIT = os.environ.get("STAGE5_ARC_AGI_SPLIT", "evaluation")
LIMIT = int(os.environ.get("STAGE5_ARC_AGI_LIMIT", "20"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
SELECTION_STRATEGY = os.environ.get("STAGE5_ARC_AGI_SELECTION_STRATEGY", "heuristic")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_GATE_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    include_symbolic: bool
    symbolic_position: str = "after_model"
    symbolic_candidate_format: str = "grid"
    checkpoint: Path | None = None


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout=stdout, stderr=None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def mount_drive_if_possible() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:  # pragma: no cover - Colab only
        print(f"Drive mount skipped/failed: {exc}")


def restore_phase1_checkpoint() -> None:
    if PHASE1_CKPT.exists():
        return
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    candidates = sorted(drive_root.rglob("phase1_step_500.pt")) if drive_root.exists() else []
    for candidate in candidates:
        if BASE_RUN_ID in str(candidate):
            PHASE1_CKPT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, PHASE1_CKPT)
            print(f"restored_phase1_checkpoint={candidate} -> {PHASE1_CKPT}")
            return
    if candidates:
        PHASE1_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], PHASE1_CKPT)
        print(f"restored_phase1_checkpoint={candidates[0]} -> {PHASE1_CKPT}")
        return
    raise FileNotFoundError(f"Missing Phase1 checkpoint: {PHASE1_CKPT}")


def clone_or_update(repo_url: str, target: Path) -> None:
    if target.exists() and (target / ".git").exists():
        run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", repo_url, str(target)])


def resolve_tasks_path() -> Path:
    if user_path := os.environ.get("STAGE5_ARC_AGI_TASKS_PATH"):
        return Path(user_path)
    if ARC_VERSION == "2":
        repo_dir = DATA_ROOT / "ARC-AGI-2"
        clone_or_update(ARC_AGI_2_REPO, repo_dir)
    else:
        repo_dir = DATA_ROOT / "ARC-AGI"
        clone_or_update(ARC_AGI_1_REPO, repo_dir)

    candidates = [
        repo_dir / "data" / ARC_SPLIT,
        repo_dir / ARC_SPLIT,
        repo_dir / "data" / f"{ARC_SPLIT}_challenges",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find ARC-AGI split {ARC_SPLIT!r} under {repo_dir}")


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def requested_variants() -> list[str] | None:
    value = os.environ.get("STAGE5_ARC_AGI_GATE_VARIANTS")
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_variants() -> list[Variant]:
    all_variants = [
        Variant("symbolic_only", "base", True, "only"),
        Variant("symbolic_program_only", "base", True, "only", "program"),
        Variant("base_model_only", "base", False),
        Variant("base_hybrid_symbolic_first", "base", True, "before_model"),
        Variant("base_hybrid_program_first", "base", True, "before_model", "program"),
        Variant("phase1_model_only", "phase1", False, checkpoint=PHASE1_CKPT),
        Variant("phase1_hybrid_symbolic_first", "phase1", True, "before_model", checkpoint=PHASE1_CKPT),
        Variant("phase1_hybrid_program_first", "phase1", True, "before_model", "program", PHASE1_CKPT),
    ]
    names = requested_variants()
    if names is None:
        return all_variants
    by_name = {variant.name: variant for variant in all_variants}
    unknown = set(names) - set(by_name)
    if unknown:
        raise ValueError(f"Unknown gate variants: {sorted(unknown)}")
    return [by_name[name] for name in names]


def eval_variant(variant: Variant, tasks_path: Path) -> dict[str, Any]:
    summary_json = RUN_DIR / f"{variant.name}_summary.json"
    cmd = [
        sys.executable,
        "eval/eval_arc_agi.py",
        "--tasks_path",
        str(tasks_path),
        "--limit",
        str(LIMIT),
        "--mode",
        variant.mode,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--grid_format",
        GRID_FORMAT,
        "--selection_strategy",
        SELECTION_STRATEGY,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--output_jsonl",
        path_for_cli(RUN_DIR / f"{variant.name}_candidates.jsonl"),
        "--summary_json",
        path_for_cli(summary_json),
        "--summary_md",
        path_for_cli(RUN_DIR / f"{variant.name}_summary.md"),
    ]
    if variant.mode != "base":
        assert variant.checkpoint is not None
        cmd += [
            "--checkpoint",
            path_for_cli(variant.checkpoint),
            "--max_loops",
            "4",
            "--num_candidates",
            "1",
        ]
    if variant.include_symbolic:
        cmd += [
            "--include_symbolic_candidates",
            "--symbolic_position",
            variant.symbolic_position,
            "--symbolic_candidate_format",
            variant.symbolic_candidate_format,
        ]
    run(cmd, log_name=f"{variant.name}_eval.log")
    payload = read_summary(summary_json)
    return {
        "variant": variant.name,
        "mode": variant.mode,
        "include_symbolic_candidates": variant.include_symbolic,
        "symbolic_position": variant.symbolic_position if variant.include_symbolic else None,
        "symbolic_candidate_format": variant.symbolic_candidate_format if variant.include_symbolic else None,
        "summary": payload["summary"],
        "candidate_source_summary": payload.get("candidate_source_summary", {}),
        "parse_method_summary": payload.get("parse_method_summary", {}),
        "program_verifier_summary": payload.get("program_verifier_summary", {}),
    }


def compact_row(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    return {
        "variant": result["variant"],
        "first": summary["first_exact"],
        "selected": summary["selected_exact"],
        "best": summary["best_of_k_exact"],
        "examples": summary["examples_with_targets"],
        "tasks_solved_best": summary["tasks_solved_best_of_k"],
        "tasks": summary["tasks_with_targets"],
        "valid_rate": summary["valid_candidate_rate"],
    }


def analyze_symbolic_coverage(tasks_path: Path) -> dict[str, Any]:
    summary_json = RUN_DIR / "symbolic_coverage.json"
    run(
        [
            sys.executable,
            "eval/analyze_arc_agi_symbolic.py",
            "--tasks_path",
            str(tasks_path),
            "--limit",
            str(LIMIT),
            "--summary_json",
            path_for_cli(summary_json),
            "--summary_md",
            path_for_cli(RUN_DIR / "symbolic_coverage.md"),
        ],
        log_name="symbolic_coverage.log",
    )
    return read_summary(summary_json)["summary"]


def write_comparison(results: list[dict[str, Any]], metadata: dict[str, Any], coverage: dict[str, Any]) -> None:
    rows = [compact_row(result) for result in results]
    payload = {"metadata": metadata, "symbolic_coverage": coverage, "rows": rows, "results": results}
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 5 ARC-AGI Candidate Gate - {RUN_ID}",
        "",
        f"- ARC version: `{ARC_VERSION}`",
        f"- Split: `{ARC_SPLIT}`",
        f"- Limit: `{LIMIT}`",
        f"- Grid format: `{GRID_FORMAT}`",
        f"- Selection strategy: `{SELECTION_STRATEGY}`",
        f"- Phase1 checkpoint: `{path_for_cli(PHASE1_CKPT)}`",
        f"- Symbolic exact coverage: `{coverage['exact_symbolic']}` / `{coverage['examples_with_targets']}` = "
        f"`{coverage['exact_symbolic_rate']}`",
        f"- Symbolic task solve coverage: `{coverage['tasks_solved_symbolic']}` / `{coverage['tasks_with_targets']}` = "
        f"`{coverage['task_solve_rate_symbolic']}`",
        "",
        "## Comparison",
        "",
        "| Variant | First | Selected | Best-of-K | Tasks best | Valid rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['first']}/{row['examples']} | "
            f"{row['selected']}/{row['examples']} | {row['best']}/{row['examples']} | "
            f"{row['tasks_solved_best']}/{row['tasks']} | {row['valid_rate']:.4f} |"
        )
    lines += ["", "## Candidate Source Summaries"]
    for result in results:
        lines += ["", f"### {result['variant']}"]
        source_summary = result.get("candidate_source_summary", {})
        if not source_summary:
            lines.append("- No candidate source summary.")
            continue
        for source, stats in sorted(source_summary.items()):
            lines.append(
                f"- `{source}`: count `{stats['count']}`, valid `{stats['valid']}`, "
                f"exact `{stats['exact']}`, selected `{stats['selected']}`, "
                f"selected_exact `{stats['selected_exact']}`"
            )
    lines += [
        "",
        "Interpretation guide: symbolic-only tells us how much simple transform coverage exists. "
        "Hybrid-symbolic-first is not a deployable verifier; it is a value gate for whether useful "
        "non-neural candidates exist on this slice.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def backup_to_drive() -> None:
    mount_drive_if_possible()
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No ARC-AGI candidate gate outputs changed.")
        return
    run(["git", "commit", "-m", "Record Stage 5 ARC-AGI candidate gate"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Run from Colab. Optional STAGE5_ARC_AGI_GATE_VARIANTS can restrict "
            "variants, e.g. symbolic_program_only,base_model_only,phase1_hybrid_program_first."
        )
        return 0

    tasks_path = resolve_tasks_path()
    restore_phase1_checkpoint()
    variants = build_variants()
    metadata = {
        "run_id": RUN_ID,
        **model_metadata(MODEL_NAME),
        "arc_version": ARC_VERSION,
        "arc_split": ARC_SPLIT,
        "limit": LIMIT,
        "tasks_path": str(tasks_path),
        "phase1_checkpoint": path_for_cli(PHASE1_CKPT),
        "grid_format": GRID_FORMAT,
        "selection_strategy": SELECTION_STRATEGY,
        "dtype": DTYPE,
        "adapter_dtype": ADAPTER_DTYPE,
        "variants": [variant.name for variant in variants],
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    coverage = analyze_symbolic_coverage(tasks_path)
    results = [eval_variant(variant, tasks_path) for variant in variants]
    write_comparison(results, metadata, coverage)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
