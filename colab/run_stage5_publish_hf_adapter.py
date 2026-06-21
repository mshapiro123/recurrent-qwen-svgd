"""Export and optionally upload a recovered recurrent-Qwen adapter artifact.

The recurrent experiments store trainable checkpoints rather than full base
model weights. This runner packages the selected checkpoint with enough metadata
to reproduce it:

* trainable checkpoint copied as ``recurrent_adapter_checkpoint.pt``;
* source run summary, if available;
* machine-readable ``recurrent_adapter_config.json``;
* a Hugging Face-style ``README.md`` model card.

Set ``STAGE5_HF_REPO_ID=mshapiro123/<repo>`` and provide ``HF_TOKEN`` or
``HUGGINGFACE_HUB_TOKEN`` to upload. Uploads are private by default.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_HF_EXPORT_RUN_ID") or time.strftime("stage5_hf_export_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "hf_exports" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_SUMMARY = os.environ.get("STAGE5_HF_SOURCE_SUMMARY", "")
EXPLICIT_CHECKPOINT = os.environ.get("STAGE5_HF_CHECKPOINT", "") or os.environ.get("STAGE5_ARC_AGI_RECOVERED_CKPT", "")
HF_REPO_ID = os.environ.get("STAGE5_HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
HF_PRIVATE = os.environ.get("STAGE5_HF_PRIVATE", "1").strip().lower() in {"1", "true", "yes", "y"}
HF_UPLOAD = os.environ.get("STAGE5_HF_UPLOAD", "auto").strip().lower()
MODEL_NAME_DEFAULT = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
RECIPE_CONTROL_SUMMARY = os.environ.get("STAGE5_HF_RECIPE_CONTROL_SUMMARY", "")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}\n{proc.stdout}")
    return proc


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_source_summary() -> Path | None:
    candidates: list[Path] = []
    for path in list((ROOT / "outputs" / "stage5").glob("*/summary.json")) + list(
        (ROOT / "outputs" / "hf_exports").glob("*/summary.json")
    ):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if checkpoint_value_from_payload(payload):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def checkpoint_value_from_payload(payload: dict[str, Any]) -> str | None:
    candidates = [
        payload.get("metadata", {}).get("recovered_checkpoint"),
        payload.get("metadata", {}).get("checkpoint"),
        payload.get("compact", {}).get("final_checkpoint"),
        payload.get("autopilot_compact", {}).get("final_checkpoint"),
        payload.get("final_checkpoint"),
        payload.get("tuned_checkpoint"),
        payload.get("selected_checkpoint", {}).get("checkpoint"),
    ]
    stages = payload.get("stages") or []
    if stages:
        candidates.append(stages[-1].get("selected_checkpoint", {}).get("checkpoint"))
    curriculum = payload.get("curriculum") or {}
    if curriculum:
        candidates.append(curriculum.get("final_checkpoint"))
        curriculum_stages = curriculum.get("stages") or []
        if curriculum_stages:
            candidates.append(curriculum_stages[-1].get("selected_checkpoint", {}).get("checkpoint"))
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def is_recipe_control_assessment(payload: dict[str, Any]) -> bool:
    return payload.get("gate") == "stage5_same_recipe_architecture"


def latest_recipe_control_summary() -> Path | None:
    candidates: list[Path] = []
    for path in (ROOT / "outputs" / "stage5").glob("*/summary.json"):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if is_recipe_control_assessment(payload):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def resolve_recipe_control_summary(source_summary: Path | None, source_payload: dict[str, Any] | None) -> Path | None:
    if RECIPE_CONTROL_SUMMARY:
        path = resolve_path(RECIPE_CONTROL_SUMMARY)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if source_summary and source_payload and is_recipe_control_assessment(source_payload):
        return source_summary
    return latest_recipe_control_summary()


def resolve_source_summary() -> Path | None:
    if SOURCE_SUMMARY:
        path = resolve_path(SOURCE_SUMMARY)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    return latest_source_summary()


def resolve_checkpoint(source_payload: dict[str, Any] | None) -> Path:
    if EXPLICIT_CHECKPOINT:
        checkpoint = resolve_path(EXPLICIT_CHECKPOINT)
    elif source_payload:
        value = checkpoint_value_from_payload(source_payload)
        if not value:
            raise ValueError("source summary did not contain a checkpoint path")
        checkpoint = resolve_path(value)
    else:
        raise ValueError("Set STAGE5_HF_CHECKPOINT or provide a source summary with a checkpoint path.")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str]) -> str | None:
    proc = run(["git", *args], check=False)
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return {
        "phase": checkpoint.get("phase") if isinstance(checkpoint, dict) else None,
        "step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
        "config": config,
        "trainable_key_count": len(checkpoint.get("trainable_state_dict", {})) if isinstance(checkpoint, dict) else None,
    }


def extract_eval_snapshot(source_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not source_payload:
        return {}
    keys = [
        "compact",
        "autopilot_compact",
        "deltas",
        "base",
        "phase1_start",
        "recovered",
        "candidate_distill_gate",
    ]
    return {key: source_payload[key] for key in keys if key in source_payload}


def compact_recipe_control_evidence(path: Path | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    decision = payload.get("decision_evidence") or {}
    return {
        "summary_path": path_for_cli(path) if path else None,
        "status": payload.get("status"),
        "passed": bool(payload.get("passed", False)),
        "reason": payload.get("reason"),
        "next_step": payload.get("next_step"),
        "dense_summary": payload.get("dense_summary"),
        "recurrent_summary": payload.get("recurrent_summary"),
        "hard_bucket": payload.get("hard_bucket"),
        "aggregate_selected": decision.get("aggregate"),
        "hard_selected": decision.get("hard"),
        "aggregate_best_of_k": decision.get("aggregate_best_of_k"),
        "hard_best_of_k": decision.get("hard_best_of_k"),
    }


def build_export_metadata(
    *,
    checkpoint: Path,
    checkpoint_metadata: dict[str, Any],
    source_summary: Path | None,
    source_payload: dict[str, Any] | None,
    recipe_control_summary: Path | None = None,
    recipe_control_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = checkpoint_metadata.get("config") or {}
    return {
        "artifact_type": "recurrent_qwen_trainable_adapter_checkpoint",
        "base_model": config.get("model_name", MODEL_NAME_DEFAULT),
        "checkpoint_file": "recurrent_adapter_checkpoint.pt",
        "checkpoint_source_path": path_for_cli(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint": checkpoint_metadata,
        "architecture": {
            "wrapper": "RecurrentQwenForCausalLM",
            "layer_split": config.get("layer_split", "6,18"),
            "max_loops": config.get("max_loops", 4),
            "lora": config.get("lora", {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0}),
            "adapter_dtype": config.get("adapter_dtype", "float32"),
        },
        "source": {
            "repo": git_value(["config", "--get", "remote.origin.url"]),
            "commit": git_value(["rev-parse", "HEAD"]),
            "summary_path": path_for_cli(source_summary) if source_summary else None,
        },
        "eval_snapshot": extract_eval_snapshot(source_payload),
        "architecture_evidence": compact_recipe_control_evidence(
            recipe_control_summary,
            recipe_control_payload,
        ),
    }


def delta_fragment(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "missing"
    return (
        f"delta {stats.get('delta_exact', 0)} "
        f"({stats.get('wins', 0)}/{stats.get('losses', 0)}/{stats.get('ties', 0)} W/L/T)"
    )


def render_model_card(metadata: dict[str, Any]) -> str:
    base_model = metadata["base_model"]
    architecture = metadata["architecture"]
    source = metadata["source"]
    eval_snapshot = metadata.get("eval_snapshot") or {}
    architecture_evidence = metadata.get("architecture_evidence") or {}
    deltas = eval_snapshot.get("deltas") or {}
    compact = eval_snapshot.get("compact") or eval_snapshot.get("autopilot_compact") or {}
    lines = [
        "---",
        "license: other",
        "base_model:",
        f"- {base_model}",
        "tags:",
        "- recurrent-depth",
        "- qwen",
        "- arc-agi",
        "- adapter",
        "library_name: transformers",
        "---",
        "",
        "# Recurrent-Depth Qwen Adapter",
        "",
        "This repository contains a trainable checkpoint for a GRAM-inspired recurrent-depth Qwen wrapper.",
        "It does **not** contain the base Qwen weights. Load the base model separately, wrap it with this project's `RecurrentQwenForCausalLM`, apply LoRA to the recurrent block, then load `recurrent_adapter_checkpoint.pt`.",
        "",
        "## Base And Architecture",
        "",
        f"- Base model: `{base_model}`",
        f"- Wrapper: `{architecture['wrapper']}`",
        f"- Layer split: `{architecture['layer_split']}`",
        f"- Max loops: `{architecture['max_loops']}`",
        f"- LoRA config: `{architecture['lora']}`",
        f"- Adapter dtype: `{architecture['adapter_dtype']}`",
        f"- Source commit: `{source.get('commit')}`",
        f"- Source repo: `{source.get('repo')}`",
        "",
        "## Evaluation Snapshot",
        "",
    ]
    if compact:
        lines.append(f"- Autopilot compact summary: `{compact}`")
    if deltas:
        lines.append(f"- Benchmark deltas: `{deltas}`")
    if not compact and not deltas:
        lines.append("- No benchmark summary was packaged with this export.")
    lines.extend(["", "## Same-Recipe Architecture Evidence", ""])
    if architecture_evidence:
        lines.extend(
            [
                f"- Architecture gate summary: `{architecture_evidence.get('summary_path')}`",
                f"- Status: `{architecture_evidence.get('status')}`",
                f"- Passed: `{architecture_evidence.get('passed')}`",
                f"- Reason: {architecture_evidence.get('reason')}",
                f"- Next step: {architecture_evidence.get('next_step')}",
                f"- Aggregate selected recurrent-vs-dense: {delta_fragment(architecture_evidence.get('aggregate_selected'))}",
                f"- Hard selected recurrent-vs-dense: {delta_fragment(architecture_evidence.get('hard_selected'))}",
                f"- Aggregate best-of-K recurrent-vs-dense: {delta_fragment(architecture_evidence.get('aggregate_best_of_k'))}",
                f"- Hard best-of-K recurrent-vs-dense: {delta_fragment(architecture_evidence.get('hard_best_of_k'))}",
            ]
        )
    else:
        lines.append(
            "- No same-recipe dense-vs-recurrent architecture assessment was packaged with this export."
        )
    lines.extend(
        [
            "",
            "## Loading Sketch",
            "",
            "```python",
            "import torch",
            "from transformers import AutoModelForCausalLM",
            "from models.recurrent_wrapper import RecurrentQwenForCausalLM",
            "from models.lora import apply_lora_to_recurrent_block",
            "from eval.eval_identity import parse_split, resolve_dtype",
            "from training.checkpointing import load_trainable_checkpoint",
            "",
            f"base = AutoModelForCausalLM.from_pretrained({base_model!r}, dtype=torch.bfloat16)",
            f"wrapper = RecurrentQwenForCausalLM(base, layer_split=parse_split({architecture['layer_split']!r}))",
            "apply_lora_to_recurrent_block(wrapper, rank=8, alpha=16, dropout=0.0, adapter_dtype=torch.float32)",
            "wrapper.set_trainable_modules_dtype(torch.float32)",
            "load_trainable_checkpoint(wrapper, 'recurrent_adapter_checkpoint.pt')",
            "```",
            "",
            "## Caveats",
            "",
            "- This is an experimental architecture-modification checkpoint, not a standalone chat model.",
            "- Use the source repository commit listed above for exact wrapper code.",
            "- ARC-AGI results should be interpreted with the packaged run summary and split/limit metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def should_upload(repo_id: str = HF_REPO_ID, token: str = HF_TOKEN, upload_flag: str = HF_UPLOAD) -> bool:
    if upload_flag in {"0", "false", "no", "n", "off"}:
        return False
    if upload_flag in {"1", "true", "yes", "y", "on"}:
        return bool(repo_id and token)
    return bool(repo_id and token)


def write_export(
    *,
    checkpoint: Path,
    source_summary: Path | None,
    source_payload: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> None:
    shutil.copy2(checkpoint, RUN_DIR / "recurrent_adapter_checkpoint.pt")
    (RUN_DIR / "recurrent_adapter_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (RUN_DIR / "README.md").write_text(render_model_card(metadata), encoding="utf-8")
    if source_summary:
        shutil.copy2(source_summary, RUN_DIR / "source_summary.json")
        source_md = source_summary.with_suffix(".md")
        if source_md.exists():
            shutil.copy2(source_md, RUN_DIR / "source_summary.md")
    architecture_summary = (metadata.get("architecture_evidence") or {}).get("summary_path")
    if architecture_summary:
        architecture_summary_path = resolve_path(str(architecture_summary))
        if architecture_summary_path.exists():
            shutil.copy2(architecture_summary_path, RUN_DIR / "architecture_assessment_summary.json")
            architecture_md = architecture_summary_path.with_suffix(".md")
            if architecture_md.exists():
                shutil.copy2(architecture_md, RUN_DIR / "architecture_assessment_summary.md")
    summary = {
        "run_id": RUN_ID,
        "export_dir": path_for_cli(RUN_DIR),
        "checkpoint": metadata["checkpoint_source_path"],
        "hf_repo_id": HF_REPO_ID or None,
        "uploaded": False,
        "metadata": metadata,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def upload_export() -> str | None:
    if not should_upload():
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, private=HF_PRIVATE, exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO_ID,
        folder_path=str(RUN_DIR),
        path_in_repo=".",
        commit_message=f"Upload recurrent Qwen adapter export {RUN_ID}",
    )
    return f"https://huggingface.co/{HF_REPO_ID}"


def update_summary_after_upload(url: str | None) -> None:
    summary_path = RUN_DIR / "summary.json"
    summary = read_json(summary_path)
    summary["uploaded"] = url is not None
    summary["hf_url"] = url
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_report(url: str | None) -> None:
    summary = read_json(RUN_DIR / "summary.json")
    lines = [
        f"# Stage 5 HF Adapter Export - {RUN_ID}",
        "",
        f"- Export dir: `{summary['export_dir']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- HF repo id: `{summary['hf_repo_id']}`",
        f"- Uploaded: `{summary['uploaded']}`",
        f"- HF URL: `{url}`",
        "",
        "Files:",
        "",
        "- `recurrent_adapter_checkpoint.pt`",
        "- `recurrent_adapter_config.json`",
        "- `README.md`",
        "- `source_summary.json` if a source run summary was found",
        "- `architecture_assessment_summary.json` if a same-recipe architecture gate was found",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Export a recurrent adapter checkpoint and optionally upload to Hugging Face.")
        return 0
    source_summary = resolve_source_summary()
    source_payload = read_json(source_summary) if source_summary else None
    recipe_control_summary = resolve_recipe_control_summary(source_summary, source_payload)
    recipe_control_payload = read_json(recipe_control_summary) if recipe_control_summary else None
    checkpoint = resolve_checkpoint(source_payload)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint)
    metadata = build_export_metadata(
        checkpoint=checkpoint,
        checkpoint_metadata=checkpoint_metadata,
        source_summary=source_summary,
        source_payload=source_payload,
        recipe_control_summary=recipe_control_summary,
        recipe_control_payload=recipe_control_payload,
    )
    write_export(
        checkpoint=checkpoint,
        source_summary=source_summary,
        source_payload=source_payload,
        metadata=metadata,
    )
    url = upload_export()
    update_summary_after_upload(url)
    write_report(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
