"""Run the authorized D0 pre-lock rejection-density probe."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs
from eval.eval_internal_think_token_t1_lite import restore_checkpoint
from training.internal_think_token_runtime import install_internal_control_tokens, split_internal_control_token_rows
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_spec import (
    D0ExecutionPolicy,
    DRAFTER_MODEL,
    DRAFTER_MODEL_REVISION,
    DRAFTER_CHECKPOINT_SHA256,
    TEACHER_7B,
    TEACHER_7B_REVISION,
    prelock_contract,
)
from training.train_unfrozen_recurrent import prepare_wrapper


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input(row: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    attention = torch.ones_like(values)
    return values, attention


@torch.no_grad()
def drafter_predictions(
    rows: list[dict[str, Any]],
    *,
    checkpoint: str | Path,
    model_name: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    revision: str = DRAFTER_MODEL_REVISION,
) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        **model_load_kwargs(dtype, attn_implementation),
    ).to(device)
    resize = install_internal_control_tokens(tokenizer, model)
    split_internal_control_token_rows(model, original_vocab_size=resize.original_vocab_size)
    config = {
        "layer_split": "6,18",
        "initial_halt_prob": 0.15,
        "bridge_projection_mode": "split",
        "adapter_dtype": "float32",
        "training_mode": "full_block",
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "train_auxiliary": {"bridge": True, "halting": False, "reentry_adapter": False, "latent": False},
    }
    wrapper, _ = prepare_wrapper(model, config, device=device)
    wrapper.base_model.get_input_embeddings().control_rows.requires_grad_(True)
    restore_checkpoint(wrapper, checkpoint)
    wrapper.eval()
    predictions: list[list[int]] = []
    for index, row in enumerate(rows):
        input_ids, attention = _input(row, device)
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            labels=None,
            max_loops=1,
            use_cache=False,
            return_dict=True,
        )
        predictions.append(output.logits[0, :-1].argmax(dim=-1).cpu().tolist())
        if (index + 1) % 32 == 0:
            print(f"d0_density_drafter_rows={index + 1}/{len(rows)}", flush=True)
    del wrapper, model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


@torch.no_grad()
def teacher_predictions(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    revision: str = TEACHER_7B_REVISION,
) -> list[list[int]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        **model_load_kwargs(dtype, attn_implementation),
    ).to(device)
    model.eval()
    predictions: list[list[int]] = []
    for index, row in enumerate(rows):
        input_ids, attention = _input(row, device)
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        predictions.append(logits[0, :-1].argmax(dim=-1).cpu().tolist())
        if (index + 1) % 32 == 0:
            print(f"d0_density_teacher_rows={index + 1}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def score_density(
    rows: list[dict[str, Any]],
    drafter: list[list[int]],
    teacher: list[list[int]],
) -> dict[str, Any]:
    if not (len(rows) == len(drafter) == len(teacher)):
        raise ValueError("D0 density arrays have different row counts")
    disagreements = 0
    prediction_positions = 0
    for row, draft_values, teacher_values in zip(rows, drafter, teacher, strict=True):
        expected = max(0, len(row["input_ids"]) - 1)
        if len(draft_values) != expected or len(teacher_values) != expected:
            raise RuntimeError("D0 density prediction length drifted")
        prediction_positions += expected
        disagreements += sum(left != right for left, right in zip(draft_values, teacher_values, strict=True))
    return {
        "rows": len(rows),
        "input_tokens": sum(int(row["token_count"]) for row in rows),
        "prediction_positions": prediction_positions,
        "disagreements": disagreements,
        "rejection_density_per_1000_tokens": disagreements / prediction_positions * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True, action="append")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--drafter_model", default=DRAFTER_MODEL)
    parser.add_argument("--drafter_revision", default=DRAFTER_MODEL_REVISION)
    parser.add_argument("--teacher_model", default=TEACHER_7B)
    parser.add_argument("--teacher_revision", default=TEACHER_7B_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    args = parser.parse_args()

    contract = prelock_contract()
    D0ExecutionPolicy(density_probe_authorized=True).assert_allowed(
        density_probe=True, labeling=False, training=False
    )
    observed_checkpoint_sha = sha256_file(args.checkpoint)
    if observed_checkpoint_sha != DRAFTER_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"D0 drafter SHA mismatch: {observed_checkpoint_sha} != {DRAFTER_CHECKPOINT_SHA256}"
        )
    sources: list[tuple[str, list[dict[str, Any]]]] = []
    all_rows: list[dict[str, Any]] = []
    for source in args.data_jsonl:
        rows = read_jsonl(source)
        label = str(rows[0].get("stratum") if rows else Path(source).stem)
        sources.append((label, rows))
        all_rows.extend(rows)
    draft = drafter_predictions(
        all_rows,
        checkpoint=args.checkpoint,
        model_name=args.drafter_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        revision=args.drafter_revision,
    )
    teacher = teacher_predictions(
        all_rows,
        model_name=args.teacher_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        revision=args.teacher_revision,
    )
    per_stratum: dict[str, Any] = {}
    offset = 0
    for label, rows in sources:
        stop = offset + len(rows)
        per_stratum[label] = score_density(rows, draft[offset:stop], teacher[offset:stop])
        offset = stop
    summary = {
        "kind": "paper2_d0_prelock_density_probe",
        "status": "finished_uncitable_prelock_probe",
        "contract": contract,
        "data_jsonl": list(args.data_jsonl),
        "data_sha256": {str(path): sha256_file(path) for path in args.data_jsonl},
        "drafter_checkpoint_sha256": observed_checkpoint_sha,
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "teacher_reloaded_after_labeling": False,
        "models_used_for_labeling_proper": False,
        "optimizer_steps": 0,
        "strata": per_stratum,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
