"""Train recurrent Phase 1 adapters with an MCQ option-score objective.

This is a narrow repair tool for MCQ surface failures.  Unlike causal SFT, it
scores every answer option under the same prompt/completion surface used by
``eval/eval_mcq.py`` and optimizes the correct option to outrank distractors.
It can also distill the base model's option-score distribution on those same
options, which is useful when the selected rows are base-correct direct cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype  # noqa: E402
from eval.eval_mcq import MCQExample, format_completion, format_prompt, normalize_answer, option_items, sequence_logprobs  # noqa: E402
from models.lora import apply_lora_to_recurrent_block  # noqa: E402
from models.recurrent_wrapper import RecurrentQwenForCausalLM  # noqa: E402
from training.checkpointing import save_trainable_checkpoint  # noqa: E402
from training.reentry_repair import apply_reentry_repair_controls  # noqa: E402
from training.stability import (  # noqa: E402
    assert_finite_trainable_gradients,
    assert_finite_trainable_parameters,
    assert_finite_training_state,
)
from training.train_phase1_ponder import cfg_float, cfg_int, optimizer_parameters  # noqa: E402


@dataclass(frozen=True)
class EncodedOptions:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    target_index: int
    target_loop_count: int
    routing_type: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def mcq_example(row: dict[str, Any]) -> MCQExample:
    choices = option_items(row)
    answer = normalize_answer(row.get("answer") or row.get("label") or row.get("target"), choices)
    return MCQExample(
        id=str(row.get("id") or ""),
        question=str(row["question"]),
        choices=choices,
        answer=answer,
    )


def encode_options(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    pad_token_id: int,
    default_prompt_style: str,
    default_score_target: str,
    max_train_loops: int,
) -> EncodedOptions:
    example = mcq_example(row)
    prompt_style = str(row.get("prompt_style") or default_prompt_style)
    score_target = str(row.get("score_target") or default_score_target)
    prompt = format_prompt(example, prompt_style)
    encoded_ids: list[list[int]] = []
    encoded_labels: list[list[int]] = []
    target_index = -1
    for idx, (label, text) in enumerate(example.choices):
        if label == example.answer:
            target_index = idx
        completion = format_completion(label, text, score_target)
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        ids = tokenizer(
            prompt + completion,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        labels = list(ids)
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        if all(item == -100 for item in labels):
            raise ValueError(f"Completion was fully truncated for row {row.get('id')}")
        encoded_ids.append(ids)
        encoded_labels.append(labels)
    if target_index < 0:
        raise ValueError(f"Answer {example.answer!r} not found in choices for row {row.get('id')}")

    seq_len = max(len(ids) for ids in encoded_ids)
    input_ids = torch.full((len(encoded_ids), seq_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_ids), seq_len), dtype=torch.long)
    labels = torch.full((len(encoded_ids), seq_len), -100, dtype=torch.long)
    for idx, (ids, option_labels) in enumerate(zip(encoded_ids, encoded_labels)):
        length = len(ids)
        input_ids[idx, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[idx, :length] = 1
        labels[idx, :length] = torch.tensor(option_labels, dtype=torch.long)

    return EncodedOptions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        target_index=target_index,
        target_loop_count=max(1, min(int(row.get("target_loop_count", 1)), int(max_train_loops))),
        routing_type=str(row.get("routing_type") or ""),
    )


def option_scores_from_logits(logits: torch.Tensor, labels: torch.Tensor, *, normalize: bool) -> torch.Tensor:
    return sequence_logprobs(logits, labels, normalize=normalize).float()


def halting_target_nll(output: Any, target_loop_count: int, *, max_loops: int) -> torch.Tensor | None:
    weights = getattr(output, "halting_weights", None)
    if weights is None:
        return None
    flat = weights.reshape(-1, weights.shape[-1]).float().clamp_min(1e-8)
    target = torch.full(
        (flat.shape[0],),
        max(0, min(int(target_loop_count), int(max_loops)) - 1),
        dtype=torch.long,
        device=flat.device,
    )
    return F.nll_loss(flat.log(), target)


def option_distribution_kl(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """KL over one MCQ option distribution, not over options-as-batch."""

    return F.kl_div(
        F.log_softmax((student_scores.float() / float(temperature)).unsqueeze(0), dim=-1),
        F.softmax((teacher_scores.float() / float(temperature)).unsqueeze(0), dim=-1),
        reduction="batchmean",
    ) * (float(temperature) ** 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        **model_load_kwargs(cfg.get("dtype", "auto"), cfg.get("attn_implementation", "default")),
    ).to(args.device)
    wrapper = RecurrentQwenForCausalLM(
        model,
        layer_split=parse_split(cfg.get("layer_split", "auto")),
        initial_halt_prob=cfg_float(cfg, "initial_halt_prob", 0.25),
    ).to(args.device)
    adapter_dtype = resolve_dtype(cfg.get("adapter_dtype", "float32"))
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        replaced = apply_lora_to_recurrent_block(
            wrapper,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=float(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            adapter_dtype=adapter_dtype,
        )
        print(f"lora_recurrent_modules={replaced}")
    wrapper.freeze_base_model()
    wrapper.set_latent_trainable(False)
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    if cfg.get("resume_from"):
        from training.checkpointing import load_trainable_checkpoint

        load_info = load_trainable_checkpoint(wrapper, cfg["resume_from"])
        print(f"loaded_checkpoint={cfg['resume_from']} loaded_keys={len(load_info['loaded_keys'])}")
    repair_info = apply_reentry_repair_controls(wrapper, cfg)
    if repair_info["applied"]:
        print("reentry_repair_controls=" + " ".join(f"{key}={value}" for key, value in repair_info.items()))
    assert_finite_trainable_parameters(wrapper, step=0)

    distill_cfg = cfg.get("score_distillation", {})
    teacher = None
    if distill_cfg.get("enabled", False):
        teacher_name = distill_cfg.get("teacher_model_name", cfg["model_name"])
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            **model_load_kwargs(distill_cfg.get("dtype", cfg.get("dtype", "auto")), cfg.get("attn_implementation", "default")),
        ).to(args.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        print(f"score_distillation_teacher={teacher_name}")

    rows = read_jsonl(args.train_jsonl)
    if not rows:
        raise ValueError(f"No rows in {args.train_jsonl}")

    optimizer_params = optimizer_parameters(wrapper, cfg)
    print(f"optimizer_parameter_tensors={len(optimizer_params)}")
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=cfg_float(cfg, "learning_rate", 1e-5),
        weight_decay=cfg_float(cfg, "weight_decay", 0.0),
    )
    wrapper.train()
    wrapper.zero_grad(set_to_none=True)

    max_loops = cfg_int(cfg, "max_loops", 4)
    max_steps = cfg_int(cfg, "max_steps", 50)
    save_every = cfg_int(cfg, "save_every", 0) if cfg.get("save_every", 0) else 0
    score_temperature = cfg_float(cfg, "score_temperature", 1.0)
    margin = cfg_float(cfg, "score_margin", 0.0)
    margin_weight = cfg_float(cfg, "score_margin_weight", 0.0)
    halt_nll_weight = cfg_float(cfg, "halt_target_nll_weight", 0.0)
    distill_weight = float(distill_cfg.get("weight", 0.0))
    distill_temperature = float(distill_cfg.get("temperature", 1.0))
    normalize_option_score = bool(cfg.get("normalize_option_score", True))

    step = 0
    while step < max_steps:
        for row in rows:
            encoded = encode_options(
                row,
                tokenizer,
                max_length=cfg_int(cfg, "max_length", 512),
                pad_token_id=int(tokenizer.pad_token_id),
                default_prompt_style=str(cfg.get("prompt_style", "question_only")),
                default_score_target=str(cfg.get("score_target", "option_text")),
                max_train_loops=max_loops,
            )
            batch = {
                "input_ids": encoded.input_ids.to(args.device),
                "attention_mask": encoded.attention_mask.to(args.device),
            }
            labels = encoded.labels.to(args.device)
            output = wrapper(
                **batch,
                labels=None,
                max_loops=max_loops,
                reentry_rescale_mode=cfg.get("reentry_rescale_mode", "none"),
                use_reentry_adapter=cfg.get("use_reentry_adapter", False),
                reentry_adapter_mode=cfg.get("reentry_adapter_mode", "affine"),
                use_cache=False,
                return_dict=True,
            )
            scores = option_scores_from_logits(output.logits, labels, normalize=normalize_option_score)
            target = torch.tensor([encoded.target_index], dtype=torch.long, device=scores.device)
            score_ce = F.cross_entropy((scores / score_temperature).unsqueeze(0), target)
            loss = cfg_float(cfg, "score_ce_weight", 1.0) * score_ce
            metrics: dict[str, torch.Tensor] = {"score_ce": score_ce.detach()}

            if margin_weight and margin:
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask[encoded.target_index] = False
                wrong_best = scores[mask].max()
                margin_loss = F.relu(torch.tensor(margin, device=scores.device) - (scores[encoded.target_index] - wrong_best))
                loss = loss + margin_weight * margin_loss
                metrics["score_margin_loss"] = margin_loss.detach()

            if teacher is not None and distill_weight:
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        return_dict=True,
                    )
                    teacher_scores = option_scores_from_logits(
                        teacher_out.logits,
                        labels,
                        normalize=normalize_option_score,
                    )
                score_kl = option_distribution_kl(
                    scores,
                    teacher_scores,
                    temperature=distill_temperature,
                )
                loss = loss + distill_weight * score_kl
                metrics["base_score_kl"] = score_kl.detach()

            if halt_nll_weight:
                halt_loss = halting_target_nll(output, encoded.target_loop_count, max_loops=max_loops)
                if halt_loss is not None:
                    loss = loss + halt_nll_weight * halt_loss
                    metrics["halting_target_nll"] = halt_loss.detach()

            metrics["correct_score"] = scores[encoded.target_index].detach()
            metrics["best_wrong_score"] = scores[torch.arange(scores.numel(), device=scores.device) != encoded.target_index].max().detach()
            metrics["loss"] = loss.detach()
            assert_finite_training_state(wrapper, loss, metrics, step)
            loss.backward()
            assert_finite_trainable_gradients(wrapper, step)
            torch.nn.utils.clip_grad_norm_(
                optimizer_params,
                cfg_float(cfg, "max_grad_norm", 0.3),
                error_if_nonfinite=True,
            )
            optimizer.step()
            wrapper.zero_grad(set_to_none=True)
            assert_finite_trainable_parameters(wrapper, step + 1)

            if step % cfg_int(cfg, "log_every", 10) == 0:
                metric_text = " ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
                print(f"step={step} routing_type={encoded.routing_type} {metric_text}")
            step += 1
            output_dir = cfg.get("output_dir")
            if output_dir and save_every and step % save_every == 0 and step < max_steps:
                checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "phase1", step, cfg)
                print(f"saved_checkpoint={checkpoint_path}")
            if step >= max_steps:
                break
    output_dir = cfg.get("output_dir")
    if output_dir:
        checkpoint_path = save_trainable_checkpoint(wrapper, output_dir, "phase1", step, cfg)
        print(f"saved_checkpoint={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
