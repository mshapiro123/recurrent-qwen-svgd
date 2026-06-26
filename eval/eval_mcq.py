"""Multiple-choice likelihood scorer for base and recurrent Qwen variants.

Input JSONL rows should contain:

    {"id": "...", "question": "...", "choices": {"A": "...", ...}, "answer": "A"}

`choices` may also be a list, in which case labels A-F are assigned in order.
The answer may be either the label or the exact choice text.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype
from eval.eval_best_of_k_jsonl import parse_optional_float
from models.lora import apply_lora_to_qwen_layers, apply_lora_to_recurrent_block
from models.recurrent_wrapper import RecurrentQwenForCausalLM
from training.checkpointing import load_trainable_checkpoint


LABELS = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class MCQExample:
    id: str
    question: str
    choices: list[tuple[str, str]]
    answer: str


@dataclass(frozen=True)
class CompletionScore:
    scores: torch.Tensor
    diagnostics: dict[str, Any]


def read_examples(path: str | Path) -> list[MCQExample]:
    examples: list[MCQExample] = []
    for idx, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        choices = option_items(row)
        answer = normalize_answer(row.get("answer") or row.get("label") or row.get("target"), choices)
        question = row.get("question") or row.get("prompt")
        if question is None:
            raise ValueError(f"Row {idx} missing question/prompt")
        examples.append(
            MCQExample(
                id=str(row.get("id") or row.get("name") or idx),
                question=str(question),
                choices=choices,
                answer=answer,
            )
        )
    return examples


def option_items(row: dict[str, Any]) -> list[tuple[str, str]]:
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, dict):
        return [(str(label).strip(), str(text)) for label, text in choices.items()]
    if isinstance(choices, list):
        if len(choices) > len(LABELS):
            raise ValueError(f"Too many choices ({len(choices)}); max supported is {len(LABELS)}")
        return list(zip(LABELS, [str(item) for item in choices]))
    raise ValueError("Each row must contain choices/options as a list or dict")


def normalize_answer(answer: Any, choices: list[tuple[str, str]]) -> str:
    if answer is None:
        raise ValueError("Each row must contain answer/label/target")
    raw = str(answer).strip()
    labels = {label for label, _ in choices}
    if raw in labels:
        return raw
    raw_folded = raw.casefold()
    for label, text in choices:
        if raw_folded == text.strip().casefold():
            return label
    raise ValueError(f"Answer {raw!r} does not match labels or choice text")


def format_prompt(example: MCQExample, prompt_style: str) -> str:
    if prompt_style == "question_only":
        return example.question.rstrip() + "\nAnswer:"
    if prompt_style != "with_options":
        raise ValueError(f"Unknown prompt_style={prompt_style}")
    rendered = "\n".join(f"{label}. {text}" for label, text in example.choices)
    return f"{example.question.rstrip()}\n{rendered}\nAnswer:"


def format_completion(label: str, text: str, score_target: str) -> str:
    if score_target == "label":
        return f" {label}"
    if score_target == "option_text":
        return f" {text}"
    if score_target == "label_and_text":
        return f" {label}. {text}"
    raise ValueError(f"Unknown score_target={score_target}")


def parse_base_lora_layer_range(value: str, num_layers: int) -> tuple[int, int]:
    if value.lower() in {"", "auto", "all"}:
        return 0, num_layers
    left, right = value.split(",", maxsplit=1)
    start, end = int(left), int(right)
    if not 0 <= start < end <= num_layers:
        raise ValueError(f"Invalid base LoRA layer range {value!r} for {num_layers} layers")
    return start, end


def load_base_model(args: argparse.Namespace, *, load_dense_lora_checkpoint: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    if load_dense_lora_checkpoint and args.checkpoint:
        start, end = parse_base_lora_layer_range(args.base_lora_layer_range, len(model.model.layers))
        replaced = apply_lora_to_qwen_layers(
            model,
            start_layer=start,
            end_layer=end,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            adapter_dtype=resolve_dtype(args.adapter_dtype),
        )
        print(f"dense_lora_modules={replaced} layer_range={start},{end}")
        load_info = load_trainable_checkpoint(model, args.checkpoint)
        print(f"loaded_base_lora_checkpoint={args.checkpoint} loaded_keys={len(load_info['loaded_keys'])}")
        if load_info["skipped"]:
            print(f"skipped_keys={len(load_info['skipped'])}")
    model.eval()
    return model


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_recurrent_wrapper(args: argparse.Namespace, checkpoint: str | None) -> RecurrentQwenForCausalLM:
    model = load_base_model(args, load_dense_lora_checkpoint=False)
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split(args.split)).to(args.device)
    adapter_dtype = resolve_dtype(args.adapter_dtype)
    replaced = apply_lora_to_recurrent_block(
        wrapper,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
        adapter_dtype=adapter_dtype,
    )
    print(f"lora_recurrent_modules={replaced}")
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    if checkpoint:
        load_info = load_trainable_checkpoint(wrapper, checkpoint)
        print(f"loaded_checkpoint={checkpoint} loaded_keys={len(load_info['loaded_keys'])}")
        if load_info["skipped"]:
            print(f"skipped_keys={len(load_info['skipped'])}")
    wrapper.eval()
    return wrapper


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, normalize: bool) -> torch.Tensor:
    if logits.shape[0] != labels.shape[0]:
        if labels.shape[0] != 1 or logits.shape[0] % labels.shape[0] != 0:
            raise ValueError(f"Cannot align logits {tuple(logits.shape)} with labels {tuple(labels.shape)}")
        labels = labels.expand(logits.shape[0], -1).contiguous()
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    scores = (token_logprobs * mask).sum(dim=-1)
    if normalize:
        scores = scores / mask.sum(dim=-1).clamp_min(1)
    return scores


def select_forced_loop_logits(output: Any, forced_loop_count: int) -> torch.Tensor:
    """Return logits for one 1-based loop from a recurrent output.

    ``RecurrentQwenForCausalLM`` exposes loop logits as
    ``[batch, trajectories, loops, seq, vocab]`` when ``return_loop_logits`` is
    enabled. The MCQ scorer expects trajectory-flattened logits.
    """

    loop_logits = getattr(output, "loop_logits", None)
    if loop_logits is None:
        raise RuntimeError("Expected loop_logits when --forced_loop_count is set")
    loop_index = int(forced_loop_count) - 1
    if loop_index < 0 or loop_index >= loop_logits.shape[2]:
        raise RuntimeError(
            f"Cannot select forced loop {forced_loop_count}; loop_logits shape is {tuple(loop_logits.shape)}"
        )
    selected = loop_logits[:, :, loop_index]
    return selected.reshape(-1, *selected.shape[2:])


def score_completion(
    model_or_wrapper,
    tokenizer,
    prompt: str,
    completion: str,
    args: argparse.Namespace,
) -> CompletionScore:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    encoded = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=True).to(args.device)
    labels = encoded["input_ids"].clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100

    forced_loop_count = int(args.forced_loop_count or 0)
    forward_max_loops = max(int(args.max_loops), forced_loop_count) if forced_loop_count else int(args.max_loops)

    with torch.no_grad():
        if args.mode == "base":
            output = model_or_wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                use_cache=False,
                return_dict=True,
            )
        else:
            output = model_or_wrapper(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=None,
                max_loops=forward_max_loops,
                num_trajectories=args.num_trajectories,
                sample_latents=args.sample_latents,
                latent_injection_mode=args.latent_injection_mode,
                particle_update_mode=args.particle_update_mode,
                particle_init_noise=args.particle_init_noise,
                svgd_eps=args.svgd_eps,
                svgd_repulsion_scale=args.svgd_repulsion_scale,
                svgd_bandwidth=args.svgd_bandwidth,
                svgd_bandwidth_floor=args.svgd_bandwidth_floor,
                svgd_repulsion_max_norm=args.svgd_repulsion_max_norm,
                svgd_kernel_projection_dim=args.svgd_kernel_projection_dim,
                svgd_kernel_projection_path=args.svgd_kernel_projection_path,
                svgd_kernel_geometry=args.svgd_kernel_geometry,
                svgd_projection_seed=args.svgd_projection_seed,
                use_learned_loop_control=args.use_learned_loop_control,
                reentry_tail_damper_path=args.reentry_tail_damper_path,
                reentry_tail_damper_strength=args.reentry_tail_damper_strength,
                return_loop_logits=bool(forced_loop_count),
                use_cache=False,
                return_dict=True,
            )
    if forced_loop_count:
        if args.mode == "base":
            raise RuntimeError("--forced_loop_count is only valid for recurrent modes")
        logits = select_forced_loop_logits(output, forced_loop_count)
    else:
        logits = output.logits
        trajectory_logits = getattr(output, "trajectory_logits", None)
        if args.num_trajectories > 1 and trajectory_logits is None:
            raise RuntimeError("Expected trajectory_logits when num_trajectories > 1")
        if trajectory_logits is not None:
            logits = trajectory_logits.reshape(-1, *trajectory_logits.shape[2:])
    diagnostics = extract_loop_diagnostics(output) if args.include_loop_diagnostics else {}
    if forced_loop_count and args.include_loop_diagnostics:
        diagnostics["forced_loop_count"] = forced_loop_count
    scores = sequence_logprobs(logits, labels, normalize=args.normalize_option_score).cpu()
    if args.num_trajectories > 1 and scores.numel() != args.num_trajectories:
        raise RuntimeError(f"Expected {args.num_trajectories} trajectory scores, got {scores.numel()}")
    return CompletionScore(scores=scores, diagnostics=diagnostics)


def scalarize_tensor(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().float().cpu()
        if detached.numel() == 1:
            return float(detached.item())
        return [float(item) for item in detached.flatten().tolist()]
    return value


def extract_loop_diagnostics(output: Any) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    metrics = getattr(output, "metrics", {}) or {}
    for key in ("mean_expected_loops", "mean_halt_entropy", "trajectory_diversity"):
        if key in metrics:
            diagnostics[key] = scalarize_tensor(metrics[key])
    expected_loops = getattr(output, "expected_loops", None)
    if expected_loops is not None:
        diagnostics["expected_loops"] = scalarize_tensor(expected_loops)
    halting_weights = getattr(output, "halting_weights", None)
    if halting_weights is not None:
        diagnostics["mean_halting_weights"] = scalarize_tensor(halting_weights.detach().float().mean(dim=tuple(range(halting_weights.dim() - 1))))
    return diagnostics


def aggregate(scores: torch.Tensor, method: str) -> float:
    if method == "mean":
        return float(scores.mean().item())
    if method == "max":
        return float(scores.max().item())
    raise ValueError(f"Unknown aggregate={method}")


def predict_from_scores(option_scores: dict[str, torch.Tensor], aggregate_method: str) -> tuple[str, dict[str, float]]:
    if aggregate_method == "vote":
        labels = list(option_scores)
        per_traj = torch.stack([option_scores[label] for label in labels], dim=0)
        winners = per_traj.argmax(dim=0)
        votes = {label: int(winners.eq(idx).sum().item()) for idx, label in enumerate(labels)}
        mean_scores = {label: float(option_scores[label].mean().item()) for label in labels}
        prediction = max(labels, key=lambda label: (votes[label], mean_scores[label]))
        return prediction, mean_scores
    scalar_scores = {label: aggregate(scores, aggregate_method) for label, scores in option_scores.items()}
    return max(scalar_scores, key=scalar_scores.get), scalar_scores


def mean_numeric(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, (int, float))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def aggregate_loop_diagnostics(
    option_diagnostics: dict[str, dict[str, Any]],
    answer: str,
    prediction: str,
) -> dict[str, Any]:
    if not option_diagnostics:
        return {}
    all_expected = [
        diagnostics.get("mean_expected_loops")
        for diagnostics in option_diagnostics.values()
        if diagnostics.get("mean_expected_loops") is not None
    ]
    all_entropy = [
        diagnostics.get("mean_halt_entropy")
        for diagnostics in option_diagnostics.values()
        if diagnostics.get("mean_halt_entropy") is not None
    ]
    result: dict[str, Any] = {
        "mean_expected_loops": mean_numeric(all_expected),
        "mean_halt_entropy": mean_numeric(all_entropy),
    }
    if answer in option_diagnostics:
        answer_diag = option_diagnostics[answer]
        result["answer_expected_loops"] = answer_diag.get("mean_expected_loops")
        result["answer_halt_entropy"] = answer_diag.get("mean_halt_entropy")
    if prediction in option_diagnostics:
        pred_diag = option_diagnostics[prediction]
        result["prediction_expected_loops"] = pred_diag.get("mean_expected_loops")
        result["prediction_halt_entropy"] = pred_diag.get("mean_halt_entropy")
    return {key: value for key, value in result.items() if value is not None}


def append_jsonl(path: str | Path | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--mode", choices=("base", "phase1", "phase2"), default="base")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Trainable checkpoint. For phase1/phase2 this is a recurrent adapter; "
            "for base mode this may be a dense base-model LoRA checkpoint."
        ),
    )
    parser.add_argument(
        "--allow_untrained_recurrent",
        action="store_true",
        help=(
            "Permit phase1/phase2 scoring without a checkpoint. This is intended only for "
            "no-training surgery viability probes; normal trained evaluations should pass --checkpoint."
        ),
    )
    parser.add_argument("--output_jsonl")
    parser.add_argument(
        "--quiet_rows",
        action="store_true",
        help="Do not print each scored row to stdout. Rows are still written when --output_jsonl is set.",
    )
    parser.add_argument("--prompt_style", choices=("with_options", "question_only"), default="with_options")
    parser.add_argument("--score_target", choices=("label", "option_text", "label_and_text"), default="label")
    parser.add_argument("--aggregate", choices=("mean", "max", "vote"), default="mean")
    parser.add_argument(
        "--aggregates",
        help="Optional comma-separated aggregate list, e.g. mean,max,vote. Reuses option scores in one model pass.",
    )
    parser.add_argument("--normalize_option_score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include_loop_diagnostics",
        action="store_true",
        help="For recurrent modes, record expected loop and halting telemetry for each scored option.",
    )
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument(
        "--forced_loop_count",
        type=int,
        help=(
            "For recurrent modes, ignore halting weights and score completions from this "
            "1-based recurrent loop's logits. Used for forced-depth diagnostics."
        ),
    )
    parser.add_argument("--num_trajectories", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--base_lora_layer_range",
        default="6,18",
        help="Layer range for loading a dense base-mode LoRA checkpoint. Use all/auto or start,end.",
    )
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--sample_latents", action="store_true")
    parser.add_argument("--latent_injection_mode", default="post", choices=("pre", "post", "both"))
    parser.add_argument("--particle_update_mode", default="none", choices=("none", "svgd"))
    parser.add_argument("--particle_init_noise", type=float, default=0.0)
    parser.add_argument("--svgd_eps", type=float, default=1.0)
    parser.add_argument("--svgd_repulsion_scale", type=float, default=0.5)
    parser.add_argument("--svgd_bandwidth", default="median")
    parser.add_argument("--svgd_bandwidth_floor", type=float, default=1e-6)
    parser.add_argument("--svgd_repulsion_max_norm", type=parse_optional_float)
    parser.add_argument("--svgd_kernel_projection_dim", type=int)
    parser.add_argument("--svgd_kernel_projection_path")
    parser.add_argument("--svgd_kernel_geometry", default="euclidean", choices=("euclidean", "spherical"))
    parser.add_argument("--svgd_projection_seed", type=int, default=0)
    parser.add_argument("--use_learned_loop_control", action="store_true")
    parser.add_argument("--reentry_tail_damper_path")
    parser.add_argument("--reentry_tail_damper_strength", type=float, default=0.0)
    args = parser.parse_args()

    if args.mode != "base" and not args.checkpoint and not args.allow_untrained_recurrent:
        raise SystemExit("--checkpoint is required for phase1/phase2 modes")
    if args.mode == "phase1" and args.num_trajectories != 1:
        raise SystemExit("phase1 mode expects --num_trajectories 1")
    if args.forced_loop_count is not None:
        if args.forced_loop_count < 1:
            raise SystemExit("--forced_loop_count must be >= 1")
        if args.mode == "base":
            raise SystemExit("--forced_loop_count is only valid for phase1/phase2 modes")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_or_wrapper = (
        load_base_model(args, load_dense_lora_checkpoint=True)
        if args.mode == "base"
        else load_recurrent_wrapper(args, args.checkpoint)
    )

    examples = read_examples(args.data_jsonl)
    aggregates = (
        [item.strip() for item in args.aggregates.split(",") if item.strip()]
        if args.aggregates
        else [args.aggregate]
    )
    allowed_aggregates = {"mean", "max", "vote"}
    unknown = set(aggregates) - allowed_aggregates
    if unknown:
        raise SystemExit(f"Unknown aggregates: {sorted(unknown)}")

    correct = {aggregate_name: 0 for aggregate_name in aggregates}
    for example in examples:
        prompt = format_prompt(example, args.prompt_style)
        option_scores: dict[str, torch.Tensor] = {}
        option_loop_diagnostics: dict[str, dict[str, Any]] = {}
        for label, text in example.choices:
            completion = format_completion(label, text, args.score_target)
            completion_score = score_completion(model_or_wrapper, tokenizer, prompt, completion, args)
            option_scores[label] = completion_score.scores
            if completion_score.diagnostics:
                option_loop_diagnostics[label] = completion_score.diagnostics
        trajectory_scores = {label: [float(x) for x in scores.tolist()] for label, scores in option_scores.items()}
        for aggregate_name in aggregates:
            prediction, scalar_scores = predict_from_scores(option_scores, aggregate_name)
            hit = prediction == example.answer
            correct[aggregate_name] += int(hit)
            row = {
                "id": example.id,
                "mode": args.mode,
                "checkpoint": args.checkpoint,
                "prompt_style": args.prompt_style,
                "score_target": args.score_target,
                "aggregate": aggregate_name,
                "num_trajectories": args.num_trajectories,
                "forced_loop_count": args.forced_loop_count,
                "prediction": prediction,
                "answer": example.answer,
                "hit": hit,
                "scores": scalar_scores,
                "trajectory_scores": trajectory_scores,
            }
            if option_loop_diagnostics:
                row["option_loop_diagnostics"] = option_loop_diagnostics
                row["loop_diagnostics"] = aggregate_loop_diagnostics(
                    option_loop_diagnostics,
                    answer=example.answer,
                    prediction=prediction,
                )
            if not args.quiet_rows:
                print(json.dumps(row, ensure_ascii=False))
            append_jsonl(args.output_jsonl, row)

    for aggregate_name in aggregates:
        accuracy = correct[aggregate_name] / max(len(examples), 1)
        print(f"aggregate={aggregate_name} accuracy={accuracy:.4f} correct={correct[aggregate_name]} total={len(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
