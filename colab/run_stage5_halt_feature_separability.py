"""Probe whether recurrent hidden states contain direct/deep routing signal.

This diagnostic extracts masked-mean recurrent hidden features for a curriculum
checkpoint at loop depths 1..N, then trains tiny cross-validated linear probes
to classify ``curriculum_mode == deep_narrow`` versus ``direct``. It answers a
specific question: is the halt predictor failing despite separable features, or
are the features it sees not informative enough for routing?
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs, parse_split, resolve_dtype  # noqa: E402
from models.halting import masked_mean  # noqa: E402
from models.lora import apply_lora_to_recurrent_block  # noqa: E402
from models.recurrent_wrapper import RecurrentQwenForCausalLM  # noqa: E402
from training.checkpointing import load_trainable_checkpoint  # noqa: E402
from training.dataset import JsonlCausalDataset, collate_causal_batch  # noqa: E402


RUN_ID = os.environ.get("STAGE5_HALT_FEATURE_PROBE_RUN_ID") or time.strftime(
    "stage5_halt_feature_probe_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_SUMMARY = os.environ.get("STAGE5_HALT_FEATURE_PROBE_SOURCE_SUMMARY", "").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
MAX_FOLDS = int(os.environ.get("STAGE5_HALT_FEATURE_PROBE_FOLDS", "5"))
EPOCHS = int(os.environ.get("STAGE5_HALT_FEATURE_PROBE_EPOCHS", "300"))
WEIGHT_DECAY = float(os.environ.get("STAGE5_HALT_FEATURE_PROBE_WEIGHT_DECAY", "0.01"))
LR = float(os.environ.get("STAGE5_HALT_FEATURE_PROBE_LR", "0.05"))
SEED = int(os.environ.get("STAGE5_HALT_FEATURE_PROBE_SEED", "13"))
PUSH_RESULTS = os.environ.get("STAGE5_HALT_FEATURE_PROBE_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def current_source_summary() -> Path:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if SOURCE_SUMMARY:
        return resolve_path(SOURCE_SUMMARY)
    if pointer.exists():
        for line in pointer.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return resolve_path(stripped)
    raise FileNotFoundError("Set STAGE5_HALT_FEATURE_PROBE_SOURCE_SUMMARY or config/stage5_current_source_summary.txt")


def update_current_source_summary(summary_path: Path) -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def source_positive_sft(payload: dict[str, Any]) -> Path:
    dataset = payload.get("dataset") or {}
    if dataset.get("source_positive_sft"):
        return resolve_path(str(dataset["source_positive_sft"]))
    config = payload.get("config") or {}
    if config.get("work_dir"):
        return resolve_path(str(config["work_dir"])) / "positive_sft.jsonl"
    raise KeyError("source summary lacks dataset.source_positive_sft and config.work_dir")


def mode_label(row: dict[str, Any]) -> int | None:
    mode = str(row.get("curriculum_mode") or row.get("routing_type") or "")
    if mode == "direct":
        return 0
    if mode == "deep_narrow":
        return 1
    return None


def stratified_folds(labels: torch.Tensor, *, folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), []).append(index)
    actual_folds = max(2, min(folds, *(len(indices) for indices in by_label.values())))
    split: list[list[int]] = [[] for _ in range(actual_folds)]
    for indices in by_label.values():
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            split[offset % actual_folds].append(index)
    return [sorted(fold) for fold in split if fold]


def standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (train_x - mean) / std, (test_x - mean) / std


def linear_probe_accuracy(features: torch.Tensor, labels: torch.Tensor, *, seed: int) -> dict[str, float]:
    folds = stratified_folds(labels, folds=MAX_FOLDS, seed=seed)
    correct = 0
    total = 0
    losses: list[float] = []
    for fold_index, test_indices in enumerate(folds):
        test_set = set(test_indices)
        train_indices = [index for index in range(labels.numel()) if index not in test_set]
        train_x, test_x = standardize(features[train_indices], features[test_indices])
        train_y = labels[train_indices].float()
        test_y = labels[test_indices]
        generator = torch.Generator().manual_seed(seed + fold_index)
        weight = torch.zeros(train_x.shape[1], requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        torch.nn.init.normal_(weight, mean=0.0, std=0.01, generator=generator)
        opt = torch.optim.AdamW([weight, bias], lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(EPOCHS):
            logits = train_x @ weight + bias
            loss = F.binary_cross_entropy_with_logits(logits, train_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = test_x @ weight + bias
            pred = (logits >= 0).long()
            correct += int((pred == test_y).sum().item())
            total += int(test_y.numel())
            losses.append(float(F.binary_cross_entropy_with_logits(logits, test_y.float()).item()))
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "mean_fold_bce": sum(losses) / max(len(losses), 1),
        "folds": len(folds),
    }


def centroid_probe_accuracy(features: torch.Tensor, labels: torch.Tensor, *, seed: int) -> dict[str, float]:
    folds = stratified_folds(labels, folds=MAX_FOLDS, seed=seed)
    correct = 0
    total = 0
    for test_indices in folds:
        test_set = set(test_indices)
        train_indices = [index for index in range(labels.numel()) if index not in test_set]
        train_x, test_x = standardize(features[train_indices], features[test_indices])
        train_y = labels[train_indices]
        centroids = torch.stack([train_x[train_y == label].mean(dim=0) for label in (0, 1)])
        distances = torch.cdist(test_x, centroids)
        pred = distances.argmin(dim=1)
        test_y = labels[test_indices]
        correct += int((pred == test_y).sum().item())
        total += int(test_y.numel())
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total, "folds": len(folds)}


def extract_features(
    *,
    payload: dict[str, Any],
    checkpoint: Path,
    positive_sft: Path,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_load_kwargs(DTYPE, "default"),
    ).to(DEVICE)
    wrapper = RecurrentQwenForCausalLM(model, layer_split=parse_split("6,18")).to(DEVICE)
    adapter_dtype = resolve_dtype(ADAPTER_DTYPE)
    replaced = apply_lora_to_recurrent_block(wrapper, rank=8, alpha=16, dropout=0.0, adapter_dtype=adapter_dtype)
    print(f"lora_recurrent_modules={replaced}", flush=True)
    wrapper.set_trainable_modules_dtype(adapter_dtype)
    load_info = load_trainable_checkpoint(wrapper, checkpoint)
    print(f"loaded_checkpoint={path_for_cli(checkpoint)} loaded_keys={len(load_info['loaded_keys'])}", flush=True)
    wrapper.eval()

    max_loops = int((payload.get("config") or {}).get("max_loops", 4))
    max_length = int((payload.get("config") or {}).get("max_length", 512))
    dataset = JsonlCausalDataset(
        positive_sft,
        tokenizer=tokenizer,
        max_length=max_length,
        max_train_loops=max_loops,
        train_on_prompt=False,
    )
    keep_indices = [idx for idx, row in enumerate(dataset.rows) if mode_label(row) is not None]
    rows = [dataset.rows[idx] for idx in keep_indices]
    labels = torch.tensor([mode_label(row) for row in rows], dtype=torch.long)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(collate_causal_batch, pad_token_id=tokenizer.pad_token_id),
    )
    features: dict[int, list[torch.Tensor]] = {loop: [] for loop in range(1, max_loops + 1)}
    keep_set = set(keep_indices)
    with torch.no_grad():
        for row_index, batch in enumerate(loader):
            if row_index not in keep_set:
                continue
            model_batch = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
            for loop in range(1, max_loops + 1):
                output = wrapper(
                    **model_batch,
                    max_loops=loop,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = output.final_recurrent_hidden[:, 0]
                pooled = masked_mean(hidden, model_batch["attention_mask"]).detach().float().cpu()[0]
                features[loop].append(pooled)
    return {loop: torch.stack(items) for loop, items in features.items()}, labels, rows


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "config", "user.email", "colab-runner@local"], check=False)
    run(["git", "config", "user.name", "Colab Runner"], check=False)
    run(["git", "add", "-f", path_for_cli(RUN_DIR)], check=False)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No halt feature probe outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 halt feature probe {RUN_ID}"])
    run(["git", "pull", "--rebase", "origin", "main"], check=False)
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    source_summary = current_source_summary()
    payload = read_json(source_summary)
    checkpoint = resolve_path(str(payload["phase1_checkpoint"]))
    positive_sft = source_positive_sft(payload)
    features_by_loop, labels, rows = extract_features(
        payload=payload,
        checkpoint=checkpoint,
        positive_sft=positive_sft,
    )
    mode_counts = {
        "direct": int((labels == 0).sum().item()),
        "deep_narrow": int((labels == 1).sum().item()),
    }
    majority = max(mode_counts.values()) / max(sum(mode_counts.values()), 1)
    loop_results: dict[str, Any] = {}
    for loop, features in features_by_loop.items():
        loop_results[str(loop)] = {
            "linear_probe": linear_probe_accuracy(features, labels, seed=SEED + loop),
            "centroid_probe": centroid_probe_accuracy(features, labels, seed=SEED + 100 + loop),
            "feature_dim": int(features.shape[1]),
            "examples": int(features.shape[0]),
        }
    best_linear = max(loop_results.items(), key=lambda item: item[1]["linear_probe"]["accuracy"])
    summary = {
        "run_id": RUN_ID,
        "kind": "stage5_halt_feature_separability",
        "source_summary": path_for_cli(source_summary),
        "checkpoint": path_for_cli(checkpoint),
        "positive_sft": path_for_cli(positive_sft),
        "mode_counts": mode_counts,
        "majority_baseline_accuracy": majority,
        "probe_config": {
            "folds": MAX_FOLDS,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        "loop_results": loop_results,
        "best_linear_loop": int(best_linear[0]),
        "best_linear_accuracy": best_linear[1]["linear_probe"]["accuracy"],
        "interpretation": (
            "features_linearly_separable"
            if best_linear[1]["linear_probe"]["accuracy"] >= majority + 0.10
            else "no_clear_linear_separability"
        ),
    }
    write_json(RUN_DIR / "summary.json", summary)
    update_current_source_summary(RUN_DIR / "summary.json")
    lines = [
        f"# Stage 5 Halt Feature Separability - {RUN_ID}",
        "",
        f"- Source summary: `{summary['source_summary']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Positive SFT: `{summary['positive_sft']}`",
        f"- Mode counts: `{mode_counts}`",
        f"- Majority baseline accuracy: `{majority:.4f}`",
        f"- Best linear loop: `{summary['best_linear_loop']}`",
        f"- Best linear accuracy: `{summary['best_linear_accuracy']:.4f}`",
        f"- Interpretation: `{summary['interpretation']}`",
        "",
        "| Loop | Linear acc | Centroid acc |",
        "|---:|---:|---:|",
    ]
    for loop, result in loop_results.items():
        lines.append(
            f"| {loop} | {result['linear_probe']['accuracy']:.4f} | {result['centroid_probe']['accuracy']:.4f} |"
        )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
