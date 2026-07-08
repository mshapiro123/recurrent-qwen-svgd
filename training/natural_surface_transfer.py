"""Natural-surface transfer data for the synthetic-depth mechanism.

The symbolic mechanism is still an iterated function.  This module changes the
surface only: tables become natural-language premises and symbols become names.
The emitted rows preserve the existing synthetic-depth schema so the active
label, final-symbol, and probe evaluators can run unchanged with
``value_prefix=name:``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from training.synthetic_depth_task import (
    NAME_SYMBOLS,
    SyntheticDepthConfig,
    SyntheticDepthInstance,
    build_dataset,
    build_instance,
    serialized_mapping,
    symbol,
)


Family = Literal["relay", "pointer"]


@dataclass(frozen=True)
class NaturalSurfaceConfig:
    n_symbols: int = 20
    train_max_depth: int = 8
    eval_max_depth: int = 12
    train_rows_per_depth: int = 256
    val_rows_per_depth: int = 64
    eval_rows_per_depth: int = 128
    seed: int = 910_031
    max_target_loops: int = 12
    value_prefix: str = "name:"


def _rng_for(*, seed: int, family: str, split: str, depth: int, row_index: int) -> random.Random:
    family_offsets = {"relay": 0, "pointer": 10_000_000}
    split_offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
    return random.Random(
        int(seed)
        + family_offsets.get(family, 20_000_000)
        + split_offsets.get(split, 3_000_000)
        + int(depth) * 10_000
        + int(row_index)
    )


def _instance_seed(*, seed: int, family: str, split: str, depth: int, row_index: int) -> int:
    return _rng_for(seed=seed, family=family, split=split, depth=depth, row_index=row_index).randrange(2**31)


def verbalize_relay(instance: SyntheticDepthInstance, rng: random.Random) -> dict[str, Any]:
    """Temporal counter surface: one key pass per day."""

    name = lambda idx: symbol(idx, prefix="name:")
    premises = [
        f"{name(idx)} always passes the key to {name(instance.mapping[idx])}."
        for idx in range(instance.n_symbols)
    ]
    rng.shuffle(premises)
    question = (
        "\n".join(premises)
        + "\n\n"
        + f"{name(instance.start)} starts with the key. Each day it is passed once. "
        + f"After {instance.depth} days, who has the key?"
    )
    steps = [
        f"After day {loop_idx}, {name(target)} has the key."
        for loop_idx, target in enumerate(instance.orbit[1:], start=1)
    ]
    return {
        "premises": premises,
        "question": question,
        "step_sentences": steps,
        "answer_text": name(instance.target),
        "k_star": instance.depth,
        "latent_targets": [name(target) for target in instance.orbit[1:]],
    }


def verbalize_pointer(instance: SyntheticDepthInstance, rng: random.Random) -> dict[str, Any]:
    """Spatial counter surface: one note reference per hop."""

    name = lambda idx: symbol(idx, prefix="name:")
    premises = [
        f"{name(idx)}'s note points to {name(instance.mapping[idx])}."
        for idx in range(instance.n_symbols)
    ]
    rng.shuffle(premises)
    question = (
        "\n".join(premises)
        + "\n\n"
        + f"Start at {name(instance.start)} and follow the notes exactly {instance.depth} times. "
        + "Where do you end up?"
    )
    steps = [
        f"After hop {loop_idx}, you are at {name(target)}."
        for loop_idx, target in enumerate(instance.orbit[1:], start=1)
    ]
    return {
        "premises": premises,
        "question": question,
        "step_sentences": steps,
        "answer_text": name(instance.target),
        "k_star": instance.depth,
        "latent_targets": [name(target) for target in instance.orbit[1:]],
    }


def verbalize(instance: SyntheticDepthInstance, *, family: Family, rng: random.Random) -> dict[str, Any]:
    if family == "relay":
        return verbalize_relay(instance, rng)
    if family == "pointer":
        return verbalize_pointer(instance, rng)
    raise ValueError(f"unknown verbal family: {family!r}")


def build_verbal_chain_row(
    instance: SyntheticDepthInstance,
    *,
    family: Family,
    split: str,
    rng: random.Random,
    max_target_loops: int,
) -> dict[str, Any]:
    rendered = verbalize(instance, family=family, rng=rng)
    max_active = min(int(max_target_loops), int(instance.depth))
    name_orbit = [symbol(value, prefix="name:") for value in instance.orbit]
    return {
        "id": instance.instance_id,
        "instance_id": instance.instance_id,
        "question": rendered["question"],
        "target": symbol(instance.target, prefix="name:"),
        "depth": int(instance.depth),
        "synthetic_depth": int(instance.depth),
        "start": symbol(instance.start, prefix="name:"),
        "mapping": serialized_mapping(instance, value_prefix="name:"),
        "orbit": name_orbit,
        "n_symbols": int(instance.n_symbols),
        "score_target": "full_symbols",
        "prompt_style": "question_only",
        "intermediate_chain_supervision": True,
        "chain_symbol_by_loop": {
            str(loop_idx): name_orbit[loop_idx]
            for loop_idx in range(1, max_active + 1)
        },
        "chain_answer_by_loop": {
            str(loop_idx): name_orbit[loop_idx]
            for loop_idx in range(1, max_active + 1)
        },
        "loop_completions": [f" {name_orbit[loop_idx]}" for loop_idx in range(1, max_active + 1)],
        "completion": f" {name_orbit[max_active]}",
        "target_loop_count": max_active,
        "synthetic_task": "natural_surface_iterated_function",
        "verbal_surface_family": family,
        "split": split,
        "k_star": int(rendered["k_star"]),
        "answer_text": rendered["answer_text"],
        "latent_targets": rendered["latent_targets"][:max_active],
        "step_sentences": rendered["step_sentences"][:max_active],
    }


def build_verbal_rows(
    *,
    family: Family,
    split: str,
    n_symbols: int,
    max_depth: int,
    rows_per_depth: int,
    seed: int,
    max_target_loops: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth in range(1, max_depth + 1):
        for row_index in range(rows_per_depth):
            instance_id = f"{split}_{family}_d{depth:02d}_{row_index:05d}"
            instance = build_instance(
                instance_id=instance_id,
                n_symbols=n_symbols,
                depth=depth,
                seed=_instance_seed(
                    seed=seed,
                    family=family,
                    split=split,
                    depth=depth,
                    row_index=row_index,
                ),
                split=split,
                num_choices=4,
            )
            rows.append(
                build_verbal_chain_row(
                    instance,
                    family=family,
                    split=split,
                    rng=_rng_for(seed=seed, family=family, split=split, depth=depth, row_index=row_index),
                    max_target_loops=max_target_loops,
                )
            )
    return rows


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def manifest_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("id") or row.get("instance_id")) for row in rows]
    return {
        "rows": len(rows),
        "depth_counts": {
            str(depth): sum(1 for row in rows if int(row["depth"]) == depth)
            for depth in sorted({int(row["depth"]) for row in rows})
        },
        "row_id_sha256": sha256_lines(ids),
        "row_sha256": sha256_lines([canonical_row(row) for row in rows]),
    }


def assert_verbal_row_invariants(row: dict[str, Any]) -> None:
    depth = int(row["depth"])
    orbit = list(row["orbit"])
    if len(orbit) != depth + 1:
        raise AssertionError(f"orbit length does not equal depth+1 for {row.get('id')}")
    if len(set(orbit)) != len(orbit):
        raise AssertionError(f"orbit prefix is not distinct for {row.get('id')}")
    if row["answer_text"] != orbit[depth]:
        raise AssertionError(f"answer_text does not match orbit target for {row.get('id')}")
    if int(row["k_star"]) != depth:
        raise AssertionError(f"k_star does not match depth for {row.get('id')}")
    targets = list(row["latent_targets"])
    if targets != orbit[1 : min(depth, int(row["target_loop_count"])) + 1]:
        raise AssertionError(f"latent targets do not match orbit for {row.get('id')}")
    for target, sentence in zip(targets, row["step_sentences"]):
        if target not in sentence:
            raise AssertionError(f"step sentence does not name latent target {target!r} for {row.get('id')}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_synthetic_rehearsal_rows(config: NaturalSurfaceConfig) -> list[dict[str, Any]]:
    synthetic_config = SyntheticDepthConfig(
        n_symbols=min(16, config.n_symbols),
        max_depth=config.train_max_depth,
        rows_per_depth=config.train_rows_per_depth,
        seed=config.seed + 50_000_000,
        num_choices=4,
        max_target_loops=config.train_max_depth,
        value_prefix="letter:",
    )
    rows = []
    for instance in build_dataset(synthetic_config, split="train"):
        from training.synthetic_depth_task import build_chain_symbol_sft_row

        rows.append(
            build_chain_symbol_sft_row(
                instance,
                max_target_loops=synthetic_config.max_target_loops,
                value_prefix=synthetic_config.value_prefix,
            )
            | {"curriculum_family": "synthetic_rehearsal"}
        )
    return rows


def write_natural_surface_dataset(*, output_dir: str | Path, config: NaturalSurfaceConfig) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_relay = build_verbal_rows(
        family="relay",
        split="train",
        n_symbols=config.n_symbols,
        max_depth=config.train_max_depth,
        rows_per_depth=config.train_rows_per_depth,
        seed=config.seed,
        max_target_loops=config.train_max_depth,
    )
    val_relay = build_verbal_rows(
        family="relay",
        split="val",
        n_symbols=config.n_symbols,
        max_depth=config.train_max_depth,
        rows_per_depth=config.val_rows_per_depth,
        seed=config.seed,
        max_target_loops=config.train_max_depth,
    )
    eval_relay = build_verbal_rows(
        family="relay",
        split="test",
        n_symbols=config.n_symbols,
        max_depth=config.eval_max_depth,
        rows_per_depth=config.eval_rows_per_depth,
        seed=config.seed,
        max_target_loops=config.eval_max_depth,
    )
    eval_pointer = build_verbal_rows(
        family="pointer",
        split="test",
        n_symbols=config.n_symbols,
        max_depth=config.eval_max_depth,
        rows_per_depth=config.eval_rows_per_depth,
        seed=config.seed,
        max_target_loops=config.eval_max_depth,
    )
    synthetic_rehearsal = build_synthetic_rehearsal_rows(config)
    rung0_train_mix = [
        row | {"curriculum_family": "relay_verbal"}
        for row in train_relay
    ] + synthetic_rehearsal
    random.Random(config.seed + 77_777).shuffle(rung0_train_mix)

    all_sets = {
        "train_relay_chain_symbol_sft": train_relay,
        "val_relay_chain_symbol_sft": val_relay,
        "relay_test_chain_mcq": eval_relay,
        "pointer_test_chain_mcq": eval_pointer,
        "synthetic_rehearsal_chain_symbol_sft": synthetic_rehearsal,
        "rung0_train_mix_chain_symbol_sft": rung0_train_mix,
    }
    for label, rows in all_sets.items():
        for row in rows:
            if row.get("verbal_surface_family") in {"relay", "pointer"}:
                assert_verbal_row_invariants(row)
        write_jsonl(out / f"{label}.jsonl", rows)

    files = {label: f"{label}.jsonl" for label in all_sets}
    manifests = {label: manifest_for_rows(rows) for label, rows in all_sets.items()}
    summary = {
        "kind": "stage5_natural_surface_transfer_dataset",
        "status": "finished",
        "config": asdict(config),
        "name_symbols": list(NAME_SYMBOLS[: config.n_symbols]),
        "families": {
            "relay": "temporal key passing, trained at rung zero",
            "pointer": "spatial note following, held out for zero-shot cross-template read",
        },
        "files": files,
        "manifests": manifests,
        "gates": {
            "relay_train_depths": [1, config.train_max_depth],
            "relay_eval_depths": [1, config.eval_max_depth],
            "pointer_eval_depths": [1, config.eval_max_depth],
            "active_label_bar": {"correct": 91, "total": 128, "accuracy": 91 / 128},
            "synthetic_nonregression_floor_delta": 0.03,
        },
        "standing_rules": {
            "text_born_from_steps": True,
            "answer_space": "single-token names must be tokenizer-verified before training",
            "segmentation_required": False,
            "prontoqa": "queued behind verbal generator as weak-supervision external-data track",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def verify_single_token_names(tokenizer: Any, *, n_symbols: int) -> dict[str, Any]:
    rows = []
    all_pass = True
    for name in NAME_SYMBOLS[:n_symbols]:
        bare = tokenizer(name, add_special_tokens=False)["input_ids"]
        spaced = tokenizer(" " + name, add_special_tokens=False)["input_ids"]
        row = {
            "symbol": name,
            "bare_token_count": len(bare),
            "space_prefixed_token_count": len(spaced),
            "pass": len(bare) == 1 and len(spaced) == 1,
        }
        rows.append(row)
        all_pass = all_pass and bool(row["pass"])
    return {"all_single_token": all_pass, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_symbols", type=int, default=20)
    parser.add_argument("--train_max_depth", type=int, default=8)
    parser.add_argument("--eval_max_depth", type=int, default=12)
    parser.add_argument("--train_rows_per_depth", type=int, default=256)
    parser.add_argument("--val_rows_per_depth", type=int, default=64)
    parser.add_argument("--eval_rows_per_depth", type=int, default=128)
    parser.add_argument("--seed", type=int, default=910_031)
    args = parser.parse_args()
    summary = write_natural_surface_dataset(
        output_dir=args.output_dir,
        config=NaturalSurfaceConfig(
            n_symbols=args.n_symbols,
            train_max_depth=args.train_max_depth,
            eval_max_depth=args.eval_max_depth,
            train_rows_per_depth=args.train_rows_per_depth,
            val_rows_per_depth=args.val_rows_per_depth,
            eval_rows_per_depth=args.eval_rows_per_depth,
            seed=args.seed,
            max_target_loops=max(args.train_max_depth, args.eval_max_depth),
        ),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
