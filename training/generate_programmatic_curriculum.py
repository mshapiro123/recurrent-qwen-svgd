"""Generate verified direct/deep-narrow curriculum records without model calls.

These records implement Stage 7 of the curriculum plan: use constructed
procedures when we need reliable depth labels and verified answers. The output
is the typed curriculum schema consumed by ``training/prepare_curriculum_jsonl.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Operation:
    verb: str
    symbol: str
    value: int

    def apply(self, current: int) -> int:
        if self.symbol == "+":
            return current + self.value
        if self.symbol == "-":
            return current - self.value
        if self.symbol == "*":
            return current * self.value
        raise ValueError(f"Unsupported operation {self.symbol!r}")

    def phrase(self, first: bool = False) -> str:
        prefix = "" if first else "Then "
        return f"{prefix}{self.verb} {self.value}."

    def step_text(self, before: int, after: int) -> str:
        return f"{before} {self.symbol} {self.value} = {after}"


def parse_range(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected MIN,MAX")
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 1 or hi < lo:
        raise argparse.ArgumentTypeError("Expected 1 <= MIN <= MAX")
    return lo, hi


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def random_operation(rng: random.Random) -> Operation:
    kind = rng.choices(["+", "-", "*"], weights=[5, 4, 2], k=1)[0]
    if kind == "+":
        return Operation("Add", "+", rng.randint(2, 19))
    if kind == "-":
        return Operation("Subtract", "-", rng.randint(2, 19))
    return Operation("Multiply by", "*", rng.randint(2, 5))


def generate_operations(
    rng: random.Random,
    *,
    steps: int,
    start: int,
    max_abs_value: int,
) -> tuple[list[Operation], list[tuple[int, int, Operation]]]:
    for _attempt in range(500):
        current = start
        operations: list[Operation] = []
        trace: list[tuple[int, int, Operation]] = []
        ok = True
        for _ in range(steps):
            op = random_operation(rng)
            nxt = op.apply(current)
            if abs(nxt) > max_abs_value:
                ok = False
                break
            operations.append(op)
            trace.append((current, nxt, op))
            current = nxt
        if ok:
            return operations, trace
    raise RuntimeError(f"Could not generate bounded operation chain with steps={steps}")


def depth_to_target_loop(mode: str, steps: int, *, max_target_loops: int) -> int:
    if mode == "direct":
        return 1
    return max(2, min(max_target_loops, 1 + (steps + 1) // 3))


def solution_text(trace: list[tuple[int, int, Operation]]) -> str:
    lines = ["Follow the operations in order:"]
    for index, (before, after, op) in enumerate(trace, start=1):
        lines.append(f"{index}. {op.step_text(before, after)}")
    lines.append(f"ANSWER: {trace[-1][1]}")
    return "\n".join(lines)


def problem_statement(start: int, operations: list[Operation]) -> str:
    operation_text = " ".join(op.phrase(first=index == 0) for index, op in enumerate(operations))
    return f"Start with {start}. {operation_text} What is the final value?"


def make_arithmetic_record(
    *,
    rng: random.Random,
    index: int,
    mode: str,
    step_range: tuple[int, int],
    max_abs_value: int,
    max_target_loops: int,
) -> dict[str, Any]:
    steps = rng.randint(step_range[0], step_range[1])
    start = rng.randint(2, 25)
    operations, trace = generate_operations(rng, steps=steps, start=start, max_abs_value=max_abs_value)
    answer = trace[-1][1]
    answer_normalized = str(answer)
    target_loop = depth_to_target_loop(mode, steps, max_target_loops=max_target_loops)
    role = "positive_direct" if mode == "direct" else "positive_depth"
    difficulty_pass_rate = 0.9 if mode == "direct" else max(0.05, round(0.9 - 0.08 * steps, 3))

    return {
        "id": f"programmatic-arithmetic-{mode}-{index:06d}",
        "domain": "math",
        "statement": problem_statement(start, operations),
        "answer": {
            "value": str(answer),
            "verified_by": ["constructed", "python_eval"],
            "confidence": "high",
        },
        "difficulty": {
            "pass_rate": difficulty_pass_rate,
            "reference_model": "constructed_proxy",
        },
        "width_signature": {
            "methods": ["arithmetic_chain"],
            "width": 1,
        },
        "depth": {
            "per_method": {"arithmetic_chain": steps},
            "min_steps": steps,
        },
        "mode": mode,
        "target_loop_count": target_loop,
        "decontaminated": True,
        "source_dataset": "programmatic_arithmetic_curriculum",
        "traces": [
            {
                "role": role,
                "method": "arithmetic_chain",
                "correct": True,
                "natural": True,
                "steps": steps,
                "source_model": "programmatic_generator",
                "answer_match": {
                    "matched": True,
                    "source": "constructed_python_eval",
                    "parsed_answer": answer_normalized,
                    "parsed_answer_normalized": answer_normalized,
                    "verified_answer_normalized": answer_normalized,
                },
                "text": solution_text(trace),
            }
        ],
    }


def generate_records(
    *,
    num_direct: int,
    num_deep_narrow: int,
    direct_steps: tuple[int, int],
    deep_steps: tuple[int, int],
    seed: int,
    max_abs_value: int,
    max_target_loops: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for index in range(num_direct):
        records.append(
            make_arithmetic_record(
                rng=rng,
                index=index,
                mode="direct",
                step_range=direct_steps,
                max_abs_value=max_abs_value,
                max_target_loops=max_target_loops,
            )
        )
    for index in range(num_deep_narrow):
        records.append(
            make_arithmetic_record(
                rng=rng,
                index=index,
                mode="deep_narrow",
                step_range=deep_steps,
                max_abs_value=max_abs_value,
                max_target_loops=max_target_loops,
            )
        )
    rng.shuffle(records)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, int] = {}
    target_loops: dict[str, list[int]] = {}
    depths: dict[str, list[int]] = {}
    for record in records:
        mode = str(record["mode"])
        by_mode[mode] = by_mode.get(mode, 0) + 1
        target_loops.setdefault(mode, []).append(int(record["target_loop_count"]))
        depths.setdefault(mode, []).append(int(record["depth"]["min_steps"]))
    return {
        "records": len(records),
        "by_mode": by_mode,
        "target_loop_ranges": {
            mode: {"min": min(values), "max": max(values)}
            for mode, values in sorted(target_loops.items())
        },
        "depth_ranges": {
            mode: {"min": min(values), "max": max(values)}
            for mode, values in sorted(depths.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--num_direct", type=int, default=100)
    parser.add_argument("--num_deep_narrow", type=int, default=100)
    parser.add_argument("--direct_steps", type=parse_range, default=(1, 2))
    parser.add_argument("--deep_steps", type=parse_range, default=(5, 9))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_abs_value", type=int, default=500)
    parser.add_argument("--max_target_loops", type=int, default=4)
    args = parser.parse_args(argv)

    records = generate_records(
        num_direct=args.num_direct,
        num_deep_narrow=args.num_deep_narrow,
        direct_steps=args.direct_steps,
        deep_steps=args.deep_steps,
        seed=args.seed,
        max_abs_value=args.max_abs_value,
        max_target_loops=args.max_target_loops,
    )
    write_jsonl(args.output_jsonl, records)
    report = summarize(records)
    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"records={report['records']}")
    print(f"by_mode={report['by_mode']}")
    print(f"target_loop_ranges={report['target_loop_ranges']}")
    print(f"depth_ranges={report['depth_ranges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
