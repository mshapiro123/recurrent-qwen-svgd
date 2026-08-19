"""Resumable data and checkpoint primitives for the Stage 2B-D campaign."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, payload: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return sha256_file(destination)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class DeterministicCycleSampler:
    """Without-replacement cycles with a fully serializable cursor."""

    rows: int
    seed: int
    order: list[int]
    cursor: int
    cycle: int
    generator: torch.Generator

    @classmethod
    def create(cls, rows: int, seed: int) -> "DeterministicCycleSampler":
        if rows < 1:
            raise ValueError("sampler requires at least one row")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(rows, generator=generator).tolist()
        return cls(rows, seed, order, 0, 0, generator)

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "DeterministicCycleSampler":
        sampler = cls.create(int(payload["rows"]), int(payload["seed"]))
        sampler.order = [int(value) for value in payload["order"]]
        sampler.cursor = int(payload["cursor"])
        sampler.cycle = int(payload["cycle"])
        sampler.generator.set_state(payload["generator_state"])
        sampler._validate()
        return sampler

    def _validate(self) -> None:
        if sorted(self.order) != list(range(self.rows)):
            raise RuntimeError("sampler order is not a permutation")
        if not 0 <= self.cursor <= self.rows:
            raise RuntimeError("sampler cursor is outside its cycle")

    def take(self, count: int) -> list[int]:
        if count < 1:
            raise ValueError("batch count must be positive")
        selected: list[int] = []
        while len(selected) < count:
            if self.cursor == self.rows:
                self.order = torch.randperm(
                    self.rows, generator=self.generator
                ).tolist()
                self.cursor = 0
                self.cycle += 1
            available = min(count - len(selected), self.rows - self.cursor)
            selected.extend(self.order[self.cursor : self.cursor + available])
            self.cursor += available
        return selected

    def state_dict(self) -> dict[str, Any]:
        self._validate()
        return {
            "rows": self.rows,
            "seed": self.seed,
            "order": self.order,
            "cursor": self.cursor,
            "cycle": self.cycle,
            "generator_state": self.generator.get_state(),
        }


class ShardedTeacherCache:
    """Integrity-checked random access over compact teacher-cache shards."""

    KIND = "paper2_stage2b_teacher_cache_index_v1"

    def __init__(self, index_path: str | Path, *, preload: bool = True) -> None:
        self.index_path = Path(index_path)
        self.root = self.index_path.parent
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index.get("kind") != self.KIND or self.index.get("status") != "complete":
            raise RuntimeError("Stage 2B teacher cache index is not complete")
        self.rows = int(self.index["rows"])
        self._locations: list[tuple[Path, int]] = []
        for shard in self.index["shards"]:
            path = self.root / shard["file"]
            if not path.is_file() or sha256_file(path) != shard["sha256"]:
                raise RuntimeError(f"Stage 2B teacher shard failed integrity: {path}")
            for local_index in range(int(shard["rows"])):
                self._locations.append((path, local_index))
        if len(self._locations) != self.rows:
            raise RuntimeError("Stage 2B teacher-cache row count changed")
        self._loaded_path: Path | None = None
        self._loaded_rows: list[dict[str, Any]] | None = None
        self._rows: list[dict[str, Any]] | None = None
        if preload:
            self._rows = []
            for shard in self.index["shards"]:
                payload = torch.load(
                    self.root / shard["file"], map_location="cpu", weights_only=False
                )
                self._rows.extend(payload["rows"])
            if len(self._rows) != self.rows:
                raise RuntimeError("preloaded Stage 2B teacher-cache row count changed")

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._rows is not None:
            return self._rows[int(index)]
        path, local = self._locations[int(index)]
        if path != self._loaded_path:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("kind") != "paper2_stage2b_teacher_cache_shard_v1":
                raise RuntimeError("Stage 2B teacher shard kind changed")
            self._loaded_path = path
            self._loaded_rows = payload["rows"]
        assert self._loaded_rows is not None
        return self._loaded_rows[local]


def collate_teacher_rows(
    rows: Sequence[Mapping[str, Any]], *, device: str
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("cannot collate an empty Stage 2B batch")
    lengths = [int(row["input_ids"].numel()) for row in rows]
    width = max(lengths)
    batch = len(rows)
    inputs = torch.zeros((batch, width), dtype=torch.long)
    attention = torch.zeros((batch, width), dtype=torch.long)
    top_ids = torch.zeros((batch, width - 1, 128), dtype=torch.long)
    # Padding is masked downstream, but it must remain finite because the KL is
    # computed before masking and NaN multiplied by zero is still NaN.
    top_logits = torch.zeros((batch, width - 1, 128), dtype=torch.bfloat16)
    mask = torch.zeros((batch, width - 1), dtype=torch.bool)
    for index, (row, length) in enumerate(zip(rows, lengths)):
        ids = row["input_ids"].long()
        positions = length - 1
        if row["teacher_topk_token_ids"].shape != (positions, 128):
            raise RuntimeError("teacher-cache token lattice shape changed")
        inputs[index, :length] = ids
        attention[index, :length] = 1
        top_ids[index, :positions] = row["teacher_topk_token_ids"].long()
        top_logits[index, :positions] = row["teacher_topk_logits"].to(torch.bfloat16)
        mask[index, :positions] = True
    return {
        "input_ids": inputs.to(device),
        "attention_mask": attention.to(device),
        "teacher_topk_token_ids": top_ids.to(device),
        "teacher_topk_logits": top_logits.to(device),
        "teacher_tokens": top_ids[..., 0].to(device),
        "loss_mask": mask.to(device),
    }


def schedule_digest(indexes: Iterable[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(index)) for index in indexes).encode("ascii")
    ).hexdigest()
