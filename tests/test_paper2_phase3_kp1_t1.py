from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import colab.run_stage5_paper2_phase3_kp1_t1 as runner

from training.paper2_phase3_kp1_t1 import (
    T1_CELL_DIM,
    T1_CORE_CELLS,
    assemble_core_cells,
    canonical_sha256,
    core_cell_mask,
    knowledge_gap_rows,
    ridge_embedding_probe,
    row_reindex,
    stratified_probe_split,
    token_ranks,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gap_population_and_stratified_split_are_deterministic() -> None:
    panel = [
        {"item_id": "a", "battery": "x", "partition": "dev"},
        {"item_id": "b", "battery": "x", "partition": "dev"},
        {"item_id": "c", "battery": "y", "partition": "dev"},
        {"item_id": "d", "battery": "y", "partition": "dev"},
    ]
    references = [
        {"item_id": "a", "partition": "dev", "teacher_14b_correct": True, "base_correct": False},
        {"item_id": "b", "partition": "dev", "teacher_14b_correct": False, "base_correct": False},
        {"item_id": "c", "partition": "dev", "teacher_14b_correct": True, "base_correct": False},
        {"item_id": "d", "partition": "dev", "teacher_14b_correct": True, "base_correct": True},
    ]
    gap = knowledge_gap_rows(panel, references)
    assert [row["item_id"] for row in gap] == ["a", "c"]
    split = stratified_probe_split(gap, seed=7, eval_fraction=0.5)
    assert split == stratified_probe_split(gap, seed=7, eval_fraction=0.5)
    assert set(split) == {"a", "c"}
    assert set(split.values()) == {"probe_eval"}
    assert canonical_sha256(["a", "c"]) == canonical_sha256(["a", "c"])


def test_t1_core_cells_and_masks_preserve_the_fixed_schema() -> None:
    prelude = torch.randn(2, 8, T1_CELL_DIM)
    recurrent = [torch.randn_like(prelude) for _ in range(4)]
    layers = torch.randn(2, 4, T1_CELL_DIM)
    cells = assemble_core_cells(prelude, recurrent, layers)
    assert cells.shape == (2, T1_CORE_CELLS, T1_CELL_DIM)
    mask = core_cell_mask(loop_count=2, batch=2, device=torch.device("cpu"))
    assert mask.shape == (2, T1_CORE_CELLS)
    assert mask[0].sum().item() == 8 + 16 + 4
    assert mask[:, 24:40].sum().item() == 0


def test_ridge_probe_recovers_an_affine_mapping_without_optimizer() -> None:
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(12, 3, generator=generator)
    weight = torch.randn(3, 5, generator=generator)
    bias = torch.randn(1, 5, generator=generator)
    y = x @ weight + bias
    prediction = ridge_embedding_probe(x[:9], y[:9], x[9:], ridge=1e-6)
    assert torch.allclose(prediction, y[9:], atol=2e-3, rtol=2e-3)


def test_token_ranks_use_one_based_full_vocabulary_order() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [4.0, 2.0, 3.0]])
    assert token_ranks(logits, torch.tensor([2, 0])).tolist() == [2, 1]


def test_row_reindex_aligns_panel_order_to_locked_item_order() -> None:
    indexes = row_reindex(["b", "c", "a"], ["a", "b", "c"])
    assert indexes == [2, 0, 1]
    values = torch.tensor([20, 30, 10])
    assert values[indexes].tolist() == [10, 20, 30]


def test_lock_preserves_authority_seals_and_four_endpoint_hashes() -> None:
    lock = json.loads(
        (ROOT / "training/paper2_phase3_kp1_t1_lock.json").read_text(encoding="utf-8")
    )
    assert lock["status"] == "authorized_score_only"
    assert lock["authority"]["drive_id"] == "1l8yDmL97eI3a4iTybM0m9yq989xvMBrz"
    assert lock["sealed_partitions"]["remain_sealed"] is True
    assert lock["sealed_partitions"]["confirm_scored"] is False
    assert lock["sealed_partitions"]["eval_e_scored"] is False
    assert len(lock["t1"]["checkpoints"]) == 4
    assert lock["t1"]["core_schema"]["cells"] == 44
    assert lock["optimizer_constructed"] is False
    assert lock["optimizer_steps"] == 0


def test_stage_chain_can_reuse_an_independently_verified_p34(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "migrated": b"migrated",
        "p33": b"p33",
        "i1": b"i1",
        "p34": b"p34",
    }
    digest = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    monkeypatch.setattr(runner, "MIGRATED_SHA", {1: digest["migrated"]})
    monkeypatch.setattr(runner, "P33_SHA", {1: digest["p33"]})
    monkeypatch.setattr(runner, "I1_SHA", {1: digest["i1"]})

    copied_sources: list[str] = []

    def fake_rsync(source: Path, destination: Path) -> None:
        copied_sources.append(str(source))
        if "migrated" in destination.name:
            destination.write_bytes(payloads["migrated"])
        elif "p33" in destination.name:
            destination.write_bytes(payloads["p33"])
        elif "i1" in destination.name:
            destination.write_bytes(payloads["i1"])
        else:
            raise AssertionError(f"unexpected source: {source}")

    monkeypatch.setattr(runner, "rsync", fake_rsync)
    p34 = tmp_path / "seed_1_p34_step_4000.pt"
    p34.write_bytes(payloads["p34"])
    paths = runner.stage_chain_with_verified_p34(
        tmp_path, seed=1, expected_p34=digest["p34"]
    )

    assert paths["p34"] == p34
    assert len(copied_sources) == 3
    assert all("p34" not in source for source in copied_sources)
