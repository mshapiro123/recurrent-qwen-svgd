from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from colab.run_stage5_paper2_stage2b_depth import archive_previous_failure
from eval.eval_paper2_stage2b_campaign import Stage2BTaskInferenceGraph
from training.paper2_stage2b_runtime import (
    DeterministicCycleSampler,
    ShardedTeacherCache,
    atomic_json,
    atomic_torch_save,
    collate_teacher_rows,
)


def test_failed_colab_status_is_archived_before_retry(tmp_path: Path) -> None:
    status = tmp_path / "receipts" / "cache_status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"status": "failed", "updated_at_unix": 1234.9}) + "\n",
        encoding="utf-8",
    )
    archive = archive_previous_failure(status)
    assert archive == status.parent / "archaeology/cache_status_failed_1234.json"
    assert json.loads(archive.read_text(encoding="utf-8"))["status"] == "failed"
    assert archive_previous_failure(status) == archive


def _teacher_row(index: int, length: int) -> dict:
    return {
        "row_index": index,
        "input_ids": torch.arange(length, dtype=torch.int32),
        "teacher_topk_token_ids": torch.arange(128, dtype=torch.int32)
        .view(1, 128)
        .expand(length - 1, -1)
        .clone(),
        "teacher_topk_logits": torch.zeros((length - 1, 128), dtype=torch.bfloat16),
    }


def test_cycle_sampler_resumes_across_cycle_boundary() -> None:
    first = DeterministicCycleSampler.create(5, 17)
    prefix = first.take(7)
    restored = DeterministicCycleSampler.restore(first.state_dict())
    assert restored.take(13) == first.take(13)
    assert sorted(prefix[:5]) == list(range(5))


def test_sharded_teacher_cache_checks_hashes_and_collates_finite_padding(
    tmp_path: Path,
) -> None:
    rows = [_teacher_row(0, 3), _teacher_row(1, 5)]
    shard = tmp_path / "shard_0000.pt"
    sha = atomic_torch_save(
        shard,
        {"kind": "paper2_stage2b_teacher_cache_shard_v1", "rows": rows},
    )
    atomic_json(
        tmp_path / "index.json",
        {
            "kind": "paper2_stage2b_teacher_cache_index_v1",
            "status": "complete",
            "rows": 2,
            "shards": [
                {"file": shard.name, "start": 0, "stop": 2, "rows": 2, "sha256": sha}
            ],
        },
    )
    cache = ShardedTeacherCache(tmp_path / "index.json")
    batch = collate_teacher_rows([cache[0], cache[1]], device="cpu")
    assert batch["loss_mask"].sum(dim=1).tolist() == [2, 4]
    assert torch.isfinite(batch["teacher_topk_logits"]).all()
    shard.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="integrity"):
        ShardedTeacherCache(tmp_path / "index.json")


class DummyWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_kwargs):
        batch, length = input_ids.shape
        logits = torch.zeros((batch, 1, 4, length, 7), device=input_ids.device)
        for row in range(batch):
            position = int(attention_mask[row].nonzero()[-1])
            logits[row, 0, -1, position, row + 1] = 5
            logits[row, 0, 0, position, row + 2] = 4
        return SimpleNamespace(
            loop_logits=logits,
            metrics={
                "stage2b_position_gate_mean": torch.tensor(0.2),
                "stage2b_writeback_ratio_mean": torch.tensor(0.03),
            },
        )


def test_stage2b_task_graph_selects_each_rows_current_position() -> None:
    graph = Stage2BTaskInferenceGraph(DummyWrapper(), "M3", 0.05)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention = torch.tensor([[1, 1, 0], [1, 1, 1]])
    output = graph.next_token(input_ids=input_ids, attention_mask=attention)
    assert output.augmented_logits.argmax(dim=-1).tolist() == [1, 2]
    assert output.base_logits.argmax(dim=-1).tolist() == [2, 3]


def test_runner_keeps_signed_assertion_immediately_before_optimizer() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "training/run_paper2_stage2b_depth.py").read_text(encoding="utf-8")
    needle = (
        "assert_optimizer_construction_authorized(args.lock)\n"
        "    optimizer = torch.optim.AdamW"
    )
    assert needle in source
    assert "awaiting_step_5000_strategy_adjudication" in source
    assert '"confirm_scored": False' in source
    assert '"eval_e_scored": False' in source
