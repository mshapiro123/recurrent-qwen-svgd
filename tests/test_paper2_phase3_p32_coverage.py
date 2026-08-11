from __future__ import annotations

import json

import pytest
import torch

import eval.eval_paper2_phase3_p32_coverage as coverage_module
from eval.eval_paper2_phase3_p32_coverage import (
    _flatten_teacher_field,
    _valid_resumed_shard,
    coverage_surface,
    load_agreement_records_resumable,
    sha256_file,
)


def _row(
    index: int,
    *,
    flip: bool,
    covered: bool,
    concurrent: bool,
    teachability: float,
    margin: float,
) -> dict[str, object]:
    return {
        "record_id": str(index),
        "source": "old" if index < 3 else "new",
        "anchor_index": index,
        "horizon": 1,
        "stratum": "general" if index % 2 == 0 else "code",
        "flip_candidate_14b": flip,
        "cascade_covered": covered,
        "cross_scale_consistent": concurrent,
        "teachability": teachability,
        "confident_agreement_margin": margin,
    }


def test_coverage_surface_separates_strict_writes_and_extension() -> None:
    records = [
        _row(0, flip=True, covered=True, concurrent=True, teachability=0.9, margin=0.0),
        _row(1, flip=True, covered=False, concurrent=False, teachability=0.9, margin=0.0),
        _row(2, flip=True, covered=True, concurrent=False, teachability=0.9, margin=0.0),
        _row(3, flip=False, covered=False, concurrent=False, teachability=0.1, margin=2.5),
    ]
    result = coverage_surface(
        records,
        teachability_thresholds=[0.8],
        margin_thresholds=[2.0],
    )
    write = result["strict_write_surface"][0]
    assert write["14b_flip_candidates"] == 3
    assert write["strict_concurrent_write_candidates"] == 1
    assert write["cross_scale_conflicts"] == 1
    assert write["targeted_32b_extension_candidates"] == 1
    negative = result["permissive_negative_surface"][0]
    assert negative["confident_agreement_negatives"] == 1
    assert negative["14b_only_admissible"] == 1
    assert result["thresholds_selected_for_p33"] is False


def test_teacher_row_groups_flatten_in_declared_sample_order() -> None:
    payload = {
        "sample_indices": torch.tensor([3, 7, 11]),
        "rows": [
            {
                "sample_indices": torch.tensor([3, 7]),
                "topk_log_probs": torch.tensor([[1.0, 0.0], [2.0, 1.0]]),
            },
            {
                "sample_indices": torch.tensor([11]),
                "topk_log_probs": torch.tensor([[3.0, 2.0]]),
            },
        ],
    }
    observed = _flatten_teacher_field(payload, "topk_log_probs")
    assert torch.equal(
        observed,
        torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]),
    )


def test_teacher_row_groups_reject_changed_sample_order() -> None:
    payload = {
        "sample_indices": torch.tensor([3, 7]),
        "rows": [
            {
                "sample_indices": torch.tensor([7, 3]),
                "topk_log_probs": torch.zeros(2, 2),
            }
        ],
    }
    with pytest.raises(RuntimeError, match="sample ordering changed"):
        _flatten_teacher_field(payload, "topk_log_probs")


def test_resumable_shard_requires_matching_lineage_and_output(tmp_path) -> None:
    output = tmp_path / "shard.jsonl"
    receipt = tmp_path / "shard.receipt.json"
    output.write_text('{"record_id":"a"}\n', encoding="utf-8")
    receipt.write_text(
        "{\n"
        '  "source": "old",\n'
        '  "shard_index": 3,\n'
        '  "lattice_sha256": "lattice",\n'
        '  "teacher_sha256": "teacher",\n'
        f'  "output_sha256": "{sha256_file(output)}",\n'
        '  "records": 1\n'
        "}\n",
        encoding="utf-8",
    )
    arguments = {
        "output_path": output,
        "receipt_path": receipt,
        "source": "old",
        "shard_index": 3,
        "lattice_sha256": "lattice",
        "teacher_sha256": "teacher",
    }
    assert _valid_resumed_shard(**arguments)
    output.write_text('{"record_id":"changed"}\n', encoding="utf-8")
    assert not _valid_resumed_shard(**arguments)


def test_resumable_loader_skips_durable_completed_shard(tmp_path, monkeypatch) -> None:
    private = tmp_path / "private"
    lattice = private / "lattice" / "shard.pt"
    teacher = private / "model_cache" / "teacher_14b" / "shard.pt"
    lattice.parent.mkdir(parents=True)
    teacher.parent.mkdir(parents=True)
    lattice.write_bytes(b"lattice")
    teacher.write_bytes(b"teacher")
    (private / "sample_manifest.jsonl").write_text(
        '{"anchor_index":0,"document_id":"doc","stratum":"general","row_id":"row"}\n',
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "lattice": {
                    "shards": [{"path": str(lattice), "sha256": sha256_file(lattice)}]
                },
                "model_caches": {
                    "teacher_14b": {
                        "shards": [
                            {"path": str(teacher), "sha256": sha256_file(teacher)}
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        coverage_module,
        "_training_anchor_mask",
        lambda samples, source: (
            torch.tensor([True]),
            {0: {"document_id": "doc", "stratum": "general", "row_id": "row"}},
        ),
    )

    calls = {"copy": 0, "process": 0}

    def copy_shard(source, destination):
        calls["copy"] += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    def process_shard(**kwargs):
        calls["process"] += 1
        return ([{"record_id": "record"}], [
            {"path": str(kwargs["lattice_path"]), "sha256": sha256_file(kwargs["lattice_path"])},
            {"path": str(kwargs["teacher_path"]), "sha256": sha256_file(kwargs["teacher_path"])},
        ])

    monkeypatch.setattr(coverage_module, "_copy_shard", copy_shard)
    monkeypatch.setattr(coverage_module, "_agreement_shard_records", process_shard)
    arguments = {
        "source": "new",
        "summary_path": summary,
        "private_root": private,
        "resume_shard_dir": tmp_path / "resume",
        "scratch_dir": tmp_path / "scratch",
    }
    first, _ = load_agreement_records_resumable(**arguments)
    second, _ = load_agreement_records_resumable(**arguments)
    assert first == second == [{"record_id": "record"}]
    assert calls == {"copy": 2, "process": 1}
