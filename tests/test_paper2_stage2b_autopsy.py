from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import eval.eval_paper2_stage2b_autopsy as autopsy_eval

from models.paper2_dc2_student import Phase3StudentModules
from models.paper2_stage2b_depth import Stage2BDepthAttachment
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from training.paper2_stage2b_autopsy import (
    discrete_mutual_information,
    decision_mapping,
    margin_correlation_receipt,
    normalized_gram_eigengap,
    spherical_kmeans,
    stable_dev2_subsample,
    validate_autopsy_lock,
)
from eval.eval_paper2_stage2b_campaign import Stage2BTaskInferenceGraph


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "training/paper2_stage2b_autopsy_lock.json"


class _ProjectionWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[int, bool]] = []

    def forward(self, input_ids: torch.Tensor, **kwargs):
        keep = int(kwargs["logits_to_keep"])
        sparse = bool(kwargs["stage2b_score_only_sparse_logits"])
        self.calls.append((keep, sparse))
        batch, width = input_ids.shape
        logits = torch.arange(batch * 4 * width * 7, dtype=torch.float32).reshape(
            batch, 1, 4, width, 7
        )
        if keep:
            logits = logits[:, :, :, -keep:, :]
        return type(
            "Output",
            (),
            {
                "loop_logits": logits,
                "metrics": {
                    "stage2b_position_gate_mean": torch.tensor(0.25),
                    "stage2b_writeback_ratio_mean": torch.tensor(0.5),
                },
            },
        )()


def test_stage2b_task_graph_sparse_loop_projection_is_exact() -> None:
    wrapper = _ProjectionWrapper()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention = torch.ones_like(input_ids)
    full = Stage2BTaskInferenceGraph(
        wrapper=wrapper,
        stage="M2",
        amplitude=0.05,
        last_token_projection=False,
        sparse_loop_projection=False,
    ).next_token(input_ids=input_ids, attention_mask=attention)
    sparse = Stage2BTaskInferenceGraph(
        wrapper=wrapper,
        stage="M2",
        amplitude=0.05,
        last_token_projection=False,
        sparse_loop_projection=True,
    ).next_token(input_ids=input_ids, attention_mask=attention)
    assert wrapper.calls == [(0, False), (0, True)]
    assert torch.equal(full.augmented_logits, sparse.augmented_logits)
    assert torch.equal(full.base_logits, sparse.base_logits)


def test_stage2b_incremental_cache_preserves_greedy_transport() -> None:
    torch.manual_seed(20260820)
    config = Qwen2Config(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_cache=True,
    )
    base = Qwen2ForCausalLM(config).eval()
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=1, recurrent_end=3)
    ).eval()
    sidecar = Phase3StudentModules(
        tied_embedding=base.model.embed_tokens,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        control_dim=4,
        draft_rank=4,
        max_steps=4,
        rms_cap=0.5,
    ).eval()
    wrapper.install_stage2b_depth_attachment(Stage2BDepthAttachment.from_phase3(sidecar))
    common = {
        "wrapper": wrapper,
        "stage": "M2",
        "amplitude": 0.05,
        "flow_loops": 4,
    }
    recompute = Stage2BTaskInferenceGraph(**common, incremental_cache=False)
    cached = Stage2BTaskInferenceGraph(**common, incremental_cache=True)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    recompute_state, recompute_output = recompute.prefill_cached(
        input_ids=input_ids, attention_mask=attention_mask
    )
    cached_state, cached_output = cached.prefill_cached(
        input_ids=input_ids, attention_mask=attention_mask
    )
    for _step in range(3):
        recompute_token = recompute_output.augmented_logits.argmax(dim=-1)
        cached_token = cached_output.augmented_logits.argmax(dim=-1)
        assert torch.equal(recompute_token, cached_token)
        recompute_state, recompute_output = recompute.advance_cached(
            state=recompute_state, selected_tokens=recompute_token
        )
        cached_state, cached_output = cached.advance_cached(
            state=cached_state, selected_tokens=recompute_token
        )


def test_autopsy_signed_lock_is_score_only_and_sealed() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    validate_autopsy_lock(lock, require_signature=True)
    assert lock["status"] == "SIGNED"
    assert lock["mark_signed"] is True
    assert lock["locked_before_model_contact"] is True
    assert lock["authority"]["signature_record_drive_id"] == (
        "1OSaglrQTMNkf_hWDLudeMIXYnnNLdrwK"
    )
    assert lock["authority"]["signature_record_sha256"] == (
        "bbdd5c05d08e6e6e9fc2c4d2a3d128b657f7b4b479c185c18b089b756aee481b"
    )
    assert lock["optimizer_steps_allowed"] == 0
    assert lock["training_authorized"] is False
    assert lock["sealed_partitions"]["remain_sealed"] is True

    unsigned = copy.deepcopy(lock)
    unsigned["status"] = "DRAFT_UNEXECUTABLE"
    unsigned["mark_signed"] = False
    with pytest.raises(RuntimeError, match="unsigned"):
        validate_autopsy_lock(unsigned, require_signature=True)


def test_autopsy_lock_rejects_training_or_seal_contact() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    training = copy.deepcopy(lock)
    training["optimizer_steps_allowed"] = 1
    with pytest.raises(RuntimeError, match="score-only"):
        validate_autopsy_lock(training, require_signature=False)
    unsealed = copy.deepcopy(lock)
    unsealed["sealed_partitions"]["confirm_scored"] = True
    with pytest.raises(RuntimeError, match="sealed-partition"):
        validate_autopsy_lock(unsealed, require_signature=False)


def test_dev2_subsample_is_deterministic_and_stratified() -> None:
    rows = [
        {"item_id": f"{battery}-{index}", "battery": battery}
        for battery, count in (("gsm8k", 80), ("mbpp", 10), ("mmlu", 8), ("tier1", 2))
        for index in range(count)
    ]
    first = stable_dev2_subsample(rows, size=25)
    second = stable_dev2_subsample(list(reversed(rows)), size=25)
    assert first == second
    counts = Counter(row["battery"] for row in first)
    assert len(first) == 25
    assert set(counts) == {"gsm8k", "mbpp", "mmlu", "tier1"}


def test_margin_correlation_reports_pearson_and_spearman() -> None:
    rows = [
        {"per_loop_mean_teacher_token_margin": [value, 0.0, 0.0, 2.0 * value]}
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    receipt = margin_correlation_receipt(rows)
    assert receipt["pearson"] == pytest.approx(1.0)
    assert receipt["spearman"] == pytest.approx(1.0)


def test_decision_mapping_composes_hypotheses() -> None:
    assert decision_mapping({"h_b_magnitude": True, "h_a_attractor": True}) == [
        "radius_control_successor",
        "task_preservation_anchor_required",
    ]


def test_arm6_geometry_primitives_detect_separated_directions() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn((16, 8), generator=generator) * 0.01
    right = torch.randn((16, 8), generator=generator) * 0.01
    left[:, 0] += 1.0
    right[:, 0] -= 1.0
    values = torch.cat([left, right])
    labels, silhouette = spherical_kmeans(
        values, clusters=2, restarts=4, iterations=20, seed=11
    )
    assert silhouette > 0.9
    gap = normalized_gram_eigengap(values, max_rank=4)
    assert gap["maximum"] > 0.0
    association = discrete_mutual_information(
        labels.tolist(), ["left"] * 16 + ["right"] * 16
    )
    assert association["normalized_by_battery_entropy"] == pytest.approx(1.0)


def test_autopsy_runner_contains_no_optimizer_or_sealed_partition_path() -> None:
    evaluator = (ROOT / "eval/eval_paper2_stage2b_autopsy.py").read_text(encoding="utf-8")
    orchestrator = (ROOT / "colab/run_stage5_paper2_stage2b_autopsy.py").read_text(
        encoding="utf-8"
    )
    assert "torch.optim" not in evaluator
    assert "optimizer.step" not in evaluator
    assert "stage5_paper2_phase3_confirm" not in orchestrator.lower()
    assert "stage5_paper2_eval_e" not in orchestrator.lower()
    assert 'f"receipts/seed_{seed}/summary.json"' in orchestrator
    assert "validate_autopsy_lock(lock, require_signature=True)" in orchestrator
    assert '"optimizer_steps": 0' in evaluator
    assert '"optimizer_steps": 0' in orchestrator


def test_dev1_condition_resumes_from_atomic_generation_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows_by_id = [
        ("arc_easy", "arc_easy"),
        ("arc_challenge", "arc_challenge"),
        ("mmlu", "mmlu"),
        *((f"gsm8k-{index}", "gsm8k") for index in range(8)),
        ("mbpp", "mbpp"),
        ("tier1", "tier1"),
    ]
    panel = [{"item_id": item_id, "battery": battery} for item_id, battery in rows_by_id]
    comparators = {
        item_id: {"item_id": item_id, "correct": False, "augmented_correct": False}
        for item_id, _battery in rows_by_id
    }

    class FakeGraph:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def scored(rows: list[dict[str, str]]) -> list[dict[str, object]]:
        return [
            {
                "item_id": row["item_id"],
                "battery": row["battery"],
                "augmented_correct": True,
                "prediction": "ok",
            }
            for row in rows
        ]

    calls: list[list[str]] = []

    def interrupted_generation(
        _graph: object,
        _tokenizer: object,
        rows: list[dict[str, str]],
        *,
        batch_size: int,
        emit_batch: object,
    ) -> list[dict[str, object]]:
        del batch_size
        calls.append([row["item_id"] for row in rows])
        emit_batch(scored(rows[:8]))
        raise RuntimeError("simulated backend loss")

    monkeypatch.setattr(autopsy_eval, "Stage2BTaskInferenceGraph", FakeGraph)
    monkeypatch.setattr(
        autopsy_eval,
        "score_mcq",
        lambda _graph, _tokenizer, rows, *, batch_size: scored(rows),
    )
    monkeypatch.setattr(autopsy_eval, "score_generation", interrupted_generation)
    with pytest.raises(RuntimeError, match="backend loss"):
        autopsy_eval._score_dev1_condition(
            wrapper=object(),
            tokenizer=object(),
            panel=panel,
            base_rows=comparators,
            initialization_rows=comparators,
            seed=0,
            gamma=0.05,
            mode="standard",
            condition="resume_test",
            private_dir=tmp_path,
            mcq_batch_size=8,
            generation_batch_size=2,
        )
    partial = tmp_path / "dev1__resume_test.partial.jsonl"
    assert partial.is_file()
    assert len(autopsy_eval.read_jsonl(partial)) == 11

    def completed_generation(
        _graph: object,
        _tokenizer: object,
        rows: list[dict[str, str]],
        *,
        batch_size: int,
        emit_batch: object,
    ) -> list[dict[str, object]]:
        del batch_size
        calls.append([row["item_id"] for row in rows])
        result = scored(rows)
        emit_batch(result)
        return result

    monkeypatch.setattr(autopsy_eval, "score_generation", completed_generation)
    rows, summary = autopsy_eval._score_dev1_condition(
        wrapper=object(),
        tokenizer=object(),
        panel=panel,
        base_rows=comparators,
        initialization_rows=comparators,
        seed=0,
        gamma=0.05,
        mode="standard",
        condition="resume_test",
        private_dir=tmp_path,
        mcq_batch_size=8,
        generation_batch_size=2,
    )
    expected_generation = [f"gsm8k-{index}" for index in range(8)] + ["mbpp", "tier1"]
    assert calls == [expected_generation, ["mbpp", "tier1"]]
    assert [row["item_id"] for row in rows] == [item_id for item_id, _battery in rows_by_id]
    assert summary["rows"] == len(rows_by_id)
    assert not partial.exists()
    assert (tmp_path / "dev1__resume_test.jsonl").is_file()


def test_dev1_condition_reuses_hash_pinned_precomputed_rows(tmp_path: Path) -> None:
    batteries = (
        "arc_easy",
        "arc_challenge",
        "mmlu",
        "gsm8k",
        "mbpp",
        "tier1",
    )
    panel = [
        {"item_id": f"item-{index}", "battery": battery}
        for index, battery in enumerate(batteries)
    ]
    comparators = {
        row["item_id"]: {"item_id": row["item_id"], "augmented_correct": False}
        for row in panel
    }
    precomputed = [
        {**row, "augmented_correct": True, "prediction": "ok"}
        for row in panel
    ]
    rows, summary = autopsy_eval._score_dev1_condition(
        wrapper=object(),
        tokenizer=object(),
        panel=panel,
        base_rows=comparators,
        initialization_rows=comparators,
        seed=0,
        gamma=0.02,
        mode="standard",
        condition="initialization__gamma_0p02",
        private_dir=tmp_path,
        mcq_batch_size=8,
        generation_batch_size=2,
        precomputed_rows=precomputed,
        precomputed_source={"sha256": "abc", "scorer_path": "registered"},
    )
    assert len(rows) == len(panel)
    assert summary["reused_precomputed_rows"]["sha256"] == "abc"
    assert not (tmp_path / "dev1__initialization__gamma_0p02.partial.jsonl").exists()


def test_dev1_condition_archives_rows_from_superseded_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batteries = (
        "arc_easy",
        "arc_challenge",
        "mmlu",
        "gsm8k",
        "mbpp",
        "tier1",
    )
    panel = [
        {"item_id": f"item-{index}", "battery": battery}
        for index, battery in enumerate(batteries)
    ]
    comparators = {
        row["item_id"]: {"item_id": row["item_id"], "augmented_correct": False}
        for row in panel
    }
    partial = tmp_path / "dev1__transport_test.partial.jsonl"
    legacy = [{
        "kind": "paper2_stage2b_dev1_row_v1",
        "seed": 0,
        "look": 1000,
        "item_id": "item-0",
        "battery": "arc_easy",
        "current_correct": True,
        "base_correct": False,
        "initialization_correct": False,
        "augmented_correct": True,
        "autopsy_condition": "transport_test",
    }]
    autopsy_eval.write_jsonl(partial, legacy)
    legacy_sha = autopsy_eval.sha256_file(partial)

    class FakeGraph:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def scored(rows: list[dict[str, str]]) -> list[dict[str, object]]:
        return [
            {
                "item_id": row["item_id"],
                "battery": row["battery"],
                "augmented_correct": True,
            }
            for row in rows
        ]

    monkeypatch.setattr(autopsy_eval, "Stage2BTaskInferenceGraph", FakeGraph)
    monkeypatch.setattr(
        autopsy_eval,
        "score_mcq",
        lambda _graph, _tokenizer, rows, *, batch_size: scored(rows),
    )

    def score_all_generation(
        _graph: object,
        _tokenizer: object,
        rows: list[dict[str, str]],
        *,
        batch_size: int,
        emit_batch: object,
    ) -> list[dict[str, object]]:
        del batch_size
        result = scored(rows)
        emit_batch(result)
        return result

    monkeypatch.setattr(autopsy_eval, "score_generation", score_all_generation)
    rows, _summary = autopsy_eval._score_dev1_condition(
        wrapper=object(),
        tokenizer=object(),
        panel=panel,
        base_rows=comparators,
        initialization_rows=comparators,
        seed=0,
        gamma=0.05,
        mode="standard",
        condition="transport_test",
        private_dir=tmp_path,
        mcq_batch_size=8,
        generation_batch_size=2,
    )
    archived = list((tmp_path / "superseded").glob("*.jsonl"))
    assert len(archived) == 1
    assert autopsy_eval.sha256_file(archived[0]) == legacy_sha
    assert all(
        row["serving_transport"] == "exact_mcq_incremental_generation_v1"
        for row in rows
    )


def test_k_sweep_reuses_identical_k4_amplitude_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [{"item_id": "row-1", "battery": "gsm8k"}]
    calls = []

    class FakeGraph:
        def __init__(self, **kwargs: object) -> None:
            calls.append(int(kwargs["flow_loops"]))

    def fake_score(
        _graph: object,
        _tokenizer: object,
        _rows: list[dict[str, str]],
        *,
        batch_size: int,
    ) -> list[dict[str, object]]:
        del batch_size
        return [{"item_id": "row-1", "battery": "gsm8k", "augmented_correct": False}]

    monkeypatch.setattr(autopsy_eval, "Stage2BTaskInferenceGraph", FakeGraph)
    monkeypatch.setattr(autopsy_eval, "score_generation", fake_score)
    summary = autopsy_eval._k_sweep(
        wrapper=object(),
        tokenizer=object(),
        rows=rows,
        seed=0,
        condition="stop",
        private_dir=tmp_path,
        batch_size=2,
        precomputed_k4=[
            {"item_id": "row-1", "battery": "gsm8k", "augmented_correct": True}
        ],
    )
    assert calls == [1, 2, 3]
    assert summary["4"]["correct"] == 1
    assert summary["4"]["reused_identical_amplitude_cell"] is True
