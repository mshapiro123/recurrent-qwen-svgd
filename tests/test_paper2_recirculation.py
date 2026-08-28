from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from eval.eval_paper2_recirculation_phase0 import (
    adjudicate_battery_anchor,
    file_receipt,
    projection_receipt,
    sha256_file,
    write_json,
    write_jsonl,
)
from eval.eval_paper2_recirculation_phase_a import (
    CellSpec,
    canonical_lf_receipt,
    checkpoint_overrun,
    coarse_pairs,
    coarse_specs,
    expected_total_seconds,
    rank_unique_configurations,
    refinement_specs,
    select_contiguous_region,
)
from colab.run_stage5_paper2_recirculation_phase0 import (
    PANEL,
    PANEL_CANONICAL_LF_SHA256,
    canonical_lf_sha256,
    force_add_public_receipts,
)
from colab.run_stage5_paper2_recirculation_phase_a import (
    force_add_public_receipts as force_add_phase_a_public_receipts,
)
from models.recirculation import (
    PaperNativeRecirculationEvaluator,
    RecirculationConfig,
    graph_receipt,
)


def _qwen() -> Qwen2ForCausalLM:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config).eval()


def _gemma() -> Gemma3ForCausalLM:
    torch.manual_seed(11)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=3,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    return Gemma3ForCausalLM(config).eval()


def _tokens() -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    return input_ids, torch.ones_like(input_ids)


def test_graph_receipt_freezes_first_pass_readout_and_kv_ownership() -> None:
    receipt = graph_receipt(sequence_length=3, num_layers=4, destination_layer=2)
    rows = receipt["rows"]
    assert receipt["readout"] == "first_iteration_only"
    assert receipt["tap_convention"] == "post_block_hidden_states_index_equals_paper_layer"
    assert any(
        row["position"] == 0
        and row["layer"] == 3
        and row["architecture_copy"] == 0
        and row["status"] == "provisional_then_discarded"
        for row in rows
    )
    assert any(
        row["position"] == 0
        and row["layer"] == 3
        and row["architecture_copy"] == 1
        and row["status"] == "committed"
        for row in rows
    )
    final_rows = [row for row in rows if row["position"] == 2]
    assert all(row["architecture_copy"] == 0 for row in final_rows)
    assert all(row["kv_owner"] == "scored_stack" for row in final_rows)


def test_qwen_alpha_zero_is_bit_exact_for_logits_and_committed_cache() -> None:
    input_ids, attention_mask = _tokens()
    evaluator = PaperNativeRecirculationEvaluator(
        _qwen(), RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.0)
    )
    receipt = evaluator.identity_receipt(
        input_ids=input_ids, attention_mask=attention_mask
    )
    assert receipt["bit_exact"] is True
    assert receipt["scored_logits_maximum_absolute_difference"] == 0.0
    assert receipt["committed_cache"]["maximum_absolute_difference"] == 0.0


def test_gemma_sliding_cache_alpha_zero_is_bit_exact() -> None:
    input_ids, attention_mask = _tokens()
    evaluator = PaperNativeRecirculationEvaluator(
        _gemma(), RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.0)
    )
    assert evaluator.identity_receipt(
        input_ids=input_ids, attention_mask=attention_mask
    )["bit_exact"]


def test_nonzero_recirculation_changes_only_later_scored_positions() -> None:
    input_ids, attention_mask = _tokens()
    model = _qwen()
    intact = PaperNativeRecirculationEvaluator(
        model, RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.0)
    )
    recirculated = PaperNativeRecirculationEvaluator(
        model, RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.2)
    )
    intact_logits, _ = intact.forward_sequence(
        input_ids=input_ids, attention_mask=attention_mask
    )
    recirculated_logits, _ = recirculated.forward_sequence(
        input_ids=input_ids, attention_mask=attention_mask
    )
    differences = (intact_logits - recirculated_logits).abs().amax(dim=-1)
    assert differences[0, 0].item() == 0.0
    assert bool((differences[0, 1:] > 0).all())


def test_identity_normalization_is_a_distinct_registered_nonzero_arm() -> None:
    input_ids, attention_mask = _tokens()
    model = _qwen()
    matched = PaperNativeRecirculationEvaluator(
        model,
        RecirculationConfig(
            source_layer=4,
            destination_layer=2,
            alpha=0.2,
            normalization_mode="norm_matched",
        ),
    )
    identity = PaperNativeRecirculationEvaluator(
        model,
        RecirculationConfig(
            source_layer=4,
            destination_layer=2,
            alpha=0.2,
            normalization_mode="identity",
        ),
    )
    matched_logits, _ = matched.forward_sequence(
        input_ids=input_ids, attention_mask=attention_mask
    )
    identity_logits, _ = identity.forward_sequence(
        input_ids=input_ids, attention_mask=attention_mask
    )
    assert not torch.equal(matched_logits[:, 1:], identity_logits[:, 1:])
    with pytest.raises(ValueError, match="normalization_mode"):
        RecirculationConfig(
            source_layer=4,
            destination_layer=2,
            alpha=0.2,
            normalization_mode="unknown",
        ).validate(4)


def test_prefill_and_incremental_advance_use_future_cache_only() -> None:
    input_ids, attention_mask = _tokens()
    evaluator = PaperNativeRecirculationEvaluator(
        _qwen(), RecirculationConfig(source_layer=4, destination_layer=2, alpha=0.2)
    )
    state, output = evaluator.prefill_cached(
        input_ids=input_ids[:, :3], attention_mask=attention_mask[:, :3]
    )
    assert output.augmented_logits.shape == (1, 64)
    state, advanced = evaluator.advance_cached(
        state=state, selected_tokens=torch.tensor([4])
    )
    assert state.processed_positions == 4
    assert state.attention_mask.shape == (1, 4)
    assert advanced.augmented_logits.shape == (1, 64)


def test_cost_projection_prices_the_complete_registered_phase_a() -> None:
    receipt = projection_receipt(
        phase0_elapsed=100.0,
        qwen_pilot={"recirculated": {"elapsed_seconds": 50.0}},
        battery_elapsed=25.0,
    )
    assert receipt["coarse_pairs"] == 32
    assert receipt["coarse_cells"] == 96
    assert receipt["refinement_perplexity_cells"] == 13
    assert receipt["battery_cells"] == 2
    assert receipt["projected_total_seconds"] > 100.0
    assert receipt["ceiling_a100_hours"] == 8.0


def test_frozen_panel_hash_is_line_ending_portable() -> None:
    root = Path(__file__).resolve().parents[1]
    assert canonical_lf_sha256(root / PANEL) == PANEL_CANONICAL_LF_SHA256


def test_canonical_panel_hash_accepts_only_crlf_to_lf_transport(
    tmp_path,
) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    invalid = tmp_path / "invalid.jsonl"
    lf.write_bytes(b'{"row":1}\n{"row":2}\n')
    crlf.write_bytes(b'{"row":1}\r\n{"row":2}\r\n')
    invalid.write_bytes(b'{"row":1}\r{"row":2}\n')
    assert canonical_lf_sha256(lf) == canonical_lf_sha256(crlf)
    with pytest.raises(RuntimeError, match="unauthorized carriage return"):
        canonical_lf_sha256(invalid)


def test_battery_adjudication_preserves_v1_and_reuses_exact_rows(tmp_path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    v1_path = tmp_path / "battery_anchor.json"
    v2_path = tmp_path / "battery_anchor_v2_adjudicated.json"
    write_jsonl(
        rows_path,
        [
            {"item_id": "a", "augmented_correct": True},
            {"item_id": "b", "augmented_correct": False},
        ],
    )
    write_json(
        v1_path,
        {
            "kind": "paper2_recirculation_battery_anchor_v1",
            "rows": 2,
            "correct": 1,
            "expected_correct": 2,
            "elapsed_seconds": 12.5,
            "row_receipt": file_receipt(rows_path),
            "passed": False,
        },
    )
    authority = {
        "drive_id": "ruling",
        "filename": "ruling.md",
        "bytes": 10,
        "sha256": "authority-sha",
    }
    lock = {
        "authorities": [authority],
        "gates": {
            "battery_anchor_rows": 2,
            "battery_additive_delta": 20,
            "battery_additive_threshold": 21,
            "battery_neutral_lower_delta": -9,
            "battery_neutral_lower_threshold": -8,
        },
        "comparator_adjudication": {
            "authority_filename": "ruling.md",
            "authority_sha256": "authority-sha",
            "paper_native_correct": 1,
            "prior_correct": 2,
            "prior_evaluator": "prior",
            "row_receipt": file_receipt(rows_path),
        },
    }
    v1_sha = sha256_file(v1_path)
    receipt = adjudicate_battery_anchor(
        lock=lock, v1_path=v1_path, rows_path=rows_path, v2_path=v2_path
    )
    assert sha256_file(v1_path) == v1_sha
    assert receipt["passed"] is True
    assert receipt["generation_replayed"] is False
    assert receipt["correct"] == 1
    assert receipt["source_v1_passed"] is False
    assert receipt["authority"] == authority
    assert adjudicate_battery_anchor(
        lock=lock, v1_path=v1_path, rows_path=rows_path, v2_path=v2_path
    ) == receipt

    rows_path.write_text('{"item_id":"changed","augmented_correct":true}\n')
    with pytest.raises(RuntimeError, match="row receipt identity changed"):
        adjudicate_battery_anchor(
            lock=lock, v1_path=v1_path, rows_path=rows_path, v2_path=v2_path
        )


def test_phase0_publication_force_adds_only_lightweight_receipts(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    public_dir = root / "outputs" / "stage5" / "phase0"
    public_dir.mkdir(parents=True)
    summary = public_dir / "summary.json"
    status = public_dir / "status.json"
    checkpoint = public_dir / "checkpoint.pt"
    summary.write_text("{}\n", encoding="utf-8")
    status.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"not-for-git")
    commands: list[tuple[list[str], Path]] = []

    def capture(command: list[str], *, cwd: Path) -> None:
        commands.append((command, cwd))

    monkeypatch.setattr(
        "colab.run_stage5_paper2_recirculation_phase0.run", capture
    )
    paths = force_add_public_receipts(root=root, public_dir=public_dir)

    assert paths == [status, summary]
    assert commands == [
        (["git", "add", "-f", str(status.relative_to(root))], root),
        (["git", "add", "-f", str(summary.relative_to(root))], root),
    ]


def _phase_a_lock() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "training/paper2_recirculation_phase_a_lock.json").read_text(
            encoding="utf-8"
        )
    )


def _coarse_row(spec: CellSpec, score: float, index: int) -> dict:
    return {
        "run_index": index,
        "stage": "coarse",
        "config": {
            "source_layer": spec.source_layer,
            "destination_layer": spec.destination_layer,
            "alpha": spec.alpha,
            "beta_mode": spec.beta_mode,
            "ramp_tokens": spec.ramp_tokens,
            "normalization_mode": spec.normalization_mode,
        },
        "recirculated": {"mean_nll": 2.0 - score / 100.0},
        "perplexity_reduction_percent": score,
    }


def test_phase_a_grid_and_refinement_counts_are_locked() -> None:
    lock = _phase_a_lock()
    pairs = coarse_pairs(lock)
    specs = coarse_specs(lock)
    assert len(pairs) == 32
    assert len(specs) == 96
    assert len({spec.key() for spec in specs}) == 96

    rows = [_coarse_row(spec, 0.0, index) for index, spec in enumerate(specs, 1)]
    selection = select_contiguous_region(rows, lock)
    assert len(selection["selected_pairs"]) == 3
    assert len(refinement_specs(rows, selection, lock)) == 13
    assert expected_total_seconds(lock, 111) == pytest.approx(
        lock["phase0"]["projected_total_seconds"], abs=1e-9
    )
    root = Path(__file__).resolve().parents[1]
    summary = lock["phase0"]["public_summary"]
    assert canonical_lf_receipt(root / summary["path"]) == {
        "bytes": summary["canonical_lf_bytes"],
        "sha256": summary["canonical_lf_sha256"],
    }
    phase0 = json.loads((root / summary["path"]).read_text(encoding="utf-8"))
    assert lock["phase0"]["corpus_token_windows"] == phase0["private_receipts"][
        "corpus_token_windows.pt"
    ]


def test_phase_a_resume_does_not_recheck_banked_overrun_checkpoints() -> None:
    arguments = {
        "resume_completed": 72,
        "actual_total_seconds": 7200.0,
        "checkpoint_set": {24, 48, 72, 96},
        "overrun_multiplier": 1.25,
        "cost_ceiling_seconds": 8.0 * 3600.0,
    }
    assert not checkpoint_overrun(
        completed=24,
        expected_total_seconds_at_checkpoint=3600.0,
        **arguments,
    )
    assert checkpoint_overrun(
        completed=96,
        expected_total_seconds_at_checkpoint=3600.0,
        **arguments,
    )
    assert checkpoint_overrun(
        completed=24,
        expected_total_seconds_at_checkpoint=3600.0,
        **{**arguments, "actual_total_seconds": 8.0 * 3600.0 + 1.0},
    )


def test_phase_a_selector_prefers_a_connected_region_over_an_isolated_peak() -> None:
    lock = _phase_a_lock()
    rows = []
    connected = {(4, 10): 2.0, (4, 12): 1.9, (6, 12): 1.8}
    isolated = {(14, 18): 5.0}
    for index, spec in enumerate(coarse_specs(lock), 1):
        pair = (spec.destination_layer, spec.source_layer)
        score = connected.get(pair, isolated.get(pair, 0.0))
        rows.append(_coarse_row(spec, score, index))
    selected = select_contiguous_region(rows, lock)
    observed = {
        (row["destination_layer"], row["source_layer"])
        for row in selected["selected_pairs"]
    }
    assert observed == set(connected)
    assert (
        selected["best_pair"]["destination_layer"],
        selected["best_pair"]["source_layer"],
    ) == (14, 18)


def test_phase_a_ranking_deduplicates_exact_cells_and_rejects_drift() -> None:
    spec = CellSpec(source_layer=12, destination_layer=4, alpha=0.1)
    first = _coarse_row(spec, 1.0, 1)
    duplicate = _coarse_row(spec, 1.0, 2)
    duplicate["recirculated"]["mean_nll"] = first["recirculated"]["mean_nll"]
    assert len(rank_unique_configurations([first, duplicate])) == 1
    duplicate["recirculated"]["mean_nll"] += 1e-6
    with pytest.raises(RuntimeError, match="duplicate deterministic cell changed"):
        rank_unique_configurations([first, duplicate])


def test_phase_a_publication_allows_figures_but_excludes_checkpoints(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    public_dir = root / "outputs" / "stage5" / "phase_a"
    public_dir.mkdir(parents=True)
    files = [
        public_dir / "summary.json",
        public_dir / "heatmap.png",
        public_dir / "heatmap.svg",
    ]
    for path in files:
        path.write_bytes(b"receipt")
    checkpoint = public_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"not-for-git")
    commands: list[tuple[list[str], Path]] = []

    def capture(command: list[str], *, cwd: Path) -> None:
        commands.append((command, cwd))

    monkeypatch.setattr(
        "colab.run_stage5_paper2_recirculation_phase_a.run", capture
    )
    paths = force_add_phase_a_public_receipts(root=root, public_dir=public_dir)
    assert paths == sorted(files)
    assert checkpoint not in paths
    assert commands == [
        (["git", "add", "-f", str(path.relative_to(root))], root)
        for path in sorted(files)
    ]
