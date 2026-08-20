from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from eval.eval_paper2_stage2b_m0_stability import centered_gain, stability_verdict
from eval.eval_paper2_stage2b_riders import (
    compare_fixed_prompt_logits,
    fixed_prompt_comparison_receipt,
    runtime_discordance_audit,
    seed_ensemble_probe,
)
from eval.prepare_paper2_stage2b_dev2 import (
    REGISTERED_POWER_AT_PLUS_30,
    build_dev2,
    merge_reference_scores,
    registered_power_at_plus_30,
    write_receipt,
)
from models.lora import LoopScopedLoRALinear
from models.paper2_dc2_student import Phase3StudentModules, SharedResidualFlow
from models.paper2_stage2b_depth import (
    MultiLaneScratchFlow,
    Stage2BDepthAttachment,
    lane_effective_rank,
    log_sinkhorn,
    routing_residuals,
)
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from tests.test_recurrent_wrapper_tiny import TinyCausalLM
from training.paper2_stage2b_depth import (
    DepthObjectiveWeights,
    STAGE_ALLOCATIONS,
    amended_kill_gate_verdict,
    bootstrap_mean_interval,
    calibrated_gradient_share_weights,
    depth_objective,
    kill_gate_seed_read,
    kill_gate_trend_read,
    kill_gate_verdict,
    load_and_validate_lock,
    paired_sign_test_power,
    stage2b_learning_rate,
    stage_for_step,
)
from training.run_paper2_stage2b_depth import (
    audit_stage_gradients,
    retention_artifact_receipt,
    save_retained_checkpoint,
)
from training.paper2_stage2b_data import build_training_corpus, select_calibration_rows


def test_log_sinkhorn_is_doubly_stochastic() -> None:
    torch.manual_seed(0)
    matrix = log_sinkhorn(torch.randn(5, 4, 4), iterations=20)
    row, column = routing_residuals(matrix)
    assert row.max().item() < 1e-5
    assert column.max().item() < 1e-5
    assert torch.linalg.matrix_norm(matrix, ord=2).max().item() <= 1.00001


def test_retention_law_writes_and_verifies_due_named_artifacts(tmp_path: Path) -> None:
    save_retained_checkpoint(tmp_path, 0, {"step": 0, "value": torch.tensor([1.0])})
    save_retained_checkpoint(tmp_path, 1_000, {"step": 1_000, "value": torch.tensor([2.0])})
    receipt = retention_artifact_receipt(tmp_path, 1_000)
    assert [row["step"] for row in receipt] == [0, 1_000]
    assert all(row["exists"] and len(row["sha256"]) == 64 for row in receipt)


def test_multilane_m1_reproduces_single_lane_flow() -> None:
    torch.manual_seed(3)
    base_flow = SharedResidualFlow(
        latent_dim=8, context_dim=12, n_slots=8, max_steps=4, hidden_dim=24
    )
    multilane = MultiLaneScratchFlow(
        context_dim=12,
        latent_dim=8,
        n_slots=8,
        n_lanes=4,
        max_steps=4,
        base_flow=base_flow,
    )
    scratch = torch.randn(2, 8, 8)
    context = torch.randn(2, 12)
    expected = base_flow(scratch, context, steps=4).state
    actual = multilane(
        multilane.replicate(scratch),
        context,
        steps=4,
        dynamic_routing=False,
        constitutive_active=False,
        forced_lane_one=True,
    )
    assert torch.equal(actual.read_state, expected)


def test_lane_effective_rank_is_permutation_invariant() -> None:
    state = torch.randn(3, 4, 8, 7)
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(lane_effective_rank(state), lane_effective_rank(state[:, permutation]))


def test_loop_scoped_lora_keeps_first_pass_exact_after_training() -> None:
    torch.manual_seed(4)
    base = nn.Linear(7, 5)
    layer = LoopScopedLoRALinear(base, rank=3, alpha=3)
    with torch.no_grad():
        layer.lora_b.weight.normal_()
    inputs = torch.randn(2, 7)
    layer.set_loop_index(0)
    assert torch.equal(layer(inputs), base(inputs))
    layer.set_loop_index(1)
    assert not torch.equal(layer(inputs), base(inputs))


def test_stage2b_attachment_preserves_one_pass_and_reports_reentry() -> None:
    torch.manual_seed(5)
    base = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=1, recurrent_end=3)
    ).eval()
    sidecar = Phase3StudentModules(
        tied_embedding=base.model.embed_tokens,
        hidden_size=8,
        latent_dim=8,
        n_slots=8,
        control_dim=4,
        draft_rank=4,
        max_steps=4,
        rms_cap=0.5,
    ).eval()
    wrapper.install_stage2b_depth_attachment(Stage2BDepthAttachment.from_phase3(sidecar))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        plain = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=1,
            return_dict=True,
        )
        attached = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=1,
            stage2b_depth_enabled=True,
            stage2b_stage="M1",
            return_dict=True,
        )
        deeper = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=3,
            stage2b_depth_enabled=True,
            stage2b_stage="M1",
            return_dict=True,
        )
    assert torch.equal(plain.logits, attached.logits)
    assert deeper.metrics["stage2b_reentry_steps"].item() == 2
    assert deeper.metrics["stage2b_sinkhorn_row_residual_max"].item() == 0.0
    assert deeper.metrics["stage2b_sinkhorn_column_residual_max"].item() == 0.0


def _tiny_stage2b_wrapper() -> tuple[RecurrentQwenForCausalLM, torch.Tensor, torch.Tensor]:
    torch.manual_seed(51)
    base = TinyCausalLM().eval()
    wrapper = RecurrentQwenForCausalLM(
        base, layer_split=LayerSplit(prelude_end=1, recurrent_end=3)
    ).eval()
    sidecar = Phase3StudentModules(
        tied_embedding=base.model.embed_tokens,
        hidden_size=8,
        latent_dim=8,
        n_slots=8,
        control_dim=4,
        draft_rank=4,
        max_steps=4,
        rms_cap=0.5,
    ).eval()
    wrapper.install_stage2b_depth_attachment(Stage2BDepthAttachment.from_phase3(sidecar))
    input_ids = torch.tensor([[1, 2, 3, 4]])
    return wrapper, input_ids, torch.ones_like(input_ids)


def test_stage2b_zero_write_is_checkpoint_independent() -> None:
    wrapper, input_ids, attention_mask = _tiny_stage2b_wrapper()
    with torch.no_grad():
        before = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.0,
            stage2b_diagnostic_mode="zero_write",
            return_dict=True,
        ).logits
        wrapper.stage2b_depth_attachment.flow.hidden_innovation.weight.normal_()
        after = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.0,
            stage2b_diagnostic_mode="zero_write",
            return_dict=True,
        ).logits
    assert torch.equal(before, after)


def test_stage2b_component_modes_are_live_and_distinct() -> None:
    wrapper, input_ids, attention_mask = _tiny_stage2b_wrapper()
    with torch.no_grad():
        wrapper.stage2b_depth_attachment.flow.hidden_innovation.weight.normal_(std=0.2)
        wrapper.stage2b_depth_attachment.flow.prompt_gate.weight.normal_(std=0.2)
        outputs = {}
        for mode in (
            "standard",
            "constitutive_off",
            "fresh_state_each_loop",
            "inherited_flow_off",
        ):
            result = wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_loops=4,
                stage2b_depth_enabled=True,
                stage2b_stage="M2",
                stage2b_amplitude=0.05,
                stage2b_diagnostic_mode=mode,
                return_dict=True,
            )
            outputs[mode] = result.logits
            metrics = result.metrics
            assert metrics["stage2b_flow_update_loop_1_max_abs"].item() == 0.0
            assert metrics["stage2b_constitutive_update_loop_1_max_abs"].item() == 0.0
            assert metrics["stage2b_carry_contribution_loop_1_max_abs"].item() == 0.0
            if mode == "constitutive_off":
                assert all(
                    metrics[f"stage2b_constitutive_update_loop_{loop}_max_abs"].item() == 0.0
                    for loop in range(1, 5)
                )
            if mode == "fresh_state_each_loop":
                assert all(
                    metrics[f"stage2b_carry_contribution_loop_{loop}_max_abs"].item() == 0.0
                    for loop in range(1, 5)
                )
            if mode == "inherited_flow_off":
                assert all(
                    metrics[f"stage2b_flow_update_loop_{loop}_max_abs"].item() == 0.0
                    for loop in range(1, 5)
                )
    for mode in outputs:
        assert torch.isfinite(outputs[mode]).all()
    assert not torch.equal(outputs["standard"], outputs["constitutive_off"])
    assert not torch.equal(outputs["standard"], outputs["fresh_state_each_loop"])
    assert not torch.equal(outputs["standard"], outputs["inherited_flow_off"])


def test_stage2b_diagnostic_mode_rejects_unregistered_values() -> None:
    wrapper, input_ids, attention_mask = _tiny_stage2b_wrapper()
    with pytest.raises(ValueError, match="diagnostic mode"):
        wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=2,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_diagnostic_mode="invented",
            return_dict=True,
        )


def test_stage_schedule_and_landing_are_bound() -> None:
    assert stage_for_step(1) == "M2"
    assert stage_for_step(2501) == "M3"
    assert stage_for_step(5001) == "M4"
    assert stage2b_learning_rate(1, peak=5e-4) == pytest.approx(1e-6)
    assert stage2b_learning_rate(24000, peak=5e-4) == pytest.approx(0.0, abs=1e-15)
    assert STAGE_ALLOCATIONS["M4"] == [5001, 24000]


def test_m2_gradient_audit_allows_only_registered_dormant_paths() -> None:
    active = nn.Parameter(torch.tensor([1.0]))
    active.grad = torch.tensor([0.5])
    router = nn.Parameter(torch.tensor([1.0]))
    lora = nn.Parameter(torch.tensor([1.0]))
    groups = {"new_modules": [active, router], "gates": [], "loop_lora": [lora]}
    names = {
        id(active): "stage2b_depth_attachment.flow.hidden_innovation.weight",
        id(router): "stage2b_depth_attachment.flow.router.weight",
        id(lora): "base.model.layers.6.self_attn.q_proj.lora_a.weight",
    }
    audit = audit_stage_gradients(groups=groups, parameter_names=names, stage="M2")
    assert audit["pass"]
    assert len(audit["missing_expected"]) == 2
    assert not audit["missing_active"]


def test_m3_gradient_audit_rejects_missing_or_nonfinite_active_paths() -> None:
    missing = nn.Parameter(torch.tensor([1.0]))
    nonfinite = nn.Parameter(torch.tensor([1.0]))
    nonfinite.grad = torch.tensor([float("nan")])
    groups = {"new_modules": [missing], "gates": [nonfinite], "loop_lora": []}
    names = {id(missing): "flow.router.weight", id(nonfinite): "bridge.gate_logits"}
    audit = audit_stage_gradients(groups=groups, parameter_names=names, stage="M3")
    assert not audit["pass"]
    assert audit["missing_active"] == [{"group": "new_modules", "name": "flow.router.weight"}]
    assert audit["nonfinite"] == [{"group": "gates", "name": "bridge.gate_logits"}]


def test_depth_objective_uses_sparse_top128_and_equal_example_weighting() -> None:
    torch.manual_seed(12)
    batch, positions, vocabulary = 2, 4, 160
    base = torch.randn(batch, positions, vocabulary)
    loop_logits = [base.clone().requires_grad_() for _ in range(4)]
    teacher_ids = torch.arange(128).view(1, 1, 128).expand(batch, positions, -1)
    teacher_logits = base.detach().gather(-1, teacher_ids) + 0.2
    teacher_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    loss_mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
    weights = DepthObjectiveWeights(ce=0.3, kl=0.5, monotonicity=0.2)

    total, components = depth_objective(
        loop_logits=loop_logits,
        teacher_topk_token_ids=teacher_ids,
        teacher_topk_logits=teacher_logits,
        teacher_tokens=teacher_tokens,
        loss_mask=loss_mask,
        weights=weights,
        hinge_delta=0.01,
    )

    per_token = torch.nn.functional.cross_entropy(
        base.reshape(-1, vocabulary), teacher_tokens.reshape(-1), reduction="none"
    ).reshape(batch, positions)
    expected_ce = torch.stack([per_token[0, 0], per_token[1, :3].mean()]).mean()
    assert components["ce"].item() == pytest.approx(expected_ce.item())
    assert components["kl"].item() == pytest.approx(0.0, abs=1e-6)
    assert components["monotonicity"].item() == pytest.approx(0.03, abs=1e-6)
    total.backward()
    assert all(logits.grad is not None for logits in loop_logits)


def test_depth_objective_rejects_dense_teacher_logits() -> None:
    logits = [torch.randn(1, 2, 160) for _ in range(4)]
    with pytest.raises(ValueError, match="128"):
        depth_objective(
            loop_logits=logits,
            teacher_topk_token_ids=torch.zeros(1, 2, 160, dtype=torch.long),
            teacher_topk_logits=torch.zeros(1, 2, 160),
            teacher_tokens=torch.zeros(1, 2, dtype=torch.long),
            loss_mask=torch.ones(1, 2, dtype=torch.bool),
            weights=DepthObjectiveWeights(ce=0.3, kl=0.5, monotonicity=0.2),
            hinge_delta=0.01,
        )


def test_monotonicity_estimator_is_invariant_to_microbatch_partition() -> None:
    torch.manual_seed(121)
    batch, positions, vocabulary = 4, 3, 160
    loops = [torch.randn(batch, positions, vocabulary) for _ in range(4)]
    teacher_ids = torch.arange(128).view(1, 1, 128).expand(batch, positions, -1)
    teacher_logits = torch.randn(batch, positions, 128)
    targets = teacher_ids[..., 0]
    mask = torch.ones((batch, positions), dtype=torch.bool)
    weights = DepthObjectiveWeights(ce=0.3, kl=0.5, monotonicity=0.2)
    full = depth_objective(
        loop_logits=loops,
        teacher_topk_token_ids=teacher_ids,
        teacher_topk_logits=teacher_logits,
        teacher_tokens=targets,
        loss_mask=mask,
        weights=weights,
        hinge_delta=0.01,
    )[1]["monotonicity"]
    split = torch.stack(
        [
            depth_objective(
                loop_logits=[value[index : index + 1] for value in loops],
                teacher_topk_token_ids=teacher_ids[index : index + 1],
                teacher_topk_logits=teacher_logits[index : index + 1],
                teacher_tokens=targets[index : index + 1],
                loss_mask=mask[index : index + 1],
                weights=weights,
                hinge_delta=0.01,
            )[1]["monotonicity"]
            for index in range(batch)
        ]
    ).mean()
    assert full == pytest.approx(float(split), abs=1e-7)


def test_stage2b_full_sequence_corpus_reuses_document_split() -> None:
    old = [
        {"document_id": f"old-{index}", "row_id": f"o{index}", "stratum": "general", "input_ids": [1, 2, 3]}
        for index in range(100)
    ]
    new = [
        {"document_id": f"new-{index}", "row_id": f"n{index}", "stratum": "code", "input_ids": [4, 5, 6, 7]}
        for index in range(20)
    ]
    first, receipt = build_training_corpus(old, new)
    second, _ = build_training_corpus(old, new)
    assert first == second
    assert receipt["counts_by_source"]["option_b_new_train"] == 20
    assert receipt["old_document_split"]["selected_training_rows"] < 100
    assert receipt["next_token_positions"] == sum(len(row["input_ids"]) - 1 for row in first)


def test_stage2b_calibration_panel_is_balanced_and_deterministic() -> None:
    rows = [
        {
            "document_id": f"{stratum}-{index}",
            "row_id": f"{stratum}-{index}",
            "stratum": stratum,
            "source_partition": "test",
            "input_ids": [1, 2, 3],
        }
        for stratum in ("general", "code")
        for index in range(40)
    ]
    first, receipt = select_calibration_rows(rows)
    second, _ = select_calibration_rows(rows)
    assert first == second
    assert receipt["rows"] == 32
    assert receipt["counts_by_stratum"] == {"code": 16, "general": 16}


def test_gradient_share_calibration_hits_registered_targets() -> None:
    weights = calibrated_gradient_share_weights(
        {"ce": 2.0, "kl": 10.0, "monotonicity": 0.5}
    )
    weighted = {
        "ce": weights.ce * 2.0,
        "kl": weights.kl * 10.0,
        "monotonicity": weights.monotonicity * 0.5,
    }
    total = sum(weighted.values())
    assert weighted["ce"] / total == pytest.approx(0.3)
    assert weighted["kl"] / total == pytest.approx(0.5)
    assert weighted["monotonicity"] / total == pytest.approx(0.2)


def test_kill_gate_continues_on_one_separating_seed() -> None:
    rows = 64
    separating = [[0.0] * rows, [0.01] * rows, [0.03] * rows, [0.06] * rows]
    flat = [[0.0] * rows for _ in range(4)]
    seed0 = kill_gate_seed_read(separating, draws=500)
    seed1 = kill_gate_seed_read(flat, draws=500)
    assert seed0["separating"]
    assert not seed1["separating"]
    assert kill_gate_verdict({0: seed0, 1: seed1}) == "continue_m4"
    assert kill_gate_verdict({0: seed1, 1: seed1}) == "terminate_and_bank_boundary"


def test_amended_kill_gate_defers_once_on_replicated_positive_trends() -> None:
    series = {
        seed: {
            "k2_to_k3": [[0.00] * 64, [0.01] * 64, [0.02] * 64],
            "k3_to_k4": [[0.00] * 64, [0.02] * 64, [0.04] * 64],
        }
        for seed in (0, 1)
    }
    trend = kill_gate_trend_read(series, draws=200)
    flat = {seed: {"separating": False} for seed in (0, 1)}
    assert trend["positive_trend"]
    assert amended_kill_gate_verdict(flat, step=5000, trend_read=trend) == (
        "defer_once_to_step_8000"
    )
    assert amended_kill_gate_verdict(flat, step=8000, trend_read=trend) == (
        "terminate_and_bank_boundary"
    )


def test_power_arithmetic_is_monotone_in_effect() -> None:
    smaller = paired_sign_test_power(rows=2048, net_improvement=20, discordance_rate=0.2)
    larger = paired_sign_test_power(rows=2048, net_improvement=30, discordance_rate=0.2)
    assert larger["power"] > smaller["power"]


def test_dev2_registered_power_serialization_is_cross_platform_stable() -> None:
    receipt = registered_power_at_plus_30(2048)
    assert [entry["power"] for entry in receipt] == list(
        REGISTERED_POWER_AT_PLUS_30.values()
    )


def test_dev2_receipt_uses_signed_crlf_bytes(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    write_receipt(output, {"a": 1, "b": [2, 3]})
    payload = output.read_bytes()
    assert payload.endswith(b"\r\n")
    assert payload.count(b"\n") == payload.count(b"\r\n")


def test_dev2_is_deterministic_and_excludes_dev1() -> None:
    rows = []
    for index in range(11_733):
        if index < 1_519:
            partition = "dev"
        elif index < 10_231:
            partition = "verified_train"
        else:
            partition = "confirm"
        rows.append(
            {
                "item_id": f"row-{index}",
                "document_id": f"doc-{index}",
                "battery": ("gsm8k", "arc_challenge", "mbpp")[index % 3],
                "battery_role": "target_primary",
                "partition": partition,
                "content_sha256": f"{index:064x}",
                "base_correct": index % 2 == 0,
                "teacher_14b_correct": index % 3 == 0,
            }
        )
    dev1 = [{"item_id": f"row-{index}"} for index in range(1024)]
    first, receipt = build_dev2(rows, dev1)
    second, _ = build_dev2(rows, dev1)
    assert first == second
    assert len(first) == 2048
    assert receipt["candidate_rows"] == 9207
    assert "confirm_rows_excluded" not in receipt
    assert not ({row["item_id"] for row in first} & {row["item_id"] for row in dev1})
    assert not any(row["source_partition"] == "confirm" for row in first)


def test_dev2_rejects_partition_table_before_score_join() -> None:
    rows = [
        {
            "item_id": "row-0",
            "document_id": "doc-0",
            "battery": "gsm8k",
            "battery_role": "target_primary",
            "partition": "dev",
            "content_sha256": "0" * 64,
        }
    ]
    with pytest.raises(RuntimeError, match="joined P3.1 DEV/verified scores"):
        build_dev2(rows, [])


def test_dev2_score_join_leaves_confirm_sealed() -> None:
    partition = [
        {
            "item_id": "dev",
            "document_id": "dev-doc",
            "battery": "gsm8k",
            "battery_role": "target_primary",
            "partition": "dev",
            "content_sha256": "0" * 64,
        },
        {
            "item_id": "confirm",
            "document_id": "confirm-doc",
            "battery": "gsm8k",
            "battery_role": "target_primary",
            "partition": "confirm",
            "content_sha256": "1" * 64,
        },
    ]
    scores = [{**partition[0], "base_correct": False, "teacher_14b_correct": True}]
    merged = merge_reference_scores(partition, scores)
    assert merged[0]["teacher_14b_correct"] is True
    assert "base_correct" not in merged[1]
    assert "teacher_14b_correct" not in merged[1]


def test_stage2b_runner_freezes_dev2_from_scored_reference() -> None:
    runner = Path("colab/run_stage5_paper2_stage2b_depth.py").read_text(
        encoding="utf-8"
    )
    assert "p31_merged_dev_verified_scores.jsonl" in runner
    assert '"--reference-scores", str(reference_scores)' in runner


def test_seed_ensemble_uses_registered_margin_rule() -> None:
    seed0 = [
        {"item_id": "a", "battery": "gsm8k", "answer_token_margin_minimum": 0.1, "augmented_correct": True, "prediction": "1"},
        {"item_id": "b", "battery": "gsm8k", "answer_token_margin_minimum": 0.3, "augmented_correct": False, "prediction": "2"},
    ]
    seed1 = [
        {"item_id": "a", "battery": "gsm8k", "answer_token_margin_minimum": 0.2, "augmented_correct": False, "prediction": "2"},
        {"item_id": "b", "battery": "gsm8k", "answer_token_margin_minimum": 0.2, "augmented_correct": True, "prediction": "1"},
    ]
    rows, summary = seed_ensemble_probe(seed0, seed1)
    assert [row["selected_seed"] for row in rows] == [1, 0]
    assert summary["counts"]["ensemble"] == 0


def test_r1_finds_first_token_divergence() -> None:
    source = [{"item_id": "a", "battery": "gsm8k", "answer_token_margin_minimum": 0.001, "augmented_correct": True, "generated_token_ids": [1, 2, 3]}]
    diagnostic = [{"item_id": "a", "battery": "gsm8k", "answer_token_margin_minimum": 0.002, "augmented_correct": False, "generated_token_ids": [1, 9, 8]}]
    result = runtime_discordance_audit(source, diagnostic)
    assert result["prediction_changed_rows"] == 1
    assert result["first_divergence_histogram"] == {"1": 1}
    assert result["changed_rows_above_0p01_margin"] == 0


def test_r1_fixed_prompt_comparison_requires_same_top_token() -> None:
    aligned = compare_fixed_prompt_logits(
        torch.tensor([0.0, 2.0, 1.0]), torch.tensor([0.1, 1.9, 1.0])
    )
    discordant = compare_fixed_prompt_logits(
        torch.tensor([0.0, 2.0, 1.0]), torch.tensor([0.0, 1.0, 2.0])
    )
    assert aligned["status"] == "runtime_aligned"
    assert discordant["status"] == "runtime_discordant"


def test_fixed_prompt_comparison_receipt_hashes_inputs(tmp_path: Path) -> None:
    a100 = tmp_path / "a100.pt"
    l4 = tmp_path / "l4.pt"
    torch.save(torch.tensor([0.0, 2.0, 1.0]), a100)
    torch.save(torch.tensor([0.1, 1.9, 1.0]), l4)

    receipt = fixed_prompt_comparison_receipt(a100, l4)

    assert receipt["status"] == "runtime_aligned"
    assert set(receipt["source_sha256"]) == {"a100_40gb", "l4"}
    assert receipt["optimizer_steps"] == 0


def test_m0_stability_verdict_uses_catastrophe_tripwires() -> None:
    receipt = {
        "identity": {
            "maximum_absolute_logit_difference": 0.0,
            "trained_adapter_maximum_absolute_difference": 0.0,
        },
        "m1": {"single_lane_bit_exact": True},
        "routing": {"row_residual_maximum": 1e-7, "column_residual_maximum": 1e-7},
        "finite_horizon": {"centered_rms_gains": [0.8, 1.1, 1.4, 1.6]},
    }
    passed, failures = stability_verdict(receipt)
    assert passed and not failures
    receipt["finite_horizon"]["centered_rms_gains"][-1] = 101.0
    passed, failures = stability_verdict(receipt)
    assert not passed
    assert "catastrophe" in failures[0]


def test_centered_gain_preserves_serving_dtype() -> None:
    state = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    direction = torch.randn_like(state, dtype=torch.float32)

    def function(value: torch.Tensor) -> torch.Tensor:
        assert value.dtype == torch.bfloat16
        return 2.0 * value

    gain = centered_gain(function, state, direction, epsilon=0.02)
    assert gain == pytest.approx(2.0, rel=0.05)


def test_signed_lock_authorizes_optimizer_and_keeps_seals_closed() -> None:
    path = Path(__file__).resolve().parents[1] / "training/paper2_stage2b_depth_executed_lock.json"
    lock = load_and_validate_lock(path, require_signature=True)
    assert lock["training_authorized"] is True
    assert lock["sealed_partitions"]["remain_sealed"] is True


def test_stage2b_preflight_target_is_wired_and_score_only() -> None:
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (root / "colab/STAGE5_PAPER2_STAGE2B_PREFLIGHT_CELL.py").read_text(encoding="utf-8")
    runner = (root / "colab/run_stage5_paper2_stage2b_preflight.py").read_text(encoding="utf-8")
    assert "paper2_stage2b_preflight" in bootstrap
    assert "paper2_stage2b_preflight_v1" in cell
    assert "optimizer_constructed\": False" in runner
    assert "confirm_scored\": False" in runner
    assert "reusable_m0" in runner
    assert "M0 receipt is not passing" in runner
    assert "training.run" not in runner


def test_stage2b_loss_calibration_target_is_wired_and_no_optimizer() -> None:
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (root / "colab/STAGE5_PAPER2_STAGE2B_LOSS_CALIBRATION_CELL.py").read_text(
        encoding="utf-8"
    )
    runner = (root / "colab/run_stage5_paper2_stage2b_loss_calibration.py").read_text(
        encoding="utf-8"
    )
    assert "paper2_stage2b_loss_calibration" in bootstrap
    assert "paper2_stage2b_loss_calibration_v1" in cell
    assert '"optimizer_constructed": False' in runner
    assert '"confirm_scored": False' in runner
    assert "torch.optim" not in runner


def test_stage2b_executed_lock_is_signed_and_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = json.loads(
        (root / "training/paper2_stage2b_depth_executed_lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["status"] == "SIGNED"
    assert lock["unresolved_lock_fields"] == []
    assert lock["training_authorized"] is True
    assert lock["locked_before_training"] is True
    assert lock["mark_signed"] is True
    assert lock["signature"]["approval_handoff"]["drive_id"] == (
        "1qUJ-ZaW5W_c1aLRxf4_H70ggsE8lJtWD"
    )
    for seed in ("0", "1"):
        weights = lock["training"]["objective"]["weights_by_seed"][seed]
        assert sum(weights.values()) == pytest.approx(1.0)
    calibration = lock["training"]["objective"]["calibration"]
    assert calibration["optimizer_steps"] == 0
    assert calibration["confirm_scored"] is False
    assert calibration["eval_e_scored"] is False
