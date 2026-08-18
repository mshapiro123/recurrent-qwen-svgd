from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from eval.eval_paper2_stage2b_m0_stability import centered_gain, stability_verdict
from eval.eval_paper2_stage2b_riders import (
    compare_fixed_prompt_logits,
    runtime_discordance_audit,
    seed_ensemble_probe,
)
from eval.prepare_paper2_stage2b_dev2 import build_dev2
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
    STAGE_ALLOCATIONS,
    bootstrap_mean_interval,
    kill_gate_seed_read,
    kill_gate_verdict,
    load_and_validate_lock,
    paired_sign_test_power,
    stage2b_learning_rate,
    stage_for_step,
)


def test_log_sinkhorn_is_doubly_stochastic() -> None:
    torch.manual_seed(0)
    matrix = log_sinkhorn(torch.randn(5, 4, 4), iterations=20)
    row, column = routing_residuals(matrix)
    assert row.max().item() < 1e-5
    assert column.max().item() < 1e-5
    assert torch.linalg.matrix_norm(matrix, ord=2).max().item() <= 1.00001


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


def test_stage_schedule_and_landing_are_bound() -> None:
    assert stage_for_step(1) == "M2"
    assert stage_for_step(2501) == "M3"
    assert stage_for_step(5001) == "M4"
    assert stage2b_learning_rate(1, peak=5e-4) == pytest.approx(1e-6)
    assert stage2b_learning_rate(24000, peak=5e-4) == pytest.approx(0.0, abs=1e-15)
    assert STAGE_ALLOCATIONS["M4"] == [5001, 24000]


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


def test_power_arithmetic_is_monotone_in_effect() -> None:
    smaller = paired_sign_test_power(rows=2048, net_improvement=20, discordance_rate=0.2)
    larger = paired_sign_test_power(rows=2048, net_improvement=30, discordance_rate=0.2)
    assert larger["power"] > smaller["power"]


def test_dev2_is_deterministic_and_excludes_dev1() -> None:
    rows = []
    for index in range(10_231):
        rows.append(
            {
                "item_id": f"row-{index}",
                "document_id": f"doc-{index}",
                "battery": ("gsm8k", "arc_challenge", "mbpp")[index % 3],
                "battery_role": "target_primary",
                "partition": "dev" if index < 1519 else "verified_train",
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
    assert not ({row["item_id"] for row in first} & {row["item_id"] for row in dev1})


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


def test_draft_lock_cannot_authorize_optimizer() -> None:
    path = Path(__file__).resolve().parents[1] / "training/paper2_stage2b_depth_executed_lock.draft.json"
    lock = load_and_validate_lock(path, require_signature=False)
    assert lock["training_authorized"] is False
    with pytest.raises(RuntimeError, match="pending Mark's signature"):
        load_and_validate_lock(path, require_signature=True)


def test_stage2b_preflight_target_is_wired_and_score_only() -> None:
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / "colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    cell = (root / "colab/STAGE5_PAPER2_STAGE2B_PREFLIGHT_CELL.py").read_text(encoding="utf-8")
    runner = (root / "colab/run_stage5_paper2_stage2b_preflight.py").read_text(encoding="utf-8")
    assert "paper2_stage2b_preflight" in bootstrap
    assert "paper2_stage2b_preflight_v1" in cell
    assert "optimizer_constructed\": False" in runner
    assert "confirm_scored\": False" in runner
    assert "training.run" not in runner
