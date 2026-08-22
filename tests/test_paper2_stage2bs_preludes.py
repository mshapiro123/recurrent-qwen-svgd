from __future__ import annotations

from pathlib import Path

import torch

from models.paper2_dc2_student import Phase3StudentModules
from models.paper2_stage2b_depth import Stage2BDepthAttachment
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM
from tests.test_recurrent_wrapper_tiny import TinyCausalLM
from training.paper2_stage2bs_preludes import (
    dependency_verdict,
    load_lock,
    noise_verdict,
    prelude1_decision,
    starvation_verdict,
    transplant_verdict,
)


ROOT = Path(__file__).resolve().parents[1]


def _wrapper(batch: int = 2, *, randomize_innovation: bool = True):
    torch.manual_seed(20260821)
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
    if randomize_innovation:
        with torch.no_grad():
            wrapper.stage2b_depth_attachment.flow.hidden_innovation.weight.normal_(std=0.2)
            wrapper.stage2b_depth_attachment.flow.prompt_gate.weight.normal_(std=0.2)
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])[:batch]
    return wrapper, ids, torch.ones_like(ids)


def test_prelude_lock_records_the_ratified_f1_estimator() -> None:
    lock = load_lock(ROOT / "training/paper2_stage2bs_preludes_lock.json")
    assert lock["status"] == "SIGNED"
    assert lock["mark_signed"] is True
    assert lock["prelude_2"]["f1_estimator"] == "absolute_delta_WP_over_absolute_delta_WH"
    assert lock["amendment_authority"]["drive_id"] == "1EdkabZdjO-bhlKfVaXzVzyZXoO-rBQ94"
    assert lock["optimizer_steps_allowed"] == 0
    assert lock["sealed"] == {"confirm_scored": False, "eval_e_scored": False}


def test_preflight_partials_are_durable_and_resumable() -> None:
    runner = (ROOT / "colab/run_stage5_paper2_stage2bs_preludes.py").read_text(
        encoding="utf-8"
    )
    assert 'private = DRIVE_RUN / f"private/preflight/seed_{seed}"' in runner
    assert 'private = scratch / f"preflight_seed_{seed}"' not in runner


def test_score_blind_initialization_phase_is_available_for_desk_recovery() -> None:
    evaluator = (ROOT / "eval/eval_paper2_stage2bs_preludes.py").read_text(
        encoding="utf-8"
    )
    assert 'choices=("initialize", "preflight", "probes", "desk")' in evaluator
    assert '"status": "complete_score_blind"' in evaluator
    assert '"task_rows_scored": 0' in evaluator


def test_registered_wp_initialization_has_zero_relative_denominator() -> None:
    wrapper, _ids, _attention = _wrapper(randomize_innovation=False)
    weight = wrapper.stage2b_depth_attachment.flow.prompt_gate.weight
    assert torch.count_nonzero(weight).item() == 0


def test_per_loop_inherited_flow_suppression_is_exact_and_bridge_remains_live() -> None:
    wrapper, ids, attention = _wrapper()
    with torch.no_grad():
        output = wrapper(
            input_ids=ids,
            attention_mask=attention,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_loop_diagnostic_modes={1: "inherited_flow_off", 2: "inherited_flow_off"},
            return_dict=True,
        )
    metrics = output.metrics
    assert metrics["stage2b_flow_update_loop_2_max_abs"].item() == 0.0
    assert metrics["stage2b_flow_update_loop_3_max_abs"].item() == 0.0
    assert metrics["stage2b_flow_update_loop_4_max_abs"].item() > 0.0
    assert metrics["stage2b_writeback_ratio_loop_2_max"].item() > 0.0
    assert metrics["stage2b_writeback_ratio_loop_3_max"].item() > 0.0


def test_noise_and_transplant_controls_are_opt_in_and_live() -> None:
    wrapper, ids, attention = _wrapper()
    noise = torch.ones((2, 4, 8))
    permutation = torch.tensor([1, 0])
    common = dict(
        input_ids=ids,
        attention_mask=attention,
        max_loops=4,
        stage2b_depth_enabled=True,
        stage2b_stage="M2",
        return_dict=True,
    )
    with torch.no_grad():
        baseline = wrapper(**common).logits
        noisy = wrapper(**common, stage2b_recurrent_state_noise={1: (noise, 0.03)}).logits
        transplanted = wrapper(
            **common, stage2b_recurrent_state_permutations={3: permutation}
        ).logits
    assert not torch.equal(noisy, baseline)
    assert not torch.equal(transplanted, baseline)


def test_registered_prelude_decision_rules_cover_boundaries() -> None:
    assert noise_verdict({0.001: 0.8, 0.003: 0.7, 0.01: 0.6, 0.03: 0.5, 0.1: 0.4}) == "SMOOTH"
    assert noise_verdict({0.001: 0.49, 0.003: 0.4, 0.01: 0.2, 0.03: 0.1, 0.1: 0.0}) == "SHATTERS"
    assert dependency_verdict(160, 128) == "SURVIVES"
    assert dependency_verdict(160, 80) == "DEPENDENT"
    assert transplant_verdict(128, 64) == "GRACEFUL"
    assert transplant_verdict(128, 19) == "CATASTROPHIC"
    reusable = [{"noise": "SMOOTH", "dependency": "SURVIVES", "transplant": "GRACEFUL"}] * 2
    assert prelude1_decision(reusable) == "REUSABLE_COMPUTATION"
    mixed = [reusable[0], {"noise": "SMOOTH", "dependency": "DEPENDENT", "transplant": "MIXED"}]
    assert prelude1_decision(mixed) == "ESCALATE_STRATEGY"
    assert starvation_verdict([0.25, 0.2]) == "STARVED"
    assert starvation_verdict([0.3, 0.8]) == "NOT_STARVED"
