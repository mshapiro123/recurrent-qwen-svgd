import math
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.paper2_dc2_student import (
    DetachedMultiProbePool,
    Phase3StudentModules,
    ProbeControlState,
    install_probe_control_reader,
)
from eval.eval_paper2_phase3_p35_persistence import _generate_batch
from eval.repair_paper2_phase3_serving_oracle_cache import subset_prior_cache
from training.paper2_phase3_p35 import (
    P35LandingContract,
    assert_source_anchor_identity,
    initialize_ema,
    landing_learning_rate,
    margin_summary,
    repaired_oracle_payload,
    reanchored_directions,
    set_p35_trainable,
    update_ema,
)
from training.paper2_phase2_matched_alpha import build_adamw_groups
from training.run_paper2_phase3_p35 import (
    _audit_ema_state,
    _adamw_group_names,
    _probe_attachment_identity,
    _restore_optimizer_by_name,
    _validate_executed_lock,
    run,
)


def test_probe_pool_is_mean_preserving_at_attach_and_stops_cell_gradient() -> None:
    pool = DetachedMultiProbePool(cell_dim=8, n_probes=4)
    cells = torch.randn(3, 5, 8, requires_grad=True)
    output = pool(cells)
    assert torch.allclose(output, cells.mean(dim=1), atol=1e-6)
    output.square().sum().backward()
    assert cells.grad is None
    assert pool.probes.grad is not None
    assert pool.out.weight.grad is not None


def test_probe_control_attach_matches_mean_control() -> None:
    torch.manual_seed(4)
    embedding = nn.Embedding(31, 16)
    mean = Phase3StudentModules(
        tied_embedding=embedding, hidden_size=16, latent_dim=8, control_dim=6
    )
    probe = Phase3StudentModules(
        tied_embedding=embedding,
        hidden_size=16,
        latent_dim=8,
        control_dim=6,
        control_reader="probe",
    )
    probe.control.position_embedding.load_state_dict(mean.control.position_embedding.state_dict())
    probe.control.cell.load_state_dict(mean.control.cell.state_dict())
    assert isinstance(probe.control, ProbeControlState)
    scratch = torch.randn(2, 8, 8)
    common = {
        "scratch": scratch,
        "previous": None,
        "innovation_norm": torch.randn(2),
        "student_entropy": torch.randn(2),
        "top2_margin": torch.randn(2),
        "position_bucket": torch.tensor([1, 4]),
    }
    assert torch.allclose(mean.control(**common), probe.control(**common), atol=1e-6)


def test_real_module_probe_attachment_is_identity() -> None:
    torch.manual_seed(41)
    embedding = nn.Embedding(31, 16)
    module = Phase3StudentModules(
        tied_embedding=embedding, hidden_size=16, latent_dim=8, control_dim=6
    )
    assert _probe_attachment_identity(module) <= 1e-6
    assert isinstance(module.control, ProbeControlState)
    assert all(
        not torch.is_inference(parameter)
        for parameter in module.control.reader.parameters()
    )
    trainable = set_p35_trainable(module, arm="probe_reader")
    assert "control.reader.probes" in trainable


def test_landing_contract_lr_and_ema() -> None:
    P35LandingContract().validate()
    assert landing_learning_rate(4001) < 3e-4
    assert landing_learning_rate(4200) == pytest.approx(1.5e-4)
    assert landing_learning_rate(4400) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        landing_learning_rate(4000)
    state = {"a": torch.tensor([1.0])}
    ema = initialize_ema(state)
    update_ema(ema, {"a": torch.tensor([3.0])}, decay=0.5)
    assert ema["a"].item() == 2.0


def test_arm_trainable_surfaces() -> None:
    embedding = nn.Embedding(31, 16)
    mean = Phase3StudentModules(tied_embedding=embedding, hidden_size=16, latent_dim=8)
    probe = Phase3StudentModules(
        tied_embedding=embedding, hidden_size=16, latent_dim=8, control_reader="probe"
    )
    assert all(
        name.startswith(("bridge.", "control."))
        for name in set_p35_trainable(mean, arm="stabilized")
    )
    names = set(set_p35_trainable(probe, arm="probe_reader"))
    assert "control.reader.probes" in names
    assert "control.reader.out.weight" in names


def test_repaired_oracle_is_exact_for_pinned_reader() -> None:
    torch.manual_seed(2)
    hidden = torch.randn(7, 8)
    weight = torch.randn(19, 8)
    prior = {
        "kind": "paper2_phase3_agreement_oracle_direction_cache_v1",
        "record_ids": [str(index) for index in range(7)],
        "target_tokens": torch.arange(7, dtype=torch.int32) + 7,
    }
    repaired = repaired_oracle_payload(
        prior=prior, selected_hidden=hidden, lm_head_weight=weight
    )
    receipt = assert_source_anchor_identity(
        cache=repaired, selected_hidden=hidden, lm_head_weight=weight
    )
    assert receipt["identity_rate"] == 1.0
    changed = dict(repaired)
    changed["source_tokens"] = repaired["source_tokens"].clone()
    changed["source_tokens"][0] = (changed["source_tokens"][0] + 1) % 19
    with pytest.raises(RuntimeError):
        assert_source_anchor_identity(
            cache=changed, selected_hidden=hidden, lm_head_weight=weight
        )


def test_serving_cache_repair_subsets_registered_rows_in_audit_order() -> None:
    prior = {
        "kind": "paper2_phase3_agreement_oracle_direction_cache_v1",
        "record_ids": ["a", "b", "c", "d"],
        "documents": ["da", "db", "dc", "dd"],
        "directions": torch.arange(12).reshape(4, 3),
        "teachability": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "horizons": torch.tensor([1, 2, 3, 4]),
        "sources": ["old", "new", "old", "new"],
        "strata": ["x", "y", "z", "w"],
        "source_tokens": torch.tensor([11, 12, 13, 14]),
        "target_tokens": torch.tensor([21, 22, 23, 24]),
        "global_receipt": {"kept": True},
    }
    subset = subset_prior_cache(prior, ["d", "b"])
    assert subset["record_ids"] == ["d", "b"]
    assert subset["documents"] == ["dd", "db"]
    assert subset["sources"] == ["new", "new"]
    assert subset["target_tokens"].tolist() == [24, 22]
    assert subset["directions"].tolist() == [[9, 10, 11], [3, 4, 5]]
    assert subset["global_receipt"] == {"kept": True}
    assert subset["source_cache_rows"] == 4
    assert subset["selected_audit_rows"] == 2


def test_reanchoring_uses_current_source_and_margin_summary_is_stratified() -> None:
    torch.manual_seed(9)
    hidden = torch.randn(3, 8)
    weight = torch.randn(17, 8)
    target = torch.tensor([3, 4, 5])
    source, directions = reanchored_directions(
        current_hidden=hidden, target_tokens=target, lm_head_weight=weight
    )
    assert source.shape == (3,)
    assert directions.shape == hidden.shape
    assert torch.allclose(directions.norm(dim=-1), torch.ones(3), atol=1e-6)
    summary = margin_summary(
        [
            {
                "battery": "gsm8k",
                "answer_token_margins": [0.2, 0.1],
                "answer_token_margin_minimum": 0.1,
            },
            {
                "battery": "mbpp",
                "answer_token_margins": [0.4],
                "answer_token_margin_minimum": 0.4,
            },
        ]
    )
    assert summary["pooled"]["rows"] == 2
    assert summary["by_battery"]["gsm8k"]["tokens"] == 2


def test_probe_arm_restores_existing_optimizer_state_and_adds_reader_state_lazily() -> None:
    embedding = nn.Embedding(31, 16)
    module = Phase3StudentModules(tied_embedding=embedding, hidden_size=16, latent_dim=8)
    set_p35_trainable(module, arm="stabilized")
    old_names = _adamw_group_names(module)
    old_optimizer = torch.optim.AdamW(build_adamw_groups(module, weight_decay=0.01))
    loss = sum(parameter.square().sum() for parameter in module.parameters() if parameter.requires_grad)
    loss.backward()
    old_optimizer.step()
    saved = old_optimizer.state_dict()
    install_probe_control_reader(module, n_probes=4)
    set_p35_trainable(module, arm="probe_reader")
    new_names = _adamw_group_names(module)
    new_optimizer = torch.optim.AdamW(build_adamw_groups(module, weight_decay=0.01))
    receipt = _restore_optimizer_by_name(
        optimizer=new_optimizer,
        saved=saved,
        old_group_names=old_names,
        new_group_names=new_names,
    )
    assert receipt["restored_parameter_states"] > 0
    assert receipt["new_parameter_states"] == 2


def test_ema_audit_restores_raw_parameters(monkeypatch) -> None:
    module = nn.Linear(2, 1, bias=False)
    module.weight.requires_grad_(True)
    raw = module.weight.detach().clone()
    seen = {}

    def fake_audit_model(**kwargs):
        seen["weight"] = kwargs["module"].weight.detach().clone()
        return {"pi_dir": {"point": 0.2}}, []

    monkeypatch.setattr(
        "training.run_paper2_phase3_p35.audit_model", fake_audit_model
    )
    ema = {"weight": torch.full_like(module.weight, 3.0)}
    result, rows = _audit_ema_state(
        module=module,
        ema_state=ema,
        material={},
        direction_index={},
        directions=torch.empty(0),
        seed=0,
        step=4100,
        device="cpu",
    )
    assert result["pi_dir"]["point"] == 0.2
    assert rows == []
    assert torch.equal(seen["weight"], ema["weight"])
    assert torch.equal(module.weight, raw)


def test_persistence_batch_stops_each_path_at_its_own_eos() -> None:
    class Batch(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, _prompts, **_kwargs):
            return Batch(
                input_ids=torch.ones(2, 2, dtype=torch.long),
                attention_mask=torch.ones(2, 2, dtype=torch.long),
            )

    def output(tokens, sources):
        augmented = torch.full((2, 10), -1.0)
        base = torch.full((2, 10), -1.0)
        augmented[torch.arange(2), torch.tensor(tokens)] = 1.0
        base[torch.arange(2), torch.tensor(sources)] = 1.0
        return SimpleNamespace(augmented_logits=augmented, base_logits=base)

    class Graph:
        def prefill_cached(self, **_kwargs):
            return 0, output([9, 1], [2, 3])

        def advance_cached(self, *, state, selected_tokens):
            assert selected_tokens.tolist() == [9, 1]
            return state + 1, output([5, 9], [4, 6])

    generated, sources = _generate_batch(
        graph=Graph(),
        tokenizer=Tokenizer(),
        prompts=["a", "b"],
        caps=[4, 4],
        device="cpu",
    )
    assert generated == [[9], [1, 9]]
    assert sources == [[2], [3, 6]]


def test_runner_refuses_unratified_lock_before_reading_run_inputs(tmp_path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "kind": "paper2_phase3_p35_executed_lock_v1",
                "status": "draft",
                "locked_before_training": False,
                "training_authorized": False,
                "mark_ratified": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="remains disabled"):
        run(Namespace(lock=lock))


def test_executed_lock_matches_code_and_is_ratified() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = json.loads(
        (root / "training" / "paper2_phase3_p35_preregistration.draft.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = _validate_executed_lock(lock)
    assert lock["training_authorized"] is True
    assert lock["locked_before_training"] is True
    assert lock["mark_ratified"] is True
    assert lock["status"] == "approved_for_training"
    assert lock["landing"]["landing_steps"] == 400
    assert lock["landing"]["look_steps"] == [4100, 4200, 4300, 4400]
    assert lock["checkpoint_policy"]["ema_decay"] == 0.995
    assert lock["evaluation"]["primary_gate_ceiling"] == 0.02
    assert lock["estimator_repair"]["required_identity_rate"] == 1.0
    assert lock["arms"]["R"]["probe_count"] == 4
    assert lock["unresolved_before_ratification"] == []
    assert receipt["primary_evaluation_ceiling"] == 0.02
    assert receipt["causal_instrument"] == "repaired v2 only"
    assert receipt["execution_build_commit"] == "6071d8b23b66bd74ccf188c2d3fe0637042b1c50"
