from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from models.ablation_lm import AblationLM, AblationLMConfig
from models.ablation_lm.memory import ReadOnlyLatentMemory
from models.ablation_lm.observational_invariance import (
    CompleteObservationalInvarianceReceipt,
    DenseAnchorCell,
    OBS_INV_BACKENDS,
    OBS_INV_DTYPES,
    OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS,
    OBS_INV_K_AXIS_BACKGROUND,
    OBS_INV_K_VALUES,
    OBS_INV_MATERIALIZED_MODULES,
    OBS_INV_MODULE_SPECS,
    ObservationalInvarianceBlocked,
    compare_dense_anchor,
    compare_module_off_idempotence,
    observational_invariance_matrix,
    pending_dense_anchor_cell,
    promote_complete_observational_invariance,
    typed_observational_invariance_nonpass,
)


_TOKENS = torch.tensor([[1, 5, 9, 14, 21]], dtype=torch.long)
_LABELS = torch.tensor([[5, 9, 14, 21, 3]], dtype=torch.long)


def _config(recurrent_steps: int = 1, **updates: object) -> AblationLMConfig:
    base = AblationLMConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=16,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        use_recurrence=recurrent_steps > 1,
        recurrent_steps=recurrent_steps,
        max_recurrent_steps=8,
        hadamard_experts=2,
        scratch_width=2,
        engram_hashes_per_order=1,
        engram_table_size=17,
        engram_row_dim=2,
        long_term_memory_slots=3,
        long_term_memory_width=2,
        max_sequence_length=8,
    )
    return replace(base, **updates)


def _memory(config: AblationLMConfig) -> ReadOnlyLatentMemory:
    keys = torch.tensor(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)))
    values = torch.tensor(((0.5, -0.25), (-0.5, 0.75), (0.1, 0.2)))
    return ReadOnlyLatentMemory(
        config.d_model,
        keys=keys,
        values=values,
        provenance_ids=torch.arange(3),
        layer_scale=config.long_term_memory_layer_scale,
        norm_eps=config.norm_eps,
        initialization_seed=config.initialization_seed,
    )


def _model(config: AblationLMConfig) -> AblationLM:
    memory = _memory(config) if config.use_long_term_memory else None
    return AblationLM(config, long_term_memory=memory).eval()


def _manual_dense_output(
    model: AblationLM,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.token_embedding(_TOKENS)
    positions = torch.arange(_TOKENS.shape[1]).view(1, -1)
    for block in model.prelude_blocks:
        hidden = block(hidden, position_ids=positions)
    for block in model.core_blocks:
        hidden = block(hidden, position_ids=positions)
    for block in model.coda_blocks:
        hidden = block(hidden, position_ids=positions)
    logits = model.lm_head(model.final_norm(hidden))
    loss = model._language_model_loss(logits, _LABELS, None, None)
    return logits, loss


def _forward(model: AblationLM) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(_TOKENS, labels=_LABELS)
    assert output.loss is not None
    return output.logits, output.loss


def _background_config(module_name: str, recurrent_steps: int) -> AblationLMConfig:
    updates: dict[str, object] = {}
    if module_name in ("static_kv_core", "reentry_bridge"):
        updates["use_recurrence"] = True
    if module_name in ("bicameral_core", "bicameral_combiner"):
        updates.update(
            use_recurrence=True,
            use_bicameral_core=True,
            d_model=64,
            d_ff=64,
            scratch_width=8,
        )
    if module_name == "lane_carrier":
        updates["use_scratch"] = True
    return _config(recurrent_steps, **updates)


def _active_config(module_name: str, recurrent_steps: int) -> AblationLMConfig:
    background = _background_config(module_name, recurrent_steps)
    switches: dict[str, object] = {
        "front_hadamard_experts": {"use_front_hadamard_experts": True},
        "static_kv_core": {"use_static_kv_core": True},
        "bicameral_core": {},
        "bicameral_combiner": {},
        "reentry_bridge": {"use_reentry_bridge": True},
        "scratch": {"use_scratch": True},
        "lane_carrier": {"use_lane_carrier": True},
        "engram": {"use_engram": True},
        "long_term_memory": {"use_long_term_memory": True},
    }[module_name]
    return replace(background, **switches)


def _delete_attached_module(model: AblationLM, module_name: str) -> None:
    """Physically remove only the named attachment from a copied ON graph."""

    updates: dict[str, object]
    if module_name == "front_hadamard_experts":
        model.front_hadamard = None
        updates = {"use_front_hadamard_experts": False}
    elif module_name == "static_kv_core":
        updates = {"use_static_kv_core": False}
    elif module_name == "bicameral_core":
        with torch.no_grad():
            for block in model.core_blocks:
                for projection_name in (
                    "q_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ):
                    projection = getattr(block, projection_name)
                    projection.dU.zero_()
                    projection.dV.zero_()
        updates = {}
    elif module_name == "bicameral_combiner":
        assert model.bicameral_combiner is not None
        with torch.no_grad():
            model.bicameral_combiner.theta.zero_()
        updates = {}
    elif module_name == "reentry_bridge":
        model.reentry_bridge = None
        updates = {"use_reentry_bridge": False}
    elif module_name == "scratch":
        model.scratch = None
        updates = {"use_scratch": False, "use_lane_carrier": False}
    elif module_name == "lane_carrier":
        assert model.scratch is not None
        model.scratch.carrier = None
        updates = {"use_lane_carrier": False}
    elif module_name == "engram":
        model.engram = None
        updates = {"use_engram": False}
    elif module_name == "long_term_memory":
        model.long_term_memory = None
        updates = {"use_long_term_memory": False}
    else:  # pragma: no cover - guarded by the registry in every caller
        raise AssertionError(f"no integrated A7 deletion rule for {module_name}")
    model.config = replace(model.config, **updates)


def _activate_attached_module(model: AblationLM, module_name: str) -> None:
    """Plant the A7 positive control when the default is structural OFF."""

    if module_name != "bicameral_combiner":
        return
    assert model.bicameral_combiner is not None
    with torch.no_grad():
        model.bicameral_combiner.theta.copy_(
            torch.linspace(
                0.125,
                0.5,
                model.bicameral_combiner.num_bands,
                dtype=model.bicameral_combiner.theta.dtype,
                device=model.bicameral_combiner.theta.device,
            )
        )


def _cpu_anchor_cells() -> list[DenseAnchorCell]:
    cells: list[DenseAnchorCell] = []
    for dtype in (torch.float32, torch.bfloat16):
        model = _model(_config()).to(dtype=dtype)
        with torch.no_grad():
            observed_logits, observed_loss = _forward(model)
            reference_logits, reference_loss = _manual_dense_output(model)
        cells.append(
            compare_dense_anchor(
                reference_logits,
                reference_loss,
                observed_logits,
                observed_loss,
                backend="cpu",
                reason="PF-3.4 K=1 all-optionals-OFF 4/2/4 dense anchor",
            )
        )
    return cells


def _cpu_module_cells():
    cells = []
    for spec in OBS_INV_MODULE_SPECS:
        for recurrent_steps in OBS_INV_K_VALUES:
            for dtype in (torch.float32, torch.bfloat16):
                dtype_name = str(dtype).removeprefix("torch.")
                if not spec.integrated or not spec.eligible_at(recurrent_steps):
                    cells.append(
                        typed_observational_invariance_nonpass(
                            module_name=spec.module_name,
                            requested_recurrent_steps=recurrent_steps,
                            dtype=dtype_name,
                            backend="cpu",
                            reason=(
                                spec.eligibility_reason
                                if spec.integrated
                                else "production module is absent"
                            ),
                        )
                    )
                    continue

                active = _model(
                    _active_config(spec.module_name, recurrent_steps)
                )
                _activate_attached_module(active, spec.module_name)
                off_after_on = copy.deepcopy(active)
                _delete_attached_module(off_after_on, spec.module_name)
                all_off = _model(
                    _background_config(spec.module_name, recurrent_steps)
                )
                if spec.module_name in ("bicameral_core", "bicameral_combiner"):
                    _delete_attached_module(all_off, spec.module_name)
                assert off_after_on.config == all_off.config
                active.to(dtype=dtype)
                off_after_on.to(dtype=dtype)
                all_off.to(dtype=dtype)
                with torch.no_grad():
                    all_off_logits, all_off_loss = _forward(all_off)
                    off_logits, off_loss = _forward(off_after_on)
                    on_logits, on_loss = _forward(active)
                cells.append(
                    compare_module_off_idempotence(
                        all_off_logits,
                        all_off_loss,
                        off_logits,
                        off_loss,
                        on_logits,
                        on_loss,
                        module_name=spec.module_name,
                        requested_recurrent_steps=recurrent_steps,
                        executed_recurrent_steps=recurrent_steps,
                        backend="cpu",
                        reason="PF-3.4 matched-background structural deletion",
                    )
                )
    return cells


@pytest.fixture(scope="module")
def cpu_a7_matrix():
    anchor_cells = _cpu_anchor_cells()
    anchor_cells.extend(
        pending_dense_anchor_cell(
            dtype=dtype,
            backend="cuda_deterministic",
            reason="awaits execution under the PRE-FLIGHT meter",
        )
        for dtype in OBS_INV_DTYPES
    )
    cells = _cpu_module_cells()
    cells.extend(
        typed_observational_invariance_nonpass(
            module_name=spec.module_name,
            requested_recurrent_steps=recurrent_steps,
            dtype=dtype,
            backend="cuda_deterministic",
            reason="awaits execution under the PRE-FLIGHT meter",
        )
        for spec in OBS_INV_MODULE_SPECS
        for recurrent_steps in OBS_INV_K_VALUES
        for dtype in OBS_INV_DTYPES
    )
    return observational_invariance_matrix(anchor_cells, cells)


def test_a7_k1_dense_424_anchor_is_bit_identical_for_logits_and_loss(
    cpu_a7_matrix,
) -> None:
    cpu_anchors = tuple(
        cell for cell in cpu_a7_matrix.anchor_cells if cell.backend == "cpu"
    )
    assert {cell.dtype for cell in cpu_anchors} == set(OBS_INV_DTYPES)
    assert all(cell.status == "passed" for cell in cpu_anchors)
    assert all(cell.logits_bit_identical is True for cell in cpu_anchors)
    assert all(cell.loss_bit_identical is True for cell in cpu_anchors)
    assert all(cell.logits_mismatch_count == 0 for cell in cpu_anchors)
    assert all(cell.loss_mismatch_count == 0 for cell in cpu_anchors)


def test_a7_every_eligible_integrated_cpu_cell_is_exact_and_nontrivial(
    cpu_a7_matrix,
) -> None:
    passed = tuple(
        cell
        for cell in cpu_a7_matrix.cells
        if cell.backend == "cpu" and cell.integrated and cell.eligible
    )
    assert len(passed) == 70
    assert all(cell.status == "passed" for cell in passed)
    assert all(cell.off_logits_bit_identical is True for cell in passed)
    assert all(cell.off_loss_bit_identical is True for cell in passed)
    assert all(cell.on_logits_nontrivial is True for cell in passed)
    assert all(cell.off_logits_mismatch_count == 0 for cell in passed)
    assert all(cell.off_loss_mismatch_count == 0 for cell in passed)
    assert all(cell.on_logits_mismatch_count > 0 for cell in passed)
    assert cpu_a7_matrix.cpu_passed
    cpu_a7_matrix.require_cpu_passed()


def test_a7_eligibility_and_absence_are_typed_nonpasses(cpu_a7_matrix) -> None:
    assert OBS_INV_MATERIALIZED_MODULES == (
        "front_hadamard_experts",
        "static_kv_core",
        "bicameral_core",
        "bicameral_combiner",
        "reentry_bridge",
        "scratch",
        "lane_carrier",
        "engram",
        "long_term_memory",
    )
    assert "recurrence" not in OBS_INV_MATERIALIZED_MODULES
    assert OBS_INV_K_AXIS_BACKGROUND == ("recurrence",)
    assert OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS == (
        "integrated_rotor_carrier",
        "per_band_callosum",
        "sidecar",
    )
    assert len(cpu_a7_matrix.cells) == 192
    statuses = {}
    for cell in cpu_a7_matrix.cells:
        statuses[cell.status] = statuses.get(cell.status, 0) + 1
    assert statuses == {
        "absent": 48,
        "ineligible": 4,
        "passed": 70,
        "pending": 70,
    }

    by_coordinate = {cell.coordinate: cell for cell in cpu_a7_matrix.cells}
    for dtype in OBS_INV_DTYPES:
        for backend in OBS_INV_BACKENDS:
            assert by_coordinate[("reentry_bridge", 1, dtype, backend)].status == (
                "ineligible"
            )
            assert by_coordinate[("static_kv_core", 1, dtype, backend)].status == (
                "passed" if backend == "cpu" else "pending"
            )
            assert by_coordinate[("sidecar", 1, dtype, backend)].status == "absent"
            assert by_coordinate[("sidecar", 2, dtype, backend)].eligible is False
            assert by_coordinate[("sidecar", 4, dtype, backend)].eligible is True


def test_a7_cuda_cells_remain_pending_and_block_complete_promotion(
    cpu_a7_matrix,
) -> None:
    assert not cpu_a7_matrix.complete
    assert len(cpu_a7_matrix.deferred_cells) == 70
    assert all(
        cell.backend == "cuda_deterministic"
        for cell in cpu_a7_matrix.deferred_cells
    )
    with pytest.raises(ObservationalInvarianceBlocked, match="pending"):
        cpu_a7_matrix.require_complete()
    with pytest.raises(ObservationalInvarianceBlocked, match="pending"):
        promote_complete_observational_invariance(cpu_a7_matrix)


def test_a7_receipts_cannot_be_promoted_by_dataclass_mutation(
    cpu_a7_matrix,
) -> None:
    passed = next(cell for cell in cpu_a7_matrix.cells if cell.status == "passed")
    anchor = next(
        cell for cell in cpu_a7_matrix.anchor_cells if cell.status == "passed"
    )
    with pytest.raises(TypeError, match="factory-sealed"):
        replace(passed, status="failed")
    with pytest.raises(TypeError, match="factory-sealed"):
        replace(anchor, status="pending")
    with pytest.raises(TypeError, match="factory-sealed"):
        replace(cpu_a7_matrix, cells=cpu_a7_matrix.cells)
    with pytest.raises(TypeError, match="factory-sealed"):
        CompleteObservationalInvarianceReceipt(
            matrix_digest="0" * 64,
            required_cell_count=1,
        )
    with pytest.raises(TypeError, match="factory-sealed"):
        DenseAnchorCell(
            dtype="float32",
            backend="cpu",
            status="passed",
            logits_bit_identical=True,
            loss_bit_identical=True,
            logits_mismatch_count=0,
            loss_mismatch_count=0,
            max_absolute_logit_difference=0.0,
            max_absolute_loss_difference=0.0,
            reason="forged",
        )


def test_a7_matrix_coverage_is_fail_closed(cpu_a7_matrix) -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        observational_invariance_matrix(
            cpu_a7_matrix.anchor_cells,
            cpu_a7_matrix.cells[:-1],
        )
    with pytest.raises(ValueError, match="anchor coverage mismatch"):
        observational_invariance_matrix(
            cpu_a7_matrix.anchor_cells[:-1],
            cpu_a7_matrix.cells,
        )


def test_a7_positive_control_rejects_a_dead_on_switch() -> None:
    logits = torch.zeros(1, 2, 3)
    loss = torch.zeros(())
    cell = compare_module_off_idempotence(
        logits,
        loss,
        logits.clone(),
        loss.clone(),
        logits.clone(),
        loss.clone(),
        module_name="engram",
        requested_recurrent_steps=1,
        executed_recurrent_steps=1,
        backend="cpu",
        reason="planted dead switch",
    )
    assert cell.status == "failed"
    assert cell.off_logits_bit_identical is True
    assert cell.off_loss_bit_identical is True
    assert cell.on_logits_nontrivial is False


def test_a7_positive_control_requires_loss_and_logits_to_be_nontrivial() -> None:
    logits = torch.zeros(1, 2, 3)
    changed_logits = logits.clone()
    changed_logits[0, 0, 0] = 1.0
    loss = torch.zeros(())
    cell = compare_module_off_idempotence(
        logits,
        loss,
        logits.clone(),
        loss.clone(),
        changed_logits,
        loss.clone(),
        module_name="engram",
        requested_recurrent_steps=1,
        executed_recurrent_steps=1,
        backend="cpu",
        reason="planted logits-only switch",
    )
    assert cell.status == "failed"
    assert cell.on_logits_nontrivial is True
    assert cell.on_loss_nontrivial is False


def test_a7_executed_cuda_label_cannot_masquerade_with_cpu_tensors() -> None:
    logits = torch.zeros(1, 2, 3)
    changed_logits = logits.clone()
    changed_logits[0, 0, 0] = 1.0
    loss = torch.zeros(())
    changed_loss = torch.ones(())
    with pytest.raises(ValueError, match="must come from CUDA tensors"):
        compare_dense_anchor(
            logits,
            loss,
            logits.clone(),
            loss.clone(),
            backend="cuda_deterministic",
            reason="planted backend masquerade",
        )
    with pytest.raises(ValueError, match="must come from CUDA tensors"):
        compare_module_off_idempotence(
            logits,
            loss,
            logits.clone(),
            loss.clone(),
            changed_logits,
            changed_loss,
            module_name="engram",
            requested_recurrent_steps=1,
            executed_recurrent_steps=1,
            backend="cuda_deterministic",
            reason="planted backend masquerade",
        )


def test_a7_recurrence_is_background_and_static_kv_is_live_at_k1() -> None:
    """Document the two same-K/non-triviality boundaries in executable form."""

    tokens = _TOKENS
    dense = _model(_config()).eval()
    recurrent_k1 = _model(_config(use_recurrence=True)).eval()
    static_k1 = _model(
        _config(use_recurrence=True, use_static_kv_core=True)
    ).eval()
    with torch.no_grad():
        dense_logits = dense(tokens).logits
        recurrent_logits = recurrent_k1(tokens).logits
        static_logits = static_k1(tokens).logits
    assert torch.equal(dense_logits, recurrent_logits)
    assert not torch.equal(recurrent_logits, static_logits)
    assert "recurrence" not in OBS_INV_MATERIALIZED_MODULES
    static_spec = next(
        spec for spec in OBS_INV_MODULE_SPECS if spec.module_name == "static_kv_core"
    )
    assert static_spec.eligible_at(1)
    assert static_spec.eligible_at(2)


def test_a7_bicameral_core_and_s2_combiner_have_explicit_off_surfaces(
    cpu_a7_matrix,
) -> None:
    by_coordinate = {cell.coordinate: cell for cell in cpu_a7_matrix.cells}
    for module_name in ("bicameral_core", "bicameral_combiner"):
        spec = next(
            spec for spec in OBS_INV_MODULE_SPECS if spec.module_name == module_name
        )
        assert spec.integrated
        assert spec.eligible_at(1)
        assert spec.eligible_at(8)
        for recurrent_steps in OBS_INV_K_VALUES:
            for dtype in OBS_INV_DTYPES:
                cpu_cell = by_coordinate[
                    (module_name, recurrent_steps, dtype, "cpu")
                ]
                assert cpu_cell.status == "passed"
                assert cpu_cell.off_logits_bit_identical is True
                assert cpu_cell.off_loss_bit_identical is True
                assert cpu_cell.on_logits_nontrivial is True
                assert cpu_cell.on_loss_nontrivial is True
                assert by_coordinate[
                    (module_name, recurrent_steps, dtype, "cuda_deterministic")
                ].status == "pending"
