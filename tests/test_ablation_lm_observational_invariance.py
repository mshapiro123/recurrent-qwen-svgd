from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from models.ablation_lm import AblationLM, AblationLMConfig
from models.ablation_lm.observational_invariance import (
    ObservationalInvarianceBlocked,
    compare_observational_invariance,
    deferred_observational_invariance_cell,
    observational_invariance_matrix,
)


def _config(**updates: object) -> AblationLMConfig:
    base = AblationLMConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=16,
        n_prelude_layers=1,
        n_core_blocks=1,
        n_coda_layers=1,
        max_recurrent_steps=8,
        hadamard_experts=2,
        engram_hashes_per_order=1,
        engram_table_size=17,
        engram_row_dim=2,
        max_sequence_length=8,
    )
    return replace(base, **updates)


def test_a7_each_module_off_alone_is_not_the_dense_baseline() -> None:
    """Counterexample to one literal reading; this is not an A7 pass cell."""

    tokens = torch.tensor([[1, 5, 9, 14, 21]])
    dense = AblationLM(_config()).eval()
    engram_off_but_hadamard_on = AblationLM(
        _config(use_front_hadamard_experts=True, use_engram=False)
    ).eval()

    with torch.no_grad():
        dense_logits = dense(tokens).logits
        module_off_logits = engram_off_but_hadamard_on(tokens).logits

    assert not torch.equal(dense_logits, module_off_logits)
    assert torch.count_nonzero(dense_logits.ne(module_off_logits)) > 0


def test_a7_all_optional_modules_off_at_k2_is_not_the_k1_dense_baseline() -> None:
    """Counterexample to the other literal reading; K changes the graph."""

    tokens = torch.tensor([[1, 5, 9, 14, 21]])
    dense = AblationLM(_config()).eval()
    recurrent_dense_core = AblationLM(
        _config(use_recurrence=True, recurrent_steps=2)
    ).eval()

    with torch.no_grad():
        dense_logits = dense(tokens).logits
        recurrent_logits = recurrent_dense_core(tokens).logits

    assert not torch.equal(dense_logits, recurrent_logits)
    assert torch.count_nonzero(dense_logits.ne(recurrent_logits)) > 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_a7_candidate_matched_background_semantics_is_executable(
    dtype: torch.dtype,
) -> None:
    """Evidence for the minimal semantic proposed back to strategy.

    The independently initialized structural-OFF graph is compared with the
    same matched background after deleting the attached module.  This proves
    that exact cells are executable without calling them the registered A7
    matrix before strategy chooses this reading.
    """

    tokens = torch.tensor([[1, 5, 9, 14, 21]])
    attached = AblationLM(_config(use_engram=True)).eval()
    deletion_reference = copy.deepcopy(attached)
    deletion_reference.engram = None
    deletion_reference.config = _config(use_engram=False)
    structural_off = AblationLM(_config(use_engram=False)).eval()
    deletion_reference.to(dtype=dtype)
    structural_off.to(dtype=dtype)

    with torch.no_grad():
        reference_logits = deletion_reference(tokens).logits
        observed_logits = structural_off(tokens).logits

    cell = compare_observational_invariance(
        reference_logits,
        observed_logits,
        module_name="engram",
        requested_recurrent_steps=1,
        executed_recurrent_steps=1,
        backend="cpu",
        reason=(
            "candidate matched-background semantics; not a registered A7 pass "
            "until strategy binds the reading"
        ),
    )
    assert cell.status == "passed"
    assert cell.bit_identical is True


def test_a7_comparison_is_exact_and_reports_one_changed_element() -> None:
    reference = torch.zeros(2, 3, dtype=torch.float32)
    observed = reference.clone()
    observed[0, 1] = torch.finfo(torch.float32).eps

    cell = compare_observational_invariance(
        reference,
        observed,
        module_name="engram",
        requested_recurrent_steps=4,
        executed_recurrent_steps=4,
        backend="cpu",
        reason="positive control",
    )

    assert cell.status == "failed"
    assert cell.bit_identical is False
    assert cell.mismatch_count == 1
    assert cell.max_absolute_difference == torch.finfo(torch.float32).eps


def test_a7_deferred_cells_cannot_be_promoted_to_passes() -> None:
    deferred = deferred_observational_invariance_cell(
        module_name="engram",
        requested_recurrent_steps=1,
        executed_recurrent_steps=1,
        dtype="float32",
        backend="cuda_deterministic",
        reason="awaits the dedicated PRE-FLIGHT meter",
    )

    assert deferred.status == "deferred"
    assert deferred.bit_identical is None
    with pytest.raises(ValueError, match="coverage mismatch"):
        observational_invariance_matrix((deferred,))


def test_a7_matrix_object_never_treats_deferred_as_complete() -> None:
    # The full Cartesian matrix is intentionally not minted while catch #27 is
    # open.  This direct guard keeps the status semantics independently tested.
    from models.ablation_lm.observational_invariance import (
        ObservationalInvarianceMatrix,
    )

    cell = deferred_observational_invariance_cell(
        module_name="engram",
        requested_recurrent_steps=1,
        executed_recurrent_steps=1,
        dtype="float32",
        backend="cuda_deterministic",
        reason="awaits the dedicated PRE-FLIGHT meter",
    )
    partial = ObservationalInvarianceMatrix(
        cells=(cell,),
        excluded_absent_integrations=(
            "integrated_rotor_carrier",
            "per_band_callosum",
            "sidecar",
        ),
    )

    assert not partial.complete
    with pytest.raises(ObservationalInvarianceBlocked, match="deferred"):
        partial.require_complete()
