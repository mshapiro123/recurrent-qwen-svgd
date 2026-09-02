"""PF-3.3 joint-state Jacobian measurement on the current recurrent visit.

This is a standalone governed measurement harness. It does not mutate or
promote ``AblationLM``'s pre-PF-3 hidden/scratch Jacobian diagnostic, whose lane
coordinates are dimension-reweighted. The PF-3 metric is instead the literal
Euclidean direct sum ``z=[h; lanes; scratch; carrier-when-integrated]``. On the
current bring-up graph only ``h`` and position-aligned ``scratch`` exist, so the
other components are omitted and reported as absent rather than fabricated.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from analysis.weft1_preflight_c2 import (
    C2_RATIFIED_VISITS,
    C2_ROOT_SEED,
    build_c2_current_toy_model,
    c2_fixed_batch,
)
from models.ablation_lm.certificates import (
    AdapterCertificateReceipt,
    JointStateLoopLipschitzReceipt,
    absent_weft1_adapter_placeholders,
    certify_callosum_factor,
    certify_state_translation_factor,
    compose_adapter_certificate,
    estimate_empirical_joint_state_core_factor,
    make_joint_state_loop_lipschitz_receipt,
)


CJAC1_AUTHORITY = "docs/STRATEGY_PREFLIGHT_AMENDMENT_PF3_20260902.md#3-PF-3.3"
CJAC1_AUTHORITY_SHA256 = (
    "7f0081e504366ce98f8bf183b7e14c0bed47647aa381196a9d5e9540b5334cef"
)
CJAC1_AUTHORITY_BYTES = 14_632
CJAC1_TERMINAL_VISIT = C2_RATIFIED_VISITS
CJAC1_CURRENT_COMPONENTS = ("h", "scratch")
CJAC1_ABSENT_COMPONENTS = ("lanes", "carrier")
CJAC1_LEGACY_DIAGNOSTIC_STATUS = (
    "non_governing_pre_pf3_dimension_reweighted_hidden_scratch_probe"
)
CJAC1_SCRATCH_COMPONENT_MAPPING = (
    "PositionAlignedScratch_[B,S,2,scratch_width]_is_the_PF-3_scratch_component;_"
    "the_current_private_API_name_lanes_describes_its_axis_but_does_not_make_it_"
    "the_absent_full_width_bicameral_lanes_component"
)
CJAC1_STATE_DEFINITION = "z=[h;lanes;scratch;carrier_when_integrated]"


@dataclass(frozen=True)
class CurrentVisitFactorTreatment:
    name: str
    treatment: str
    reason: str

    def __post_init__(self) -> None:
        if not self.name or not self.treatment or not self.reason:
            raise ValueError("C-JAC-1 factor treatment fields must be non-empty")


@dataclass(frozen=True)
class CurrentGraphCJac1Receipt:
    authority: str
    authority_sha256: str
    authority_bytes: int
    authority_byte_verified: bool
    measurement_status: str
    root_seed: int
    terminal_visit: int
    visits_materialized_before_primal: int
    state_definition: str
    scratch_component_mapping: str
    current_graph_components: tuple[str, ...]
    absent_ratified_components: tuple[str, ...]
    factor_treatments: tuple[CurrentVisitFactorTreatment, ...]
    loop_receipt: JointStateLoopLipschitzReceipt
    lambda_adapters: float
    lambda_hat_core: float
    transition_output_matches_direct_visit: bool
    convergence_required: bool
    convergence_passed: bool
    production_certificate_authorized: bool
    production_alarm_authorized: bool
    legacy_model_diagnostic_status: str
    training_performed: bool
    checkpoint_used: bool
    sealed_data_touched: bool
    cpu_runtime: str
    torch_version: str
    a100_hours: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.authority != CJAC1_AUTHORITY
            or self.authority_sha256 != CJAC1_AUTHORITY_SHA256
            or self.authority_bytes != CJAC1_AUTHORITY_BYTES
            or not self.authority_byte_verified
        ):
            raise ValueError("C-JAC-1 authority receipt is inconsistent")
        if self.terminal_visit != CJAC1_TERMINAL_VISIT:
            raise ValueError("C-JAC-1 terminal visit is inconsistent")
        if self.root_seed != C2_ROOT_SEED:
            raise ValueError("C-JAC-1 root seed is inconsistent")
        if self.visits_materialized_before_primal != self.terminal_visit - 1:
            raise ValueError("C-JAC-1 primal materialization count is inconsistent")
        if self.state_definition != CJAC1_STATE_DEFINITION:
            raise ValueError("C-JAC-1 joint-state definition is inconsistent")
        if self.current_graph_components != CJAC1_CURRENT_COMPONENTS:
            raise ValueError("C-JAC-1 current joint-state components are inconsistent")
        if self.scratch_component_mapping != CJAC1_SCRATCH_COMPONENT_MAPPING:
            raise ValueError("C-JAC-1 scratch/lane semantic mapping is inconsistent")
        if self.absent_ratified_components != CJAC1_ABSENT_COMPONENTS:
            raise ValueError("C-JAC-1 absent joint-state components are inconsistent")
        core = self.loop_receipt.joint_core_estimate
        if (
            core.current_graph_components != self.current_graph_components
            or core.absent_ratified_components != self.absent_ratified_components
        ):
            raise ValueError("C-JAC-1 nested joint-state layout is inconsistent")
        if (
            self.lambda_adapters != self.loop_receipt.lambda_adapters
            or self.lambda_hat_core != self.loop_receipt.lambda_hat_core
        ):
            raise ValueError("C-JAC-1 two-number line is inconsistent")
        expected_convergence = core.core_estimate.converged
        if not self.convergence_required or self.convergence_passed is not expected_convergence:
            raise ValueError("C-JAC-1 convergence disposition is inconsistent")
        expected_status = (
            "current_graph_terminal_visit_joint_state_measurement_converged"
            if expected_convergence
            else "current_graph_terminal_visit_joint_state_measurement_not_converged"
        )
        if self.measurement_status != expected_status:
            raise ValueError("C-JAC-1 measurement status is inconsistent")
        expected_treatments = (
            CurrentVisitFactorTreatment(
                "anchored_reentry_bridge",
                "included_in_full_joint_visit",
                "fixed-prelude state translation; analytic factor also logged",
            ),
            CurrentVisitFactorTreatment(
                "position_aligned_scratch_update_and_injection",
                "included_in_full_joint_visit",
                "nonlinear coupled h/scratch map measured rather than falsely certified",
            ),
            CurrentVisitFactorTreatment(
                "two_lane_birkhoff_scratch_carrier",
                "certified_live_adapter_factor",
                "analytic two-lane Birkhoff operator norm equals one",
            ),
            CurrentVisitFactorTreatment(
                "loop_embedding",
                "included_in_full_joint_visit",
                "visit-fixed state translation; analytic factor also logged",
            ),
            CurrentVisitFactorTreatment(
                "two_shared_core_blocks",
                "included_in_full_joint_visit",
                "attention/MLP nonlinearities require empirical local estimation",
            ),
        )
        if self.factor_treatments != expected_treatments:
            raise ValueError("C-JAC-1 live factor enumeration is inconsistent")
        if not self.transition_output_matches_direct_visit:
            raise ValueError("C-JAC-1 packed transition did not match the direct visit")
        if (
            self.production_certificate_authorized
            or self.production_alarm_authorized
            or self.loop_receipt.alarm_threshold is not None
            or self.loop_receipt.alarm_fired is not None
        ):
            raise ValueError("C-JAC-1 cannot promote an incomplete production topology")
        if self.legacy_model_diagnostic_status != CJAC1_LEGACY_DIAGNOSTIC_STATUS:
            raise ValueError("C-JAC-1 legacy diagnostic cannot be promoted")
        if self.training_performed or self.checkpoint_used or self.sealed_data_touched:
            raise ValueError("C-JAC-1 CPU measurement cannot consume run-axis artifacts")
        if self.a100_hours != 0.0:
            raise ValueError("C-JAC-1 CPU measurement cannot report GPU spend")
        if self.cpu_runtime != platform.platform() or self.torch_version != torch.__version__:
            raise ValueError("C-JAC-1 runtime provenance is inconsistent")

    def require_converged_measurement(self) -> None:
        if not self.convergence_passed:
            raise RuntimeError("C-JAC-1 empirical power iteration did not converge")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _verify_authority_bytes() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "STRATEGY_PREFLIGHT_AMENDMENT_PF3_20260902.md"
    )
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != CJAC1_AUTHORITY_BYTES or actual_sha256 != CJAC1_AUTHORITY_SHA256:
        raise RuntimeError(
            f"C-JAC-1 authority drift at {path.name}: bytes={len(payload)}, "
            f"sha256={actual_sha256}"
        )


def _current_adapter_certificate(model: torch.nn.Module) -> AdapterCertificateReceipt:
    if model.reentry_bridge is None or model.loop_embedding is None or model.scratch is None:
        raise ValueError("C-JAC-1 current factor inventory is incomplete")
    if model.scratch.carrier is None:
        raise ValueError("C-JAC-1 expects the current narrow scratch carrier")
    factors = (
        certify_state_translation_factor(name="anchored_reentry_bridge"),
        certify_callosum_factor(
            model.scratch.carrier.rho(),
            name="two_lane_birkhoff_scratch_carrier",
        ),
        certify_state_translation_factor(name="loop_embedding"),
    )
    return compose_adapter_certificate(
        factors,
        placeholders=absent_weft1_adapter_placeholders(),
    )


def run_current_graph_cjac1(
    *,
    max_iterations: int = 64,
    minimum_iterations: int = 3,
    convergence_tolerance: float = 1e-3,
    randomized_probe_pairs: int = 4,
) -> CurrentGraphCJac1Receipt:
    """Measure PF-3.3 on the terminal K=8 current-graph recurrent visit."""

    _verify_authority_bytes()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = build_c2_current_toy_model().cpu().eval()
        tokens, _record_ids = c2_fixed_batch(model.config)
        with torch.no_grad():
            hidden = model.token_embedding(tokens)
            if model.front_hadamard is not None:
                hidden = model.front_hadamard(hidden)
            for index, block in enumerate(model.prelude_blocks):
                hidden = block(hidden)
                if index == 0 and model.engram is not None:
                    hidden, _ = model.engram(
                        hidden,
                        tokens,
                        document_ids=None,
                        enabled=True,
                    )
            prelude = hidden.detach()
            if model.scratch is None:
                raise ValueError("C-JAC-1 current graph requires position-aligned scratch")
            scratch = model.scratch.initialize(prelude).detach()
            core_kv_cache = model._project_core_kv(prelude, position_ids=None)
            position_ids = core_kv_cache[0].position_ids
            alpha = model.config.recurrence_scale(C2_RATIFIED_VISITS)
            for step_index in range(CJAC1_TERMINAL_VISIT - 1):
                hidden, next_scratch = model._run_recurrent_visit(
                    hidden,
                    prelude=prelude,
                    lanes=scratch,
                    core_kv_cache=core_kv_cache,
                    step_index=step_index,
                    alpha=alpha,
                    attention_mask=None,
                    position_ids=position_ids,
                    document_ids=None,
                    force_math_attention=True,
                )
                if next_scratch is None:
                    raise RuntimeError("C-JAC-1 scratch disappeared during materialization")
                hidden = hidden.detach()
                scratch = next_scratch.detach()

        terminal_step_index = CJAC1_TERMINAL_VISIT - 1

        def terminal_visit(
            values: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            state_hidden, state_scratch = values
            next_hidden, next_scratch = model._run_recurrent_visit(
                state_hidden,
                prelude=prelude,
                lanes=state_scratch,
                core_kv_cache=core_kv_cache,
                step_index=terminal_step_index,
                alpha=alpha,
                attention_mask=None,
                position_ids=position_ids,
                document_ids=None,
                force_math_attention=True,
            )
            if next_scratch is None:
                raise RuntimeError("C-JAC-1 scratch disappeared from the terminal visit")
            return next_hidden, next_scratch

        joint_core = estimate_empirical_joint_state_core_factor(
            terminal_visit,
            (("h", hidden), ("scratch", scratch)),
            max_iterations=max_iterations,
            minimum_iterations=minimum_iterations,
            convergence_tolerance=convergence_tolerance,
            randomized_probe_pairs=randomized_probe_pairs,
            seed=C2_ROOT_SEED + terminal_step_index,
        )
        adapter_certificate = _current_adapter_certificate(model)
        loop_receipt = make_joint_state_loop_lipschitz_receipt(
            adapter_certificate,
            joint_core,
        )
        direct_output = terminal_visit((hidden, scratch))
        packed_output = joint_core.layout.pack(direct_output)
        unpacked_output = joint_core.layout.unpack(packed_output)
        output_matches = all(
            torch.equal(expected, observed)
            for expected, observed in zip(direct_output, unpacked_output, strict=True)
        )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)

    factor_treatments = (
        CurrentVisitFactorTreatment(
            name="anchored_reentry_bridge",
            treatment="included_in_full_joint_visit",
            reason="fixed-prelude state translation; analytic factor also logged",
        ),
        CurrentVisitFactorTreatment(
            name="position_aligned_scratch_update_and_injection",
            treatment="included_in_full_joint_visit",
            reason="nonlinear coupled h/scratch map measured rather than falsely certified",
        ),
        CurrentVisitFactorTreatment(
            name="two_lane_birkhoff_scratch_carrier",
            treatment="certified_live_adapter_factor",
            reason="analytic two-lane Birkhoff operator norm equals one",
        ),
        CurrentVisitFactorTreatment(
            name="loop_embedding",
            treatment="included_in_full_joint_visit",
            reason="visit-fixed state translation; analytic factor also logged",
        ),
        CurrentVisitFactorTreatment(
            name="two_shared_core_blocks",
            treatment="included_in_full_joint_visit",
            reason="attention/MLP nonlinearities require empirical local estimation",
        ),
    )
    converged = joint_core.core_estimate.converged
    return CurrentGraphCJac1Receipt(
        authority=CJAC1_AUTHORITY,
        authority_sha256=CJAC1_AUTHORITY_SHA256,
        authority_bytes=CJAC1_AUTHORITY_BYTES,
        authority_byte_verified=True,
        measurement_status=(
            "current_graph_terminal_visit_joint_state_measurement_converged"
            if converged
            else "current_graph_terminal_visit_joint_state_measurement_not_converged"
        ),
        root_seed=C2_ROOT_SEED,
        terminal_visit=CJAC1_TERMINAL_VISIT,
        visits_materialized_before_primal=CJAC1_TERMINAL_VISIT - 1,
        state_definition=CJAC1_STATE_DEFINITION,
        scratch_component_mapping=CJAC1_SCRATCH_COMPONENT_MAPPING,
        current_graph_components=CJAC1_CURRENT_COMPONENTS,
        absent_ratified_components=CJAC1_ABSENT_COMPONENTS,
        factor_treatments=factor_treatments,
        loop_receipt=loop_receipt,
        lambda_adapters=loop_receipt.lambda_adapters,
        lambda_hat_core=loop_receipt.lambda_hat_core,
        transition_output_matches_direct_visit=output_matches,
        convergence_required=True,
        convergence_passed=converged,
        production_certificate_authorized=False,
        production_alarm_authorized=False,
        legacy_model_diagnostic_status=CJAC1_LEGACY_DIAGNOSTIC_STATUS,
        training_performed=False,
        checkpoint_used=False,
        sealed_data_touched=False,
        cpu_runtime=platform.platform(),
        torch_version=torch.__version__,
    )


def main() -> None:
    print(json.dumps(run_current_graph_cjac1().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
