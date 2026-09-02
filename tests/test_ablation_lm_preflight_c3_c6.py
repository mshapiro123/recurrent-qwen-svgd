from __future__ import annotations

from dataclasses import replace
from itertools import product

import torch

from models.ablation_lm import AblationLM, AblationLMConfig
from models.ablation_lm.memory import ReadOnlyLatentMemory
from models.ablation_lm.observational_invariance import (
    OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS,
)
from models.ablation_lm.rng import ModuleRNGStream
from training.weft1_gtok_contract import GTOK_PROXY_TOPOLOGY


CPU = torch.device("cpu")
_C6_MATERIALIZED_SWITCHES = (
    "use_recurrence",
    "use_static_kv_core",
    "static_kv_midpoint_refresh",
    "use_front_hadamard_experts",
    "use_reentry_bridge",
    "use_scratch",
    "use_lane_carrier",
    "use_engram",
    "use_long_term_memory",
)
_DEFERRED_C3_C6_CELLS = (
    (
        "c3.cuda_deterministic_replay",
        "requires an authorized GPU run under the dedicated PRE-FLIGHT meter",
    ),
    (
        "c3.nonzero_dropout_forward_isolation",
        "AblationLMConfig rejects p>0 until a generator-aware kernel exists",
    ),
    (
        "c3.stochastic_k_replay",
        "no WEFT-1 STOCH-K sampler or registered sampling stream is materialized",
    ),
    (
        "c6.absent_integration_off_combinations",
        "integrated rotor carrier, per-band callosum, and sidecar are absent",
    ),
)


def _tiny_424_config(**updates: object) -> AblationLMConfig:
    base = AblationLMConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=16,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        recurrent_steps=1,
        max_recurrent_steps=4,
        max_sequence_length=8,
        hadamard_experts=2,
        scratch_width=4,
        engram_hashes_per_order=1,
        engram_table_size=17,
        engram_row_dim=2,
        long_term_memory_slots=2,
        long_term_memory_width=4,
        initialization_seed=20_260_902,
        run_seed=20_260_902,
    )
    return replace(base, **updates)


def _model(config: AblationLMConfig) -> AblationLM:
    memory = None
    if config.use_long_term_memory:
        generator = torch.Generator(device=CPU).manual_seed(7_301)
        memory = ReadOnlyLatentMemory(
            config.d_model,
            keys=torch.randn(
                config.long_term_memory_slots,
                config.long_term_memory_width,
                generator=generator,
            ),
            values=torch.randn(
                config.long_term_memory_slots,
                config.long_term_memory_width,
                generator=generator,
            ),
            provenance_ids=torch.arange(config.long_term_memory_slots),
            layer_scale=config.long_term_memory_layer_scale,
            norm_eps=config.norm_eps,
            initialization_seed=config.initialization_seed,
        )
    return AblationLM(config, long_term_memory=memory)


def _rng_streams(model: AblationLM) -> dict[str, ModuleRNGStream]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ModuleRNGStream)
    }


def _valid_c6_switch_assignment(values: tuple[bool, ...]) -> bool:
    assignment = dict(zip(_C6_MATERIALIZED_SWITCHES, values, strict=True))
    if assignment["use_static_kv_core"] and not assignment["use_recurrence"]:
        return False
    if (
        assignment["static_kv_midpoint_refresh"]
        and not assignment["use_static_kv_core"]
    ):
        return False
    if assignment["use_reentry_bridge"] and not assignment["use_recurrence"]:
        return False
    if assignment["use_lane_carrier"] and not assignment["use_scratch"]:
        return False
    return True


def test_c3_same_seed_cpu_training_step_replays_bit_identically() -> None:
    """CPU-only C3 cell; this does not claim the deferred GPU cell."""

    config = _tiny_424_config(
        use_recurrence=True,
        use_static_kv_core=True,
        use_front_hadamard_experts=True,
        use_reentry_bridge=True,
        use_scratch=True,
        use_lane_carrier=True,
        use_engram=True,
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    torch.manual_seed(99_001)
    ambient_before = torch.random.get_rng_state().clone()
    first = _model(config).train()
    second = _model(config).train()
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=1e-3, foreach=False)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=1e-3, foreach=False)

    def step(
        model: AblationLM,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        optimizer.zero_grad(set_to_none=True)
        output = model(tokens, labels=tokens, return_diagnostics=True)
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()
        return (
            output.loss.detach().clone(),
            output.logits.detach().clone(),
            {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            },
        )

    first_loss, first_logits, first_state = step(first, first_optimizer)
    second_loss, second_logits, second_state = step(second, second_optimizer)

    assert torch.equal(first_loss, second_loss)
    assert torch.equal(first_logits, second_logits)
    assert first_state.keys() == second_state.keys()
    assert all(
        torch.equal(first_state[name], second_state[name]) for name in first_state
    )
    assert torch.equal(torch.random.get_rng_state(), ambient_before)


def test_c3_toggling_one_model_stream_consumption_cannot_shift_any_other() -> None:
    """Exercise every materialized 4/2/4 attention stream as the toggled source.

    Nonzero attention dropout remains structurally forbidden, so this is the
    bound O-9 stream-isolation cell rather than a deferred p>0 forward claim.
    """

    config = _tiny_424_config()
    expected_stream_count = (
        config.n_prelude_layers + config.n_core_blocks + config.n_coda_layers
    )
    ambient_before = torch.random.get_rng_state().clone()

    for toggled_name in _rng_streams(_model(config)):
        treatment = _rng_streams(_model(config))
        control = _rng_streams(_model(config))
        assert treatment.keys() == control.keys()
        assert len(treatment) == expected_stream_count == 10

        torch.rand(
            23,
            generator=treatment[toggled_name].next_generator(CPU, coordinate=0),
        )
        for other_name in treatment.keys() - {toggled_name}:
            treatment_draw = torch.rand(
                23,
                generator=treatment[other_name].next_generator(CPU, coordinate=0),
            )
            control_draw = torch.rand(
                23,
                generator=control[other_name].next_generator(CPU, coordinate=0),
            )
            assert torch.equal(treatment_draw, control_draw), (
                toggled_name,
                other_name,
            )

        assert treatment[toggled_name].draw_indices == (1, 0, 0, 0)
        assert control[toggled_name].draw_indices == (0, 0, 0, 0)

    assert torch.equal(torch.random.get_rng_state(), ambient_before)


def test_c6_every_valid_materialized_off_combination_executes_ten_blocks_at_k1(
) -> None:
    """Count calls, not merely configured ModuleList lengths, on the 4/2/4 toy.

    This enumerates the structural switches the current graph materializes.
    The integrated rotor, per-band callosum, and sidecar remain absent and are
    explicitly outside this CPU cell rather than being silently counted green.
    """

    assert (
        GTOK_PROXY_TOPOLOGY.n_prelude_layers,
        GTOK_PROXY_TOPOLOGY.n_core_blocks,
        GTOK_PROXY_TOPOLOGY.n_coda_layers,
        GTOK_PROXY_TOPOLOGY.executing_block_count,
    ) == (4, 2, 4, 10)
    assert OBS_INV_EXCLUDED_ABSENT_INTEGRATIONS == (
        "integrated_rotor_carrier",
        "per_band_callosum",
        "sidecar",
    )

    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    executed_combinations = 0
    for values in product((False, True), repeat=len(_C6_MATERIALIZED_SWITCHES)):
        if not _valid_c6_switch_assignment(values):
            continue
        switches = dict(zip(_C6_MATERIALIZED_SWITCHES, values, strict=True))
        config = _tiny_424_config(**switches)
        model = _model(config).eval()
        calls = {"prelude": 0, "core": 0, "coda": 0}
        handles = []

        def count(stage: str):
            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[object, ...],
                _output: object,
            ) -> None:
                calls[stage] += 1

            return hook

        for stage, blocks in (
            ("prelude", model.prelude_blocks),
            ("core", model.core_blocks),
            ("coda", model.coda_blocks),
        ):
            handles.extend(
                block.register_forward_hook(count(stage)) for block in blocks
            )
        try:
            with torch.no_grad():
                output = model(tokens, recurrent_steps=1, return_diagnostics=True)
        finally:
            for handle in handles:
                handle.remove()

        assert calls == {"prelude": 4, "core": 2, "coda": 4}, switches
        assert sum(calls.values()) == 10
        assert output.diagnostics["executed_core_visits"] == 1
        assert output.diagnostics["executed_core_block_passes"] == 2
        executed_combinations += 1

    assert executed_combinations == 168


def test_c3_c6_deferred_cells_are_typed_nonpasses_not_pytest_skips() -> None:
    """Keep unavailable cells inspectable without weakening the full-suite gate."""

    assert tuple(name for name, _ in _DEFERRED_C3_C6_CELLS) == (
        "c3.cuda_deterministic_replay",
        "c3.nonzero_dropout_forward_isolation",
        "c3.stochastic_k_replay",
        "c6.absent_integration_off_combinations",
    )
    assert all(reason.strip() for _, reason in _DEFERRED_C3_C6_CELLS)
