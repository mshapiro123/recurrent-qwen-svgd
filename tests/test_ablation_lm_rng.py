from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from models.ablation_lm.config import AblationLMConfig
from models.ablation_lm.memory import ReadOnlyLatentMemory
from models.ablation_lm.model import AblationLM
from models.ablation_lm.rng import (
    ModuleRNGStream,
    construct_with_isolated_rng,
    derive_module_seed,
    isolated_module_rng,
)


CPU = torch.device("cpu")


def _tiny_model_config(**updates: object) -> AblationLMConfig:
    base = AblationLMConfig(
        vocab_size=32,
        d_model=8,
        n_heads=2,
        n_kv_heads=1,
        d_ff=16,
        n_prelude_layers=1,
        n_core_blocks=1,
        n_coda_layers=1,
        use_recurrence=True,
        recurrent_steps=1,
        max_recurrent_steps=4,
        max_sequence_length=8,
        use_scratch=True,
        scratch_width=4,
        hadamard_experts=2,
    )
    return replace(base, **updates)


def _draw(stream: ModuleRNGStream) -> torch.Tensor:
    generator = stream.next_generator(CPU)
    assert generator.device == CPU
    return torch.randn(17, generator=generator)


def test_module_seed_and_draw_sequence_are_reproducible() -> None:
    first = ModuleRNGStream(20260826, "model.core.0.attention.dropout", replica=1)
    second = ModuleRNGStream(20260826, "model.core.0.attention.dropout", replica=1)

    assert derive_module_seed(20260826, "model.core.0.attention.dropout", 1) == (
        6870707043680624902
    )
    torch.testing.assert_close(_draw(first), _draw(second), rtol=0, atol=0)
    torch.testing.assert_close(_draw(first), _draw(second), rtol=0, atol=0)
    assert first.draw_index == second.draw_index == 2


def test_sources_and_replicas_have_distinct_streams() -> None:
    source_a = ModuleRNGStream(7, "model.engram.dropout", replica=0)
    source_b = ModuleRNGStream(7, "model.sidecar.dropout", replica=0)
    replica_b = ModuleRNGStream(7, "model.engram.dropout", replica=1)

    assert derive_module_seed(7, source_a.source_key, 0) != derive_module_seed(
        7, source_b.source_key, 0
    )
    assert derive_module_seed(7, source_a.source_key, 0) != derive_module_seed(
        7, replica_b.source_key, 1
    )
    assert not torch.equal(_draw(source_a), _draw(source_b))
    assert not torch.equal(
        _draw(ModuleRNGStream(7, "model.engram.dropout", replica=0)),
        _draw(replica_b),
    )


def test_extra_draw_from_one_source_cannot_perturb_another() -> None:
    stream_a = ModuleRNGStream(11, "model.hadamard.router")
    stream_b = ModuleRNGStream(11, "model.callosum.dropout")
    control_b = ModuleRNGStream(11, "model.callosum.dropout")

    _draw(stream_a)
    _draw(stream_a)

    torch.testing.assert_close(_draw(stream_b), _draw(control_b), rtol=0, atol=0)
    assert stream_a.draw_index == 2
    assert stream_b.draw_index == control_b.draw_index == 1


def test_state_dict_roundtrip_reproduces_the_next_random_tensor() -> None:
    source = ModuleRNGStream(23, "model.memory.address_noise", replica=3)
    _draw(source)
    checkpoint = copy.deepcopy(source.state_dict())
    assert set(checkpoint) == {"rng_identity", "rng_draw_counts"}
    assert all(type(value) is torch.Tensor for value in checkpoint.values())
    expected_next = _draw(source)

    restored = ModuleRNGStream(23, "model.memory.address_noise", replica=3)
    restored.load_state_dict(checkpoint)

    assert restored.draw_index == 1
    torch.testing.assert_close(_draw(restored), expected_next, rtol=0, atol=0)
    assert restored.draw_index == 2


def test_state_dict_rejects_wrong_stream_identity_and_counter_schema() -> None:
    source = ModuleRNGStream(23, "model.memory.address_noise", replica=3)
    _draw(source)
    checkpoint = copy.deepcopy(source.state_dict())

    wrong_source = ModuleRNGStream(23, "model.memory.router_noise", replica=3)
    with pytest.raises(RuntimeError, match="identity does not match"):
        wrong_source.load_state_dict(checkpoint)
    assert wrong_source.draw_index == 0

    invalid_counter = copy.deepcopy(checkpoint)
    invalid_counter["rng_draw_counts"] = torch.tensor([1.0])
    restored = ModuleRNGStream(23, "model.memory.address_noise", replica=3)
    with pytest.raises(RuntimeError, match="must be int64"):
        restored.load_state_dict(invalid_counter)
    assert restored.draw_index == 0

    missing = copy.deepcopy(checkpoint)
    del missing["rng_identity"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(missing)
    unexpected = copy.deepcopy(checkpoint)
    unexpected["rng_undeclared"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(unexpected)


def test_rng_checkpoint_roundtrips_through_tensor_only_safetensors() -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    source = ModuleRNGStream(27, "model.scratch.lane_noise", substreams=4)
    source.next_generator(CPU, coordinate=0)
    source.next_generator(CPU, coordinate=3)

    serialized = safetensors.save(source.state_dict())
    restored_state = safetensors.load(serialized)
    restored = ModuleRNGStream(27, "model.scratch.lane_noise", substreams=4)
    restored.load_state_dict(restored_state)

    assert restored.draw_indices == (1, 0, 0, 1)


def test_rank_zero_counter_checkpoint_restores_to_distinct_ddp_replica_stream() -> None:
    rank_zero = ModuleRNGStream(29, "model.core.0.attention.dropout", replica=0)
    _draw(rank_zero)
    checkpoint = copy.deepcopy(rank_zero.state_dict())

    rank_one = ModuleRNGStream(29, "model.core.0.attention.dropout", replica=1)
    rank_one.load_state_dict(checkpoint)

    assert rank_one.draw_index == rank_zero.draw_index == 1
    assert not torch.equal(_draw(rank_zero), _draw(rank_one))
    assert rank_one.draw_index == rank_zero.draw_index == 2


def test_isolated_construction_and_reset_leave_ambient_cpu_rng_unchanged() -> None:
    torch.manual_seed(101)
    ambient_before = torch.random.get_rng_state().clone()

    first = construct_with_isolated_rng(
        lambda: nn.Linear(8, 5),
        base_seed=31,
        source_key="model.prelude.0.projection",
    )
    torch.testing.assert_close(torch.random.get_rng_state(), ambient_before, rtol=0, atol=0)

    second = nn.Linear(8, 5)
    ambient_before_reset = torch.random.get_rng_state().clone()
    with isolated_module_rng(31, "model.prelude.0.projection"):
        second.reset_parameters()

    torch.testing.assert_close(
        torch.random.get_rng_state(), ambient_before_reset, rtol=0, atol=0
    )
    torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    torch.testing.assert_close(first.bias, second.bias, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((True, "model.dropout", 0), TypeError),
        ((1, "Model.dropout", 0), ValueError),
        ((1, "model..dropout", 0), ValueError),
        ((1, "model.dropout", True), TypeError),
        ((1, "model.dropout", -1), ValueError),
    ],
)
def test_stream_identity_validation_is_fail_closed(
    args: tuple[object, object, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ModuleRNGStream(*args)  # type: ignore[arg-type]


def test_generator_device_requires_an_exact_supported_torch_device() -> None:
    stream = ModuleRNGStream(1, "model.dropout")

    with pytest.raises(TypeError, match="torch.device"):
        stream.next_generator("cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CPU and CUDA"):
        stream.next_generator(torch.device("meta"))
    with pytest.raises(TypeError, match="coordinate"):
        stream.next_generator(CPU, coordinate=True)
    with pytest.raises(ValueError, match="configured substreams"):
        stream.next_generator(CPU, coordinate=1)


def test_optional_arm_construction_preserves_ambient_rng_and_shared_weights() -> None:
    config = _tiny_model_config(use_front_hadamard_experts=False)
    torch.manual_seed(404)
    ambient = torch.random.get_rng_state().clone()

    control = AblationLM(config)
    torch.testing.assert_close(torch.random.get_rng_state(), ambient, rtol=0, atol=0)
    challenger = AblationLM(replace(config, use_front_hadamard_experts=True))
    torch.testing.assert_close(torch.random.get_rng_state(), ambient, rtol=0, atol=0)

    challenger_parameters = dict(challenger.named_parameters())
    for name, parameter in control.named_parameters():
        assert name in challenger_parameters
        torch.testing.assert_close(
            parameter,
            challenger_parameters[name],
            rtol=0,
            atol=0,
        )
    stream_keys = [
        module.source_key
        for module in challenger.modules()
        if isinstance(module, ModuleRNGStream)
    ]
    assert len(stream_keys) == len(set(stream_keys))


def test_recurrent_visit_coordinates_preserve_common_draws_across_k_arms() -> None:
    short = ModuleRNGStream(
        41,
        "model.core.0.attention.dropout",
        substreams=4,
    )
    deep = ModuleRNGStream(
        41,
        "model.core.0.attention.dropout",
        substreams=4,
    )

    for _minibatch in range(2):
        short_visit_zero = torch.rand(
            31,
            generator=short.next_generator(CPU, coordinate=0),
        )
        deep_visit_zero = torch.rand(
            31,
            generator=deep.next_generator(CPU, coordinate=0),
        )
        torch.testing.assert_close(short_visit_zero, deep_visit_zero, rtol=0, atol=0)
        for visit in (1, 2, 3):
            torch.rand(31, generator=deep.next_generator(CPU, coordinate=visit))

    assert short.draw_indices == (2, 0, 0, 0)
    assert deep.draw_indices == (2, 2, 2, 2)


def test_dropout_free_model_exposes_coordinate_stream_receipts_without_draws() -> None:
    config = _tiny_model_config(use_scratch=False)
    short = AblationLM(config).train()
    deep = AblationLM(config).train()
    tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    torch.manual_seed(505)
    ambient = torch.random.get_rng_state().clone()
    short_output = short(tokens, recurrent_steps=1, return_diagnostics=True)
    deep_output = deep(tokens, recurrent_steps=4, return_diagnostics=True)
    torch.testing.assert_close(torch.random.get_rng_state(), ambient, rtol=0, atol=0)

    assert short.core_blocks[0].attention.dropout_rng.draw_indices == (0, 0, 0, 0)
    assert deep.core_blocks[0].attention.dropout_rng.draw_indices == (0, 0, 0, 0)
    assert short.coda_blocks[0].attention.dropout_rng.draw_indices == (0, 0, 0, 0)
    assert deep.coda_blocks[0].attention.dropout_rng.draw_indices == (0, 0, 0, 0)
    assert short_output.diagnostics["rng_run_seed"] == config.run_seed
    assert short_output.diagnostics["rng_replica"] == config.rng_replica
    short_receipt = dict(short_output.diagnostics["rng_stream_draw_indices_by_name"])
    deep_receipt = dict(deep_output.diagnostics["rng_stream_draw_indices_by_name"])
    assert short_receipt["core_blocks.0.attention.dropout_rng"] == (0, 0, 0, 0)
    assert deep_receipt["core_blocks.0.attention.dropout_rng"] == (0, 0, 0, 0)
    assert short_receipt["coda_blocks.0.attention.dropout_rng"] == (0, 0, 0, 0)
    assert deep_receipt["coda_blocks.0.attention.dropout_rng"] == (0, 0, 0, 0)


def test_long_term_memory_initialization_is_namespaced_and_rng_inert() -> None:
    keys = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    values = keys.flip(0)
    provenance_ids = torch.arange(4)
    torch.manual_seed(606)
    ambient = torch.random.get_rng_state().clone()

    first = ReadOnlyLatentMemory(
        8,
        keys=keys,
        values=values,
        provenance_ids=provenance_ids,
        layer_scale=1e-3,
        norm_eps=1e-5,
        initialization_seed=77,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), ambient, rtol=0, atol=0)
    second = ReadOnlyLatentMemory(
        8,
        keys=keys,
        values=values,
        provenance_ids=provenance_ids,
        layer_scale=1e-3,
        norm_eps=1e-5,
        initialization_seed=77,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), ambient, rtol=0, atol=0)
    torch.testing.assert_close(first.query.weight, second.query.weight, rtol=0, atol=0)
    torch.testing.assert_close(first.output.weight, second.output.weight, rtol=0, atol=0)
