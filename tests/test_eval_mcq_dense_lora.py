from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from eval import eval_mcq


class FakeBaseModel:
    def __init__(self, num_layers: int = 32) -> None:
        self.model = SimpleNamespace(layers=[object() for _ in range(num_layers)])
        self.device = None
        self.eval_called = False

    def to(self, device: str):
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True


def _args(**overrides):
    values = {
        "model_name": "fake-qwen",
        "dtype": "float32",
        "attn_implementation": "default",
        "device": "cpu",
        "checkpoint": None,
        "base_lora_layer_range": "6,18",
        "lora_rank": 8,
        "lora_alpha": 16,
        "adapter_dtype": "float32",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parse_base_lora_layer_range_accepts_auto_and_explicit() -> None:
    assert eval_mcq.parse_base_lora_layer_range("auto", 32) == (0, 32)
    assert eval_mcq.parse_base_lora_layer_range("all", 32) == (0, 32)
    assert eval_mcq.parse_base_lora_layer_range("6,18", 32) == (6, 18)


def test_parse_base_lora_layer_range_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid base LoRA layer range"):
        eval_mcq.parse_base_lora_layer_range("18,6", 32)


def test_base_mode_checkpoint_loads_dense_lora(monkeypatch) -> None:
    model = FakeBaseModel()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        eval_mcq.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_name, **kwargs: model,
    )

    def fake_apply_lora(model_arg, *, start_layer, end_layer, rank, alpha, dropout, adapter_dtype):
        calls["apply_lora"] = {
            "model": model_arg,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "adapter_dtype": adapter_dtype,
        }
        return 84

    def fake_load_checkpoint(model_arg, checkpoint):
        calls["load_checkpoint"] = {"model": model_arg, "checkpoint": checkpoint}
        return {"loaded_keys": ["a", "b"], "skipped": []}

    monkeypatch.setattr(eval_mcq, "apply_lora_to_qwen_layers", fake_apply_lora)
    monkeypatch.setattr(eval_mcq, "load_trainable_checkpoint", fake_load_checkpoint)

    loaded = eval_mcq.load_base_model(
        _args(checkpoint="dense.pt", base_lora_layer_range="4,12"),
        load_dense_lora_checkpoint=True,
    )

    assert loaded is model
    assert model.device == "cpu"
    assert model.eval_called is True
    assert calls["apply_lora"]["start_layer"] == 4
    assert calls["apply_lora"]["end_layer"] == 12
    assert calls["load_checkpoint"]["checkpoint"] == "dense.pt"


def test_base_model_does_not_load_dense_lora_when_disabled(monkeypatch) -> None:
    model = FakeBaseModel()
    monkeypatch.setattr(
        eval_mcq.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_name, **kwargs: model,
    )
    monkeypatch.setattr(
        eval_mcq,
        "apply_lora_to_qwen_layers",
        lambda *args, **kwargs: pytest.fail("dense LoRA should not be applied"),
    )

    loaded = eval_mcq.load_base_model(_args(checkpoint="recurrent.pt"), load_dense_lora_checkpoint=False)

    assert loaded is model
    assert model.eval_called is True
