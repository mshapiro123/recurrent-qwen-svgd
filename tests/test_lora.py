import pytest
import torch
from torch import nn

from models.lora import LoRALinear, apply_lora_to_qwen_layers


def test_lora_linear_zero_init_preserves_base_output():
    torch.manual_seed(0)
    base = nn.Linear(5, 3)
    wrapped = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(7, 5)
    assert torch.allclose(wrapped(x), base(x))


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.untouched = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.k_proj(x) + self.untouched(x)


class TinyDenseQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(), TinyBlock(), TinyBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_apply_lora_to_qwen_layers_wraps_requested_dense_layer_range() -> None:
    torch.manual_seed(0)
    model = TinyDenseQwen()
    x = torch.randn(2, 4)
    before = model(x)

    replaced = apply_lora_to_qwen_layers(model, start_layer=1, end_layer=3, rank=2, alpha=4)

    assert replaced == 4
    assert isinstance(model.model.layers[0].q_proj, nn.Linear)
    assert isinstance(model.model.layers[1].q_proj, LoRALinear)
    assert isinstance(model.model.layers[2].k_proj, LoRALinear)
    assert torch.allclose(model(x), before)


def test_apply_lora_to_qwen_layers_rejects_invalid_range() -> None:
    model = TinyDenseQwen()

    with pytest.raises(ValueError):
        apply_lora_to_qwen_layers(model, start_layer=2, end_layer=4)
