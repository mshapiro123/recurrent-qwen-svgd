import torch
from torch import nn

from models.lora import LoRALinear


def test_lora_linear_zero_init_preserves_base_output():
    torch.manual_seed(0)
    base = nn.Linear(5, 3)
    wrapped = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(7, 5)
    assert torch.allclose(wrapped(x), base(x))
