"""FP32 Walsh-Hadamard primitives with an explicitly verified sequency order."""

from __future__ import annotations

import math

import torch


def _validate_width(width: int) -> int:
    if type(width) is not int or width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    return width


def wht(values: torch.Tensor) -> torch.Tensor:
    """Apply the unnormalized Sylvester WHT in FP32 on the final axis."""

    if values.ndim < 1 or not values.is_floating_point():
        raise ValueError("WHT input must be a floating tensor with a final axis")
    width = _validate_width(int(values.shape[-1]))
    original_shape = values.shape
    result = values.reshape(-1, width).float()
    block = 1
    while block < width:
        shaped = result.reshape(-1, width // (2 * block), 2, block)
        left = shaped[:, :, 0, :]
        right = shaped[:, :, 1, :]
        result = torch.stack((left + right, left - right), dim=2).reshape(-1, width)
        block *= 2
    return result.reshape(original_shape)


def orthonormal_wht(values: torch.Tensor) -> torch.Tensor:
    """Apply ``W/sqrt(d)`` while retaining FP32 transform arithmetic."""

    width = _validate_width(int(values.shape[-1]) if values.ndim else 0)
    return wht(values) / math.sqrt(width)


def sequency_permutation(width: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Map sequency index to Sylvester row: bit-reversal of the Gray code."""

    width = _validate_width(width)
    bits = width.bit_length() - 1
    rows: list[int] = []
    for sequency in range(width):
        gray = sequency ^ (sequency >> 1)
        reversed_bits = 0
        for bit in range(bits):
            reversed_bits |= ((gray >> bit) & 1) << (bits - bit - 1)
        rows.append(reversed_bits)
    return torch.tensor(rows, dtype=torch.long, device=device)
