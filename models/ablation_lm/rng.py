"""Namespaced, checkpointable random-number streams for paired ablations.

The stream identity is derived with SHA-256 rather than Python's process-local
``hash``.  Logical coordinates (for example, recurrent visit indices) own
independent counters, so extra visits in a larger-K arm cannot shift the masks
for coordinates shared with a smaller-K arm.  Each draw creates a fresh
generator on the requested device; checkpoints contain only ordinary tensors,
never a device-bound ``torch.Generator`` or Python extra state.

Activation-checkpoint recomputation must explicitly snapshot and restore these
module counters around replay.  WEFT-1 does not yet enable activation
checkpointing; using an ordinary checkpoint wrapper would otherwise request a
new mask during the backward recomputation instead of replaying the old one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch
from torch import nn


_SOURCE_KEY_PATTERN = re.compile(
    r"[a-z][a-z0-9_]*(?:\.(?:[a-z][a-z0-9_]*|[0-9]+))*"
)
_SEED_DOMAIN = b"WEFT-1/module-rng/seed/v1\x00"
_DRAW_DOMAIN = b"WEFT-1/module-rng/draw/v2\x00"
_CHECKPOINT_IDENTITY_DOMAIN = b"WEFT-1/module-rng/checkpoint-identity/v1\x00"
_MAX_COUNTER = torch.iinfo(torch.int64).max
_FactoryResult = TypeVar("_FactoryResult")


def _validate_base_seed(base_seed: int) -> None:
    if type(base_seed) is not int:
        raise TypeError("base_seed must be an exact integer")


def _validate_source_key(source_key: str) -> None:
    if type(source_key) is not str:
        raise TypeError("source_key must be an exact string")
    if _SOURCE_KEY_PATTERN.fullmatch(source_key) is None:
        raise ValueError(
            "source_key must be canonical lowercase dotted module coordinates; "
            "segments may be identifiers or non-negative decimal indices"
        )


def _validate_replica(replica: int) -> None:
    if type(replica) is not int:
        raise TypeError("replica must be an exact integer")
    if replica < 0:
        raise ValueError("replica must be non-negative")


def _encode_field(value: int | str) -> bytes:
    raw = str(value).encode("utf-8")
    return len(raw).to_bytes(8, byteorder="big", signed=False) + raw


def derive_module_seed(base_seed: int, source_key: str, replica: int = 0) -> int:
    """Return the stable 64-bit seed for one canonical random source.

    The length-prefixed encoding prevents boundary ambiguities such as a key
    suffix being mistaken for part of the replica.  The exact signed decimal
    spelling of ``base_seed`` participates in the hash; no Python hash or
    process-global salt is involved.
    """

    _validate_base_seed(base_seed)
    _validate_source_key(source_key)
    _validate_replica(replica)
    payload = (
        _SEED_DOMAIN
        + _encode_field(base_seed)
        + _encode_field(source_key)
        + _encode_field(replica)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _derive_draw_seed(module_seed: int, coordinate: int, draw_index: int) -> int:
    if type(coordinate) is not int:
        raise TypeError("coordinate must be an exact integer")
    if coordinate < 0 or coordinate > _MAX_COUNTER:
        raise ValueError("coordinate must fit a non-negative int64 value")
    if type(draw_index) is not int:
        raise TypeError("draw_index must be an exact integer")
    if draw_index < 0 or draw_index > _MAX_COUNTER:
        raise ValueError("draw_index must fit a non-negative int64 counter")
    payload = (
        _DRAW_DOMAIN
        + _encode_field(module_seed)
        + _encode_field(coordinate)
        + _encode_field(draw_index)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _checkpoint_identity(base_seed: int, source_key: str) -> torch.Tensor:
    """Return a rank-agnostic tensor identity for strict checkpoint loading.

    ``replica`` is deliberately absent.  DDP ranks use distinct random streams
    but share one logical draw schedule, so a rank-zero checkpoint may restore
    the same counters onto every rank before each rank resumes its own stream.
    """

    payload = (
        _CHECKPOINT_IDENTITY_DOMAIN
        + _encode_field(base_seed)
        + _encode_field(source_key)
    )
    return torch.tensor(list(hashlib.sha256(payload).digest()), dtype=torch.uint8)


def _validate_device(device: torch.device) -> torch.device:
    if type(device) is not torch.device:
        raise TypeError("device must be an exact torch.device")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("module RNG streams support only CPU and CUDA generators")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("a CPU generator device must not carry an index")
    return device


@contextmanager
def isolated_module_rng(
    base_seed: int,
    source_key: str,
    replica: int = 0,
) -> Iterator[None]:
    """Isolate CPU RNG effects while constructing or resetting one module.

    ``devices=[]`` deliberately snapshots no accelerator generators.  Seeding
    ``torch.random.default_generator`` then affects only the forked CPU stream,
    which is restored even if the wrapped constructor or reset raises.
    """

    seed = derive_module_seed(base_seed, source_key, replica)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.default_generator.manual_seed(seed)
        yield


def construct_with_isolated_rng(
    factory: Callable[[], _FactoryResult],
    *,
    base_seed: int,
    source_key: str,
    replica: int = 0,
) -> _FactoryResult:
    """Call a zero-argument constructor without advancing ambient CPU RNG."""

    if not callable(factory):
        raise TypeError("factory must be callable")
    with isolated_module_rng(base_seed, source_key, replica):
        return factory()


class ModuleRNGStream(nn.Module):
    """Counter-derived coordinate streams whose next draws survive checkpoints.

    The custom state-dict entries are tensors so a complete model state remains
    compatible with tensor-only formats such as safetensors.  ``substreams`` is
    fixed at construction and bounds the allowed logical coordinates.
    """

    _COUNTERS_KEY = "rng_draw_counts"
    _IDENTITY_KEY = "rng_identity"

    def __init__(
        self,
        base_seed: int,
        source_key: str,
        replica: int = 0,
        *,
        substreams: int = 1,
    ) -> None:
        super().__init__()
        _validate_base_seed(base_seed)
        _validate_source_key(source_key)
        _validate_replica(replica)
        if type(substreams) is not int:
            raise TypeError("substreams must be an exact integer")
        if substreams < 1:
            raise ValueError("substreams must be positive")
        self.base_seed = base_seed
        self.source_key = source_key
        self.replica = replica
        self.substreams = substreams
        # Python counters avoid a device synchronization at every stochastic
        # site.  _save_to_state_dict materializes them as CPU int64 tensors.
        self._draw_indices = [0] * substreams

    @property
    def draw_index(self) -> int:
        """Return the next draw index for the default logical coordinate."""

        return self._draw_indices[0]

    @property
    def draw_indices(self) -> tuple[int, ...]:
        """Return the next draw index for every logical coordinate."""

        return tuple(self._draw_indices)

    def next_generator(
        self,
        device: torch.device,
        *,
        coordinate: int = 0,
    ) -> torch.Generator:
        """Return a generator for one coordinate and advance only its counter."""

        requested_device = _validate_device(device)
        if type(coordinate) is not int:
            raise TypeError("coordinate must be an exact integer")
        if coordinate < 0 or coordinate >= self.substreams:
            raise ValueError("coordinate lies outside the configured substreams")
        draw_index = self._draw_indices[coordinate]
        if draw_index == _MAX_COUNTER:
            raise OverflowError("module RNG draw counter is exhausted")
        module_seed = derive_module_seed(
            self.base_seed,
            self.source_key,
            self.replica,
        )
        generator = torch.Generator(device=requested_device)
        generator.manual_seed(_derive_draw_seed(module_seed, coordinate, draw_index))
        self._draw_indices[coordinate] = draw_index + 1
        return generator

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + self._IDENTITY_KEY] = _checkpoint_identity(
            self.base_seed,
            self.source_key,
        )
        destination[prefix + self._COUNTERS_KEY] = torch.tensor(
            self._draw_indices,
            dtype=torch.int64,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        identity_key = prefix + self._IDENTITY_KEY
        counters_key = prefix + self._COUNTERS_KEY
        expected_keys = {identity_key, counters_key}
        absent = expected_keys.difference(state_dict)
        if strict:
            missing_keys.extend(sorted(absent))
            unexpected_keys.extend(
                key
                for key in state_dict
                if key.startswith(prefix)
                and "." not in key[len(prefix) :]
                and key not in expected_keys
            )
        if absent:
            return
        identity = state_dict[identity_key]
        counters = state_dict[counters_key]
        expected_identity = _checkpoint_identity(self.base_seed, self.source_key)
        if (
            type(identity) is not torch.Tensor
            or identity.dtype != torch.uint8
            or tuple(identity.shape) != (32,)
            or not torch.equal(identity.detach().cpu(), expected_identity)
        ):
            error_msgs.append(
                f'{identity_key} does not match the destination stream identity'
            )
            return
        if type(counters) is not torch.Tensor:
            error_msgs.append(f'{counters_key} must be an exact tensor')
            return
        if counters.dtype != torch.int64 or tuple(counters.shape) != (self.substreams,):
            error_msgs.append(
                f'{counters_key} must be int64 with shape ({self.substreams},)'
            )
            return
        restored = counters.detach().cpu().tolist()
        if any(type(value) is not int or value < 0 or value > _MAX_COUNTER for value in restored):
            error_msgs.append(f'{counters_key} contains an invalid draw count')
            return
        self._draw_indices = restored
