"""Build-only machinery for the WEFT-1 Jacobian exponent panel.

This module implements the registered Tier-1 instrument without selecting a
dataset or running a panel.  The registered design is four recurrent depths,
paired within each example, with four example-owned probe directions.  In
particular, probe directions are *never* seeded by depth (P-5).

The derivatives here are fixed-routing-branch derivatives.  Callers should
return routing and hard-gate decisions as ``aux`` data and set ``has_aux=True``
so that the probe functions can assert bit-identical decisions across all
directions at one example/depth.  A routing flip under a perturbation is not
part of the reported Jacobian.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np
import torch
from torch import nn

from models.ablation_lm.rng import ModuleRNGStream


REGISTERED_DEPTHS = (1, 2, 4, 8)
MAIN_PANEL_EXAMPLES = 512
MAIN_PANEL_PROBES = 4
NORM_PANEL_EXAMPLES = 64
NORM_POWER_ITERATIONS = 10
RANK_PANEL_EXAMPLES = 64
RANK_PANEL_PROBES = 8
PILOT_EXAMPLES = 32
FULL_BOOTSTRAP_REPLICATES = 10_000
INSTRUMENT_TIER = 1
CONDITIONING_FLOOR = 0.05
DIRECTION_CLASSES = (
    "sidecar_write_directions",
    "staged_state_residual_directions",
    "isotropic_noise",
)

_PROBE_SEED_DOMAIN = b"WEFT-1/jacobian-panel/example-probe/v1\x00"
_MAX_SEED = 2**63 - 1
_Result = TypeVar("_Result")
TierName = Literal["main", "norm", "rank"]


def _validate_exact_int(value: int, *, name: str, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_seed(value: int, *, name: str) -> None:
    _validate_exact_int(value, name=name)
    if value > _MAX_SEED:
        raise ValueError(f"{name} must fit a non-negative signed int64 value")


def _encode_seed_field(value: int | str) -> bytes:
    raw = str(value).encode("utf-8")
    return len(raw).to_bytes(8, byteorder="big", signed=False) + raw


def derive_example_probe_seed(panel_seed: int, example_id: str) -> int:
    """Derive one stable probe seed from the panel seed and example identity.

    Depth is deliberately absent from the hash domain and function signature.
    This makes the P-5 binding structural rather than a caller convention.
    """

    _validate_seed(panel_seed, name="panel_seed")
    if type(example_id) is not str:
        raise TypeError("example_id must be an exact string")
    if not example_id:
        raise ValueError("example_id must not be empty")
    payload = (
        _PROBE_SEED_DOMAIN
        + _encode_seed_field(panel_seed)
        + _encode_seed_field(example_id)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % _MAX_SEED


def _autocast_is_enabled() -> bool:
    if torch.is_autocast_enabled():
        return True
    # ``torch.is_autocast_enabled()`` without a device argument does not see
    # CPU autocast.  Check both supported probe devices explicitly.
    for device_type in ("cpu", "cuda"):
        try:
            if torch.is_autocast_enabled(device_type):
                return True
        except (RuntimeError, TypeError):
            continue
    return False


def _fp32_primal(primal: torch.Tensor) -> torch.Tensor:
    if _autocast_is_enabled():
        raise RuntimeError("Jacobian probes must run outside autocast")
    if type(primal) is not torch.Tensor:
        raise TypeError("Jacobian primal must be an exact tensor")
    if not primal.is_floating_point():
        raise TypeError("Jacobian primal must be floating-point")
    if primal.numel() == 0:
        raise ValueError("Jacobian primal must not be empty")
    if not bool(torch.isfinite(primal).all()):
        raise ValueError("Jacobian primal must be finite")
    return primal.to(dtype=torch.float32)


def _validate_transition_result(
    output: torch.Tensor,
    tangent: torch.Tensor,
    primal: torch.Tensor,
) -> None:
    if type(output) is not torch.Tensor or type(tangent) is not torch.Tensor:
        raise TypeError("transition output and JVP must be exact tensors")
    if output.shape != primal.shape or tangent.shape != output.shape:
        raise ValueError("transition, primal, and JVP must have identical shapes")
    if output.device != primal.device or tangent.device != primal.device:
        raise ValueError("transition output and JVP must remain on the primal device")
    if output.dtype is not torch.float32 or tangent.dtype is not torch.float32:
        raise TypeError("Jacobian transition and JVP must remain FP32")
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(tangent).all()):
        raise ValueError("Jacobian transition and JVP must be finite")


def draw_example_probe_directions(
    primal: torch.Tensor,
    *,
    n_probe: int,
    example_probe_seed: int,
) -> torch.Tensor:
    """Draw a unit FP32 direction bank owned by one example.

    The returned shape is ``[n_probe, *primal.shape]``.  Call this once for an
    example and pass the result to every registered depth.
    """

    primal_fp32 = _fp32_primal(primal)
    _validate_exact_int(n_probe, name="n_probe", minimum=1)
    _validate_seed(example_probe_seed, name="example_probe_seed")
    generator = torch.Generator(device=primal_fp32.device)
    generator.manual_seed(example_probe_seed)
    directions = torch.randn(
        (n_probe, *primal_fp32.shape),
        generator=generator,
        device=primal_fp32.device,
        dtype=torch.float32,
    )
    flattened = directions.reshape(n_probe, -1)
    norms = flattened.norm(dim=1)
    if bool(norms.eq(0).any()):
        raise RuntimeError("probe generator produced an exactly zero direction")
    view_shape = (n_probe,) + (1,) * primal_fp32.ndim
    return directions / norms.reshape(view_shape)


def probe_bank_sha256(directions: torch.Tensor) -> str:
    """Return a receipt digest for an FP32 probe bank."""

    if type(directions) is not torch.Tensor or directions.dtype is not torch.float32:
        raise TypeError("probe bank must be an exact FP32 tensor")
    payload = directions.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class _StreamState:
    name: str
    stream: ModuleRNGStream
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class _GeneratorState:
    name: str
    generator: torch.Generator
    state: torch.Tensor


@dataclass(frozen=True)
class StochasticStateSnapshot:
    """Canonical stochastic state restored before every AD evaluation.

    Namespaced ``ModuleRNGStream`` counters and directly stored module
    generators are snapshotted.  Ambient CPU/CUDA RNG state is restored too,
    but any transition that advances it is rejected: stochastic modules must
    consume their own namespaced generators rather than rely on ambient RNG.
    """

    streams: tuple[_StreamState, ...]
    generators: tuple[_GeneratorState, ...]
    cpu_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]

    @classmethod
    def capture(cls, model: nn.Module) -> "StochasticStateSnapshot":
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        streams: list[_StreamState] = []
        generators: list[_GeneratorState] = []
        seen_generators: set[int] = set()
        for module_name, module in model.named_modules():
            if isinstance(module, ModuleRNGStream):
                state = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
                streams.append(_StreamState(module_name, module, state))
            for attribute, value in vars(module).items():
                if type(value) is not torch.Generator or id(value) in seen_generators:
                    continue
                seen_generators.add(id(value))
                qualified = f"{module_name}.{attribute}" if module_name else attribute
                generators.append(
                    _GeneratorState(qualified, value, value.get_state().clone())
                )
        cuda_states: tuple[torch.Tensor, ...] = ()
        if torch.cuda.is_available():
            cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        return cls(
            streams=tuple(streams),
            generators=tuple(generators),
            cpu_rng_state=torch.random.get_rng_state().clone(),
            cuda_rng_states=cuda_states,
        )

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(state.name for state in self.streams)

    @property
    def generator_names(self) -> tuple[str, ...]:
        return tuple(state.name for state in self.generators)

    def reset(self) -> None:
        for state in self.streams:
            state.stream.load_state_dict(copy.deepcopy(state.state_dict), strict=True)
        for state in self.generators:
            state.generator.set_state(state.state.clone())
        torch.random.set_rng_state(self.cpu_rng_state.clone())
        if self.cuda_rng_states:
            torch.cuda.set_rng_state_all([state.clone() for state in self.cuda_rng_states])

    def _ambient_rng_matches(self) -> bool:
        if not torch.equal(torch.random.get_rng_state(), self.cpu_rng_state):
            return False
        if self.cuda_rng_states:
            current = torch.cuda.get_rng_state_all()
            if len(current) != len(self.cuda_rng_states):
                return False
            if any(
                not torch.equal(observed, expected)
                for observed, expected in zip(current, self.cuda_rng_states, strict=True)
            ):
                return False
        return True

    def evaluate(self, function: Callable[[], _Result]) -> _Result:
        """Reset, evaluate once, reject ambient RNG, then restore canonically."""

        if not callable(function):
            raise TypeError("stochastic evaluation target must be callable")
        self.reset()
        succeeded = False
        try:
            result = function()
            succeeded = True
            ambient_escaped = not self._ambient_rng_matches()
        finally:
            self.reset()
        if succeeded and ambient_escaped:
            raise RuntimeError(
                "Jacobian transition advanced ambient RNG; every stochastic source "
                "must use its own snapshotted module generator"
            )
        return result


def _freeze_aux(value: Any) -> Any:
    if type(value) is torch.Tensor:
        return value.detach().cpu().clone()
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_aux(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_aux(item) for item in value)
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("routing aux mapping keys must be exact strings")
        return tuple(
            (key, _freeze_aux(value[key]))
            for key in sorted(value)
        )
    raise TypeError(
        "routing aux must be a tensor tree with string mapping keys and scalar leaves"
    )


def _aux_equal(left: Any, right: Any) -> bool:
    if type(left) is torch.Tensor or type(right) is torch.Tensor:
        return type(left) is torch.Tensor and type(right) is torch.Tensor and torch.equal(
            left, right
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _aux_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _jvp_once(
    transition: Callable[[torch.Tensor], Any],
    primal: torch.Tensor,
    tangent: torch.Tensor,
    *,
    snapshot: StochasticStateSnapshot,
    has_aux: bool,
) -> tuple[torch.Tensor, torch.Tensor, Any | None]:
    if has_aux:
        output, directional, aux = snapshot.evaluate(
            lambda: torch.func.jvp(
                transition,
                (primal,),
                (tangent,),
                has_aux=True,
            )
        )
        frozen_aux = _freeze_aux(aux)
    else:
        output, directional = snapshot.evaluate(
            lambda: torch.func.jvp(transition, (primal,), (tangent,))
        )
        frozen_aux = None
    _validate_transition_result(output, directional, primal)
    return output, directional, frozen_aux


def loop_log_gains(
    transition: Callable[[torch.Tensor], Any],
    primal: torch.Tensor,
    probe_directions: torch.Tensor,
    *,
    stochastic_snapshot: StochasticStateSnapshot,
    has_aux: bool = True,
) -> torch.Tensor:
    """Measure ``log(||J_total v|| / ||v||)`` by forward-mode AD only.

    Directions must already be unit vectors.  Supplying the bank, rather than
    a per-call seed, prevents a depth loop from accidentally redrawing it.
    When ``has_aux`` is true, ``transition`` returns ``(state, branch_aux)`` and
    branch decisions are required to be bit-identical across all probes.
    """

    if not callable(transition):
        raise TypeError("transition must be callable")
    if type(has_aux) is not bool:
        raise TypeError("has_aux must be an exact bool")
    primal_fp32 = _fp32_primal(primal)
    if type(probe_directions) is not torch.Tensor:
        raise TypeError("probe_directions must be an exact tensor")
    if probe_directions.ndim != primal_fp32.ndim + 1:
        raise ValueError("probe bank must have shape [n_probe, *primal.shape]")
    if tuple(probe_directions.shape[1:]) != tuple(primal_fp32.shape):
        raise ValueError("probe directions must align with the primal shape")
    if probe_directions.device != primal_fp32.device:
        raise ValueError("probe directions must share the primal device")
    if probe_directions.dtype is not torch.float32:
        raise TypeError("probe directions must be FP32")
    if probe_directions.shape[0] < 1:
        raise ValueError("at least one probe direction is required")
    norms = probe_directions.reshape(probe_directions.shape[0], -1).norm(dim=1)
    if not bool(torch.isfinite(norms).all()):
        raise ValueError("probe directions must be finite")
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-6, atol=1e-7):
        raise ValueError("gain probes require unit directions")

    expected_aux: Any | None = None
    log_gains: list[torch.Tensor] = []
    try:
        for index, direction in enumerate(probe_directions.unbind(0)):
            _output, directional, aux = _jvp_once(
                transition,
                primal_fp32,
                direction,
                snapshot=stochastic_snapshot,
                has_aux=has_aux,
            )
            if has_aux:
                if index == 0:
                    expected_aux = aux
                elif not _aux_equal(aux, expected_aux):
                    raise RuntimeError(
                        "expert selection or hard-gate decisions changed across probes"
                    )
            log_gains.append(torch.log(directional.norm().clamp_min(1e-30)))
    finally:
        stochastic_snapshot.reset()
    return torch.stack(log_gains)


@dataclass(frozen=True)
class ExampleDepthMeasurement:
    """Paired depth measurements for one example under one probe bank."""

    example_id: str
    example_probe_seed: int
    probe_bank_sha256: str
    depths: tuple[int, ...]
    log_gains: torch.Tensor
    lambda_hat: torch.Tensor
    fixed_branch_verified: bool

    @property
    def log_abs_lambda(self) -> torch.Tensor:
        return torch.log(self.lambda_hat.abs())

    @property
    def lambda_sign(self) -> torch.Tensor:
        return torch.sign(self.lambda_hat)


def measure_example_depths(
    transition_for_depth: Callable[[int], Callable[[torch.Tensor], Any]],
    primal: torch.Tensor,
    *,
    model: nn.Module,
    panel_seed: int,
    example_id: str,
    depths: Sequence[int] = REGISTERED_DEPTHS,
    n_probe: int = MAIN_PANEL_PROBES,
    has_aux: bool = True,
) -> ExampleDepthMeasurement:
    """Measure one example at every depth with an identical direction bank.

    This is the P-5 entry point.  There is exactly one example-derived probe
    seed and one direction draw before the depth loop; depth is not accepted by
    either seed derivation or direction drawing.
    """

    if not callable(transition_for_depth):
        raise TypeError("transition_for_depth must be callable")
    depth_tuple = tuple(depths)
    if depth_tuple != REGISTERED_DEPTHS:
        raise ValueError("WEFT-1 panel measurements require depths (1, 2, 4, 8) exactly")
    for depth in depth_tuple:
        _validate_exact_int(depth, name="depth", minimum=1)
    example_seed = derive_example_probe_seed(panel_seed, example_id)
    directions = draw_example_probe_directions(
        primal,
        n_probe=n_probe,
        example_probe_seed=example_seed,
    )
    snapshot = StochasticStateSnapshot.capture(model)
    cells: list[torch.Tensor] = []
    lambdas: list[torch.Tensor] = []
    try:
        for depth in depth_tuple:
            transition = transition_for_depth(depth)
            gains = loop_log_gains(
                transition,
                primal,
                directions,
                stochastic_snapshot=snapshot,
                has_aux=has_aux,
            )
            cells.append(gains)
            lambdas.append(gains.mean() / depth)
    finally:
        snapshot.reset()
    return ExampleDepthMeasurement(
        example_id=example_id,
        example_probe_seed=example_seed,
        probe_bank_sha256=probe_bank_sha256(directions),
        depths=depth_tuple,
        log_gains=torch.stack(cells),
        lambda_hat=torch.stack(lambdas),
        fixed_branch_verified=has_aux,
    )


def _vjp_once(
    transition: Callable[[torch.Tensor], Any],
    primal: torch.Tensor,
    *,
    snapshot: StochasticStateSnapshot,
    has_aux: bool,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor]], Any | None]:
    if has_aux:
        output, pullback, aux = snapshot.evaluate(
            lambda: torch.func.vjp(transition, primal, has_aux=True)
        )
        frozen_aux = _freeze_aux(aux)
    else:
        output, pullback = snapshot.evaluate(lambda: torch.func.vjp(transition, primal))
        frozen_aux = None
    if type(output) is not torch.Tensor or output.shape != primal.shape:
        raise ValueError("transition and primal must have identical tensor shapes")
    if output.dtype is not torch.float32 or output.device != primal.device:
        raise TypeError("Jacobian transition must remain FP32 on the primal device")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Jacobian transition must be finite")
    return output, pullback, frozen_aux


def operator_norm(
    transition: Callable[[torch.Tensor], Any],
    primal: torch.Tensor,
    *,
    model: nn.Module,
    iterations: int = NORM_POWER_ITERATIONS,
    seed: int = 0,
    has_aux: bool = False,
) -> float:
    """Estimate the top singular value of ``J_total`` via ``J.T J`` power iteration."""

    _validate_exact_int(iterations, name="iterations", minimum=1)
    _validate_seed(seed, name="seed")
    primal_fp32 = _fp32_primal(primal)
    snapshot = StochasticStateSnapshot.capture(model)
    generator = torch.Generator(device=primal_fp32.device).manual_seed(seed)
    direction = torch.randn(
        primal_fp32.shape,
        generator=generator,
        device=primal_fp32.device,
        dtype=torch.float32,
    )
    direction = direction / direction.norm().clamp_min(1e-12)
    try:
        _output, pullback, expected_aux = _vjp_once(
            transition,
            primal_fp32,
            snapshot=snapshot,
            has_aux=has_aux,
        )
        for _ in range(iterations):
            _value, directional, aux = _jvp_once(
                transition,
                primal_fp32,
                direction,
                snapshot=snapshot,
                has_aux=has_aux,
            )
            if has_aux and not _aux_equal(aux, expected_aux):
                raise RuntimeError("routing branch changed during operator-norm iteration")
            (normal_direction,) = pullback(directional)
            normal_norm = normal_direction.norm()
            if not bool(torch.isfinite(normal_norm)):
                raise ValueError("operator-norm iteration became non-finite")
            if normal_norm.item() == 0.0:
                return 0.0
            direction = normal_direction / normal_norm
        _value, directional, aux = _jvp_once(
            transition,
            primal_fp32,
            direction,
            snapshot=snapshot,
            has_aux=has_aux,
        )
        if has_aux and not _aux_equal(aux, expected_aux):
            raise RuntimeError("routing branch changed during operator-norm readout")
        return directional.norm().item()
    finally:
        snapshot.reset()


def participation_ratio(
    transition: Callable[[torch.Tensor], Any],
    primal: torch.Tensor,
    *,
    model: nn.Module,
    n_probe: int = RANK_PANEL_PROBES,
    seed: int = 0,
    has_aux: bool = False,
) -> float:
    """Estimate ``tr(A)^2 / tr(A^2)`` for ``A = J.T J`` by Hutchinson probes.

    Trace probes are unnormalised ``N(0, I)`` vectors.  Unit-normalising these
    vectors would silently divide the participation ratio by state dimension.
    """

    _validate_exact_int(n_probe, name="n_probe", minimum=1)
    _validate_seed(seed, name="seed")
    primal_fp32 = _fp32_primal(primal)
    snapshot = StochasticStateSnapshot.capture(model)
    generator = torch.Generator(device=primal_fp32.device).manual_seed(seed)
    first_trace: list[torch.Tensor] = []
    second_trace: list[torch.Tensor] = []
    try:
        _output, pullback, expected_aux = _vjp_once(
            transition,
            primal_fp32,
            snapshot=snapshot,
            has_aux=has_aux,
        )
        for _ in range(n_probe):
            # Intentionally not normalised: Hutchinson requires E[v v.T] = I.
            direction = torch.randn(
                primal_fp32.shape,
                generator=generator,
                device=primal_fp32.device,
                dtype=torch.float32,
            )
            _value, directional, aux = _jvp_once(
                transition,
                primal_fp32,
                direction,
                snapshot=snapshot,
                has_aux=has_aux,
            )
            if has_aux and not _aux_equal(aux, expected_aux):
                raise RuntimeError("routing branch changed during rank estimation")
            (normal_direction,) = pullback(directional)
            first_trace.append(directional.square().sum())
            second_trace.append(normal_direction.square().sum())
        trace_a = torch.stack(first_trace).mean()
        trace_a2 = torch.stack(second_trace).mean().clamp_min(1e-30)
        estimate = trace_a.square() / trace_a2
        if not bool(torch.isfinite(estimate)):
            raise ValueError("participation-ratio estimate is non-finite")
        return estimate.item()
    finally:
        snapshot.reset()


def _median(values: torch.Tensor, *, dim: int | None = None) -> torch.Tensor:
    """Conventional median, averaging the middle pair for even sample counts."""

    if dim is None:
        flattened = values.reshape(-1)
        ordered = flattened.sort().values
        midpoint = ordered.numel() // 2
        if ordered.numel() % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) * 0.5
    ordered = values.sort(dim=dim).values
    count = ordered.shape[dim]
    midpoint = count // 2
    if count % 2:
        return ordered.select(dim, midpoint)
    return (ordered.select(dim, midpoint - 1) + ordered.select(dim, midpoint)) * 0.5


def theil_sen_slopes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return one paired Theil-Sen slope for every example.

    ``x`` is ``[depth]`` (normally ``log2(T)``) and ``y`` is
    ``[example, depth]`` (normally ``log(abs(lambda_T))``).
    """

    if type(x) is not torch.Tensor or type(y) is not torch.Tensor:
        raise TypeError("x and y must be exact tensors")
    if x.ndim != 1 or y.ndim != 2 or y.shape[1] != x.numel():
        raise ValueError("x must be [D] and y must be [N, D]")
    if x.numel() < 2 or y.shape[0] < 1:
        raise ValueError("Theil-Sen requires at least two depths and one example")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("x and y must be floating-point")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("x and y must be finite")
    pairwise: list[torch.Tensor] = []
    for left in range(x.numel()):
        for right in range(left + 1, x.numel()):
            denominator = x[right] - x[left]
            if denominator.item() == 0.0:
                raise ValueError("Theil-Sen design points must be distinct")
            pairwise.append((y[:, right] - y[:, left]) / denominator)
    return _median(torch.stack(pairwise, dim=1), dim=1)


def p_hat(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return the negative population-median paired Theil-Sen slope."""

    return -_median(theil_sen_slopes(x, y)).item()


def pooled_sigma_w(log_gains: torch.Tensor) -> float:
    """Pool within-example, within-depth probe spread across panel cells."""

    if type(log_gains) is not torch.Tensor:
        raise TypeError("log_gains must be an exact tensor")
    if log_gains.ndim != 3 or log_gains.shape[-1] < 2:
        raise ValueError("log_gains must be [example, depth, at least 2 probes]")
    if not log_gains.is_floating_point() or not bool(torch.isfinite(log_gains).all()):
        raise ValueError("log_gains must be finite floating-point values")
    per_cell_variance = log_gains.float().var(dim=-1, unbiased=True)
    return per_cell_variance.mean().sqrt().item()


def sigma_slope_hat(
    slopes: torch.Tensor | np.ndarray,
    sigma_w: float,
    *,
    sxx: float = 5.0,
) -> tuple[float, bool]:
    """Estimate between-example slope SD after removing measurement variance."""

    if isinstance(slopes, torch.Tensor):
        values = slopes.detach().cpu().numpy().astype(np.float64, copy=False)
    elif isinstance(slopes, np.ndarray):
        values = slopes.astype(np.float64, copy=False)
    else:
        raise TypeError("slopes must be a tensor or numpy array")
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("slopes must contain at least two finite values")
    if type(sigma_w) not in {float, int} or not math.isfinite(float(sigma_w)):
        raise TypeError("sigma_w must be a finite real number")
    if float(sigma_w) < 0:
        raise ValueError("sigma_w must be non-negative")
    if type(sxx) not in {float, int} or not math.isfinite(float(sxx)):
        raise TypeError("sxx must be a finite real number")
    if float(sxx) <= 0:
        raise ValueError("sxx must be positive")
    variance = float(np.var(values, ddof=1)) - float(sigma_w) ** 2 / float(sxx)
    return math.sqrt(max(variance, 0.0)), variance < 0.0


def cluster_bootstrap_ci(
    slopes: torch.Tensor | np.ndarray,
    *,
    replicates: int = FULL_BOOTSTRAP_REPLICATES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap examples only and return ``(p_hat, ci_lo, ci_hi)``.

    Probe and depth axes never enter this function, making it impossible to
    inflate the bootstrap sample size with non-independent observations.
    """

    _validate_exact_int(replicates, name="replicates", minimum=1)
    _validate_seed(seed, name="seed")
    if type(alpha) not in {float, int} or not math.isfinite(float(alpha)):
        raise TypeError("alpha must be a finite real number")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if isinstance(slopes, torch.Tensor):
        values = slopes.detach().cpu().numpy().astype(np.float64, copy=False)
    elif isinstance(slopes, np.ndarray):
        values = slopes.astype(np.float64, copy=False)
    else:
        raise TypeError("slopes must be a tensor or numpy array")
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("slopes must contain at least two finite example clusters")
    rng = np.random.default_rng(seed)
    count = values.size
    statistics = np.empty(replicates, dtype=np.float64)
    # Bound transient allocation for the registered 10k x 512 bootstrap.
    chunk_size = min(replicates, max(1, 2_000_000 // count))
    cursor = 0
    while cursor < replicates:
        size = min(chunk_size, replicates - cursor)
        indexes = rng.integers(0, count, size=(size, count))
        statistics[cursor : cursor + size] = -np.median(values[indexes], axis=1)
        cursor += size
    lower, upper = np.percentile(
        statistics,
        [100.0 * float(alpha) / 2.0, 100.0 * (1.0 - float(alpha) / 2.0)],
    )
    return -float(np.median(values)), float(lower), float(upper)


def design_sxx(depths: Sequence[int]) -> float:
    """Return ``sum((log2(T) - mean(log2(T)))**2)`` for a depth ladder."""

    depth_tuple = tuple(depths)
    if len(depth_tuple) < 2:
        raise ValueError("at least two depths are required")
    for depth in depth_tuple:
        _validate_exact_int(depth, name="depth", minimum=1)
    x = np.log2(np.asarray(depth_tuple, dtype=np.float64))
    return float(np.square(x - x.mean()).sum())


def rejection_conditions(
    signs_by_depth: Sequence[int],
    c_l_by_depth: Sequence[float],
) -> tuple[str, ...]:
    """Evaluate the preregistered sign and conditioning conditions."""

    signs = tuple(signs_by_depth)
    conditioning = tuple(float(value) for value in c_l_by_depth)
    if not signs or len(signs) != len(conditioning):
        raise ValueError("sign and cL vectors must be non-empty and aligned")
    if any(type(sign) is not int or sign not in {-1, 0, 1} for sign in signs):
        raise ValueError("lambda signs must be exact integers in {-1, 0, 1}")
    if any(not math.isfinite(value) or value < 0 for value in conditioning):
        raise ValueError("cL values must be finite and non-negative")
    reasons: list[str] = []
    if 0 in signs or len(set(signs)) != 1:
        reasons.append("sign_inconsistency")
    if any(value < CONDITIONING_FLOOR for value in conditioning):
        reasons.append("conditioning_failure")
    return tuple(reasons)


@dataclass(frozen=True)
class JacobianPanelReport:
    """JSON-ready registered reporting contract for one seed and tier."""

    run_seed: int
    tier: TierName
    p_hat: float
    ci_lo: float
    ci_hi: float
    sigma_slope_hat: float
    sigma_slope_clipped: bool
    sigma_w_hat: float
    c_l_by_depth: tuple[float, ...]
    r_pr_by_depth: tuple[float, ...]
    lambda_sign_by_depth: tuple[int, ...]
    conditioning_flag: bool
    rejection_reasons: tuple[str, ...]
    n: int
    n_probe: int
    depths: tuple[int, ...]
    sxx: float
    direction_class_gains: tuple[tuple[str, tuple[float, ...]], ...]
    fixed_branch_verified: bool
    instrument_tier: int = INSTRUMENT_TIER
    differentiation: str = "forward_mode_jvp"
    jacobian_semantics: str = "fixed_routing_branch"
    probe_seed_scope: str = "example"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_seed": self.run_seed,
            "tier": self.tier,
            "p_hat": self.p_hat,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "sigma_slope_hat": self.sigma_slope_hat,
            "sigma_slope_clipped": self.sigma_slope_clipped,
            "sigma_w_hat": self.sigma_w_hat,
            "cL_by_depth": dict(zip(self.depths, self.c_l_by_depth, strict=True)),
            "r_PR_by_depth": dict(zip(self.depths, self.r_pr_by_depth, strict=True)),
            "sign_lambda_T_by_depth": dict(
                zip(self.depths, self.lambda_sign_by_depth, strict=True)
            ),
            "conditioning_flag": self.conditioning_flag,
            "rejection_reasons": list(self.rejection_reasons),
            "n": self.n,
            "n_probe": self.n_probe,
            "depths": list(self.depths),
            "Sxx": self.sxx,
            "direction_class_gains_by_depth": {
                name: dict(zip(self.depths, values, strict=True))
                for name, values in self.direction_class_gains
            },
            "fixed_branch_verified": self.fixed_branch_verified,
            "instrument_tier": self.instrument_tier,
            "differentiation": self.differentiation,
            "jacobian_semantics": self.jacobian_semantics,
            "routing_flip_derivative_included": False,
            "probe_seed_scope": self.probe_seed_scope,
        }


def build_panel_report(
    slopes: torch.Tensor,
    log_gains: torch.Tensor,
    *,
    run_seed: int,
    tier: TierName,
    depths: Sequence[int],
    c_l_by_depth: Sequence[float],
    r_pr_by_depth: Sequence[float],
    lambda_sign_by_depth: Sequence[int],
    direction_class_gains: Mapping[str, Sequence[float]],
    fixed_branch_verified: bool,
    bootstrap_replicates: int = FULL_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = 0,
) -> JacobianPanelReport:
    """Assemble and validate the complete registered receipt payload."""

    _validate_seed(run_seed, name="run_seed")
    if tier not in {"main", "norm", "rank"}:
        raise ValueError("tier must be main, norm, or rank")
    if type(fixed_branch_verified) is not bool:
        raise TypeError("fixed_branch_verified must be an exact bool")
    if not fixed_branch_verified:
        raise RuntimeError("panel receipt requires bit-identical routing branch evidence")
    if type(slopes) is not torch.Tensor or slopes.ndim != 1 or slopes.numel() < 2:
        raise ValueError("slopes must be a one-dimensional example vector")
    if type(log_gains) is not torch.Tensor or log_gains.ndim != 3:
        raise ValueError("log_gains must be [example, depth, probe]")
    depth_tuple = tuple(depths)
    if depth_tuple != REGISTERED_DEPTHS:
        raise ValueError("WEFT-1 panel reports require depths (1, 2, 4, 8) exactly")
    if log_gains.shape[0] != slopes.numel() or log_gains.shape[1] != len(depth_tuple):
        raise ValueError("log-gain panel must align with slopes and depths")
    c_l = tuple(float(value) for value in c_l_by_depth)
    r_pr = tuple(float(value) for value in r_pr_by_depth)
    signs = tuple(lambda_sign_by_depth)
    if not (len(c_l) == len(r_pr) == len(signs) == len(depth_tuple)):
        raise ValueError("all per-depth reporting vectors must align")
    if any(not math.isfinite(value) or value < 0 for value in c_l):
        raise ValueError("cL values must be finite and non-negative")
    if any(not math.isfinite(value) or value < 0 for value in r_pr):
        raise ValueError("r_PR values must be finite and non-negative")
    if set(direction_class_gains) != set(DIRECTION_CLASSES):
        raise ValueError("all three registered direction classes are required exactly")
    direction_rows: list[tuple[str, tuple[float, ...]]] = []
    for name in DIRECTION_CLASSES:
        values = tuple(float(value) for value in direction_class_gains[name])
        if len(values) != len(depth_tuple):
            raise ValueError("direction-class gains must align with depths")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("direction-class gains must be finite")
        direction_rows.append((name, values))
    sxx = design_sxx(depth_tuple)
    sigma_w = pooled_sigma_w(log_gains)
    sigma_slope, clipped = sigma_slope_hat(slopes, sigma_w, sxx=sxx)
    estimate, ci_lo, ci_hi = cluster_bootstrap_ci(
        slopes,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    reasons = rejection_conditions(signs, c_l)
    return JacobianPanelReport(
        run_seed=run_seed,
        tier=tier,
        p_hat=estimate,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        sigma_slope_hat=sigma_slope,
        sigma_slope_clipped=clipped,
        sigma_w_hat=sigma_w,
        c_l_by_depth=c_l,
        r_pr_by_depth=r_pr,
        lambda_sign_by_depth=signs,
        conditioning_flag="conditioning_failure" in reasons,
        rejection_reasons=reasons,
        n=slopes.numel(),
        n_probe=log_gains.shape[-1],
        depths=depth_tuple,
        sxx=sxx,
        direction_class_gains=tuple(direction_rows),
        fixed_branch_verified=fixed_branch_verified,
        differentiation=(
            "forward_mode_jvp"
            if tier == "main"
            else "forward_mode_jvp_plus_vjp"
        ),
    )


@dataclass(frozen=True)
class TierComparison:
    outcome: Literal["norm_invariance_confirmed", "return_to_strategy"]
    main_inside_norm_interval: bool
    norm_inside_main_interval: bool


def compare_main_and_norm_tiers(
    main: JacobianPanelReport,
    norm: JacobianPanelReport,
) -> TierComparison:
    """Apply P-4 without averaging or selecting between disagreeing tiers."""

    if main.tier != "main" or norm.tier != "norm":
        raise ValueError("tier comparison requires main and norm reports")
    main_inside = norm.ci_lo <= main.p_hat <= norm.ci_hi
    norm_inside = main.ci_lo <= norm.p_hat <= main.ci_hi
    outcome: Literal["norm_invariance_confirmed", "return_to_strategy"]
    if main_inside and norm_inside:
        outcome = "norm_invariance_confirmed"
    else:
        outcome = "return_to_strategy"
    return TierComparison(outcome, main_inside, norm_inside)


__all__ = [
    "CONDITIONING_FLOOR",
    "DIRECTION_CLASSES",
    "ExampleDepthMeasurement",
    "FULL_BOOTSTRAP_REPLICATES",
    "INSTRUMENT_TIER",
    "JacobianPanelReport",
    "MAIN_PANEL_EXAMPLES",
    "MAIN_PANEL_PROBES",
    "NORM_PANEL_EXAMPLES",
    "NORM_POWER_ITERATIONS",
    "PILOT_EXAMPLES",
    "RANK_PANEL_EXAMPLES",
    "RANK_PANEL_PROBES",
    "REGISTERED_DEPTHS",
    "StochasticStateSnapshot",
    "TierComparison",
    "build_panel_report",
    "cluster_bootstrap_ci",
    "compare_main_and_norm_tiers",
    "derive_example_probe_seed",
    "design_sxx",
    "draw_example_probe_directions",
    "loop_log_gains",
    "measure_example_depths",
    "operator_norm",
    "p_hat",
    "participation_ratio",
    "pooled_sigma_w",
    "probe_bank_sha256",
    "rejection_conditions",
    "sigma_slope_hat",
    "theil_sen_slopes",
]
