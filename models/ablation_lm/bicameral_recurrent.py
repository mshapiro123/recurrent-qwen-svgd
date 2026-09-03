"""Policy-explicit, separated-state bicameral recurrence seam.

The seam owns the ratified live/static/midpoint K/V source schedule and the
``alpha=c/K`` repeated execution schedule.  It deliberately leaves the
terminal pair separated; the S-2 output combiner is an independent module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol, Sequence

import torch
from torch import nn


KV_POLICIES = ("live", "static", "midpoint")


class AfterBlockHook(Protocol):
    """Typed seam for a post-block lane update without owning its semantics."""

    def __call__(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        *,
        visit: int,
        block_index: int,
        residual_scale: float,
    ) -> "AfterBlockResult": ...


@dataclass(frozen=True)
class AfterBlockResult:
    """Hook result with the exact modules that executed inside the hook."""

    h_a: torch.Tensor
    h_b: torch.Tensor
    executed_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.executed_modules, tuple) or any(
            not isinstance(name, str) or not name for name in self.executed_modules
        ):
            raise TypeError("executed_modules must be a tuple of nonempty strings")


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return converted


def expected_bicameral_visit_schedules(
    *,
    recurrent_steps: int,
    unique_core_blocks: int,
    kv_policy: str,
    after_block_modules: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return the only exact traces accepted by the Step-2 recurrence seam.

    Static and midpoint execution admit two equivalent entry modes: the seam
    may materialize the shared anchor itself, or consume a caller-materialized
    cache. Everything after that binding is identical and fully ordered.
    """

    steps = _positive_integer(recurrent_steps, name="recurrent_steps")
    blocks = _positive_integer(unique_core_blocks, name="unique_core_blocks")
    if not isinstance(kv_policy, str) or kv_policy not in KV_POLICIES:
        raise ValueError(f"kv_policy must be one of {KV_POLICIES!r}")
    if not isinstance(after_block_modules, tuple) or any(
        not isinstance(name, str) or not name for name in after_block_modules
    ):
        raise TypeError("after_block_modules must be a tuple of nonempty strings")

    setup_modes: tuple[str | None, ...] = (
        (None,)
        if kv_policy == "live"
        else (
            f"project_kv.{kv_policy}_shared",
            f"use_projected_kv.{kv_policy}",
        )
    )
    candidates: list[tuple[str, ...]] = []
    for setup_mode in setup_modes:
        events: list[str] = []
        if setup_mode is not None:
            events.extend(
                f"setup.block[{block}].{setup_mode}" for block in range(blocks)
            )
        for visit in range(steps):
            if kv_policy == "live":
                events.extend(
                    f"visit[{visit}].block[{block}].project_kv.live"
                    for block in range(blocks)
                )
            elif kv_policy == "midpoint" and steps >= 2 and visit == steps // 2:
                events.extend(
                    f"visit[{visit}].block[{block}].project_kv.midpoint_refresh"
                    for block in range(blocks)
                )
            for block in range(blocks):
                prefix = f"visit[{visit}].block[{block}]"
                events.append(f"{prefix}.attention")
                events.append(f"{prefix}.feed_forward")
                events.extend(f"{prefix}.{name}" for name in after_block_modules)
        candidates.append(tuple(events))
    return tuple(candidates)


@dataclass(frozen=True)
class BicameralRecurrenceReceipt:
    """Immutable receipt for the K/V policy and repeated execution schedule."""

    recurrent_steps: int
    unique_core_blocks: int
    executed_block_passes: int
    recurrence_c: float
    residual_scale: float
    kv_policy: str
    kv_cache_multiplier_at_serving: int
    after_block_modules: tuple[str, ...]
    visit_schedule: tuple[str, ...]
    terminal_recombination: str

    def __post_init__(self) -> None:
        steps = _positive_integer(self.recurrent_steps, name="recurrent_steps")
        blocks = _positive_integer(self.unique_core_blocks, name="unique_core_blocks")
        recurrence_c = _finite_positive(self.recurrence_c, name="recurrence_c")
        expected_scale = recurrence_c / steps
        if self.executed_block_passes != steps * blocks:
            raise ValueError("executed_block_passes does not match steps times blocks")
        if self.residual_scale != expected_scale:
            raise ValueError("residual_scale does not equal recurrence_c / recurrent_steps")
        if not isinstance(self.kv_policy, str) or self.kv_policy not in KV_POLICIES:
            raise ValueError(f"kv_policy must be one of {KV_POLICIES!r}")
        expected_multiplier = {
            "live": 2 * steps,
            "static": 1,
            "midpoint": 2,
        }[self.kv_policy]
        if (
            type(self.kv_cache_multiplier_at_serving) is not int
            or self.kv_cache_multiplier_at_serving != expected_multiplier
        ):
            raise ValueError(
                "kv_cache_multiplier_at_serving does not match the selected policy"
            )
        expected_schedules = expected_bicameral_visit_schedules(
            recurrent_steps=steps,
            unique_core_blocks=blocks,
            kv_policy=self.kv_policy,
            after_block_modules=self.after_block_modules,
        )
        if self.visit_schedule not in expected_schedules:
            raise ValueError(
                "visit_schedule is not an exact execution trace for the configured "
                "K/V policy, block count, and after-block modules"
            )
        if self.terminal_recombination != "not executed (separated-state seam)":
            raise ValueError("the recurrence seam must leave terminal recombination external")


@dataclass(frozen=True)
class BicameralRecurrentOutput:
    """The still-separated terminal hemisphere states plus schedule receipt."""

    h_a: torch.Tensor
    h_b: torch.Tensor
    receipt: BicameralRecurrenceReceipt


def _validated_blocks(
    blocks: Sequence[nn.Module] | nn.ModuleList,
) -> tuple[nn.Module, ...]:
    if isinstance(blocks, (str, bytes)) or not isinstance(
        blocks,
        (Sequence, nn.ModuleList),
    ):
        raise TypeError("blocks must be a sequence of modules")
    resolved = tuple(blocks)
    if not resolved:
        raise ValueError("blocks must contain at least one core block")
    if not all(isinstance(block, nn.Module) for block in resolved):
        raise TypeError("every core block must be an nn.Module")
    return resolved


def _validate_state_pair(h_a: torch.Tensor, h_b: torch.Tensor) -> None:
    if not isinstance(h_a, torch.Tensor) or not isinstance(h_b, torch.Tensor):
        raise TypeError("h_a and h_b must be tensors")
    if h_a.shape != h_b.shape:
        raise ValueError("h_a and h_b must have identical shapes")
    if h_a.ndim != 3:
        raise ValueError("h_a and h_b must have shape [batch, sequence, width]")
    if not h_a.is_floating_point() or not h_b.is_floating_point():
        raise TypeError("h_a and h_b must be floating point")
    if h_a.device != h_b.device or h_a.dtype != h_b.dtype:
        raise ValueError("h_a and h_b must share one dtype and device")


def execute_bicameral_recurrence(
    blocks: Sequence[nn.Module] | nn.ModuleList,
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    *,
    recurrent_steps: int,
    recurrence_c: float,
    projected_kv: Sequence[Any] | None = None,
    kv_policy: str = "live",
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    force_math_attention: bool = False,
    after_block: AfterBlockHook | None = None,
    expected_after_block_modules: tuple[str, ...] | None = None,
) -> BicameralRecurrentOutput:
    """Execute recurrence over caller-owned blocks without registering them.

    Live K/V takes one snapshot of both states at each visit entry, projects
    every unique block from that same snapshot, and then executes the blocks.
    This visit-entry rule makes live and static identical at ``K=1`` when both
    start from the shared anchor. Static either materializes the shared anchor
    inside this seam or reuses validated caller-owned payloads; midpoint does
    the same and refreshes all block payloads once at ``floor(K/2)``.

    ``after_block`` runs immediately after every unique core block. This helper
    owns only its timing and validates the returned state pair; it encodes no
    carrier, callosum, or lane-update semantics.
    """

    resolved_blocks = _validated_blocks(blocks)
    _validate_state_pair(h_a, h_b)
    steps = _positive_integer(recurrent_steps, name="recurrent_steps")
    c_value = _finite_positive(recurrence_c, name="recurrence_c")
    if not isinstance(kv_policy, str) or kv_policy not in KV_POLICIES:
        raise ValueError(f"kv_policy must be one of {KV_POLICIES!r}")
    if after_block is not None and not callable(after_block):
        raise TypeError("after_block must be callable or None")
    if after_block is None:
        if expected_after_block_modules not in (None, ()):
            raise ValueError("after-block modules require an after_block hook")
        bound_after_block_modules: tuple[str, ...] = ()
    else:
        if expected_after_block_modules is None:
            raise ValueError(
                "an after_block hook requires expected_after_block_modules"
            )
        if not isinstance(expected_after_block_modules, tuple) or any(
            not isinstance(name, str) or not name
            for name in expected_after_block_modules
        ):
            raise TypeError(
                "expected_after_block_modules must be a tuple of nonempty strings"
            )
        bound_after_block_modules = expected_after_block_modules
    execution_events: list[str] = []
    if kv_policy == "live":
        if projected_kv is not None:
            raise ValueError("live kv_policy recomputes K/V and forbids caller caches")
        active_kv: tuple[Any, ...] | None = None
    else:
        if projected_kv is None:
            active_kv = None
        else:
            if not isinstance(projected_kv, Sequence) or isinstance(
                projected_kv,
                (str, bytes),
            ):
                raise TypeError("static and midpoint projected_kv must be a sequence")
            if len(projected_kv) != len(resolved_blocks):
                raise ValueError(
                    "projected_kv must contain one entry per unique core block"
                )
            active_kv = tuple(projected_kv)
            for block_index in range(len(resolved_blocks)):
                execution_events.append(
                    f"setup.block[{block_index}].use_projected_kv.{kv_policy}"
                )

    def project_visit_entry(
        entry_a: torch.Tensor,
        entry_b: torch.Tensor | None,
        *,
        event_suffix: str,
        visit: int | None,
    ) -> tuple[Any, ...]:
        projected: list[Any] = []
        for block_index, block in enumerate(resolved_blocks):
            project = getattr(block, "project_kv", None)
            if not callable(project):
                raise TypeError("every bicameral core block must implement project_kv")
            if entry_b is None:
                projected.append(project(entry_a, position_ids=position_ids))
            else:
                projected.append(project(entry_a, entry_b, position_ids=position_ids))
            prefix = "setup" if visit is None else f"visit[{visit}]"
            execution_events.append(
                f"{prefix}.block[{block_index}].project_kv.{event_suffix}"
            )
        return tuple(projected)

    if kv_policy != "live" and active_kv is None:
        active_kv = project_visit_entry(
            h_a,
            None,
            event_suffix=f"{kv_policy}_shared",
            visit=None,
        )

    residual_scale = c_value / steps
    for visit in range(steps):
        if kv_policy == "live":
            active_kv = project_visit_entry(
                h_a,
                h_b,
                event_suffix="live",
                visit=visit,
            )
        elif kv_policy == "midpoint" and steps >= 2 and visit == steps // 2:
            active_kv = project_visit_entry(
                h_a,
                h_b,
                event_suffix="midpoint_refresh",
                visit=visit,
            )
        assert active_kv is not None
        for block_index, (block, cache) in enumerate(
            zip(resolved_blocks, active_kv, strict=True)
        ):
            event_prefix = f"visit[{visit}].block[{block_index}]"
            events_before_block = len(execution_events)
            h_a, h_b = block(
                h_a,
                h_b,
                projected_kv=cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                document_ids=document_ids,
                residual_scale=residual_scale,
                force_math_attention=force_math_attention,
                execution_events=execution_events,
                execution_prefix=event_prefix,
            )
            _validate_state_pair(h_a, h_b)
            expected_block_events = [
                f"{event_prefix}.attention",
                f"{event_prefix}.feed_forward",
            ]
            if execution_events[events_before_block:] != expected_block_events:
                raise RuntimeError(
                    "a bicameral block did not emit the exact attention-FFN trace"
                )
            if after_block is not None:
                hook_shape = h_a.shape
                hook_dtype = h_a.dtype
                hook_device = h_a.device
                hook_output = after_block(
                    h_a,
                    h_b,
                    visit=visit,
                    block_index=block_index,
                    residual_scale=residual_scale,
                )
                if not isinstance(hook_output, AfterBlockResult):
                    raise TypeError("after_block must return AfterBlockResult")
                if hook_output.executed_modules != bound_after_block_modules:
                    raise RuntimeError(
                        "after_block executed modules disagree with the bound receipt"
                    )
                h_a, h_b = hook_output.h_a, hook_output.h_b
                _validate_state_pair(h_a, h_b)
                if h_a.shape != hook_shape:
                    raise ValueError("after_block may not change the state shape")
                if h_a.dtype != hook_dtype or h_a.device != hook_device:
                    raise ValueError(
                        "after_block may not change the state dtype or device"
                    )
                execution_events.extend(
                    f"{event_prefix}.{module_name}"
                    for module_name in hook_output.executed_modules
                )

    receipt = BicameralRecurrenceReceipt(
        recurrent_steps=steps,
        unique_core_blocks=len(resolved_blocks),
        executed_block_passes=steps * len(resolved_blocks),
        recurrence_c=c_value,
        residual_scale=residual_scale,
        kv_policy=kv_policy,
        kv_cache_multiplier_at_serving={
            "live": 2 * steps,
            "static": 1,
            "midpoint": 2,
        }[kv_policy],
        after_block_modules=bound_after_block_modules,
        visit_schedule=tuple(execution_events),
        terminal_recombination="not executed (separated-state seam)",
    )
    return BicameralRecurrentOutput(h_a=h_a, h_b=h_b, receipt=receipt)


class BicameralRecurrentCore(nn.Module):
    """Standalone module wrapper around :func:`execute_bicameral_recurrence`."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_validated_blocks(blocks))

    @staticmethod
    def _validate_state_pair(h_a: torch.Tensor, h_b: torch.Tensor) -> None:
        _validate_state_pair(h_a, h_b)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        *,
        recurrent_steps: int,
        recurrence_c: float,
        projected_kv: Sequence[Any] | None = None,
        kv_policy: str = "live",
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        force_math_attention: bool = False,
        after_block: AfterBlockHook | None = None,
        expected_after_block_modules: tuple[str, ...] | None = None,
    ) -> BicameralRecurrentOutput:
        return execute_bicameral_recurrence(
            self.blocks,
            h_a,
            h_b,
            recurrent_steps=recurrent_steps,
            recurrence_c=recurrence_c,
            projected_kv=projected_kv,
            kv_policy=kv_policy,
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
            force_math_attention=force_math_attention,
            after_block=after_block,
            expected_after_block_modules=expected_after_block_modules,
        )
