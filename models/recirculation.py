"""Paper-native, score-only recirculation for decoder-only causal LMs.

The registered evaluator is serial in token position.  Its first stack owns the
scored logits.  A second, partial stack replaces only the current position's
deep K/V entries so that later positions can attend to recirculated state.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RecirculationConfig:
    source_layer: int
    destination_layer: int
    alpha: float
    beta_mode: str = "convex"
    ramp_tokens: int | None = None

    def validate(self, num_layers: int) -> None:
        if not 1 <= self.destination_layer < self.source_layer <= num_layers:
            raise ValueError(
                "recirculation requires 1 <= destination < source <= num_layers"
            )
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.beta_mode not in {"convex", "additive"}:
            raise ValueError("beta_mode must be 'convex' or 'additive'")
        if self.ramp_tokens is not None and int(self.ramp_tokens) <= 0:
            raise ValueError("ramp_tokens must be positive when supplied")


@dataclass
class RecirculationPrefix:
    attention_mask: torch.Tensor
    past_key_values: Any
    processed_positions: int


@dataclass(frozen=True)
class RecirculationStep:
    logits: torch.Tensor
    past_key_values: Any
    source_state: torch.Tensor
    destination_state: torch.Tensor
    applied_alpha: float


def _call_supported(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(fn)
    parameters = signature.parameters
    if any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()):
        supported = {key: value for key, value in kwargs.items() if value is not None}
    else:
        supported = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
            and (value is not None or key in {"use_cache", "output_attentions"})
        }
    return fn(*args, **supported)


def _clone_cache(cache: Any) -> Any:
    return copy.deepcopy(cache)


def _cache_layer_tensors(cache: Any) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    if cache is None:
        return
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for index, layer in enumerate(layers):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if torch.is_tensor(keys) and torch.is_tensor(values):
                yield index, keys, values
        return
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is None or values is None:
        raise TypeError("unsupported cache object: expected layers or key_cache/value_cache")
    for index, (key, value) in enumerate(zip(keys, values)):
        if torch.is_tensor(key) and torch.is_tensor(value):
            yield index, key, value


def cache_tensor_map(cache: Any) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for index, keys, values in _cache_layer_tensors(cache):
        tensors[f"layer_{index + 1}.key"] = keys
        tensors[f"layer_{index + 1}.value"] = values
    return tensors


def compare_cache_bitwise(left: Any, right: Any) -> dict[str, Any]:
    left_tensors = cache_tensor_map(left)
    right_tensors = cache_tensor_map(right)
    if set(left_tensors) != set(right_tensors):
        return {
            "bit_exact": False,
            "missing_left": sorted(set(right_tensors) - set(left_tensors)),
            "missing_right": sorted(set(left_tensors) - set(right_tensors)),
            "maximum_absolute_difference": None,
        }
    maximum = 0.0
    mismatched: list[str] = []
    for name in sorted(left_tensors):
        lhs = left_tensors[name]
        rhs = right_tensors[name]
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or not torch.equal(lhs, rhs):
            mismatched.append(name)
            if lhs.shape == rhs.shape:
                maximum = max(
                    maximum,
                    float((lhs.float() - rhs.float()).abs().max().item()),
                )
    return {
        "bit_exact": not mismatched,
        "mismatched_tensors": mismatched,
        "maximum_absolute_difference": maximum,
    }


def _crop_tensor_last_token(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError("cache tensors must expose a sequence dimension at -2")
    return tensor[..., :target_length, :].contiguous()


def _strip_current_deep_cache(cache: Any, *, first_deep_index: int) -> None:
    """Remove the just-scored token from deep layers in a cloned cache."""

    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers[first_deep_index:]:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not torch.is_tensor(keys) or not torch.is_tensor(values):
                continue
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = max(0, int(layer.cumulative_length) - 1)
            target_length = max(0, int(keys.shape[-2]) - 1)
            if target_length == 0:
                layer.keys = None
                layer.values = None
                if hasattr(layer, "is_initialized"):
                    layer.is_initialized = False
            else:
                layer.keys = _crop_tensor_last_token(keys, target_length)
                layer.values = _crop_tensor_last_token(values, target_length)
        return
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is None or values is None:
        raise TypeError("unsupported cache object: expected layers or key_cache/value_cache")
    for index in range(first_deep_index, len(keys)):
        target_length = max(0, int(keys[index].shape[-2]) - 1)
        keys[index] = _crop_tensor_last_token(keys[index], target_length)
        values[index] = _crop_tensor_last_token(values[index], target_length)


def _replace_deep_cache_from_prior(
    scored_cache: Any, prior_cache: Any, *, first_deep_index: int
) -> Any:
    """Keep scored shallow layers and restore exact prior deep-layer state."""

    merged = _clone_cache(scored_cache)
    merged_layers = getattr(merged, "layers", None)
    prior_layers = getattr(prior_cache, "layers", None)
    if merged_layers is not None and prior_layers is not None:
        if len(merged_layers) != len(prior_layers):
            raise RuntimeError("cache layer count changed during the scored stack")
        for index in range(first_deep_index, len(merged_layers)):
            merged_layers[index] = copy.deepcopy(prior_layers[index])
        return merged
    merged_keys = getattr(merged, "key_cache", None)
    merged_values = getattr(merged, "value_cache", None)
    prior_keys = getattr(prior_cache, "key_cache", None)
    prior_values = getattr(prior_cache, "value_cache", None)
    if merged_keys is None or merged_values is None or prior_keys is None or prior_values is None:
        raise TypeError("unsupported cache object: expected layers or key_cache/value_cache")
    for index in range(first_deep_index, len(merged_keys)):
        if index < len(prior_keys):
            merged_keys[index] = prior_keys[index].clone()
            merged_values[index] = prior_values[index].clone()
        else:
            merged_keys[index] = merged_keys[index][..., :0, :].contiguous()
            merged_values[index] = merged_values[index][..., :0, :].contiguous()
    return merged


def _trim_cache_to_length(cache: Any, target_length: int) -> None:
    """Trim every initialized layer to one common past-only length."""

    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not torch.is_tensor(keys) or not torch.is_tensor(values):
                continue
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = int(target_length)
            if int(keys.shape[-2]) <= target_length:
                continue
            if target_length == 0:
                layer.keys = None
                layer.values = None
                if hasattr(layer, "is_initialized"):
                    layer.is_initialized = False
            else:
                layer.keys = _crop_tensor_last_token(keys, target_length)
                layer.values = _crop_tensor_last_token(values, target_length)
        return
    keys = getattr(cache, "key_cache", None)
    values = getattr(cache, "value_cache", None)
    if keys is None or values is None:
        raise TypeError("unsupported cache object: expected layers or key_cache/value_cache")
    for index in range(len(keys)):
        if int(keys[index].shape[-2]) > target_length:
            keys[index] = _crop_tensor_last_token(keys[index], target_length)
            values[index] = _crop_tensor_last_token(values[index], target_length)


def graph_receipt(*, sequence_length: int, num_layers: int, destination_layer: int) -> dict[str, Any]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if not 1 <= destination_layer < num_layers:
        raise ValueError("destination_layer must be inside the model")
    rows: list[dict[str, Any]] = []
    for position in range(sequence_length):
        final_position = position == sequence_length - 1
        for layer in range(1, num_layers + 1):
            shallow = layer <= destination_layer
            rows.append(
                {
                    "position": position,
                    "layer": layer,
                    "architecture_copy": 0,
                    "input_step": position,
                    "tensor_tap": f"hidden_states[{layer}]",
                    "kv_owner": "scored_stack" if shallow or final_position else "recirculated_stack",
                    "status": (
                        "committed"
                        if shallow or final_position
                        else "provisional_then_discarded"
                    ),
                    "scored_readout_owner": layer == num_layers,
                }
            )
            if not shallow and not final_position:
                rows.append(
                    {
                        "position": position,
                        "layer": layer,
                        "architecture_copy": 1,
                        "input_step": position,
                        "tensor_tap": f"hidden_states[{destination_layer}]_mixed",
                        "kv_owner": "recirculated_stack",
                        "status": "committed",
                        "scored_readout_owner": False,
                    }
                )
    return {
        "kind": "paper_native_recirculation_graph_v1",
        "sequence_length": sequence_length,
        "num_layers": num_layers,
        "destination_layer": destination_layer,
        "readout": "first_iteration_only",
        "tap_convention": "post_block_hidden_states_index_equals_paper_layer",
        "kv_ownership": "shallow_scored_deep_recirculated",
        "final_position_recirculation": "uniformly_skipped",
        "rows": rows,
    }


class PaperNativeRecirculationEvaluator(nn.Module):
    """Correctness-first serial evaluator for one recirculation iteration."""

    def __init__(self, model: nn.Module, config: RecirculationConfig) -> None:
        super().__init__()
        self.model = model
        self.decoder = self._resolve_decoder(model)
        self.layers = self.decoder.layers
        self.config = config
        self.config.validate(len(self.layers))

    @staticmethod
    def _resolve_decoder(model: nn.Module) -> nn.Module:
        queue = [model]
        seen: set[int] = set()
        while queue:
            candidate = queue.pop(0)
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
                return candidate
            for name in ("model", "language_model", "text_model"):
                child = getattr(candidate, name, None)
                if isinstance(child, nn.Module):
                    queue.append(child)
        raise TypeError("could not locate decoder layers and embeddings")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        ids = attention_mask.long().cumsum(dim=-1) - 1
        return ids.masked_fill(attention_mask.eq(0), 0)[:, -1:]

    def _cache_position(self, cache: Any, device: torch.device) -> torch.Tensor:
        length = 0
        if cache is not None and hasattr(cache, "get_seq_length"):
            length = int(cache.get_seq_length())
        return torch.tensor([length], dtype=torch.long, device=device)

    def _causal_masks(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: Any,
        mask_past_key_values: Any | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if mask_past_key_values is None:
            mask_cache = _clone_cache(cache)
            _trim_cache_to_length(mask_cache, int(cache_position[0].item()))
        else:
            mask_cache = mask_past_key_values
        update = getattr(self.decoder, "_update_causal_mask", None)
        if update is not None:
            full = _call_supported(
                update,
                attention_mask=attention_mask,
                input_tensor=hidden_states,
                inputs_embeds=hidden_states,
                cache_position=cache_position,
                past_key_values=mask_cache,
                output_attentions=False,
            )
            return {"full_attention": full}
        try:
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

            masks: dict[str, torch.Tensor | None] = {}
            masks["full_attention"] = create_causal_mask(
                config=self.decoder.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=mask_cache,
                position_ids=position_ids,
            )
            if bool(getattr(self.decoder, "has_sliding_layers", False)):
                masks["sliding_attention"] = create_sliding_window_causal_mask(
                    config=self.decoder.config,
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=mask_cache,
                    position_ids=position_ids,
                )
            return masks
        except (ImportError, TypeError):
            pass
        batch, query = hidden_states.shape[:2]
        target = attention_mask.shape[-1]
        minimum = torch.finfo(hidden_states.dtype).min
        keys = torch.arange(target, device=hidden_states.device)
        blocked = keys.unsqueeze(0) > cache_position.unsqueeze(1)
        mask = torch.zeros(
            (query, target), device=hidden_states.device, dtype=hidden_states.dtype
        ).masked_fill(blocked, minimum)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, query, target)
        return {
            "full_attention": mask.masked_fill(
                attention_mask[:, None, None, :].eq(0), minimum
            )
        }

    def _position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str | None = None,
    ) -> Any:
        rotary = getattr(self.decoder, "rotary_emb", None)
        if rotary is None:
            return None
        try:
            if layer_type is not None:
                return rotary(hidden_states, position_ids, layer_type)
            return rotary(hidden_states, position_ids)
        except TypeError:
            try:
                return rotary(hidden_states, position_ids)
            except TypeError:
                return None

    def _run_upper_layers(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        cache: Any,
        mask_past_key_values: Any | None = None,
    ) -> Any:
        causal_masks = self._causal_masks(
            attention_mask,
            hidden_states,
            position_ids,
            cache_position,
            cache,
            mask_past_key_values,
        )
        for layer_index, layer in enumerate(
            self.layers[self.config.destination_layer :],
            start=self.config.destination_layer,
        ):
            layer_types = getattr(self.decoder.config, "layer_types", None)
            layer_type = (
                str(layer_types[layer_index])
                if layer_types is not None
                else "full_attention"
            )
            causal_mask = causal_masks.get(
                layer_type, causal_masks.get("full_attention")
            )
            position_embeddings = self._position_embeddings(
                hidden_states, position_ids, layer_type
            )
            parameters = inspect.signature(layer.forward).parameters
            cache_key = (
                "past_key_values" if "past_key_values" in parameters else "past_key_value"
            )
            outputs = _call_supported(
                layer,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                **{
                    cache_key: cache,
                    "use_cache": True,
                    "output_attentions": False,
                    "cache_position": cache_position,
                    "position_embeddings": position_embeddings,
                },
            )
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        return cache

    def _alpha_at(self, position: int) -> float:
        if self.config.ramp_tokens is None:
            return float(self.config.alpha)
        return float(self.config.alpha) * min(
            1.0, float(position) / float(self.config.ramp_tokens)
        )

    @torch.inference_mode()
    def step(
        self,
        *,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        position: int,
        recirculate_for_future: bool,
    ) -> RecirculationStep:
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("step requires one token per batch row")
        if attention_mask.shape[0] != token_ids.shape[0]:
            raise ValueError("attention mask batch size changed")
        scored_cache = _clone_cache(past_key_values)
        position_ids = self._position_ids(attention_mask)
        cache_position = self._cache_position(scored_cache, token_ids.device)
        output = self.model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=scored_cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        scored_cache = output.past_key_values
        hidden_states = output.hidden_states
        if len(hidden_states) <= self.config.source_layer:
            raise RuntimeError("model did not return the registered post-block taps")
        destination = hidden_states[self.config.destination_layer]
        source = hidden_states[self.config.source_layer]
        applied_alpha = self._alpha_at(position)
        committed = scored_cache
        if recirculate_for_future:
            committed = _replace_deep_cache_from_prior(
                scored_cache,
                past_key_values,
                first_deep_index=self.config.destination_layer,
            )
            if applied_alpha == 0.0:
                mixed = destination
            else:
                source_norm = source.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
                destination_norm = destination.float().norm(dim=-1, keepdim=True)
                matched = source.float() * (destination_norm / source_norm)
                beta = 1.0 - applied_alpha if self.config.beta_mode == "convex" else 1.0
                mixed = (
                    beta * destination.float() + applied_alpha * matched
                ).to(destination.dtype)
            committed = self._run_upper_layers(
                hidden_states=mixed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                cache=committed,
                mask_past_key_values=past_key_values,
            )
        return RecirculationStep(
            logits=output.logits[:, -1],
            past_key_values=committed,
            source_state=source,
            destination_state=destination,
            applied_alpha=applied_alpha,
        )

    def _empty_cache(self) -> Any:
        from transformers import DynamicCache

        try:
            return DynamicCache(config=getattr(self.model, "config", None))
        except TypeError:
            return DynamicCache()

    @torch.inference_mode()
    def forward_sequence(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recirculate: bool = True,
        skip_final_recirculation: bool = True,
    ) -> tuple[torch.Tensor, Any]:
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must share [batch, sequence]")
        cache = self._empty_cache()
        logits: list[torch.Tensor] = []
        for position in range(input_ids.shape[1]):
            final = position == input_ids.shape[1] - 1
            step = self.step(
                token_ids=input_ids[:, position : position + 1],
                attention_mask=attention_mask[:, : position + 1],
                past_key_values=cache,
                position=position,
                recirculate_for_future=recirculate
                and not (final and skip_final_recirculation),
            )
            logits.append(step.logits)
            cache = step.past_key_values
        return torch.stack(logits, dim=1), cache

    @torch.inference_mode()
    def identity_receipt(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, Any]:
        if float(self.config.alpha) != 0.0 or self.config.beta_mode != "convex":
            raise ValueError("identity receipt requires convex alpha=0")
        intact_logits, intact_cache = self.forward_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            recirculate=False,
        )
        recirculated_logits, recirculated_cache = self.forward_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            recirculate=True,
        )
        logits_equal = torch.equal(intact_logits, recirculated_logits)
        maximum = float(
            (intact_logits.float() - recirculated_logits.float()).abs().max().item()
        )
        cache = compare_cache_bitwise(intact_cache, recirculated_cache)
        return {
            "kind": "paper_native_recirculation_identity_v1",
            "bit_exact": bool(logits_equal and cache["bit_exact"]),
            "scored_logits_bit_exact": bool(logits_equal),
            "scored_logits_maximum_absolute_difference": maximum,
            "committed_cache": cache,
        }

    @torch.inference_mode()
    def sequence_nll(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        recirculate: bool = True,
    ) -> tuple[float, int]:
        """Accumulate next-token NLL without materializing [B, T, V] logits."""

        if input_ids.shape != attention_mask.shape or input_ids.shape[1] < 2:
            raise ValueError("sequence NLL requires matching sequences of length >= 2")
        cache = self._empty_cache()
        loss_sum = 0.0
        token_count = 0
        for position in range(input_ids.shape[1] - 1):
            step = self.step(
                token_ids=input_ids[:, position : position + 1],
                attention_mask=attention_mask[:, : position + 1],
                past_key_values=cache,
                position=position,
                recirculate_for_future=recirculate,
            )
            targets = input_ids[:, position + 1]
            valid = attention_mask[:, position + 1].bool()
            if bool(valid.any()):
                losses = F.cross_entropy(step.logits.float(), targets, reduction="none")
                loss_sum += float(losses[valid].sum().item())
                token_count += int(valid.sum().item())
            cache = step.past_key_values
        return loss_sum, token_count

    @torch.inference_mode()
    def prefill_cached(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[RecirculationPrefix, Any]:
        cache = self._empty_cache()
        last: RecirculationStep | None = None
        for position in range(input_ids.shape[1]):
            last = self.step(
                token_ids=input_ids[:, position : position + 1],
                attention_mask=attention_mask[:, : position + 1],
                past_key_values=cache,
                position=position,
                recirculate_for_future=True,
            )
            cache = last.past_key_values
        if last is None:
            raise ValueError("prefill requires at least one token")
        state = RecirculationPrefix(
            attention_mask=attention_mask,
            past_key_values=cache,
            processed_positions=input_ids.shape[1],
        )
        return state, self._task_output(last, attention_mask)

    @torch.inference_mode()
    def advance_cached(
        self, *, state: RecirculationPrefix, selected_tokens: torch.Tensor
    ) -> tuple[RecirculationPrefix, Any]:
        if selected_tokens.ndim == 1:
            selected_tokens = selected_tokens[:, None]
        ones = torch.ones(
            (selected_tokens.shape[0], 1),
            dtype=state.attention_mask.dtype,
            device=state.attention_mask.device,
        )
        attention_mask = torch.cat([state.attention_mask, ones], dim=1)
        step = self.step(
            token_ids=selected_tokens,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            position=state.processed_positions,
            recirculate_for_future=True,
        )
        updated = RecirculationPrefix(
            attention_mask=attention_mask,
            past_key_values=step.past_key_values,
            processed_positions=state.processed_positions + 1,
        )
        return updated, self._task_output(step, attention_mask)

    def _task_output(self, step: RecirculationStep, attention_mask: torch.Tensor) -> Any:
        from eval.eval_paper2_phase3_p34_task_inference import P34NextTokenOutput

        top2 = step.logits.float().topk(k=2, dim=-1).values
        batch = step.logits.shape[0]
        positions = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0)
        zeros = torch.zeros(batch, device=step.logits.device, dtype=step.logits.dtype)
        scratch = torch.zeros(
            (batch, step.destination_state.shape[-1]),
            device=step.logits.device,
            dtype=step.logits.dtype,
        )
        return P34NextTokenOutput(
            augmented_logits=step.logits,
            base_logits=step.logits,
            writeback_ratio=zeros + float(step.applied_alpha),
            position_gate=zeros + (1.0 if step.applied_alpha else 0.0),
            current_positions=positions,
            scratch_state=scratch,
            answer_token_margin=top2[:, 0] - top2[:, 1],
        )
