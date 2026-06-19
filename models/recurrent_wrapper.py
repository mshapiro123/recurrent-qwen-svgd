"""Prelude / recurrent block / coda wrapper for Qwen-style causal LMs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .bridge import IdentityGatedBridge
from .halting import (
    SequenceHaltingPredictor,
    categorical_kl,
    centered_geometric_prior,
    expected_loop_count,
    masked_mean,
    pondernet_halting_probabilities,
)
from .latent_policy import LatentTrajectoryModule
from .lora import mark_only_lora_trainable, set_lora_adapter_dtype
from .svgd import svgd_particle_update
from .trajectory_utils import (
    average_pairwise_cosine_distance,
    repeat_for_trajectories,
    unflatten_trajectories,
)


@dataclass(frozen=True)
class LayerSplit:
    """Layer partition: ``[0:prelude_end]``, ``[prelude_end:recurrent_end]``, coda."""

    prelude_end: int
    recurrent_end: int

    @classmethod
    def auto(cls, num_layers: int) -> "LayerSplit":
        if num_layers < 3:
            raise ValueError("Need at least 3 decoder layers for an automatic split")
        prelude_end = max(1, num_layers // 4)
        recurrent_end = min(num_layers - 1, max(prelude_end + 1, (3 * num_layers) // 4))
        return cls(prelude_end=prelude_end, recurrent_end=recurrent_end)

    def validate(self, num_layers: int) -> None:
        if not 0 < self.prelude_end < self.recurrent_end < num_layers:
            raise ValueError(
                "Invalid layer split. Expected 0 < prelude_end < recurrent_end < num_layers, "
                f"got prelude_end={self.prelude_end}, recurrent_end={self.recurrent_end}, "
                f"num_layers={num_layers}."
            )


@dataclass
class RecurrentQwenOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    trajectory_logits: Optional[torch.Tensor] = None
    loop_logits: Optional[torch.Tensor] = None
    halting_probs: Optional[torch.Tensor] = None
    halting_weights: Optional[torch.Tensor] = None
    expected_loops: Optional[torch.Tensor] = None
    final_recurrent_hidden: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)

    def to_tuple(self) -> tuple[Any, ...]:
        values: list[Any] = []
        if self.loss is not None:
            values.append(self.loss)
        values.append(self.logits)
        values.append(self.past_key_values)
        values.append(self.hidden_states)
        values.append(self.attentions)
        return tuple(values)


class RecurrentQwenForCausalLM(nn.Module):
    """A recurrent-depth wrapper around Hugging Face Qwen causal LMs.

    Phase 0 uses ``max_loops=1`` and ``num_trajectories=1``. The forward path
    still manually executes Prelude -> Recurrent Block -> Coda, so the identity
    eval checks the actual split rather than a shortcut through the base model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        layer_split: Optional[LayerSplit] = None,
        latent_dim: int = 256,
        initial_halt_prob: float = 0.25,
        latent_scale_init: float = 0.01,
        latent_adapter_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "model") or not hasattr(base_model.model, "layers"):
            raise TypeError("Expected a Hugging Face Qwen-style causal LM with .model.layers")
        if not hasattr(base_model, "lm_head"):
            raise TypeError("Expected the base model to expose .lm_head")

        self.base_model = base_model
        self.config = getattr(base_model, "config", None)

        num_layers = len(self.qwen.layers)
        self.layer_split = layer_split or LayerSplit.auto(num_layers)
        self.layer_split.validate(num_layers)

        hidden_size = int(
            getattr(self.config, "hidden_size", 0)
            or getattr(getattr(self.qwen, "config", None), "hidden_size", 0)
            or getattr(self.qwen.embed_tokens, "embedding_dim", 0)
        )
        if hidden_size <= 0:
            raise ValueError("Could not infer hidden size from the base model")
        self.bridge = IdentityGatedBridge(hidden_size)
        self.halt_predictor = SequenceHaltingPredictor(hidden_size, initial_halt_prob)
        self.latent_trajectory = LatentTrajectoryModule(
            hidden_size,
            latent_dim,
            latent_scale_init,
            latent_adapter_std,
        )
        self._svgd_projection_cache: dict[str, torch.Tensor] = {}
        self._align_auxiliary_modules_to_base()

    @property
    def qwen(self) -> nn.Module:
        return self.base_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.base_model.lm_head

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _align_auxiliary_modules_to_base(self) -> None:
        """Match wrapper-only modules to the base model dtype/device."""

        base_param = next(self.base_model.parameters())
        for module in (self.bridge, self.halt_predictor, self.latent_trajectory):
            module.to(device=base_param.device, dtype=base_param.dtype)

    def freeze_base_model(self) -> None:
        """Freeze Qwen weights while leaving bridge/halting/latent modules trainable."""

        for param in self.base_model.parameters():
            param.requires_grad_(False)
        mark_only_lora_trainable(self.base_model)
        for module in (self.bridge, self.halt_predictor, self.latent_trajectory):
            for param in module.parameters():
                param.requires_grad_(True)

    def set_latent_trainable(self, enabled: bool) -> None:
        for param in self.latent_trajectory.parameters():
            param.requires_grad_(enabled)

    def set_trainable_modules_dtype(self, dtype: torch.dtype) -> None:
        """Keep trainable recurrent controls/adapters in a stable optimizer dtype."""

        device = next(self.base_model.parameters()).device
        for module in (self.bridge, self.halt_predictor, self.latent_trajectory):
            module.to(device=device, dtype=dtype)
        set_lora_adapter_dtype(self.base_model, dtype)

    def trainable_component_parameters(self) -> list[nn.Parameter]:
        params = [
            param
            for param in self.base_model.parameters()
            if param.requires_grad
        ]
        params.extend(
            param
            for module in (self.bridge, self.halt_predictor, self.latent_trajectory)
            for param in module.parameters()
            if param.requires_grad
        )
        return params

    def _load_svgd_projection(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if not path:
            return None
        resolved = str(Path(path).expanduser().resolve())
        cached = self._svgd_projection_cache.get(resolved)
        if cached is not None:
            return cached

        payload = torch.load(resolved, map_location="cpu")
        projection = payload.get("projection") if isinstance(payload, dict) else payload
        if not torch.is_tensor(projection):
            raise TypeError(f"Projection file must contain a tensor or dict['projection']: {resolved}")
        projection = projection.detach().float().contiguous()
        self._svgd_projection_cache[resolved] = projection
        return projection

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        max_loops: int = 1,
        num_trajectories: int = 1,
        sample_latents: bool = False,
        beta: float = 0.0,
        eta: float = 0.0,
        rho: float = 0.0,
        latent_injection_mode: str = "pre",
        particle_update_mode: str = "none",
        particle_init_noise: float = 0.0,
        svgd_eps: float = 1.0,
        svgd_repulsion_scale: float = 1.0,
        svgd_bandwidth: str = "median",
        svgd_bandwidth_floor: float = 1e-6,
        svgd_repulsion_max_norm: Optional[float] = None,
        svgd_kernel_projection_dim: Optional[int] = None,
        svgd_kernel_projection_path: Optional[str] = None,
        svgd_kernel_geometry: str = "euclidean",
        svgd_projection_seed: int = 0,
        target_loop_counts: Optional[torch.Tensor] = None,
        target_loop_prior: Optional[torch.Tensor] = None,
        return_loop_logits: bool = False,
        force_base_model: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **_: Any,
    ) -> RecurrentQwenOutput | tuple[Any, ...]:
        trajectory_batch_size = None
        inputs_are_trajectories = False
        if input_ids is not None and input_ids.dim() == 3:
            trajectory_batch_size, input_trajectories, seq_len = input_ids.shape
            if input_trajectories != num_trajectories:
                raise ValueError(
                    "3D input_ids must be shaped [batch, num_trajectories, seq_len]. "
                    f"Got num_trajectories={num_trajectories}, input shape={tuple(input_ids.shape)}."
                )
            input_ids = input_ids.reshape(trajectory_batch_size * input_trajectories, seq_len)
            attention_mask = self._flatten_optional_trajectory_tensor(attention_mask, trajectory_batch_size, input_trajectories)
            position_ids = self._flatten_optional_trajectory_tensor(position_ids, trajectory_batch_size, input_trajectories)
            labels = self._flatten_optional_trajectory_tensor(labels, trajectory_batch_size, input_trajectories)
            inputs_are_trajectories = True
        elif inputs_embeds is not None and inputs_embeds.dim() == 4:
            trajectory_batch_size, input_trajectories, seq_len, hidden = inputs_embeds.shape
            if input_trajectories != num_trajectories:
                raise ValueError(
                    "4D inputs_embeds must be shaped [batch, num_trajectories, seq_len, hidden]. "
                    f"Got num_trajectories={num_trajectories}, inputs_embeds shape={tuple(inputs_embeds.shape)}."
                )
            inputs_embeds = inputs_embeds.reshape(trajectory_batch_size * input_trajectories, seq_len, hidden)
            attention_mask = self._flatten_optional_trajectory_tensor(attention_mask, trajectory_batch_size, input_trajectories)
            position_ids = self._flatten_optional_trajectory_tensor(position_ids, trajectory_batch_size, input_trajectories)
            labels = self._flatten_optional_trajectory_tensor(labels, trajectory_batch_size, input_trajectories)
            inputs_are_trajectories = True

        if force_base_model:
            return self._call_with_supported_kwargs(
                self.base_model.forward,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "inputs_embeds": inputs_embeds,
                    "labels": labels,
                    "use_cache": use_cache,
                    "output_attentions": output_attentions,
                    "output_hidden_states": output_hidden_states,
                    "return_dict": return_dict,
                    "cache_position": cache_position,
                },
            )

        if max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be >= 1")
        if latent_injection_mode not in {"pre", "post", "both"}:
            raise ValueError("latent_injection_mode must be one of: pre, post, both")
        if particle_update_mode not in {"none", "svgd"}:
            raise ValueError("particle_update_mode must be one of: none, svgd")

        use_cache_was_explicit = use_cache is not None
        return_dict = self._default_return_dict(return_dict)
        output_attentions = self._default_config_bool(output_attentions, "output_attentions")
        output_hidden_states = self._default_config_bool(output_hidden_states, "output_hidden_states")
        use_cache = self._default_config_bool(use_cache, "use_cache")
        recurrent_cache_invalid = (
            max_loops > 1
            or num_trajectories > 1
            or sample_latents
            or particle_update_mode != "none"
        )
        if recurrent_cache_invalid and not use_cache_was_explicit:
            use_cache = False

        if use_cache and recurrent_cache_invalid:
            raise ValueError(
                "KV cache is only supported for the identity-shaped single-pass path. "
                "Set use_cache=False for recurrent loops or multi-trajectory sampling."
            )

        if self.training and getattr(self.qwen, "gradient_checkpointing", False) and use_cache:
            use_cache = False

        prepared = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )
        hidden_states = prepared["inputs_embeds"]
        attention_mask = prepared["attention_mask"]
        position_ids = prepared["position_ids"]
        cache_position = prepared["cache_position"]

        hidden_history: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []
        if output_hidden_states:
            hidden_history.append(hidden_states)

        causal_mask = self._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            past_key_values,
            output_attentions,
        )
        position_embeddings = self._rotary_embeddings(hidden_states, position_ids)

        hidden_states, attentions = self._run_layer_range(
            start=0,
            end=self.layer_split.prelude_end,
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            collect_hidden=output_hidden_states,
            hidden_history=hidden_history,
        )
        all_attentions.extend(attentions)

        batch_size = trajectory_batch_size or hidden_states.shape[0]
        flat_attention_mask = attention_mask
        flat_position_ids = position_ids
        flat_causal_mask = causal_mask
        flat_position_embeddings = position_embeddings
        labels_flat = labels

        if num_trajectories > 1 and not inputs_are_trajectories:
            hidden_states = repeat_for_trajectories(hidden_states, num_trajectories)
            if particle_update_mode == "svgd" and particle_init_noise:
                hidden_states = hidden_states + float(particle_init_noise) * torch.randn_like(hidden_states)
            flat_attention_mask = repeat_for_trajectories(attention_mask, num_trajectories)
            flat_position_ids = repeat_for_trajectories(position_ids, num_trajectories)
            labels_flat = repeat_for_trajectories(labels, num_trajectories)
            flat_causal_mask = self._update_causal_mask(
                flat_attention_mask,
                hidden_states,
                cache_position,
                None,
                output_attentions,
            )
            flat_position_embeddings = self._rotary_embeddings(hidden_states, flat_position_ids)
        elif inputs_are_trajectories and particle_update_mode == "svgd" and particle_init_noise:
            hidden_states = hidden_states + float(particle_init_noise) * torch.randn_like(hidden_states)

        loop_logits: list[torch.Tensor] = []
        per_loop_ce: list[torch.Tensor] = []
        halt_probs: list[torch.Tensor] = []
        latent_kls: list[torch.Tensor] = []
        svgd_stats_history: list[Any] = []
        recurrent_state = hidden_states

        for loop_idx in range(max_loops):
            loop_input = recurrent_state
            if loop_idx > 0:
                loop_input = self.bridge(loop_input)
            if sample_latents and latent_injection_mode in {"pre", "both"}:
                loop_input, latent_stats = self.latent_trajectory(
                    loop_input,
                    flat_attention_mask,
                    sample=True,
                )
                latent_kls.append(latent_stats.kl)

            recurrent_state, attentions = self._run_layer_range(
                start=self.layer_split.prelude_end,
                end=self.layer_split.recurrent_end,
                hidden_states=loop_input,
                causal_mask=flat_causal_mask,
                position_ids=flat_position_ids,
                past_key_values=past_key_values if num_trajectories == 1 else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=flat_position_embeddings,
                collect_hidden=output_hidden_states and num_trajectories == 1,
                hidden_history=hidden_history if num_trajectories == 1 else None,
            )
            all_attentions.extend(attentions)

            if particle_update_mode == "svgd":
                recurrent_state, svgd_stats = svgd_particle_update(
                    previous_state=loop_input,
                    standard_state=recurrent_state,
                    attention_mask=flat_attention_mask,
                    num_particles=num_trajectories,
                    eps=svgd_eps,
                    repulsion_scale=svgd_repulsion_scale,
                    bandwidth=svgd_bandwidth,
                    bandwidth_floor=svgd_bandwidth_floor,
                    repulsion_max_norm=svgd_repulsion_max_norm,
                    kernel_projection_dim=svgd_kernel_projection_dim,
                    kernel_projection=self._load_svgd_projection(svgd_kernel_projection_path),
                    kernel_geometry=svgd_kernel_geometry,
                    projection_seed=svgd_projection_seed,
                )
                svgd_stats_history.append(svgd_stats)

            if sample_latents and latent_injection_mode in {"post", "both"}:
                recurrent_state, latent_stats = self.latent_trajectory(
                    recurrent_state,
                    flat_attention_mask,
                    sample=True,
                )
                latent_kls.append(latent_stats.kl)

            pooled = masked_mean(recurrent_state, flat_attention_mask)
            halt_probs.append(self.halt_predictor(pooled).squeeze(-1))

            coda_hidden, attentions = self._run_layer_range(
                start=self.layer_split.recurrent_end,
                end=len(self.qwen.layers),
                hidden_states=recurrent_state,
                causal_mask=flat_causal_mask,
                position_ids=flat_position_ids,
                past_key_values=past_key_values if num_trajectories == 1 else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=flat_position_embeddings,
                collect_hidden=output_hidden_states and max_loops == 1 and num_trajectories == 1,
                hidden_history=hidden_history if max_loops == 1 and num_trajectories == 1 else None,
            )
            all_attentions.extend(attentions)

            normed = self.qwen.norm(coda_hidden)
            logits = self.lm_head(self._slice_for_logits(normed, logits_to_keep))
            loop_logits.append(logits)
            if labels_flat is not None:
                if not self._keeps_full_logits(logits_to_keep):
                    raise ValueError("labels require full-sequence logits; set logits_to_keep=0")
                per_loop_ce.append(self._sequence_cross_entropy(logits.float(), labels_flat))

        halt_probs_tensor = torch.stack(halt_probs, dim=-1)
        halting_weights = pondernet_halting_probabilities(halt_probs_tensor)
        logits_stack = torch.stack(loop_logits, dim=1)
        weighted_logits = (halting_weights[:, :, None, None] * logits_stack).sum(dim=1)

        trajectory_logits = unflatten_trajectories(weighted_logits, batch_size, num_trajectories)
        output_logits = trajectory_logits.mean(dim=1) if num_trajectories > 1 else weighted_logits

        final_hidden = unflatten_trajectories(recurrent_state, batch_size, num_trajectories)
        halting_probs_by_traj = unflatten_trajectories(halt_probs_tensor, batch_size, num_trajectories)
        halting_weights_by_traj = unflatten_trajectories(halting_weights, batch_size, num_trajectories)
        expected_loops_by_traj = expected_loop_count(halting_weights_by_traj)

        metrics: dict[str, torch.Tensor] = {
            "mean_expected_loops": expected_loops_by_traj.mean().detach(),
            "mean_halt_entropy": self._entropy(halting_weights).mean().detach(),
        }

        diversity = None
        if num_trajectories > 1:
            pooled_final = self._pooled_final_by_trajectory(final_hidden, attention_mask)
            diversity = average_pairwise_cosine_distance(pooled_final)
            metrics["trajectory_diversity"] = diversity.detach()
        if svgd_stats_history:
            metrics.update(
                {
                    "svgd_bandwidth": torch.stack([item.bandwidth for item in svgd_stats_history]).mean().detach(),
                    "svgd_pairwise_distance": torch.stack(
                        [item.mean_pairwise_distance for item in svgd_stats_history]
                    ).mean().detach(),
                    "svgd_drift_rms": torch.stack([item.drift_rms for item in svgd_stats_history]).mean().detach(),
                    "svgd_repulsion_rms_pre_clip": torch.stack(
                        [item.repulsion_rms_pre_clip for item in svgd_stats_history]
                    ).mean().detach(),
                    "svgd_repulsion_rms": torch.stack(
                        [item.repulsion_rms for item in svgd_stats_history]
                    ).mean().detach(),
                    "svgd_repulsion_clip_fraction": torch.stack(
                        [item.repulsion_clip_fraction for item in svgd_stats_history]
                    ).mean().detach(),
                    "svgd_velocity_rms": torch.stack([item.velocity_rms for item in svgd_stats_history]).mean().detach(),
                }
            )

        loss = None
        if labels_flat is not None:
            ce_tensor = torch.stack(per_loop_ce, dim=-1)
            loss = (ce_tensor * halting_weights).sum(dim=-1).mean()
            metrics["expected_ce"] = loss.detach()

            prior = self._resolve_target_prior(
                target_loop_counts=target_loop_counts,
                target_loop_prior=target_loop_prior,
                batch_size=batch_size,
                num_trajectories=num_trajectories,
                max_loops=max_loops,
                device=halting_weights.device,
            )
            if prior is not None and beta:
                halt_kl = categorical_kl(halting_weights, prior).mean()
                loss = loss + float(beta) * halt_kl
                metrics["halting_kl"] = halt_kl.detach()

            if latent_kls and eta:
                latent_kl = torch.stack(latent_kls, dim=-1).mean()
                loss = loss + float(eta) * latent_kl
                metrics["latent_kl"] = latent_kl.detach()

            if diversity is not None and rho:
                loss = loss - float(rho) * diversity

            metrics["loss"] = loss.detach()

        output = RecurrentQwenOutput(
            loss=loss,
            logits=output_logits,
            trajectory_logits=trajectory_logits if num_trajectories > 1 else None,
            loop_logits=(
                unflatten_trajectories(logits_stack, batch_size, num_trajectories)
                if return_loop_logits
                else None
            ),
            halting_probs=halting_probs_by_traj,
            halting_weights=halting_weights_by_traj,
            expected_loops=expected_loops_by_traj,
            final_recurrent_hidden=final_hidden,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(hidden_history) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
            metrics=metrics,
        )
        return output if return_dict else output.to_tuple()

    @staticmethod
    def _flatten_optional_trajectory_tensor(
        tensor: Optional[torch.Tensor],
        batch_size: int,
        num_trajectories: int,
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.dim() < 3:
            return tensor
        if tensor.shape[0] != batch_size or tensor.shape[1] != num_trajectories:
            raise ValueError(
                "Trajectory tensor has incompatible leading dimensions. "
                f"Expected [{batch_size}, {num_trajectories}, ...], got {tuple(tensor.shape)}."
            )
        return tensor.reshape(batch_size * num_trajectories, *tensor.shape[2:])

    def _default_return_dict(self, value: Optional[bool]) -> bool:
        if value is not None:
            return value
        return bool(getattr(self.config, "use_return_dict", True))

    def _default_config_bool(self, value: Optional[bool], name: str) -> bool:
        if value is not None:
            return bool(value)
        return bool(getattr(self.config, name, False))

    def _prepare_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Any],
        inputs_embeds: Optional[torch.Tensor],
        cache_position: Optional[torch.LongTensor],
    ) -> dict[str, torch.Tensor]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.qwen.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)

        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = int(past_key_values.get_seq_length())
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                device=device,
                dtype=torch.long,
            )
        else:
            cache_position = cache_position.to(device=device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(device=device)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Any],
        output_attentions: bool,
    ) -> Optional[torch.Tensor]:
        update_fn = getattr(self.qwen, "_update_causal_mask", None)
        if update_fn is not None:
            kwargs = {
                "attention_mask": attention_mask,
                "input_tensor": inputs_embeds,
                "inputs_embeds": inputs_embeds,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "output_attentions": output_attentions,
            }
            return self._call_with_supported_kwargs(update_fn, kwargs)

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        return self._fallback_causal_mask(attention_mask, inputs_embeds)

    def _fallback_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = inputs_embeds.shape[:2]
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        causal = torch.full(
            (seq_len, seq_len),
            min_dtype,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        causal = torch.triu(causal, diagonal=1)
        causal_mask = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        if attention_mask is not None:
            padding_mask = attention_mask[:, None, None, :].eq(0)
            causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
        return causal_mask

    def _rotary_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> Optional[Any]:
        rotary = getattr(self.qwen, "rotary_emb", None)
        if rotary is None or position_ids is None:
            return None
        try:
            return rotary(hidden_states, position_ids)
        except TypeError:
            return None

    def _run_layer_range(
        self,
        start: int,
        end: int,
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        output_attentions: bool,
        cache_position: Optional[torch.Tensor],
        position_embeddings: Optional[Any],
        collect_hidden: bool,
        hidden_history: Optional[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attentions: list[torch.Tensor] = []
        for layer in self.qwen.layers[start:end]:
            layer_outputs = self._run_decoder_layer(
                layer=layer,
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if collect_hidden and hidden_history is not None:
                hidden_history.append(hidden_states)
            if output_attentions and len(layer_outputs) > 1 and torch.is_tensor(layer_outputs[1]):
                attentions.append(layer_outputs[1])
        return hidden_states, attentions

    def _run_decoder_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        output_attentions: bool,
        cache_position: Optional[torch.Tensor],
        position_embeddings: Optional[Any],
    ) -> tuple[Any, ...]:
        kwargs = {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_values,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }

        if self.training and getattr(self.qwen, "gradient_checkpointing", False):
            def custom_forward(states: torch.Tensor) -> tuple[Any, ...]:
                return self._call_layer(layer, states, kwargs)

            checkpoint_fn = getattr(self.qwen, "_gradient_checkpointing_func", None)
            if checkpoint_fn is not None:
                outputs = checkpoint_fn(custom_forward, hidden_states)
            else:
                outputs = checkpoint(custom_forward, hidden_states, use_reentrant=False)
            return outputs if isinstance(outputs, tuple) else (outputs,)

        return self._call_layer(layer, hidden_states, kwargs)

    def _call_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        supported = self._filter_supported_kwargs(layer.forward, kwargs)
        outputs = layer(hidden_states, **supported)
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def _call_with_supported_kwargs(self, fn: Any, kwargs: dict[str, Any]) -> Any:
        supported = self._filter_supported_kwargs(fn, kwargs)
        try:
            return fn(**supported)
        except TypeError:
            ordered = [
                kwargs.get("attention_mask"),
                kwargs.get("inputs_embeds", kwargs.get("input_tensor")),
                kwargs.get("cache_position"),
                kwargs.get("past_key_values"),
                kwargs.get("output_attentions"),
            ]
            return fn(*ordered)

    @staticmethod
    def _filter_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(fn)
        params = signature.parameters
        accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        if accepts_var_kwargs:
            return {key: value for key, value in kwargs.items() if value is not None}
        return {
            key: value
            for key, value in kwargs.items()
            if key in params and (value is not None or key in {"use_cache", "output_attentions"})
        }

    @staticmethod
    def _slice_for_logits(hidden_states: torch.Tensor, logits_to_keep: int | torch.Tensor) -> torch.Tensor:
        if RecurrentQwenForCausalLM._keeps_full_logits(logits_to_keep):
            return hidden_states
        if isinstance(logits_to_keep, int):
            return hidden_states[:, -logits_to_keep:, :]
        return hidden_states[:, logits_to_keep, :]

    @staticmethod
    def _keeps_full_logits(logits_to_keep: int | torch.Tensor | None) -> bool:
        return logits_to_keep is None or (isinstance(logits_to_keep, int) and logits_to_keep == 0)

    @staticmethod
    def _sequence_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape[0], -1)
        valid = shift_labels.ne(-100).to(dtype=flat_loss.dtype)
        return (flat_loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)

    def _resolve_target_prior(
        self,
        target_loop_counts: Optional[torch.Tensor],
        target_loop_prior: Optional[torch.Tensor],
        batch_size: int,
        num_trajectories: int,
        max_loops: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if target_loop_prior is not None:
            prior = target_loop_prior.to(device=device)
            if prior.dim() == 1:
                prior = prior.unsqueeze(0).expand(batch_size * num_trajectories, -1)
            elif prior.shape[0] == batch_size and num_trajectories > 1:
                prior = prior.repeat_interleave(num_trajectories, dim=0)
            return prior
        if target_loop_counts is None:
            return None
        counts = target_loop_counts.to(device=device)
        if counts.dim() == 0:
            counts = counts.expand(batch_size)
        if counts.shape[0] == batch_size and num_trajectories > 1:
            counts = counts.repeat_interleave(num_trajectories, dim=0)
        return centered_geometric_prior(counts, max_loops)

    @staticmethod
    def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        safe = probs.clamp_min(eps)
        return -(safe * safe.log()).sum(dim=-1)

    @staticmethod
    def _pooled_final_by_trajectory(
        final_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, num_trajectories = final_hidden.shape[:2]
        flat_hidden = final_hidden.view(batch_size * num_trajectories, *final_hidden.shape[2:])
        flat_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                if attention_mask.shape[:2] != (batch_size, num_trajectories):
                    raise ValueError(
                        "3D trajectory attention mask must be shaped [batch, num_trajectories, seq_len]. "
                        f"Got {tuple(attention_mask.shape)}."
                    )
                flat_mask = attention_mask.reshape(batch_size * num_trajectories, attention_mask.shape[-1])
            elif attention_mask.dim() == 2 and attention_mask.shape[0] == batch_size:
                flat_mask = repeat_for_trajectories(attention_mask, num_trajectories)
            elif attention_mask.dim() == 2 and attention_mask.shape[0] == batch_size * num_trajectories:
                flat_mask = attention_mask
            else:
                raise ValueError(
                    "attention_mask must be batch-level or trajectory-level. "
                    f"Got shape {tuple(attention_mask.shape)}."
                )
        pooled = masked_mean(flat_hidden, flat_mask)
        return pooled.view(batch_size, num_trajectories, -1)
