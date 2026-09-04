"""Governed CPU runner for the WEFT-1 ECA-LATENT PRE-FLIGHT instrument.

The delivered ``eca_latent_loop_20260903.py`` remains an immutable
demonstrator.  This module implements the ratified D-EP-2 extension as a
restartable 72-cell campaign.  It deliberately emits the raw ridge-probe
matrix and does not invent an executor/compiler classification threshold.

Random sources follow the O-9 SHA-256 derivation contract.  Within a
``(rule, tau, replica)`` comparison, all K arms see the same examples, model
initialization, and minibatch order.  Each completed cell is written
atomically and validated by identity before it may be reused on resume.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn

from analysis.weft1_epiplexity import EvalLossInterval, build_prequential_area_receipt
from models.ablation_lm.rng import construct_with_isolated_rng
from training.weft1_seed import derive_module_seed


ECA_RULES = (30, 54, 110)
ECA_HORIZONS = (4, 8, 16)
ECA_REPLICAS = (0, 1)
ECA_ROOT_SEED = 20_260_903
ECA_SCHEMA_VERSION = "weft1.eca_latent.v2"
ECA_SELF_HASH_FIELD = "canonical_self_sha256"

DELIVERED_AUTHORITIES: tuple[tuple[str, int, str], ...] = (
    (
        "docs/Claude outputs/eca_latent_loop_20260903.py",
        4_137,
        "457045eef6c5eaabde7f9d0571e62caed4cb1b0dc944ac57ece0343a512e9ee1",
    ),
    (
        "docs/Claude outputs/STRATEGY_EPIPLEXITY_ADJUDICATION_20260903.md",
        24_723,
        "7b5786185aebdd2388fa01a8db5522a2ce58e09a5c027947807ac0a2aebd3730",
    ),
    (
        "docs/Claude outputs/STRATEGY_EP_RATIFICATION_20260903.md",
        3_848,
        "c9ecfa58f57c6904380f3edd895e92aada5efb3da77735c4d4e3d288197730ee",
    ),
)


class ECAReceiptIdentityError(RuntimeError):
    """A durable campaign artifact does not match the requested experiment."""


@dataclass(frozen=True, order=True)
class ECACellSpec:
    rule: int
    tau: int
    k: int
    replica: int

    def __post_init__(self) -> None:
        if self.rule not in ECA_RULES:
            raise ValueError(f"rule must be one of {ECA_RULES}")
        if self.tau not in ECA_HORIZONS:
            raise ValueError(f"tau must be one of {ECA_HORIZONS}")
        if self.k not in k_values(self.tau):
            raise ValueError(f"K must be one of {k_values(self.tau)} for tau={self.tau}")
        if self.replica not in ECA_REPLICAS:
            raise ValueError(f"replica must be one of {ECA_REPLICAS}")

    @property
    def filename(self) -> str:
        return (
            f"cell_rule{self.rule}_tau{self.tau}_k{self.k}_"
            f"replica{self.replica}.json"
        )


@dataclass(frozen=True)
class ECARunProfile:
    name: str
    steps: int
    train_examples: int
    eval_examples: int
    probe_fit_examples: int
    probe_eval_examples: int
    n_cells: int
    hidden_width: int
    batch_size: int
    eval_every_steps: int
    learning_rate: float
    weight_decay: float
    ridge: float
    curriculum_boundaries: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("profile name must be a non-empty string")
        for name in (
            "steps",
            "train_examples",
            "eval_examples",
            "probe_fit_examples",
            "probe_eval_examples",
            "n_cells",
            "hidden_width",
            "batch_size",
            "eval_every_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.n_cells < 2 * max(ECA_HORIZONS):
            raise ValueError("n_cells must cover the largest registered ECA radius")
        for name in ("learning_rate", "ridge"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        b1, b2, b3 = self.curriculum_boundaries
        if not (0 < b1 < b2 < b3 <= self.steps):
            raise ValueError("curriculum boundaries must increase within the run")


FULL_PROFILE = ECARunProfile(
    name="registered_full",
    steps=1_200,
    train_examples=20_000,
    eval_examples=2_000,
    probe_fit_examples=1_000,
    probe_eval_examples=1_000,
    n_cells=32,
    hidden_width=64,
    batch_size=256,
    eval_every_steps=25,
    learning_rate=2e-3,
    weight_decay=0.01,
    ridge=1e-2,
    curriculum_boundaries=(200, 400, 700),
)

SMOKE_PROFILE = ECARunProfile(
    name="tiny_smoke_not_scientific",
    steps=4,
    train_examples=32,
    eval_examples=16,
    probe_fit_examples=16,
    probe_eval_examples=16,
    n_cells=32,
    hidden_width=8,
    batch_size=8,
    eval_every_steps=1,
    learning_rate=2e-3,
    weight_decay=0.01,
    ridge=1e-2,
    curriculum_boundaries=(1, 2, 3),
)


def k_values(tau: int) -> tuple[int, int, int, int]:
    if tau not in ECA_HORIZONS:
        raise ValueError(f"tau must be one of {ECA_HORIZONS}")
    return (1, tau // 2, tau, 2 * tau)


def registered_grid() -> tuple[ECACellSpec, ...]:
    """Return the exact 3 rules x 3 horizons x 4 depths x 2 replicas grid."""

    return tuple(
        ECACellSpec(rule=rule, tau=tau, k=k, replica=replica)
        for rule in ECA_RULES
        for tau in ECA_HORIZONS
        for k in k_values(tau)
        for replica in ECA_REPLICAS
    )


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _with_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if ECA_SELF_HASH_FIELD in payload:
        raise ValueError(f"payload already contains {ECA_SELF_HASH_FIELD}")
    result = dict(payload)
    result[ECA_SELF_HASH_FIELD] = _sha256_bytes(_canonical_json_bytes(result))
    return result


def _validate_self_hash(payload: Mapping[str, Any], *, artifact: str) -> None:
    observed = payload.get(ECA_SELF_HASH_FIELD)
    if type(observed) is not str or len(observed) != 64:
        raise ECAReceiptIdentityError(f"{artifact} has no canonical self-hash")
    unhashed = {key: value for key, value in payload.items() if key != ECA_SELF_HASH_FIELD}
    expected = _sha256_bytes(_canonical_json_bytes(unhashed))
    if observed != expected:
        raise ECAReceiptIdentityError(f"{artifact} canonical self-hash mismatch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def authority_receipts(repo_root: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Verify and return byte-level identities for every governing source."""

    root = _repo_root() if repo_root is None else Path(repo_root).resolve()
    receipts: list[dict[str, Any]] = []
    for relative, expected_bytes, expected_sha in DELIVERED_AUTHORITIES:
        path = root / relative
        if not path.is_file():
            raise ECAReceiptIdentityError(f"missing ECA-LATENT authority: {relative}")
        observed_bytes = path.stat().st_size
        observed_sha = _sha256_file(path)
        if observed_bytes != expected_bytes or observed_sha != expected_sha:
            raise ECAReceiptIdentityError(
                f"ECA-LATENT authority mismatch for {relative}: "
                f"expected {expected_bytes}/{expected_sha}, "
                f"observed {observed_bytes}/{observed_sha}"
            )
        receipts.append(
            {
                "path": relative,
                "bytes": observed_bytes,
                "sha256": observed_sha,
            }
        )
    return tuple(receipts)


def eca_step(x: torch.Tensor, rule: int) -> torch.Tensor:
    """Apply one Wolfram-convention ECA step on the last-axis ring."""

    if type(x) is not torch.Tensor or x.ndim != 2:
        raise TypeError("x must be an exact rank-2 tensor [example, cell]")
    if type(rule) is not int or not 0 <= rule <= 255:
        raise ValueError("rule must be an exact integer in [0, 255]")
    if x.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("ECA states must use an integer tensor dtype")
    if bool(((x != 0) & (x != 1)).any()):
        raise ValueError("ECA states must be binary")
    left = torch.roll(x, 1, dims=-1)
    right = torch.roll(x, -1, dims=-1)
    index = (left * 4 + x * 2 + right).long()
    table = torch.tensor(
        [(rule >> bit) & 1 for bit in range(8)],
        dtype=x.dtype,
        device=x.device,
    )
    return table[index]


def eca_states(x: torch.Tensor, rule: int, tau: int) -> tuple[torch.Tensor, ...]:
    if type(tau) is not int or tau < 1:
        raise ValueError("tau must be a positive exact integer")
    states = [x]
    for _ in range(tau):
        states.append(eca_step(states[-1], rule))
    return tuple(states)


class LatentECACore(nn.Module):
    """One weight-tied residual block with full-horizon input visibility."""

    def __init__(self, hidden_width: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.kernel_size = kernel_size
        self.embedding = nn.Linear(1, hidden_width)
        self.convolution = nn.Conv1d(
            hidden_width,
            hidden_width,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_width, 2 * hidden_width),
            nn.GELU(),
            nn.Linear(2 * hidden_width, hidden_width),
        )
        self.norm1 = nn.LayerNorm(hidden_width)
        self.norm2 = nn.LayerNorm(hidden_width)
        self.readout = nn.Linear(hidden_width, 1)

    def block(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.convolution(
            self.norm1(hidden).transpose(1, 2)
        ).transpose(1, 2)
        return hidden + self.mlp(self.norm2(hidden))

    def forward(
        self,
        x: torch.Tensor,
        k: int,
        *,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if type(k) is not int or k < 1:
            raise ValueError("K must be a positive exact integer")
        hidden = self.embedding(x.float().unsqueeze(-1))
        visits: list[torch.Tensor] = []
        for _ in range(k):
            hidden = self.block(hidden)
            if return_hidden:
                visits.append(hidden)
        logits = self.readout(hidden).squeeze(-1)
        return logits, tuple(visits)


def _source_keys(spec: ECACellSpec) -> dict[str, str]:
    return {
        "data_train": "weft.preflight.eca_latent.data.train",
        "data_eval": "weft.preflight.eca_latent.data.eval",
        "data_probe_fit": "weft.preflight.eca_latent.data.probe_fit",
        "data_probe_eval": "weft.preflight.eca_latent.data.probe_eval",
        "model_init": f"weft.preflight.eca_latent.model.{spec.tau}",
        "train_order": (
            f"weft.preflight.eca_latent.train_order.{spec.rule}.{spec.tau}"
        ),
    }


def derived_seed_receipt(
    spec: ECACellSpec,
    *,
    root_seed: int = ECA_ROOT_SEED,
) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "source_key": source_key,
            "seed": derive_module_seed(root_seed, source_key, spec.replica),
        }
        for name, source_key in _source_keys(spec).items()
    }


def curriculum_enabled(k: int) -> bool:
    return k >= 4


def executed_k_at_step(
    target_k: int,
    step_index: int,
    boundaries: tuple[int, int, int],
) -> int:
    """Return the 1 -> 2 -> 4 -> K curriculum depth for a zero-based step."""

    if type(target_k) is not int or target_k < 1:
        raise ValueError("target_k must be a positive exact integer")
    if type(step_index) is not int or step_index < 0:
        raise ValueError("step_index must be a non-negative exact integer")
    if not curriculum_enabled(target_k):
        return target_k
    b1, b2, b3 = boundaries
    if step_index < b1:
        return 1
    if step_index < b2:
        return 2
    if step_index < b3:
        return 4
    return target_k


def _binary_examples(count: int, n_cells: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        0,
        2,
        (count, n_cells),
        dtype=torch.uint8,
        generator=generator,
    )


def _tensor_identity(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    )
    return _sha256_bytes(header + value.numpy().tobytes(order="C"))


def model_initial_state_sha256(model: nn.Module) -> str:
    """Hash every named initial tensor, including its name, dtype, and shape."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(
            _canonical_json_bytes(
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            )
        )
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _new_minibatch_order_digest(
    spec: ECACellSpec,
    profile: ECARunProfile,
    seed: int,
) -> Any:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "source_key": _source_keys(spec)["train_order"],
                "seed": seed,
                "steps": profile.steps,
                "batch_size": profile.batch_size,
                "train_examples": profile.train_examples,
                "dtype": "torch.int64",
            }
        )
    )
    return digest


def minibatch_order_sha256(
    spec: ECACellSpec,
    profile: ECARunProfile,
    *,
    root_seed: int = ECA_ROOT_SEED,
) -> str:
    """Replay and hash the complete minibatch-index stream for one cell."""

    seed = int(derived_seed_receipt(spec, root_seed=root_seed)["train_order"]["seed"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    digest = _new_minibatch_order_digest(spec, profile, seed)
    for _ in range(profile.steps):
        indices = torch.randint(
            0,
            profile.train_examples,
            (profile.batch_size,),
            generator=generator,
        )
        digest.update(indices.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _build_model(
    spec: ECACellSpec,
    profile: ECARunProfile,
    *,
    root_seed: int,
) -> LatentECACore:
    key = _source_keys(spec)["model_init"]
    model = construct_with_isolated_rng(
        lambda: LatentECACore(profile.hidden_width, 2 * spec.tau + 1),
        base_seed=root_seed,
        source_key=key,
        replica=spec.replica,
    )
    if model.kernel_size != 2 * spec.tau + 1:
        raise AssertionError("the ECA model kernel must expose exactly the tau radius")
    return model


@contextmanager
def _deterministic_cpu() -> Iterator[None]:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def _evaluate(
    model: LatentECACore,
    x: torch.Tensor,
    target: torch.Tensor,
    k: int,
) -> tuple[float, float]:
    with torch.no_grad():
        logits, _ = model(x, k)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target.float())
        bpc = float(loss.item() / math.log(2.0))
        accuracy = float(((logits > 0) == target.bool()).float().mean().item())
    return bpc, accuracy


def _probe_targets(states: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(states) < 2:
        raise ValueError("probe states must include F^0 and at least F^1")
    return torch.stack(tuple(states[1:]), dim=-1).reshape(-1, len(states) - 1)


def ridge_probe_matrix(
    fit_hidden: Sequence[torch.Tensor],
    fit_states: Sequence[torch.Tensor],
    eval_hidden: Sequence[torch.Tensor],
    eval_states: Sequence[torch.Tensor],
    *,
    ridge: float,
) -> tuple[tuple[float, ...], ...]:
    """Fit on one split and score all P[j, i] cells on another split."""

    fit_visits = tuple(fit_hidden)
    eval_visits = tuple(eval_hidden)
    if not fit_visits or len(fit_visits) != len(eval_visits):
        raise ValueError("fit/eval hidden traces must have the same positive K")
    fit_targets = _probe_targets(fit_states)
    eval_targets = _probe_targets(eval_states)
    tau = fit_targets.shape[1]
    if eval_targets.shape[1] != tau:
        raise ValueError("fit/eval probe targets must have the same horizon")
    if not math.isfinite(float(ridge)) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    signed_fit = 2.0 * fit_targets.double() - 1.0
    rows: list[tuple[float, ...]] = []
    for fit_h, eval_h in zip(fit_visits, eval_visits, strict=True):
        if fit_h.ndim != 3 or eval_h.ndim != 3:
            raise ValueError("probe hidden states must be [example, cell, hidden]")
        fit_design = fit_h.detach().reshape(-1, fit_h.shape[-1]).double()
        eval_design = eval_h.detach().reshape(-1, eval_h.shape[-1]).double()
        if fit_design.shape[0] != fit_targets.shape[0]:
            raise ValueError("fit hidden states and targets are not aligned")
        if eval_design.shape[0] != eval_targets.shape[0]:
            raise ValueError("eval hidden states and targets are not aligned")
        fit_design = torch.cat(
            (fit_design, torch.ones(fit_design.shape[0], 1, dtype=torch.float64)),
            dim=1,
        )
        eval_design = torch.cat(
            (eval_design, torch.ones(eval_design.shape[0], 1, dtype=torch.float64)),
            dim=1,
        )
        gram = fit_design.T @ fit_design
        gram.diagonal().add_(float(ridge))
        weights = torch.linalg.solve(gram, fit_design.T @ signed_fit)
        predicted = eval_design @ weights > 0.0
        accuracy = (predicted == eval_targets.bool()).double().mean(dim=0)
        rows.append(tuple(float(value) for value in accuracy.tolist()))
    return tuple(rows)


def _full_probe_matrix(
    model: LatentECACore,
    spec: ECACellSpec,
    profile: ECARunProfile,
    fit_states: Sequence[torch.Tensor],
    eval_states: Sequence[torch.Tensor],
) -> tuple[tuple[float, ...], ...]:
    with torch.no_grad():
        _, fit_hidden = model(fit_states[0], spec.k, return_hidden=True)
        # Fit before materializing the independent evaluation trace so the
        # largest K=32 cell does not retain two full hidden-state banks.
        fit_targets = _probe_targets(fit_states)
        signed_fit = 2.0 * fit_targets.double() - 1.0
        weights: list[torch.Tensor] = []
        for hidden in fit_hidden:
            design = hidden.reshape(-1, hidden.shape[-1]).double()
            design = torch.cat(
                (design, torch.ones(design.shape[0], 1, dtype=torch.float64)),
                dim=1,
            )
            gram = design.T @ design
            gram.diagonal().add_(profile.ridge)
            weights.append(torch.linalg.solve(gram, design.T @ signed_fit))
        del fit_hidden

        _, eval_hidden = model(eval_states[0], spec.k, return_hidden=True)
        eval_targets = _probe_targets(eval_states).bool()
        rows: list[tuple[float, ...]] = []
        for hidden, weight in zip(eval_hidden, weights, strict=True):
            design = hidden.reshape(-1, hidden.shape[-1]).double()
            design = torch.cat(
                (design, torch.ones(design.shape[0], 1, dtype=torch.float64)),
                dim=1,
            )
            accuracy = ((design @ weight > 0.0) == eval_targets).double().mean(dim=0)
            rows.append(tuple(float(value) for value in accuracy.tolist()))
    return tuple(rows)


def _runner_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "torch_threads": 1,
        "deterministic_algorithms": True,
    }


def _cell_identity_payload(
    spec: ECACellSpec,
    profile: ECARunProfile,
    *,
    root_seed: int,
    authorities: Sequence[Mapping[str, Any]],
    runner_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ECA_SCHEMA_VERSION,
        "instrument": "ECA-LATENT",
        "cell": asdict(spec),
        "profile": asdict(profile),
        "root_seed": root_seed,
        "authorities": list(authorities),
        "runner_sha256": runner_sha256,
        "runtime": _runtime_identity(),
    }


def cell_identity_sha256(
    spec: ECACellSpec,
    profile: ECARunProfile,
    *,
    root_seed: int = ECA_ROOT_SEED,
    authorities: Sequence[Mapping[str, Any]] | None = None,
    runner_sha256: str | None = None,
) -> str:
    verified = authority_receipts() if authorities is None else tuple(authorities)
    code_sha = _runner_sha256() if runner_sha256 is None else runner_sha256
    return _sha256_bytes(
        _canonical_json_bytes(
            _cell_identity_payload(
                spec,
                profile,
                root_seed=root_seed,
                authorities=verified,
                runner_sha256=code_sha,
            )
        )
    )


def run_cell(
    spec: ECACellSpec,
    *,
    profile: ECARunProfile = FULL_PROFILE,
    root_seed: int = ECA_ROOT_SEED,
    authorities: Sequence[Mapping[str, Any]] | None = None,
    runner_sha256: str | None = None,
) -> dict[str, Any]:
    """Train and measure one deterministic ECA-LATENT cell in memory."""

    if not isinstance(spec, ECACellSpec):
        raise TypeError("spec must be an ECACellSpec")
    canonical_authorities = authority_receipts()
    verified_authorities = (
        canonical_authorities
        if authorities is None
        else tuple(dict(row) for row in authorities)
    )
    if _canonical_json_bytes({"rows": verified_authorities}) != _canonical_json_bytes(
        {"rows": canonical_authorities}
    ):
        raise ECAReceiptIdentityError("supplied authorities are not the verified authorities")
    current_runner_sha = _runner_sha256()
    code_sha = current_runner_sha if runner_sha256 is None else runner_sha256
    if code_sha != current_runner_sha:
        raise ECAReceiptIdentityError("supplied runner SHA is not the executing runner")
    seeds = derived_seed_receipt(spec, root_seed=root_seed)
    registered_context = (
        profile == FULL_PROFILE
        and root_seed == ECA_ROOT_SEED
        and verified_authorities == canonical_authorities
        and code_sha == current_runner_sha
    )

    with _deterministic_cpu():
        split_sizes = {
            "train": profile.train_examples,
            "eval": profile.eval_examples,
            "probe_fit": profile.probe_fit_examples,
            "probe_eval": profile.probe_eval_examples,
        }
        inputs: dict[str, torch.Tensor] = {}
        state_banks: dict[str, tuple[torch.Tensor, ...]] = {}
        for split, size in split_sizes.items():
            seed = int(seeds[f"data_{split}"]["seed"])
            inputs[split] = _binary_examples(size, profile.n_cells, seed)
            state_banks[split] = eca_states(inputs[split], spec.rule, spec.tau)

        model = _build_model(spec, profile, root_seed=root_seed)
        initial_model_sha = model_initial_state_sha256(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=profile.learning_rate,
            weight_decay=profile.weight_decay,
        )
        order_generator = torch.Generator(device="cpu").manual_seed(
            int(seeds["train_order"]["seed"])
        )
        order_digest = _new_minibatch_order_digest(
            spec,
            profile,
            int(seeds["train_order"]["seed"]),
        )
        curve: list[dict[str, Any]] = []
        intervals: list[EvalLossInterval] = []
        prediction_tokens_since_eval = 0
        for step in range(profile.steps):
            batch_indices = torch.randint(
                0,
                profile.train_examples,
                (profile.batch_size,),
                generator=order_generator,
            )
            order_digest.update(batch_indices.numpy().tobytes(order="C"))
            executed_k = executed_k_at_step(
                spec.k,
                step,
                profile.curriculum_boundaries,
            )
            logits, _ = model(state_banks["train"][0][batch_indices], executed_k)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                state_banks["train"][spec.tau][batch_indices].float(),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            prediction_tokens_since_eval += profile.batch_size * profile.n_cells

            is_eval = (step + 1) % profile.eval_every_steps == 0 or (
                step + 1 == profile.steps
            )
            if is_eval:
                eval_bpc, eval_accuracy = _evaluate(
                    model,
                    state_banks["eval"][0],
                    state_banks["eval"][spec.tau],
                    executed_k,
                )
                point = EvalLossInterval(
                    delta_prediction_tokens=prediction_tokens_since_eval,
                    eval_bits_per_prediction_token=eval_bpc,
                    executed_k=executed_k,
                    scored_k=executed_k,
                )
                intervals.append(point)
                curve.append(
                    {
                        "step": step + 1,
                        "delta_prediction_tokens": prediction_tokens_since_eval,
                        "executed_k": executed_k,
                        "scored_k": executed_k,
                        "eval_bpc": eval_bpc,
                        "eval_accuracy": eval_accuracy,
                    }
                )
                prediction_tokens_since_eval = 0
        if prediction_tokens_since_eval != 0:
            raise AssertionError("the terminal eval must close the final token interval")
        observed_order_sha = order_digest.hexdigest()
        expected_order_sha = minibatch_order_sha256(
            spec,
            profile,
            root_seed=root_seed,
        )
        if observed_order_sha != expected_order_sha:
            raise AssertionError("consumed minibatch order does not match its replay fingerprint")

        preq = build_prequential_area_receipt(intervals)
        terminal = curve[-1]
        probe = _full_probe_matrix(
            model,
            spec,
            profile,
            state_banks["probe_fit"],
            state_banks["probe_eval"],
        )

    identity = cell_identity_sha256(
        spec,
        profile,
        root_seed=root_seed,
        authorities=verified_authorities,
        runner_sha256=code_sha,
    )
    receipt: dict[str, Any] = {
        "schema_version": ECA_SCHEMA_VERSION,
        "status": "complete",
        "instrument": "ECA-LATENT",
        "scientific_status": (
            "registered_measurement"
            if registered_context
            else "unregistered_configuration_not_scientific"
        ),
        "instrument_status": "analysis_pending",
        "cell_identity_sha256": identity,
        "cell": asdict(spec),
        "profile": asdict(profile),
        "root_seed": root_seed,
        "derived_seeds": seeds,
        "paired_stream_rule": (
            "data/model/order source keys omit K; all K arms share draws within replica"
        ),
        "authorities": list(verified_authorities),
        "runner_sha256": code_sha,
        "runtime": _runtime_identity(),
        "model_initial_state_sha256": initial_model_sha,
        "minibatch_order": {
            "sha256": observed_order_sha,
            "batches": profile.steps,
            "indices_per_batch": profile.batch_size,
            "source_key": seeds["train_order"]["source_key"],
            "seed": seeds["train_order"]["seed"],
        },
        "dataset_sha256": {
            split: _tensor_identity(value) for split, value in inputs.items()
        },
        "kernel_size": 2 * spec.tau + 1,
        "curriculum": {
            "enabled": curriculum_enabled(spec.k),
            "rule": "1_to_2_to_4_to_target_k",
            "boundaries": list(profile.curriculum_boundaries),
            "eval_depth_rule": "score_each_eval_at_executed_k_t",
        },
        "eval_population": {
            "examples": profile.eval_examples,
            "dataset_sha256": _tensor_identity(inputs["eval"]),
            "curve_and_terminal_are_identical": True,
        },
        "eval_curve": curve,
        "terminal_bpc": float(terminal["eval_bpc"]),
        "terminal_accuracy": float(terminal["eval_accuracy"]),
        "preq_area": preq.as_dict(),
        "probe": {
            "rows": "visit_j_after_1_through_K",
            "columns": "target_F_i_for_i_1_through_tau",
            "fit_examples": profile.probe_fit_examples,
            "eval_examples": profile.probe_eval_examples,
            "fit_dataset_sha256": _tensor_identity(inputs["probe_fit"]),
            "eval_dataset_sha256": _tensor_identity(inputs["probe_eval"]),
            "fit_eval_use_separate_named_rng_streams": True,
            "ridge": profile.ridge,
            "solve_dtype": "float64",
            "matrix": [list(row) for row in probe],
            "classification": None,
            "classification_reason": "threshold_not_ratified_emit_raw_matrix_only",
        },
    }
    receipt = _with_self_hash(receipt)
    validate_cell_receipt(
        receipt,
        expected_identity=identity,
        expected_spec=spec,
    )
    return receipt


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def validate_cell_receipt(
    payload: Mapping[str, Any],
    *,
    expected_identity: str,
    expected_spec: ECACellSpec,
) -> None:
    _validate_self_hash(payload, artifact="cell receipt")
    if payload.get("schema_version") != ECA_SCHEMA_VERSION:
        raise ECAReceiptIdentityError("cell receipt schema version mismatch")
    if payload.get("status") != "complete":
        raise ECAReceiptIdentityError("cell receipt is not complete")
    if payload.get("instrument") != "ECA-LATENT":
        raise ECAReceiptIdentityError("cell receipt instrument mismatch")
    if payload.get("instrument_status") != "analysis_pending":
        raise ECAReceiptIdentityError("ECA-LATENT must remain analysis_pending")
    if payload.get("cell_identity_sha256") != expected_identity:
        raise ECAReceiptIdentityError("cell receipt identity mismatch")
    if payload.get("cell") != asdict(expected_spec):
        raise ECAReceiptIdentityError("cell receipt coordinate mismatch")
    profile_payload = payload.get("profile")
    if not isinstance(profile_payload, Mapping):
        raise ECAReceiptIdentityError("cell receipt profile is invalid")
    try:
        profile_fields = dict(profile_payload)
        profile_fields["curriculum_boundaries"] = tuple(
            profile_fields["curriculum_boundaries"]
        )
        profile = ECARunProfile(**profile_fields)
    except (KeyError, TypeError, ValueError) as error:
        raise ECAReceiptIdentityError("cell receipt profile does not reconstruct") from error
    try:
        identity_payload = {
            "schema_version": payload["schema_version"],
            "instrument": payload["instrument"],
            "cell": payload["cell"],
            "profile": payload["profile"],
            "root_seed": payload["root_seed"],
            "authorities": payload["authorities"],
            "runner_sha256": payload["runner_sha256"],
            "runtime": payload["runtime"],
        }
    except KeyError as error:
        raise ECAReceiptIdentityError("cell receipt identity inputs are incomplete") from error
    reconstructed_identity = _sha256_bytes(_canonical_json_bytes(identity_payload))
    if reconstructed_identity != expected_identity:
        raise ECAReceiptIdentityError("cell receipt identity inputs do not reconstruct")
    try:
        expected_seeds = derived_seed_receipt(
            expected_spec,
            root_seed=payload["root_seed"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ECAReceiptIdentityError("cell root seed is invalid") from error
    if payload.get("derived_seeds") != expected_seeds:
        raise ECAReceiptIdentityError("cell derived-seed receipt mismatch")
    registered_context = (
        profile == FULL_PROFILE
        and payload["root_seed"] == ECA_ROOT_SEED
        and payload["authorities"] == list(authority_receipts())
        and payload["runner_sha256"] == _runner_sha256()
    )
    expected_scientific_status = (
        "registered_measurement"
        if registered_context
        else "unregistered_configuration_not_scientific"
    )
    if payload.get("scientific_status") != expected_scientific_status:
        raise ECAReceiptIdentityError("cell scientific-status claim is not authorized")
    if payload.get("kernel_size") != 2 * expected_spec.tau + 1:
        raise ECAReceiptIdentityError("cell kernel does not expose the tau radius")
    expected_initial_model_sha = model_initial_state_sha256(
        _build_model(expected_spec, profile, root_seed=payload["root_seed"])
    )
    if payload.get("model_initial_state_sha256") != expected_initial_model_sha:
        raise ECAReceiptIdentityError("initial model-state SHA does not replay")
    expected_order_sha = minibatch_order_sha256(
        expected_spec,
        profile,
        root_seed=payload["root_seed"],
    )
    order = payload.get("minibatch_order")
    expected_order = {
        "sha256": expected_order_sha,
        "batches": profile.steps,
        "indices_per_batch": profile.batch_size,
        "source_key": expected_seeds["train_order"]["source_key"],
        "seed": expected_seeds["train_order"]["seed"],
    }
    if order != expected_order:
        raise ECAReceiptIdentityError("minibatch-order fingerprint does not replay")

    expected_eval_steps = list(
        range(profile.eval_every_steps, profile.steps + 1, profile.eval_every_steps)
    )
    if not expected_eval_steps or expected_eval_steps[-1] != profile.steps:
        expected_eval_steps.append(profile.steps)
    curve = payload.get("eval_curve")
    if not isinstance(curve, list) or len(curve) != len(expected_eval_steps):
        raise ECAReceiptIdentityError("cell receipt has the wrong eval schedule")
    intervals: list[EvalLossInterval] = []
    prior_step = 0
    for point, expected_step in zip(curve, expected_eval_steps, strict=True):
        if not isinstance(point, Mapping):
            raise ECAReceiptIdentityError("cell eval point is not a mapping")
        expected_delta = (
            (expected_step - prior_step) * profile.batch_size * profile.n_cells
        )
        expected_k = executed_k_at_step(
            expected_spec.k,
            expected_step - 1,
            profile.curriculum_boundaries,
        )
        if (
            point.get("step") != expected_step
            or point.get("delta_prediction_tokens") != expected_delta
            or point.get("executed_k") != expected_k
            or point.get("scored_k") != expected_k
        ):
            raise ECAReceiptIdentityError("cell eval schedule does not reconstruct")
        bpc = point.get("eval_bpc")
        accuracy = point.get("eval_accuracy")
        if (
            isinstance(bpc, bool)
            or not isinstance(bpc, (int, float))
            or not math.isfinite(float(bpc))
            or float(bpc) < 0.0
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(float(accuracy))
            or not 0.0 <= float(accuracy) <= 1.0
        ):
            raise ECAReceiptIdentityError("cell eval metric lies outside its range")
        try:
            intervals.append(
                EvalLossInterval(
                    delta_prediction_tokens=point["delta_prediction_tokens"],
                    eval_bits_per_prediction_token=point["eval_bpc"],
                    executed_k=point["executed_k"],
                    scored_k=point["scored_k"],
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ECAReceiptIdentityError("invalid cell eval interval") from error
        prior_step = expected_step
    if sum(point.delta_prediction_tokens for point in intervals) != (
        profile.steps * profile.batch_size * profile.n_cells
    ):
        raise ECAReceiptIdentityError("eval token intervals do not cover the run")
    recomputed = build_prequential_area_receipt(intervals)
    preq = payload.get("preq_area")
    if (
        not isinstance(preq, Mapping)
        or _canonical_json_bytes({"preq": preq})
        != _canonical_json_bytes({"preq": recomputed.as_dict()})
    ):
        raise ECAReceiptIdentityError("cell prequential area does not reconstruct")
    if payload.get("terminal_bpc") != float(curve[-1]["eval_bpc"]):
        raise ECAReceiptIdentityError("terminal BPC is not the final fixed-population eval")
    if payload.get("terminal_accuracy") != float(curve[-1]["eval_accuracy"]):
        raise ECAReceiptIdentityError(
            "terminal accuracy is not the final fixed-population eval"
        )
    expected_curriculum = {
        "enabled": curriculum_enabled(expected_spec.k),
        "rule": "1_to_2_to_4_to_target_k",
        "boundaries": list(profile.curriculum_boundaries),
        "eval_depth_rule": "score_each_eval_at_executed_k_t",
    }
    if payload.get("curriculum") != expected_curriculum:
        raise ECAReceiptIdentityError("cell curriculum metadata does not reconstruct")
    dataset_hashes = payload.get("dataset_sha256")
    eval_population = payload.get("eval_population")
    split_sizes = {
        "train": profile.train_examples,
        "eval": profile.eval_examples,
        "probe_fit": profile.probe_fit_examples,
        "probe_eval": profile.probe_eval_examples,
    }
    expected_dataset_hashes = {
        split: _tensor_identity(
            _binary_examples(
                size,
                profile.n_cells,
                int(expected_seeds[f"data_{split}"]["seed"]),
            )
        )
        for split, size in split_sizes.items()
    }
    if (
        not isinstance(dataset_hashes, Mapping)
        or dict(dataset_hashes) != expected_dataset_hashes
        or not isinstance(eval_population, Mapping)
        or eval_population.get("examples") != profile.eval_examples
        or eval_population.get("dataset_sha256") != dataset_hashes.get("eval")
        or eval_population.get("curve_and_terminal_are_identical") is not True
    ):
        raise ECAReceiptIdentityError("fixed evaluation-population identity mismatch")
    probe = payload.get("probe")
    if not isinstance(probe, Mapping) or probe.get("classification") is not None:
        raise ECAReceiptIdentityError("raw probe receipt must not invent a classification")
    matrix = probe.get("matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != expected_spec.k
        or any(not isinstance(row, list) or len(row) != expected_spec.tau for row in matrix)
    ):
        raise ECAReceiptIdentityError("probe matrix shape must be K by tau")
    if (
        probe.get("fit_eval_use_separate_named_rng_streams") is not True
        or probe.get("fit_examples") != profile.probe_fit_examples
        or probe.get("eval_examples") != profile.probe_eval_examples
        or probe.get("fit_dataset_sha256") != dataset_hashes.get("probe_fit")
        or probe.get("eval_dataset_sha256") != dataset_hashes.get("probe_eval")
        or probe.get("ridge") != profile.ridge
        or probe.get("solve_dtype") != "float64"
        or probe.get("rows") != "visit_j_after_1_through_K"
        or probe.get("columns") != "target_F_i_for_i_1_through_tau"
        or probe.get("classification_reason")
        != "threshold_not_ratified_emit_raw_matrix_only"
    ):
        raise ECAReceiptIdentityError("probe split identities do not reconstruct")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for row in matrix
        for value in row
    ):
        raise ECAReceiptIdentityError("probe accuracy lies outside [0, 1]")
    if not _finite_tree(payload):
        raise ECAReceiptIdentityError("cell receipt contains a non-finite scalar")


def _read_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ECAReceiptIdentityError(
                    f"durable JSON contains duplicate key {key!r}: {path.name}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ECAReceiptIdentityError(f"cannot read durable JSON: {path.name}") from error
    if not isinstance(payload, dict):
        raise ECAReceiptIdentityError(f"durable JSON must be an object: {path.name}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = _canonical_json_bytes(payload)
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def assigned_grid(shard_index: int, shard_count: int) -> tuple[ECACellSpec, ...]:
    """Deterministically partition the full registered grid without changing it."""

    if type(shard_count) is not int or not 1 <= shard_count <= len(registered_grid()):
        raise ValueError("shard_count must be an exact integer in [1, 72]")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be an exact integer within shard_count")
    return tuple(sorted(registered_grid()))[shard_index::shard_count]


def shard_directory_name(shard_index: int, shard_count: int) -> str:
    assigned_grid(shard_index, shard_count)
    return f"shard-{shard_index:05d}-of-{shard_count:05d}"


@contextmanager
def _exclusive_writer_lock(path: Path, payload: Mapping[str, Any]) -> Iterator[None]:
    """Hold a process-lifetime advisory lock; process death releases it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\x00")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (OSError, ImportError) as error:
            raise ECAReceiptIdentityError(
                f"concurrent or unsupported writer lock: {path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, ImportError):
                pass
        handle.close()


def _campaign_plan(
    profile: ECARunProfile,
    *,
    root_seed: int,
    authorities: Sequence[Mapping[str, Any]],
    runner_sha256: str,
) -> dict[str, Any]:
    cells = tuple(sorted(registered_grid()))
    expected = {
        spec.filename: cell_identity_sha256(
            spec,
            profile,
            root_seed=root_seed,
            authorities=authorities,
            runner_sha256=runner_sha256,
        )
        for spec in cells
    }
    identity_payload = {
        "schema_version": ECA_SCHEMA_VERSION,
        "profile": asdict(profile),
        "root_seed": root_seed,
        "authorities": list(authorities),
        "runner_sha256": runner_sha256,
        "runtime": _runtime_identity(),
        "expected_cells": expected,
    }
    return _with_self_hash({
        **identity_payload,
        "campaign_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(identity_payload)
        ),
        "total_campaign_cells": len(cells),
        "instrument_status": "analysis_pending",
    })


def _campaign_layout(plan: Mapping[str, Any], shard_count: int) -> dict[str, Any]:
    assignments = {
        shard_directory_name(index, shard_count): [
            spec.filename for spec in assigned_grid(index, shard_count)
        ]
        for index in range(shard_count)
    }
    return _with_self_hash(
        {
            "schema_version": ECA_SCHEMA_VERSION,
            "campaign_identity_sha256": plan["campaign_identity_sha256"],
            "shard_count": shard_count,
            "assignments": assignments,
            "assignment_rule": "sorted_registered_grid_stride_by_shard_index",
        }
    )


def _ensure_campaign_layout(
    root: Path,
    plan: Mapping[str, Any],
    shard_count: int,
) -> dict[str, Any]:
    expected = _campaign_layout(plan, shard_count)
    path = root / "campaign_layout.json"
    with _exclusive_writer_lock(
        root / ".campaign_layout.writer.lock",
        {
            "purpose": "campaign_layout_single_writer",
            "campaign_identity_sha256": plan["campaign_identity_sha256"],
        },
    ):
        if path.exists():
            observed = _read_json(path)
            _validate_self_hash(observed, artifact="campaign layout")
            if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
                raise ECAReceiptIdentityError(
                    "existing campaign layout conflicts with requested shard scheme"
                )
        else:
            _atomic_json(path, expected)
    return expected


def _shard_plan(
    plan: Mapping[str, Any],
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    expected_all = plan["expected_cells"]
    cells = assigned_grid(shard_index, shard_count)
    expected = {spec.filename: expected_all[spec.filename] for spec in cells}
    return {
        "schema_version": ECA_SCHEMA_VERSION,
        "campaign_identity_sha256": plan["campaign_identity_sha256"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assignment_rule": "sorted_registered_grid_stride_by_shard_index",
        "expected_cells": expected,
        "shard_assignment_sha256": _sha256_bytes(
            _canonical_json_bytes({"expected_cells": expected})
        ),
        "total_campaign_cells": len(registered_grid()),
        "instrument_status": "analysis_pending",
    }


def _manifest_payload(
    shard_plan: Mapping[str, Any],
    completed: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    expected = shard_plan["expected_cells"]
    if not isinstance(expected, Mapping):
        raise TypeError("shard plan expected_cells must be a mapping")
    return _with_self_hash({
        **shard_plan,
        "status": (
            "shard_complete" if len(completed) == len(expected) else "in_progress"
        ),
        "completed_cells": len(completed),
        "shard_cells": len(expected),
        "cell_receipts": dict(sorted(completed.items())),
    })


def run_campaign(
    output_dir: Path,
    *,
    specs: Sequence[ECACellSpec] | None = None,
    profile: ECARunProfile = FULL_PROFILE,
    root_seed: int = ECA_ROOT_SEED,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Run or resume one deterministic shard of the immutable 72-cell plan."""

    selected = assigned_grid(shard_index, shard_count)
    if specs is not None and tuple(sorted(specs)) != selected:
        raise ValueError("specs must exactly equal the deterministic shard assignment")
    authorities = authority_receipts()
    runner_sha = _runner_sha256()
    plan = _campaign_plan(
        profile,
        root_seed=root_seed,
        authorities=authorities,
        runner_sha256=runner_sha,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_campaign_layout(root, plan, shard_count)
    shard_root = root / "shards" / shard_directory_name(shard_index, shard_count)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_plan = _shard_plan(plan, shard_index, shard_count)
    manifest_path = shard_root / "manifest.json"

    with _exclusive_writer_lock(
        shard_root / ".writer.lock",
        {
            "purpose": "eca_latent_shard_single_writer",
            "campaign_identity_sha256": plan["campaign_identity_sha256"],
            "shard_index": shard_index,
            "shard_count": shard_count,
            "pid": os.getpid(),
        },
    ):
        completed: dict[str, dict[str, str]] = {}
        if manifest_path.exists():
            previous = _read_json(manifest_path)
            _validate_self_hash(previous, artifact="shard manifest")
            previous_plan = {key: previous.get(key) for key in shard_plan}
            if _canonical_json_bytes(previous_plan) != _canonical_json_bytes(shard_plan):
                raise ECAReceiptIdentityError("existing shard manifest plan mismatch")
            recorded = previous.get("cell_receipts")
            if not isinstance(recorded, Mapping) or any(
                not isinstance(value, Mapping) for value in recorded.values()
            ):
                raise ECAReceiptIdentityError("existing shard receipt index is invalid")
            completed = {str(name): dict(value) for name, value in recorded.items()}

        expected_files = set(shard_plan["expected_cells"])
        observed_files = {path.name for path in shard_root.glob("cell_*.json")}
        unexpected = observed_files.difference(expected_files)
        if unexpected:
            raise ECAReceiptIdentityError(
                f"unexpected ECA-LATENT cell receipts: {sorted(unexpected)}"
            )
        missing_recorded = set(completed).difference(observed_files)
        if missing_recorded:
            raise ECAReceiptIdentityError(
                f"shard manifest names missing receipts: {sorted(missing_recorded)}"
            )

        specs_by_name = {spec.filename: spec for spec in selected}
        recovered: dict[str, dict[str, str]] = {}
        for filename in sorted(observed_files):
            path = shard_root / filename
            payload = _read_json(path)
            expected_identity = str(shard_plan["expected_cells"][filename])
            validate_cell_receipt(
                payload,
                expected_identity=expected_identity,
                expected_spec=specs_by_name[filename],
            )
            file_sha = _sha256_file(path)
            if filename in completed:
                prior = completed[filename]
                if (
                    prior.get("sha256") != file_sha
                    or prior.get("cell_identity_sha256") != expected_identity
                ):
                    raise ECAReceiptIdentityError(
                        f"durable receipt hash mismatch for {filename}"
                    )
            recovered[filename] = {
                "sha256": file_sha,
                "cell_identity_sha256": expected_identity,
                "canonical_self_sha256": str(payload[ECA_SELF_HASH_FIELD]),
            }
        completed = recovered
        _atomic_json(manifest_path, _manifest_payload(shard_plan, completed))

        for spec in selected:
            if spec.filename in completed:
                continue
            payload = run_cell(
                spec,
                profile=profile,
                root_seed=root_seed,
                authorities=authorities,
                runner_sha256=runner_sha,
            )
            path = shard_root / spec.filename
            if path.exists():
                raise ECAReceiptIdentityError(
                    f"refusing to overwrite unexpected durable receipt {spec.filename}"
                )
            _atomic_json(path, payload)
            completed[spec.filename] = {
                "sha256": _sha256_file(path),
                "cell_identity_sha256": str(payload["cell_identity_sha256"]),
                "canonical_self_sha256": str(payload[ECA_SELF_HASH_FIELD]),
            }
            _atomic_json(manifest_path, _manifest_payload(shard_plan, completed))
    return _read_json(manifest_path)


def verify_campaign(
    output_dir: Path,
    *,
    profile: ECARunProfile = FULL_PROFILE,
    root_seed: int = ECA_ROOT_SEED,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Verify all 72 cells across shards; leave scientific analysis pending."""

    authorities = authority_receipts()
    runner_sha = _runner_sha256()
    plan = _campaign_plan(
        profile,
        root_seed=root_seed,
        authorities=authorities,
        runner_sha256=runner_sha,
    )
    root = Path(output_dir)
    layout = _ensure_campaign_layout(root, plan, shard_count)
    all_receipts: dict[str, dict[str, str]] = {}
    specs_by_name = {spec.filename: spec for spec in registered_grid()}
    with _exclusive_writer_lock(
        root / ".aggregate.writer.lock",
        {
            "purpose": "eca_latent_aggregate_single_writer",
            "campaign_identity_sha256": plan["campaign_identity_sha256"],
        },
    ):
        for shard_index in range(shard_count):
            shard_plan = _shard_plan(plan, shard_index, shard_count)
            shard_root = root / "shards" / shard_directory_name(
                shard_index, shard_count
            )
            manifest_path = shard_root / "manifest.json"
            if not manifest_path.is_file():
                raise ECAReceiptIdentityError(
                    f"missing shard manifest for shard {shard_index}/{shard_count}"
                )
            manifest = _read_json(manifest_path)
            _validate_self_hash(manifest, artifact="shard manifest")
            if manifest.get("status") != "shard_complete":
                raise ECAReceiptIdentityError(
                    f"shard {shard_index}/{shard_count} is not shard_complete"
                )
            if any(manifest.get(key) != value for key, value in shard_plan.items()):
                raise ECAReceiptIdentityError("shard manifest plan mismatch")
            expected_names = set(shard_plan["expected_cells"])
            observed_names = {path.name for path in shard_root.glob("cell_*.json")}
            if observed_names != expected_names:
                raise ECAReceiptIdentityError("shard receipt set is incomplete or unexpected")
            index = manifest.get("cell_receipts")
            if not isinstance(index, Mapping) or set(index) != expected_names:
                raise ECAReceiptIdentityError("shard manifest receipt index is incomplete")
            for filename in sorted(expected_names):
                if filename in all_receipts:
                    raise ECAReceiptIdentityError(f"duplicate campaign cell {filename}")
                path = shard_root / filename
                payload = _read_json(path)
                expected_identity = str(plan["expected_cells"][filename])
                validate_cell_receipt(
                    payload,
                    expected_identity=expected_identity,
                    expected_spec=specs_by_name[filename],
                )
                file_sha = _sha256_file(path)
                recorded = index[filename]
                if (
                    not isinstance(recorded, Mapping)
                    or recorded.get("sha256") != file_sha
                    or recorded.get("cell_identity_sha256") != expected_identity
                    or recorded.get("canonical_self_sha256")
                    != payload[ECA_SELF_HASH_FIELD]
                ):
                    raise ECAReceiptIdentityError(
                        f"aggregate receipt index mismatch for {filename}"
                    )
                all_receipts[filename] = {
                    "sha256": file_sha,
                    "cell_identity_sha256": expected_identity,
                    "canonical_self_sha256": str(payload[ECA_SELF_HASH_FIELD]),
                    "shard": shard_directory_name(shard_index, shard_count),
                }
        if set(all_receipts) != set(plan["expected_cells"]):
            raise ECAReceiptIdentityError("aggregate verifier did not recover all 72 cells")
        registered_context = profile == FULL_PROFILE and root_seed == ECA_ROOT_SEED
        aggregate = _with_self_hash(
            {
                "schema_version": ECA_SCHEMA_VERSION,
                "campaign_identity_sha256": plan["campaign_identity_sha256"],
                "layout_sha256": layout[ECA_SELF_HASH_FIELD],
                "status": "analysis_pending",
                "instrument_status": "analysis_pending",
                "execution_status": "all_72_registered_cells_verified",
                "scientific_status": (
                    "registered_measurements_complete_awaiting_analysis"
                    if registered_context
                    else "unregistered_configuration_not_scientific"
                ),
                "total_campaign_cells": len(all_receipts),
                "shard_count": shard_count,
                "cell_receipts": dict(sorted(all_receipts.items())),
            }
        )
        _atomic_json(root / "aggregate_manifest.json", aggregate)
    return aggregate


def _parse_cell(value: str) -> ECACellSpec:
    try:
        rule, tau, k, replica = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "cell must be RULE,TAU,K,REPLICA"
        ) from error
    try:
        return ECACellSpec(rule=rule, tau=tau, k=k, replica=replica)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cell",
        type=_parse_cell,
        action="append",
        help="restrict to RULE,TAU,K,REPLICA; repeat for multiple cells",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the tiny non-scientific profile (grid coordinates unchanged)",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify all shard outputs and emit analysis-pending aggregate manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    profile = SMOKE_PROFILE if args.smoke else FULL_PROFILE
    if args.verify:
        if args.cell:
            raise SystemExit("--verify cannot be combined with --cell")
        manifest = verify_campaign(
            args.output_dir,
            profile=profile,
            shard_count=args.shard_count,
        )
    else:
        shard_index = args.shard_index
        shard_count = args.shard_count
        if args.cell:
            if len(args.cell) != 1:
                raise SystemExit("--cell accepts exactly one cell per writer")
            if args.shard_index != 0 or args.shard_count != 1:
                raise SystemExit("--cell cannot be combined with explicit shard options")
            ordered = tuple(sorted(registered_grid()))
            shard_index = ordered.index(args.cell[0])
            shard_count = len(ordered)
        manifest = run_campaign(
            args.output_dir,
            profile=profile,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
