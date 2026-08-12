from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from eval.eval_paper2_phase3_p34_task_inference import (
    P34TaskInferenceGraph,
    current_position_mask,
    task_graph_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_task_preflight_runner_imports_when_invoked_as_a_file() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                "runpy.run_path('colab/run_stage5_paper2_phase3_p34_task_preflight.py', "
                "run_name='p34_import_probe')"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


from models.paper2_dc2_student import Phase3StudentModules


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(31, 16)
        self.output = nn.Linear(16, 31, bias=False)
        self.output.weight = self.embedding.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.output

    def forward(self, *, input_ids, attention_mask, use_cache=False, **_kwargs):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            logits=self.output(hidden),
            hidden_states=(hidden,),
            past_key_values=("tiny-cache",) if use_cache else None,
        )


def test_current_position_mask_supports_padding_and_closes_position_zero() -> None:
    mask, positions = current_position_mask(torch.tensor([[1, 1, 0], [1, 0, 0]]))
    assert positions.tolist() == [1, 0]
    assert mask[:, :, 0].tolist() == [[False, True, False], [False, False, False]]


def test_exact_task_graph_disables_draft_and_is_repeatable() -> None:
    torch.manual_seed(5)
    base = TinyCausalLM()
    sidecar = Phase3StudentModules(
        tied_embedding=base.embedding,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        control_dim=6,
        draft_rank=4,
        max_steps=4,
        rms_cap=0.55,
    )
    graph = P34TaskInferenceGraph(base_model=base, sidecar=sidecar)
    ids = torch.tensor([[2, 3, 4], [5, 6, 0]])
    attention = torch.tensor([[1, 1, 1], [1, 1, 0]])
    receipt = task_graph_preflight(graph, input_ids=ids, attention_mask=attention)
    assert all(receipt["assertions"].values())
    assert receipt["selected_write_cells"] == 2


def test_write_mask_changes_only_current_nonzero_position() -> None:
    torch.manual_seed(7)
    base = TinyCausalLM()
    sidecar = Phase3StudentModules(
        tied_embedding=base.embedding,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        control_dim=6,
        draft_rank=4,
    )
    hidden = torch.randn(1, 4, 16)
    write_mask = torch.zeros(1, 4, 1, dtype=torch.bool)
    write_mask[:, 3] = True
    output = sidecar(
        hidden=hidden,
        previous_logits=torch.zeros(1, 1, 31),
        steps=4,
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        position_bucket=torch.tensor([2]),
        draft_active=False,
        write_position_mask=write_mask,
    )
    assert torch.equal(output.hidden[:, :3], hidden[:, :3])
    assert torch.equal(output.logits, torch.zeros(1, 1, 31))


def test_cached_prefix_matches_uncached_next_token_semantics() -> None:
    torch.manual_seed(11)
    base = TinyCausalLM()
    sidecar = Phase3StudentModules(
        tied_embedding=base.embedding,
        hidden_size=16,
        latent_dim=8,
        n_slots=8,
        control_dim=6,
        draft_rank=4,
    )
    graph = P34TaskInferenceGraph(base_model=base, sidecar=sidecar)
    ids = torch.tensor([[2, 3, 4]])
    attention = torch.ones_like(ids)
    state, cached = graph.prefill_cached(input_ids=ids, attention_mask=attention)
    uncached = graph.next_token(input_ids=ids, attention_mask=attention)
    assert torch.equal(cached.augmented_logits, uncached.augmented_logits)
    selected = cached.augmented_logits.argmax(dim=-1)
    state, advanced = graph.advance_cached(state=state, selected_tokens=selected)
    extended = torch.cat([ids, selected[:, None]], dim=1)
    extended_attention = torch.ones_like(extended)
    repeated = graph.next_token(input_ids=extended, attention_mask=extended_attention)
    assert torch.equal(advanced.augmented_logits, repeated.augmented_logits)
