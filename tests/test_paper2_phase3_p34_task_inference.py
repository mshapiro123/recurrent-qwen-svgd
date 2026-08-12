from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from eval.eval_paper2_phase3_p34_task_inference import (
    P34TaskInferenceGraph,
    current_position_mask,
    task_graph_preflight,
)
from models.paper2_dc2_student import Phase3StudentModules


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(31, 16)
        self.output = nn.Linear(16, 31, bias=False)
        self.output.weight = self.embedding.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.output

    def forward(self, *, input_ids, attention_mask, **_kwargs):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.output(hidden), hidden_states=(hidden,))


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
