import pytest
import torch
from torch_geometric.data import Batch, Data

from metrics import AreaUnderTaskMetrics
from search_space_compression import (
    OnlineTrainer,
    SharedAttributeVocab,
    StagedBetaSchedule,
    _weisfeiler_lehman_histograms,
)


class MinimalGuide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_attr_vocab = SharedAttributeVocab([], embedding_dim=4)
        self.dummy = torch.nn.Parameter(torch.zeros(1))


def test_staged_beta_schedule_phases():
    schedule = StagedBetaSchedule(
        start_beta=0.0,
        target_beta=0.2,
        warmup_epochs=2,
        ramp_epochs=2,
        hold_epochs=1,
        cycle_length=3,
        cycle_floor=0.05,
    )
    values = [schedule.value(i) for i in range(10)]
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.1)
    assert values[3] == pytest.approx(0.2)
    assert values[4] == pytest.approx(0.2)
    assert values[5] == pytest.approx(0.05)
    assert values[6] > values[5]
    assert values[7] > values[6]


def test_online_trainer_resolves_dynamic_kl_weight():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    schedule = StagedBetaSchedule(start_beta=0.0, target_beta=0.5, warmup_epochs=1, ramp_epochs=1)
    trainer.configure_kl_scheduler(schedule, reset_state=True)
    assert trainer._resolve_kl_weight(0.1) == pytest.approx(0.0)
    trainer._kl_global_epoch = 1
    assert trainer._resolve_kl_weight(0.1) == pytest.approx(0.5)
    trainer._kl_global_epoch = 5
    assert trainer._resolve_kl_weight(0.1) == pytest.approx(0.5)
    trainer.configure_kl_scheduler(schedule, reset_state=True)
    assert trainer._kl_global_epoch == 0


def _make_graph(node_count, edges):
    node_types = torch.zeros(node_count, dtype=torch.long)
    node_attributes = [{} for _ in range(node_count)]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(node_types=node_types, edge_index=edge_index, node_attributes=node_attributes)


def test_wl_histograms_distinguish_graphs():
    g_chain = _make_graph(3, [(0, 1), (1, 2)])
    g_star = _make_graph(3, [(0, 1), (0, 2)])
    batch = Batch.from_data_list([g_chain, g_star])
    hist = _weisfeiler_lehman_histograms(
        batch.node_types,
        batch.edge_index,
        batch.batch,
        batch.num_graphs,
        iterations=2,
    )
    assert hist is not None
    assert hist.shape[0] == 2
    assert not torch.allclose(hist[0], hist[1])


def test_structural_alignment_loss_handles_present_batches():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    trainer.wl_loss_weight = 1.0
    trainer.wl_kernel_iterations = 1
    g1 = _make_graph(3, [(0, 1), (1, 2)])
    g2 = _make_graph(3, [(0, 2), (2, 1)])
    batch = Batch.from_data_list([g1, g2])
    latents = torch.randn(batch.num_graphs, 4)
    loss = trainer._structural_alignment_loss(batch, latents)
    assert loss >= 0
    # Single graph batches skip the loss entirely.
    single_batch = Batch.from_data_list([g1])
    single_loss = trainer._structural_alignment_loss(single_batch, latents[:1])
    assert single_loss.item() == pytest.approx(0.0)
