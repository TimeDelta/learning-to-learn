import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

import genes
from metrics import AreaUnderTaskMetrics
from search_space_compression import (
    GraphDecoder,
    OnlineTrainer,
    SharedAttributeVocab,
    StagedBetaSchedule,
    _weisfeiler_lehman_histograms,
    build_teacher_attr_targets,
    build_teacher_attr_value_targets,
)


class _DummyAttrEncoder(torch.nn.Module):
    def forward(self, *args, **kwargs):  # pragma: no cover - not exercised in unit tests
        raise NotImplementedError

    def get_value_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.float()
        if value is None:
            return torch.tensor([0.0])
        try:
            return torch.as_tensor([float(value)], dtype=torch.float32)
        except (TypeError, ValueError):
            return torch.tensor([0.0])


class _MinimalFitness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1)
        self.fc2 = torch.nn.Linear(1, 1)
        self.log_metric_scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):  # pragma: no cover - not exercised in unit tests
        raise NotImplementedError


class MinimalGuide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_attr_vocab = SharedAttributeVocab([], embedding_dim=4)
        self.attr_encoder = _DummyAttrEncoder()
        self.graph_encoder = torch.nn.Linear(1, 4)
        self.decoder = torch.nn.Linear(4, 1)
        self.fitness_predictor = _MinimalFitness()
        self.log_alpha_g = torch.nn.Parameter(torch.zeros(4))
        self.register_buffer("graph_latent_mask", torch.ones(4))

    @property
    def graph_latent_dim(self):
        return int(self.graph_latent_mask.numel())

    def prune_latent_dims(self, num_prune: int = 1):  # pragma: no cover - simple helper
        if num_prune <= 0:
            return
        active = (self.graph_latent_mask > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return
        drop = min(num_prune, active.numel())
        self.graph_latent_mask[active[-drop:]] = 0

    def resize_bottleneck(self):  # pragma: no cover - noop for tests
        return

    def forward(self, *args, **kwargs):  # pragma: no cover - not exercised here
        raise NotImplementedError


class _AttrRegistryGuard:
    def __enter__(self):
        self._names_backup = set(genes.ATTRIBUTE_NAMES)
        self._map_backup = {k: set(v) for k, v in genes.ATTRIBUTE_NAMES_BY_KIND.items()}
        self._version_backup = genes.ATTRIBUTE_NAMES_VERSION
        genes.ATTRIBUTE_NAMES.clear()
        genes.ATTRIBUTE_NAMES_BY_KIND.clear()
        genes.ATTRIBUTE_NAMES_VERSION = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        genes.ATTRIBUTE_NAMES.clear()
        genes.ATTRIBUTE_NAMES.update(self._names_backup)
        genes.ATTRIBUTE_NAMES_BY_KIND.clear()
        for kind, names in self._map_backup.items():
            genes.ATTRIBUTE_NAMES_BY_KIND[kind] = set(names)
        genes.ATTRIBUTE_NAMES_VERSION = self._version_backup


def test_shared_attribute_vocab_rejects_non_canonical_names():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    vocab.set_allowed_names({"foo"})
    vocab.add_names(["foo"])
    with pytest.raises(ValueError):
        vocab.add_names(["bar"])


def test_teacher_value_targets_allow_string_literals_via_private_tokens():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    vocab.set_allowed_names({"foo"})
    vocab.add_names(["foo"])
    data = [{"foo": "prim::Param"}]
    outputs = build_teacher_attr_value_targets(data, batch_vec=None, shared_vocab=vocab, max_value_dim=4)
    assert outputs and outputs[0][0]
    assert "__value__prim::Param" in vocab.name_to_index


def test_teacher_attr_targets_drop_node_type_metadata():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    vocab.set_allowed_names({"foo"})
    vocab.add_names(["foo"])
    sequences = build_teacher_attr_targets([{"node_type": "aten::add", "foo": 1.0}], batch_vec=None, shared_vocab=vocab)
    assert sequences == [[[vocab.name_to_index["foo"], vocab.eos_index]]]
    assert "node_type" not in vocab.name_to_index


def test_teacher_attr_targets_skip_metadata_only_entries():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    vocab.set_allowed_names(set())
    sequences = build_teacher_attr_targets([{"node_type": "aten::add"}], batch_vec=None, shared_vocab=vocab)
    assert sequences == [[[vocab.eos_index]]]
    assert "node_type" not in vocab.name_to_index


def test_shared_attribute_vocab_tracks_allowed_set_updates():
    allowed = set()
    vocab = SharedAttributeVocab([], embedding_dim=4)
    vocab.set_allowed_names(allowed)
    allowed.add("alpha")
    vocab.add_names(["alpha"])
    assert vocab.ensure_index("alpha") == vocab.name_to_index["alpha"]


def test_wl_histograms_reflect_attribute_differences():
    node_types = torch.tensor([0, 0], dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch_vec = torch.tensor([0, 1], dtype=torch.long)
    attrs = [{"alpha": 1.0}, {"alpha": 2.0}]
    hist = _weisfeiler_lehman_histograms(
        node_types,
        edge_index,
        batch_vec,
        num_graphs=2,
        node_attributes=attrs,
        iterations=1,
    )
    assert hist is not None
    assert hist.size(0) == 2
    assert not torch.allclose(hist[0], hist[1])


def test_wl_histograms_match_when_attributes_match():
    node_types = torch.tensor([1, 1], dtype=torch.long)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch_vec = torch.tensor([0, 1], dtype=torch.long)
    attrs = [{"beta": [0.5, 1.5]}, {"beta": torch.tensor([0.5, 1.5])}]
    hist = _weisfeiler_lehman_histograms(
        node_types,
        edge_index,
        batch_vec,
        num_graphs=2,
        node_attributes=attrs,
        iterations=1,
    )
    assert hist is not None
    assert torch.allclose(hist[0], hist[1])


def test_decoder_skips_attr_loss_on_type_mismatch():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    with _AttrRegistryGuard():
        node_type = genes.NODE_TYPE_OPTIONS[0]
        genes.register_attribute_name(node_type, "foo")
        vocab.set_allowed_names(genes.ATTRIBUTE_NAMES)
        vocab.add_names(list(genes.ATTRIBUTE_NAMES))
        decoder = GraphDecoder(len(genes.NODE_TYPE_OPTIONS), latent_dim=4, shared_attr_vocab=vocab, min_pin_nodes=0)
        decoder.train()
        with torch.no_grad():
            decoder.type_head.weight.zero_()
            decoder.type_head.bias.fill_(-10.0)
            decoder.type_head.bias[0] = 10.0
            decoder.stop_head.weight.zero_()
            decoder.stop_head.bias.fill_(-10.0)
        latent = torch.zeros(1, decoder.latent_dim)
        foo_idx = vocab.name_to_index["foo"]
        teacher_attr_targets = [[[foo_idx, vocab.eos_index]]]
        teacher_attr_value_targets = [[[torch.tensor([1.0])]]]
        match_types = [torch.tensor([0], dtype=torch.long)]
        mismatch_types = [torch.tensor([1], dtype=torch.long)]
        _, aux_match = decoder(
            latent,
            teacher_attr_targets=teacher_attr_targets,
            teacher_attr_value_targets=teacher_attr_value_targets,
            teacher_node_types=match_types,
        )
        assert "loss" in aux_match
        _, aux_mismatch = decoder(
            latent,
            teacher_attr_targets=teacher_attr_targets,
            teacher_attr_value_targets=teacher_attr_value_targets,
            teacher_node_types=mismatch_types,
        )
        assert "loss" not in aux_mismatch
        assert aux_mismatch.get("attr_type_mismatch_skips") == 1


def test_graph_decoder_materializes_scalar_attribute_values():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(num_node_types=1, latent_dim=2, shared_attr_vocab=vocab)
    assert decoder._materialize_attribute_value([torch.tensor([0.7])], "bool") is True
    assert decoder._materialize_attribute_value([torch.tensor([2.4])], "int") == 2
    list_vals = decoder._materialize_attribute_value([torch.tensor([1.2]), torch.tensor([3.8])], "list[int]")
    assert list_vals == [1, 4]
    assert decoder._materialize_attribute_value(np.array([0.125], dtype=np.float32), "float") == pytest.approx(0.125)
    literal_idx = vocab.ensure_value_literal("spam")
    literal_vec = decoder.shared_attr_vocab.embedding.weight[literal_idx].detach().clone()
    assert decoder._materialize_attribute_value([literal_vec], "string") == "spam"
    tensor_vals = decoder._materialize_attribute_value(torch.tensor([1.0, 2.0]), "tensor")
    assert torch.allclose(tensor_vals, torch.tensor([1.0, 2.0]))
    fallback_tensor = decoder._materialize_attribute_value(torch.tensor([5.0, 6.0]), None)
    assert torch.allclose(fallback_tensor, torch.tensor([5.0, 6.0]))
    assert decoder._materialize_attribute_value([], "int") is None


def test_graph_decoder_masks_noncanonical_names_per_type():
    with _AttrRegistryGuard():
        kind_a = genes.NODE_TYPE_OPTIONS[0]
        kind_b = genes.NODE_TYPE_OPTIONS[1]
        genes.register_attribute_name(kind_a, "alpha")
        genes.register_attribute_name(kind_b, "beta")
        vocab = SharedAttributeVocab([], embedding_dim=4)
        vocab.set_allowed_names(genes.ATTRIBUTE_NAMES)
        vocab.add_names(list(genes.ATTRIBUTE_NAMES))
        decoder = GraphDecoder(len(genes.NODE_TYPE_OPTIONS), latent_dim=4, shared_attr_vocab=vocab)
        logits = torch.zeros(vocab.embedding.num_embeddings)
        masked = decoder._mask_logits_for_type(0, logits.clone())
        assert masked[vocab.name_to_index["alpha"]] == pytest.approx(0.0)
        assert masked[vocab.name_to_index["beta"]] < -1e3


def test_graph_decoder_allows_metadata_names_globally():
    with _AttrRegistryGuard():
        genes.register_attribute_name(genes.NODE_TYPE_OPTIONS[0], "alpha")
        vocab = SharedAttributeVocab([], embedding_dim=4)
        vocab.set_allowed_names(genes.ATTRIBUTE_NAMES)
        vocab.add_names(list(genes.ATTRIBUTE_NAMES))
        vocab.add_names(["__node_kind__"])
        decoder = GraphDecoder(len(genes.NODE_TYPE_OPTIONS), latent_dim=4, shared_attr_vocab=vocab)
        logits = torch.zeros(vocab.embedding.num_embeddings)
        masked = decoder._mask_logits_for_type(0, logits.clone())
        meta_idx = vocab.name_to_index["__node_kind__"]
        assert masked[meta_idx] == pytest.approx(0.0)
        assert decoder._is_name_allowed_for_type(0, meta_idx)


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


def test_online_trainer_resolves_kl_slice_bounds():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    trainer.kl_partial_slice_ratio = 0.5
    trainer.kl_partial_slice_start = 1
    assert trainer._resolve_kl_slice_bounds(6) == (1, 4)
    trainer.kl_partial_slice_dims = 2
    assert trainer._resolve_kl_slice_bounds(10) == (1, 3)
    trainer.kl_partial_slice_dims = 0
    assert trainer._resolve_kl_slice_bounds(10) == (1, 1)
    trainer.kl_partial_slice_start = 9
    assert trainer._resolve_kl_slice_bounds(9) is None
    trainer.kl_partial_slice_dims = -1
    trainer.kl_partial_slice_ratio = 0.25
    trainer.kl_partial_slice_start = 0
    assert trainer._resolve_kl_slice_bounds(8) == (0, 2)


def test_reduce_kl_loss_uses_configured_slice():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    trainer.kl_partial_slice_dims = 2
    trainer.kl_partial_slice_start = 1
    kl_per_dim = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], dtype=torch.float, requires_grad=True)
    sliced = trainer._reduce_kl_loss(kl_per_dim)
    expected = torch.tensor([2.0 + 3.0, 3.0 + 4.0]).mean()
    assert sliced.item() == pytest.approx(float(expected))
    trainer.kl_partial_slice_start = 10  # invalid slice falls back to full latent
    full = trainer._reduce_kl_loss(kl_per_dim)
    assert full.item() == pytest.approx(float(kl_per_dim.sum(dim=1).mean()))
    trainer.kl_partial_slice_start = 0
    trainer.kl_partial_slice_dims = 0  # explicit disable
    disabled = trainer._reduce_kl_loss(kl_per_dim)
    assert disabled.item() == pytest.approx(0.0)


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


def test_reconstruction_loss_handles_none_attribute_values():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    graph = _make_graph(1, [])
    batch = Batch.from_data_list([graph])
    decoded_graphs = [
        {
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "node_attributes": [{"scale": None}],
        }
    ]
    target_graph_attrs = [[{"scale": torch.tensor([1.0])}]]

    loss_adj, loss_feat = trainer._compute_reconstruction_losses(
        batch=batch,
        decoded_graphs=decoded_graphs,
        decoder_aux=None,
        target_graph_attrs=target_graph_attrs,
        teacher_force_weight=1.0,
    )

    assert torch.isfinite(loss_adj)
    assert torch.isfinite(loss_feat)
    assert loss_feat.item() > 0.0


def test_reconstruction_loss_skips_type_mismatched_nodes():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    graph = _make_graph(1, [])
    batch = Batch.from_data_list([graph])
    decoded_common = {
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_attributes": [{"foo": torch.tensor([0.0])}],
    }
    target_graph_attrs = [[{"foo": torch.tensor([1.0])}]]
    teacher_node_types = [torch.tensor([0], dtype=torch.long)]

    decoded_match = [{**decoded_common, "node_types": torch.tensor([0], dtype=torch.long)}]
    _loss_adj, loss_feat_match = trainer._compute_reconstruction_losses(
        batch=batch,
        decoded_graphs=decoded_match,
        decoder_aux=None,
        target_graph_attrs=target_graph_attrs,
        teacher_node_types=teacher_node_types,
    )
    assert loss_feat_match.item() > 0.0

    decoded_mismatch = [{**decoded_common, "node_types": torch.tensor([1], dtype=torch.long)}]
    _loss_adj, loss_feat_skip = trainer._compute_reconstruction_losses(
        batch=batch,
        decoded_graphs=decoded_mismatch,
        decoder_aux=None,
        target_graph_attrs=target_graph_attrs,
        teacher_node_types=teacher_node_types,
    )
    assert loss_feat_skip.item() == pytest.approx(0.0)


def test_reconstruction_loss_only_skips_mismatched_nodes():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    graph = _make_graph(2, [])
    batch = Batch.from_data_list([graph])
    decoded_graphs = [
        {
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "node_attributes": [
                {"foo": torch.tensor([0.0])},
                {"foo": torch.tensor([0.0])},
            ],
            "node_types": torch.tensor([1, 0], dtype=torch.long),
        }
    ]
    target_graph_attrs = [
        [
            {"foo": torch.tensor([1.0])},
            {"foo": torch.tensor([1.0])},
        ]
    ]
    teacher_node_types = [torch.tensor([0, 0], dtype=torch.long)]

    _loss_adj, loss_feat = trainer._compute_reconstruction_losses(
        batch=batch,
        decoded_graphs=decoded_graphs,
        decoder_aux=None,
        target_graph_attrs=target_graph_attrs,
        teacher_node_types=teacher_node_types,
    )

    assert loss_feat.item() == pytest.approx(1.0)


def test_graph_decoder_respects_node_budget(monkeypatch):
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(num_node_types=3, latent_dim=8, shared_attr_vocab=vocab, hidden_dim=4)
    decoder.max_nodes = 0  # Force the decoder to allow only a single node.
    decoder.max_edges_per_node = 1
    decoder.max_edges_per_graph = 1

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(-20.0)  # near-zero stop prob ⇒ continue forever without a budget.

    def _always_continue(prob):
        return torch.ones_like(prob)

    monkeypatch.setattr(torch, "bernoulli", _always_continue)

    latent = torch.zeros(1, decoder.latent_dim)
    graphs = decoder(latent)
    assert isinstance(graphs, list)
    assert len(graphs) == 1
    graph = graphs[0]
    assert graph["node_types"].shape[0] == 1  # budget enforces a single node even though sampling never stops.


def test_graph_decoder_marks_max_nodes_hit():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(num_node_types=3, latent_dim=8, shared_attr_vocab=vocab, hidden_dim=4)
    decoder.max_nodes = 0
    decoder.eval()

    latent = torch.zeros(1, decoder.latent_dim)
    graphs = decoder(latent)

    assert isinstance(graphs, list)
    assert graphs, "decoder returned no graphs"
    graph = graphs[0]
    assert graph.get("_max_nodes_hit") is True
    assert graph.get("_decoder_max_nodes") == 0


def test_graph_decoder_marks_max_edges_hit():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(num_node_types=3, latent_dim=8, shared_attr_vocab=vocab, hidden_dim=4)
    decoder.max_edges_per_graph = 0
    decoder.eval()

    latent = torch.zeros(1, decoder.latent_dim)
    graphs = decoder(latent)

    assert graphs, "decoder returned no graphs"
    graph = graphs[0]
    assert graph.get("_max_edges_hit") is True
    assert graph.get("_decoder_max_edges") == 0


def test_graph_decoder_soft_stop_reduces_continue_probability():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(num_node_types=3, latent_dim=8, shared_attr_vocab=vocab, hidden_dim=4)
    decoder.max_nodes = 10
    decoder.node_stop_decay_start_ratio = 0.6
    decoder.node_stop_decay_margin = 0
    decoder.node_stop_decay_min = 0.1

    baseline = torch.full((1, 1), 0.8)
    early = decoder._apply_node_continue_decay(baseline.clone(), emitted_nodes=2, minimum_required_nodes=1)
    assert torch.allclose(early, baseline)

    near_cap = decoder._apply_node_continue_decay(baseline.clone(), emitted_nodes=9, minimum_required_nodes=1)
    assert decoder.node_stop_decay_min <= near_cap.item() < baseline.item()


def test_graph_decoder_edge_soft_stop_reduces_probability():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(num_node_types=3, latent_dim=8, shared_attr_vocab=vocab, hidden_dim=4)
    decoder.max_edges_per_graph = 100
    decoder.edge_stop_decay_start_ratio = 0.5
    decoder.edge_stop_decay_margin = 0
    decoder.edge_stop_decay_min = 0.05

    baseline = torch.full((1,), 0.7)
    early = decoder._apply_edge_continue_decay(baseline.clone(), emitted_edges=10)
    assert torch.allclose(early, baseline)

    near_cap = decoder._apply_edge_continue_decay(baseline.clone(), emitted_edges=90)
    assert decoder.edge_stop_decay_min <= near_cap.item() < baseline.item()


def test_graph_decoder_enforces_min_pin_nodes():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=3,
    )
    decoder.eval()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(20.0)  # would normally stop immediately

    latent = torch.zeros(1, decoder.latent_dim)
    graphs = decoder(latent)
    graph = graphs[0]
    assert len(graph["node_attributes"]) >= 3


def test_graph_decoder_seeds_input_pin_roles():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=4,
        required_input_count=2,
    )
    decoder.eval()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(20.0)

    latent = torch.zeros(1, decoder.latent_dim)
    graphs = decoder(latent)
    graph = graphs[0]
    attrs = graph["node_attributes"]
    assert attrs[0]["pin_role"] == "input"
    assert attrs[0]["pin_slot_index"] == 0
    assert attrs[1]["pin_role"] == "input"
    assert attrs[1]["pin_slot_index"] == 1


def test_graph_decoder_seeds_output_pin_roles():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=5,
        required_input_count=2,
        required_output_slots=[0, 3],
    )
    decoder.eval()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(20.0)

    latent = torch.zeros(1, decoder.latent_dim)
    graph = decoder(latent)[0]
    outputs = [attrs for attrs in graph["node_attributes"] if attrs.get("pin_role") == "output"]
    assert any(attr.get("pin_slot_index") == 0 for attr in outputs)
    assert any(attr.get("pin_slot_index") == 3 for attr in outputs)


def test_graph_decoder_fallbacks_to_min_pin_nodes_for_inputs():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=5,
        required_input_count=1,
        required_output_slots=[0],
    )
    decoder.eval()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(20.0)

    graph = decoder(torch.zeros(1, decoder.latent_dim))[0]
    attrs = graph["node_attributes"]
    expected_inputs = 4  # min_pin_nodes (5) - len(output slots) (1)
    observed_inputs = [idx for idx, attr in enumerate(attrs) if attr.get("pin_role") == "input"]
    assert len(observed_inputs) >= expected_inputs
    for slot in range(expected_inputs):
        assert attrs[slot]["pin_role"] == "input"
        assert attrs[slot]["pin_slot_index"] == slot


def test_graph_decoder_reserves_slot_for_outputs_when_inputs_disabled():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=0,
        required_input_count=0,
        required_output_slots=[0],
    )
    decoder.eval()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(20.0)

    graph = decoder(torch.zeros(1, decoder.latent_dim))[0]
    attrs = graph["node_attributes"]
    inputs = [attr for attr in attrs if attr.get("pin_role") == "input"]
    outputs = [attr for attr in attrs if attr.get("pin_role") == "output"]

    assert inputs, "decoder should synthesize at least one input pin"
    assert outputs, "decoder must leave room for configured outputs"
    assert attrs[len(inputs)]["pin_role"] == "output"
    assert outputs[0]["pin_slot_index"] == 0


def test_graph_decoder_attaches_fallback_edge_when_output_unreferenced():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=2,
        required_input_count=1,
        required_output_slots=[0],
    )
    decoder.train()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(20.0)
        decoder.edge_head.weight.zero_()
        decoder.edge_head.bias.fill_(-20.0)

    teacher_nodes = decoder._resolved_required_input_slots() + len(decoder.required_output_slots)
    eos = decoder.attr_eos_index
    teacher_attr_targets = [[[eos] for _ in range(teacher_nodes)]]

    graphs, _ = decoder(torch.zeros(1, decoder.latent_dim), teacher_attr_targets=teacher_attr_targets)
    graph = graphs[0]
    edge_index = graph["edge_index"]
    assert edge_index.numel() > 0
    outputs = [idx for idx, attr in enumerate(graph["node_attributes"]) if attr.get("pin_role") == "output"]
    referenced = set(edge_index[1].tolist())
    for node_idx in outputs:
        assert node_idx in referenced


def test_graph_decoder_teacher_forcing_extends_minimum_nodes():
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=4,
        required_input_count=3,
        required_output_slots=[0],
    )
    decoder.train()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(-20.0)

    teacher_nodes = 2  # deliberately less than min_pin_nodes to force decoder-generated extras
    eos = decoder.attr_eos_index
    teacher_attr_targets = [[[eos] for _ in range(teacher_nodes)]]

    graphs, _ = decoder(torch.zeros(1, decoder.latent_dim), teacher_attr_targets=teacher_attr_targets)
    graph = graphs[0]
    attrs = graph["node_attributes"]

    assert len(attrs) >= 4
    inputs = [attr for attr in attrs if attr.get("pin_role") == "input"]
    outputs = [attr for attr in attrs if attr.get("pin_role") == "output"]

    assert len(inputs) >= 3
    assert outputs, "decoder must reserve at least one output slot"


def test_graph_decoder_invokes_pin_enforcement_once(monkeypatch):
    vocab = SharedAttributeVocab([], embedding_dim=4)
    decoder = GraphDecoder(
        num_node_types=3,
        latent_dim=8,
        shared_attr_vocab=vocab,
        hidden_dim=4,
        min_pin_nodes=4,
        required_input_count=3,
        required_output_slots=[0],
    )
    decoder.eval()

    with torch.no_grad():
        decoder.stop_head.weight.zero_()
        decoder.stop_head.bias.fill_(20.0)

    call_counts: list[int] = []
    original = GraphDecoder._enforce_required_pin_metadata

    def tracker(self, node_attributes):
        call_counts.append(len(node_attributes))
        return original(self, node_attributes)

    monkeypatch.setattr(GraphDecoder, "_enforce_required_pin_metadata", tracker)

    graphs = decoder(torch.zeros(1, decoder.latent_dim))
    assert len(graphs) == 1

    expected_nodes = max(
        decoder._min_required_nodes,
        decoder._resolved_required_input_slots() + len(decoder.required_output_slots),
        1,
    )
    assert len(call_counts) == 1
    node_attrs_len = len(graphs[0]["node_attributes"])
    assert call_counts[0] == node_attrs_len
    assert node_attrs_len >= expected_nodes
