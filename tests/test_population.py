import os
import sys
from collections import deque
from types import SimpleNamespace

import neat
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import population as population_module
from attributes import FloatAttribute, IntAttribute
from genes import NODE_TYPE_TO_INDEX, ConnectionGene, NodeGene
from genome import OptimizerGenome
from metrics import AreaUnderTaskMetrics, MemoryCost, TimeCost
from models import ManyLossMinimaModel
from population import GuidedPopulation
from relative_rank_stagnation import RelativeRankStagnation
from reproduction import GuidedReproduction
from search_space_compression import (
    FitnessPredictor,
    GraphEncoder,
    NodeAttributeDeepSetEncoder,
    OnlineTrainer,
    SharedAttributeVocab,
    flatten_task_features,
)
from tasks import RegressionTask


class DummyStatefulOptimizer(torch.nn.Module):
    """Mimics a TorchScript optimizer that keeps per-parameter buffers."""

    def __init__(self):
        super().__init__()
        # Intentionally create a mismatched buffer shape to simulate a prior task
        # with different dimensionality.
        self.state_buffers = {"fc1.weight": torch.zeros(1, 1)}

    def forward(self, loss, prev_loss, named_params):
        updated = {}
        for name, param in named_params:
            buf = self.state_buffers.get(name)
            if buf is None:
                buf = torch.zeros_like(param)
            if tuple(buf.shape) != tuple(param.shape):
                raise RuntimeError("optimizer buffer shape mismatch")
            updated[name] = param.detach().clone()
            # Persist the buffer we used so tests can inspect the resized shapes.
            self.state_buffers[name] = buf.clone().detach()
        return updated


class EmptyStateOptimizer:
    """Simple object mimicking TorchScript module with dict attribute."""

    def __init__(self):
        self.state = {}


class DummyGuideModel(torch.nn.Module):
    """Minimal module exposing shared_attr_vocab for OnlineTrainer tests."""

    def __init__(self):
        super().__init__()
        self.shared_attr_vocab = SharedAttributeVocab([], embedding_dim=4)
        self.dummy = torch.nn.Parameter(torch.zeros(1))


class StubFitnessPredictor(torch.nn.Module):
    output_dim = 1

    def forward(self, z):  # pragma: no cover - simple stub
        zeros = torch.zeros(z.size(0), self.output_dim, device=z.device, dtype=z.dtype)
        return zeros, torch.zeros_like(zeros), None


class StubGuide:
    def __init__(self, graph_factories, latent_dim=2):
        self.graph_encoder = SimpleNamespace(latent_dim=latent_dim)
        self.graph_latent_mask = torch.ones(latent_dim)
        self.fitness_predictor = StubFitnessPredictor()
        self.decoder = SimpleNamespace()
        self._graph_factories = list(graph_factories)
        self._decode_calls = 0

    def decode(self, latent):
        idx = min(self._decode_calls, len(self._graph_factories) - 1)
        graph = self._graph_factories[idx]()
        self._decode_calls += 1
        return [graph]


def make_config():
    config_path = os.path.join(os.path.dirname(__file__), os.pardir, "neat-config")
    return neat.Config(
        OptimizerGenome,
        GuidedReproduction,
        neat.DefaultSpeciesSet,
        RelativeRankStagnation,
        config_path,
    )


def create_simple_genome(key=0):
    genome = OptimizerGenome(key)
    ng0 = NodeGene(0, None)
    ng0.node_type = "aten::add"
    ng0.dynamic_attributes = {IntAttribute("a"): 1}
    ng1 = NodeGene(1, None)
    ng1.node_type = "aten::mul"
    ng1.dynamic_attributes = {FloatAttribute("b"): 0.5}
    genome.nodes = {0: ng0, 1: ng1}
    cg = ConnectionGene((0, 1))
    cg.enabled = True
    genome.connections = {(0, 1): cg}
    genome.next_node_id = 2
    return genome


def make_decoded_graph(bias):
    node_types = torch.tensor([NODE_TYPE_TO_INDEX["aten::add"], NODE_TYPE_TO_INDEX["aten::mul"]], dtype=torch.long)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    node_attrs = [
        {"node_type": "aten::add", "bias": torch.tensor([bias])},
        {"node_type": "aten::mul", "bias": torch.tensor([bias * 2])},
    ]
    return {
        "node_types": node_types,
        "edge_index": edge_index,
        "node_attributes": node_attrs,
    }


def make_graph_factory(bias):
    def factory():
        graph = make_decoded_graph(bias)
        return {
            "node_types": graph["node_types"].clone(),
            "edge_index": graph["edge_index"].clone(),
            "node_attributes": [dict(attrs) for attrs in graph["node_attributes"]],
        }

    return factory


def configure_stub_population(graph_factories, monkeypatch):
    config = make_config()
    pop = GuidedPopulation(config)
    pop.guide = StubGuide(graph_factories)
    pop._optimizer_updates_parameters = lambda *args, **kwargs: True
    monkeypatch.setattr(
        population_module,
        "rebuild_and_script",
        lambda *args, **kwargs: DummyStatefulOptimizer(),
    )
    return pop, config


def test_genome_to_data():
    config = make_config()
    pop = GuidedPopulation(config)
    genome = create_simple_genome()
    data = pop.genome_to_data(genome)

    assert genome.graph_dict is not None


def test_genome_to_data_preserves_graph_ir(monkeypatch):
    config = make_config()
    pop = GuidedPopulation(config)
    genome = create_simple_genome()
    genome.graph_dict = {
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "node_attributes": [{}, {}],
        "graph_ir": {"inputs": [], "outputs": [], "nodes": []},
        "module_state": {"step": 1},
        "module_type": "__torch__.Dummy",
    }

    data = pop.genome_to_data(genome)

    assert genome.graph_dict.get("graph_ir") == {"inputs": [], "outputs": [], "nodes": []}
    assert genome.graph_dict.get("module_state") == {"step": 1}
    assert genome.graph_dict.get("module_type") == "__torch__.Dummy"
    assert list(data.node_types.tolist()) == [NODE_TYPE_TO_INDEX["aten::add"], NODE_TYPE_TO_INDEX["aten::mul"]]
    assert data.edge_index.size(1) == 1
    assert data.edge_index[:, 0].tolist() == [0, 1]
    assert len(data.node_attributes) == 2


def test_online_trainer_deduplicates_graphs():
    model = DummyGuideModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])

    base_attrs = [
        {"alpha": torch.tensor([1.0])},
        {"beta": torch.tensor([-0.5, 0.25])},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    graph = Data(node_types=torch.tensor([0, 1], dtype=torch.long), edge_index=edge_index, node_attributes=base_attrs)

    trainer.add_data([graph], [{AreaUnderTaskMetrics: 0.1}])
    trainer.add_data([graph.clone()], [{AreaUnderTaskMetrics: 0.2}])
    assert len(trainer.dataset) == 1

    variant_attrs = [
        {"alpha": torch.tensor([1.0])},
        {"beta": torch.tensor([-0.25, 0.5])},
    ]
    variant_graph = Data(
        node_types=torch.tensor([0, 1], dtype=torch.long),
        edge_index=edge_index,
        node_attributes=variant_attrs,
    )
    trainer.add_data([variant_graph], [{AreaUnderTaskMetrics: 0.3}])
    assert len(trainer.dataset) == 2

    invalid_graph = Data(
        node_types=torch.tensor([0, 1], dtype=torch.long),
        edge_index=torch.tensor([[1], [0]], dtype=torch.long),
        node_attributes=base_attrs,
    )
    trainer.add_data([invalid_graph], [{AreaUnderTaskMetrics: 0.0}], invalid_flags=[True])
    assert len(trainer.invalid_dataset) == 1

    trainer.add_data([invalid_graph.clone()], [{AreaUnderTaskMetrics: 0.0}], invalid_flags=[True])
    assert len(trainer.invalid_dataset) == 1


def test_generate_guided_offspring_skips_exact_duplicates(monkeypatch):
    pop, config = configure_stub_population(
        [make_graph_factory(0.1), make_graph_factory(0.1), make_graph_factory(0.2)],
        monkeypatch,
    )
    offspring = pop.generate_guided_offspring(
        [], config, n_offspring=3, latent_steps=1, max_decode_attempts=1, decode_jitter_std=0.0
    )
    assert len(offspring) == 2


def test_generate_guided_offspring_allows_attribute_variants(monkeypatch):
    pop, config = configure_stub_population(
        [make_graph_factory(0.1), make_graph_factory(0.25)],
        monkeypatch,
    )
    offspring = pop.generate_guided_offspring(
        [], config, n_offspring=2, latent_steps=1, max_decode_attempts=1, decode_jitter_std=0.0
    )
    assert len(offspring) == 2


def test_generation_eval_steps_respects_max_cap():
    config = make_config()
    config.max_evaluation_steps = 30
    pop = GuidedPopulation(config)
    pop.generation = 50
    assert pop._generation_eval_steps() == 30

    pop.max_regression_epochs = None
    pop.generation = 5
    assert pop._generation_eval_steps() == 25


def test_graph_output_slot_precheck_detects_missing_slots():
    config = make_config()
    pop = GuidedPopulation(config)
    graph_dict = {
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_attributes": [{} for _ in range(3)],
    }

    ok, details = pop._graph_output_slot_coverage(graph_dict)
    assert not ok
    assert details["missing_slots"] == config.genome_config.output_keys

    first_output = config.genome_config.output_keys[0]
    graph_dict["edge_index"] = torch.tensor([[0], [first_output]], dtype=torch.long)

    ok2, details2 = pop._graph_output_slot_coverage(graph_dict)
    assert ok2
    assert details2["missing_slots"] == []


def test_generate_guided_offspring():
    config = make_config()
    pop = GuidedPopulation(config)
    pop.guide.decoder.max_nodes = 2
    pop.guide.decoder.max_attributes_per_node = 2

    g1 = create_simple_genome(0)
    g1.fitness = 1.0
    g2 = create_simple_genome(1)
    g2.fitness = 0.5
    pop.genome_to_data(g1)
    pop.genome_to_data(g2)

    RegressionTask.random_init(num_samples=4, silent=True)
    offspring = pop.generate_guided_offspring([g1, g2], config, n_offspring=2, latent_steps=1)

    assert isinstance(offspring, list)
    assert 1 <= len(offspring) <= 4
    for child in offspring:
        assert isinstance(child, OptimizerGenome)
        assert child.graph_dict is not None


def test_eval_genomes_penalizes_skipped_empty_graphs():
    config = make_config()
    pop = GuidedPopulation(config)
    task = RegressionTask.random_init(num_samples=4, silent=True)

    genome = create_simple_genome()
    genome.connections = {}
    genome.skip_evaluation = True
    genome.invalid_reason = "empty_graph"

    pop.eval_genomes([(0, genome)], config, task, steps=1)

    assert getattr(genome, "invalid_graph", False)
    assert genome.fitnesses[AreaUnderTaskMetrics] == GuidedPopulation.INVALID_METRIC_VALUE
    assert genome.fitnesses[TimeCost] == GuidedPopulation.INVALID_METRIC_VALUE
    assert genome.fitnesses[MemoryCost] == GuidedPopulation.INVALID_METRIC_VALUE
    assert genome.fitness == -0.1


def test_generate_guided_offspring_handles_missing_elites():
    config = make_config()
    pop = GuidedPopulation(config)
    pop.guide.decoder.max_nodes = 2
    pop.guide.decoder.max_attributes_per_node = 2

    RegressionTask.random_init(num_samples=4, silent=True)

    offspring = pop.generate_guided_offspring([], config, n_offspring=3, latent_steps=1)

    assert isinstance(offspring, list)
    assert len(offspring) >= 1
    assert len(offspring) <= 3
    for child in offspring:
        assert isinstance(child, OptimizerGenome)


def _make_empty_graph_dict(node_count: int, node_attrs):
    return {
        "node_types": torch.zeros(node_count, dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_attributes": node_attrs,
    }


def test_repair_requires_configured_io_nodes():
    config = make_config()
    pop = GuidedPopulation(config)
    num_inputs = len(config.genome_config.input_keys)
    num_outputs = len(config.genome_config.output_keys)

    # Omit one required input label.
    node_attrs = []
    for idx in range(max(1, num_inputs + num_outputs)):
        if idx < max(0, num_inputs - 1):
            node_attrs.append({"node_type": "input"})
        elif idx < max(0, num_inputs + num_outputs - 1):
            node_attrs.append({"node_type": "output"})
        else:
            node_attrs.append({"node_type": "hidden"})
    graph = _make_empty_graph_dict(len(node_attrs), node_attrs)
    assert not pop._repair_graph_dict(graph)
    assert graph["edge_index"].numel() == 0

    # Omit one required output label.
    node_attrs = []
    for idx in range(max(1, num_inputs + num_outputs)):
        if idx < num_inputs:
            node_attrs.append({"node_type": "input"})
        elif idx < num_inputs + max(0, num_outputs - 1):
            node_attrs.append({"node_type": "output"})
        else:
            node_attrs.append({"node_type": "hidden"})
    graph = _make_empty_graph_dict(len(node_attrs), node_attrs)
    assert not pop._repair_graph_dict(graph)
    assert graph["edge_index"].numel() == 0


def test_repair_connects_each_input_to_output():
    config = make_config()
    pop = GuidedPopulation(config)
    num_inputs = len(config.genome_config.input_keys)
    num_outputs = len(config.genome_config.output_keys)

    node_attrs = []
    for _ in range(num_inputs):
        node_attrs.append({"node_type": "input"})
    for _ in range(num_outputs):
        node_attrs.append({"node_type": "output"})
    node_attrs.append({"node_type": "hidden"})

    graph = _make_empty_graph_dict(len(node_attrs), node_attrs)
    assert pop._repair_graph_dict(graph)
    edge_index = graph["edge_index"]
    assert edge_index.numel() > 0

    edges = edge_index.t().tolist() if edge_index.numel() else []
    adjacency = {idx: [] for idx in range(len(node_attrs))}
    for src, dst in edges:
        adjacency[src].append(dst)

    output_nodes = [idx for idx, attrs in enumerate(node_attrs) if attrs.get("node_type") == "output"]
    input_nodes = [idx for idx, attrs in enumerate(node_attrs) if attrs.get("node_type") == "input"]

    def reachables_from(source):
        seen = {source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for dst in adjacency.get(current, []):
                if dst not in seen:
                    seen.add(dst)
                    queue.append(dst)
        return seen

    reachable_union = set()
    for inp in input_nodes:
        seen = reachables_from(inp)
        reachable_union.update(seen)
        assert any(out in seen for out in output_nodes)

    for out in output_nodes:
        assert out in reachable_union


def test_fitness_predictor_returns_per_metric_log_scales():
    predictor = FitnessPredictor(latent_dim=4, hidden_dim=8, fitness_dim=3)
    z_graph = torch.randn(5, 4)
    predictor.log_metric_scale.data = torch.tensor([0.1, -0.2, 0.05])

    preds, log_scales, convex_pred = predictor(z_graph)

    assert preds.shape == (5, 3)
    assert log_scales.shape == (5, 3)
    assert convex_pred is None
    # All rows should share the learned scale parameters
    expected = predictor.log_metric_scale.detach()
    assert torch.allclose(log_scales[0], expected)
    assert torch.allclose(log_scales, expected.unsqueeze(0).expand_as(log_scales))


def test_fitness_predictor_emits_convex_predictions():
    predictor = FitnessPredictor(latent_dim=4, hidden_dim=8, fitness_dim=2, icnn_hidden_dims=(4, 2))
    z_graph = torch.randn(3, 4)
    preds, log_scales, convex_pred = predictor(z_graph)
    assert preds.shape == (3, 2)
    assert log_scales.shape == (3, 2)
    assert convex_pred.shape == (3, 2)


def test_flatten_task_features_respects_expected_length():
    import numpy as np

    features = [np.array([1.0, 2.0]), np.array([3.0])]
    flat = flatten_task_features(features, expected_len=5)
    assert flat.shape[0] == 5
    assert np.allclose(flat[:3], [1.0, 2.0, 3.0])
    assert np.allclose(flat[3:], np.zeros(2))

    # Truncation when too long
    features_long = [np.arange(10)]
    flat_long = flatten_task_features(features_long, expected_len=4)
    assert flat_long.shape[0] == 4
    assert np.allclose(flat_long, np.arange(4))


def test_graph_encoder_pads_trailing_empty_graphs():
    shared_vocab = SharedAttributeVocab([], embedding_dim=4)
    attr_encoder = NodeAttributeDeepSetEncoder(shared_vocab, encoder_hdim=4, aggregator_hdim=4, out_dim=4)
    encoder = GraphEncoder(len(NODE_TYPE_TO_INDEX), attr_encoder, latent_dim=2, hidden_dims=[4])

    populated = Data(
        node_types=torch.tensor([NODE_TYPE_TO_INDEX["aten::add"]], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_attributes=[{IntAttribute("foo"): 1}],
    )
    empty = Data(
        node_types=torch.empty(0, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_attributes=[],
    )

    batch = Batch.from_data_list([populated, empty])
    mu, lv = encoder(
        batch.node_types,
        batch.edge_index,
        batch.node_attributes,
        batch.batch,
        num_graphs=batch.num_graphs,
    )

    assert batch.num_graphs == 2
    assert mu.shape[0] == batch.num_graphs
    assert lv.shape[0] == batch.num_graphs


def test_node_attribute_encoder_accepts_tensor_values():
    shared_vocab = SharedAttributeVocab([], embedding_dim=6)
    attr_encoder = NodeAttributeDeepSetEncoder(shared_vocab, encoder_hdim=4, aggregator_hdim=4, out_dim=3)

    tensor_value = torch.arange(6.0).reshape(3, 2)

    flattened = attr_encoder.get_value_tensor(tensor_value.clone())
    assert flattened.shape[0] == attr_encoder.max_value_dim
    assert torch.allclose(flattened[: tensor_value.numel()], tensor_value.reshape(-1))

    encoded = attr_encoder({"tensor": tensor_value})
    assert encoded.shape == (attr_encoder.out_dim,)


def test_evaluate_optimizer_resizes_state_before_execution():
    config = make_config()
    pop = GuidedPopulation(config)
    task = RegressionTask.random_init(num_samples=4, silent=True)
    model = ManyLossMinimaModel(task.train_data.num_input_features)
    optimizer = DummyStatefulOptimizer()

    # Should not raise even though optimizer buffers start with mismatched shapes.
    pop.evaluate_optimizer(optimizer, model, task, steps=1)

    assert tuple(optimizer.state_buffers["fc1.weight"].shape) == tuple(model.fc1.weight.shape)


def test_evaluate_optimizer_resets_state_and_step_each_run():
    class TrackingOptimizer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Pre-populate the dict so the reset logic has something to clear.
            self.state = {"fc1.weight": torch.ones(1, 1)}
            self.step = 7
            self.observed = []

        def forward(self, loss, prev_loss, named_parameters):
            # Record the state that evaluation sees before any updates.
            self.observed.append((self.step, dict(self.state)))
            updated = {}
            for name, param in named_parameters:
                updated[name] = param.detach().clone()
            # Store non-zero tensors so subsequent runs would show stale state if not reset.
            self.state = {name: torch.ones_like(param) for name, param in named_parameters}
            self.step += 1
            return updated

    config = make_config()
    pop = GuidedPopulation(config)
    task = RegressionTask.random_init(num_samples=4, silent=True)
    model = ManyLossMinimaModel(task.train_data.num_input_features)
    optimizer = TrackingOptimizer()

    pop.evaluate_optimizer(optimizer, model, task, steps=1)
    # Simulate leftover state before the second evaluation.
    optimizer.state = {k: v + 5 for k, v in optimizer.state.items()}
    optimizer.step = 42
    pop.evaluate_optimizer(optimizer, model, task, steps=1)

    assert [step for step, _ in optimizer.observed] == [0, 0]
    assert all(len(state) == 0 for _, state in optimizer.observed)


def test_evaluate_optimizer_marks_nan_outputs_invalid():
    class NaNOptimizer(torch.nn.Module):
        def forward(self, loss, prev_loss, named_parameters):
            return {name: torch.full_like(param, float("nan")) for name, param in named_parameters}

    config = make_config()
    pop = GuidedPopulation(config)
    task = RegressionTask.random_init(num_samples=4, silent=True)
    model = ManyLossMinimaModel(task.train_data.num_input_features)

    optimizer = NaNOptimizer()
    result = pop.evaluate_optimizer(optimizer, model, task, steps=1)

    assert result is None


def test_optimizer_state_attributes_include_empty_dicts():
    config = make_config()
    pop = GuidedPopulation(config)
    opt = EmptyStateOptimizer()

    attrs = pop._optimizer_state_attributes(opt)

    assert "state" in attrs
