import os
import sys

import neat
import numpy as np
import torch
from torch_geometric.data import Batch, Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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


def test_genome_to_data():
    config = make_config()
    pop = GuidedPopulation(config)
    genome = create_simple_genome()
    data = pop.genome_to_data(genome)

    assert genome.graph_dict is not None
    assert list(data.node_types.tolist()) == [NODE_TYPE_TO_INDEX["aten::add"], NODE_TYPE_TO_INDEX["aten::mul"]]
    assert data.edge_index.size(1) == 1
    assert data.edge_index[:, 0].tolist() == [0, 1]
    assert len(data.node_attributes) == 2
    assert "a" in pop.shared_attr_vocab.name_to_index
    assert "b" in pop.shared_attr_vocab.name_to_index


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

    task = RegressionTask.random_init(num_samples=4, silent=True)
    offspring = pop.generate_guided_offspring(task, [g1, g2], config, n_offspring=2, latent_steps=1)

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

    task = RegressionTask.random_init(num_samples=4, silent=True)

    offspring = pop.generate_guided_offspring(task, [], config, n_offspring=3, latent_steps=1)

    assert isinstance(offspring, list)
    assert len(offspring) >= 1
    assert len(offspring) <= 3
    for child in offspring:
        assert isinstance(child, OptimizerGenome)


def test_fitness_predictor_returns_per_metric_log_scales():
    predictor = FitnessPredictor(latent_dim=4, hidden_dim=8, fitness_dim=3)
    z_graph = torch.randn(5, 4)
    predictor.log_metric_scale.data = torch.tensor([0.1, -0.2, 0.05])

    preds, log_scales = predictor(z_graph)

    assert preds.shape == (5, 3)
    assert log_scales.shape == (5, 3)
    # All rows should share the learned scale parameters
    expected = predictor.log_metric_scale.detach()
    assert torch.allclose(log_scales[0], expected)
    assert torch.allclose(log_scales, expected.unsqueeze(0).expand_as(log_scales))


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
