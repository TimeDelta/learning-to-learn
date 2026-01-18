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
from models import ManyLossMinimaModel
from population import GuidedPopulation
from reproduction import GuidedReproduction
from search_space_compression import (
    GraphEncoder,
    NodeAttributeDeepSetEncoder,
    OnlineTrainer,
    SharedAttributeVocab,
    TaskConditionedFitnessPredictor,
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
        neat.DefaultStagnation,
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


def test_trainer_handles_variable_fitness_dims():
    config = make_config()
    pop = GuidedPopulation(config)
    task = RegressionTask.random_init(num_samples=4, silent=True)

    class DummyMetric:
        def __init__(self, name):
            self.name = name
            self.objective = "min"

    graph = pop.genome_to_data(create_simple_genome())
    pop.trainer.add_data([graph], [{DummyMetric("m1"): 0.1}], task.name(), task.features)
    assert pop.trainer.max_fitness_dim == 1

    graph2 = pop.genome_to_data(create_simple_genome())
    pop.trainer.add_data([graph2], [{DummyMetric("m1"): 0.2, DummyMetric("m2"): 0.3}], task.name(), task.features)
    assert pop.trainer.max_fitness_dim == 2
    assert pop.trainer.dataset[0].y.numel() == 2


def test_task_conditioned_predictor_builds_per_task_heads():
    predictor = TaskConditionedFitnessPredictor(latent_dim=4, hidden_dim=8)
    z_graph = torch.randn(3, 2)
    z_task = torch.randn(3, 2)
    task_types = torch.tensor([0, 1, 0], dtype=torch.long)
    fitness_dims = torch.tensor([1, 2, 1], dtype=torch.long)

    outputs = predictor(z_graph, z_task, task_types, fitness_dims)
    assert outputs.shape == (3, 2)
    assert torch.allclose(outputs[task_types == 0, 1], torch.zeros(2))

    head_out = predictor.predict_task(1, 2, z_graph[:1], z_task[:1])
    assert head_out.shape == (1, 2)


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


def test_online_trainer_aligns_predictions_to_mask():
    pred = torch.randn(2, 2)
    mask = torch.ones(2, 4)
    aligned = OnlineTrainer._align_pred_to_mask(pred, mask)
    assert aligned.shape[1] == 4
    assert torch.allclose(aligned[:, :2], pred)
    assert torch.allclose(aligned[:, 2:], torch.zeros(2, 2))

    mask_small = torch.ones(2, 1)
    aligned_small = OnlineTrainer._align_pred_to_mask(pred, mask_small)
    assert aligned_small.shape[1] == 1

    # handle 1-D predictions/masks gracefully
    one_d_pred = torch.randn(2)
    one_d_mask = torch.ones(4)
    aligned_one_d = OnlineTrainer._align_pred_to_mask(one_d_pred, one_d_mask)
    assert aligned_one_d.shape == (2, 4)


def test_trainer_normalizes_task_feature_shapes():
    config = make_config()
    pop = GuidedPopulation(config)
    graph = pop.genome_to_data(create_simple_genome())

    class DummyMetric:
        def __init__(self, name):
            self.name = name
            self.objective = "min"

    task = RegressionTask.random_init(num_samples=4, silent=True)
    pop.trainer.add_data([graph], [{DummyMetric("m1"): 0.1}], task.name(), task.features)

    data = pop.trainer.dataset[0]
    data.task_features = torch.randn(72)  # emulate legacy 1-D storage
    pop.trainer._normalize_task_feature_shapes()
    assert data.task_features.dim() == 2


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


def test_evaluate_optimizer_resizes_state_before_execution():
    config = make_config()
    pop = GuidedPopulation(config)
    task = RegressionTask.random_init(num_samples=4, silent=True)
    model = ManyLossMinimaModel(task.train_data.num_input_features)
    optimizer = DummyStatefulOptimizer()

    # Should not raise even though optimizer buffers start with mismatched shapes.
    pop.evaluate_optimizer(optimizer, model, task, steps=1)

    assert tuple(optimizer.state_buffers["fc1.weight"].shape) == tuple(model.fc1.weight.shape)


def test_optimizer_state_attributes_include_empty_dicts():
    config = make_config()
    pop = GuidedPopulation(config)
    opt = EmptyStateOptimizer()

    attrs = pop._optimizer_state_attributes(opt)

    assert "state" in attrs
