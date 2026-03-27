import copy
import json
import os
import sys
from collections import Counter, deque
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

tests_dir = os.path.dirname(__file__)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from factories import make_neat_config

import population as population_module
from attributes import FloatAttribute, IntAttribute
from genes import (
    NODE_TYPE_TO_INDEX,
    ConnectionGene,
    NodeGene,
    attribute_value_kind,
    canonical_attribute_names_for_kind,
    register_attribute_name,
    register_schema_argument_names,
)
from genome import OptimizerGenome
from graph_builder import rebuild_and_script
from loop_blocks import prime_registry
from main import create_initial_genome
from metrics import AreaUnderTaskMetrics, MemoryCost, TimeCost
from models import ManyLossMinimaModel
from population import (
    BLOCK_PAYLOAD_ATTR_PREFIX,
    BLOCK_REF_ATTR_PREFIX,
    GuidedPopulation,
    OptimizerValidationResult,
    ValidatorOutcome,
)
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

for _name in ("a", "b", "foo"):
    register_attribute_name(None, _name)


class _AttrRegistryGuard:
    def __enter__(self):
        import genes

        self._names = set(genes.ATTRIBUTE_NAMES)
        self._map = {kind: set(names) for kind, names in genes.ATTRIBUTE_NAMES_BY_KIND.items()}
        self._version = genes.ATTRIBUTE_NAMES_VERSION
        genes.ATTRIBUTE_NAMES.clear()
        genes.ATTRIBUTE_NAMES_BY_KIND.clear()
        genes.ATTRIBUTE_NAMES_VERSION = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        import genes

        genes.ATTRIBUTE_NAMES.clear()
        genes.ATTRIBUTE_NAMES.update(self._names)
        genes.ATTRIBUTE_NAMES_BY_KIND.clear()
        for kind, names in self._map.items():
            genes.ATTRIBUTE_NAMES_BY_KIND[kind] = set(names)
        genes.ATTRIBUTE_NAMES_VERSION = self._version


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


class _DummyFitnessHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1)
        self.fc2 = torch.nn.Linear(1, 1)
        self.log_metric_scale = torch.nn.Parameter(torch.zeros(1))
        self.icnn = None

    def forward(self, *args, **kwargs):  # pragma: no cover - not needed for tests
        raise NotImplementedError


class DummyGuideModel(torch.nn.Module):
    """Minimal module exposing trainer-facing attributes without heavy encoders."""

    def __init__(self):
        super().__init__()
        latent_dim = 2
        self.shared_attr_vocab = SharedAttributeVocab([], embedding_dim=4)
        self.attr_encoder = torch.nn.Linear(1, 1)
        self.graph_encoder = torch.nn.Linear(latent_dim, latent_dim)
        self.graph_encoder.latent_dim = latent_dim
        self.decoder = torch.nn.Linear(latent_dim, latent_dim)
        self.fitness_predictor = _DummyFitnessHead()
        self.log_alpha_g = torch.nn.Parameter(torch.zeros(latent_dim))
        self.register_buffer("graph_latent_mask", torch.ones(latent_dim))


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


class RelaxDecoderStub(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim=4):
        super().__init__()
        self.node_hidden_state_init = torch.nn.Linear(latent_dim, hidden_dim)
        self.stop_head = torch.nn.Linear(hidden_dim, 1)
        self.edge_head = torch.nn.Linear(hidden_dim, 1)
        self.required_output_slots = [0, 1]
        self._min_required_nodes = 3
        self.max_nodes = 8
        self._resolved_inputs = 1

    def _resolved_required_input_slots(self):
        return self._resolved_inputs

    def node_rnn(self, zero_input, hidden):  # pragma: no cover - simple activation
        return torch.tanh(hidden)

    def edge_rnn(self, edge_input, hidden):  # pragma: no cover - simple activation
        del edge_input
        return torch.tanh(hidden)


class RecordingReporters:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message):
        self.messages.append(str(message))

    def __getattr__(self, name):  # pragma: no cover - passthrough for unused hooks
        return lambda *args, **kwargs: None


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


def test_registers_schema_argument_names_for_aten_ops():
    add = torch.jit.trace(lambda x, y: x + y, (torch.ones(1), torch.ones(1)))
    with _AttrRegistryGuard():
        target = None
        for node in add.graph.nodes():
            if node.kind() == "aten::add":
                target = node
                break
        assert target is not None
        register_schema_argument_names(target)
        names = canonical_attribute_names_for_kind("aten::add")
        assert {"self", "other"}.issubset(names)
        kind = attribute_value_kind("aten::add", "self")
        assert kind == "tensor"


def make_decoded_graph(bias):
    input_type = NODE_TYPE_TO_INDEX.get("prim::GetAttr", 0)
    add_type = NODE_TYPE_TO_INDEX.get("aten::add", 0)
    mul_type = NODE_TYPE_TO_INDEX.get("aten::mul", 0)
    output_type = NODE_TYPE_TO_INDEX.get("prim::Return", 0)
    node_types = torch.tensor([input_type, input_type, input_type, add_type, mul_type, output_type], dtype=torch.long)
    node_attrs = [
        {"pin_role": "input", "pin_slot_index": 0},
        {"pin_role": "input", "pin_slot_index": 1},
        {"pin_role": "input", "pin_slot_index": 2},
        {"node_type": "aten::add", "bias": torch.tensor([bias])},
        {"node_type": "aten::mul", "bias": torch.tensor([bias * 2])},
        {"pin_role": "output", "pin_slot_index": 0},
    ]
    edges = (
        torch.tensor(
            [
                [0, 3],
                [1, 3],
                [2, 3],
                [3, 4],
                [4, 5],
            ],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )
    return {
        "node_types": node_types,
        "edge_index": edges,
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


def make_invalid_graph_factory():
    def factory():
        node_types = torch.tensor(
            [
                NODE_TYPE_TO_INDEX.get("prim::GetAttr", 0),
                NODE_TYPE_TO_INDEX.get("prim::Return", 0),
                NODE_TYPE_TO_INDEX.get("aten::add", 0),
            ],
            dtype=torch.long,
        )
        node_attrs = [
            {"pin_role": "input", "pin_slot_index": 0},
            {"pin_role": "output", "pin_slot_index": 0},
            {},
        ]
        edges = torch.empty((2, 0), dtype=torch.long)
        return {"node_types": node_types, "edge_index": edges, "node_attributes": node_attrs}

    return factory


def configure_stub_population(graph_factories, monkeypatch):
    config = make_neat_config()
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
    config = make_neat_config()
    pop = GuidedPopulation(config)
    genome = create_simple_genome()
    data = pop.genome_to_data(genome)

    assert genome.graph_dict is not None


def test_genome_to_data_preserves_graph_ir(monkeypatch):
    config = make_neat_config()
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
    offspring = pop.generate_guided_offspring([], config, n_offspring=3, latent_steps=1)
    assert len(offspring) == 2


def test_generate_guided_offspring_allows_attribute_variants(monkeypatch):
    pop, config = configure_stub_population(
        [make_graph_factory(0.1), make_graph_factory(0.25)],
        monkeypatch,
    )
    offspring = pop.generate_guided_offspring([], config, n_offspring=2, latent_steps=1)
    assert len(offspring) == 2


def test_generate_guided_offspring_buffers_near_misses(monkeypatch):
    pop, config = configure_stub_population(
        [make_graph_factory(0.1)],
        monkeypatch,
    )
    pop._optimizer_updates_parameters = lambda *args, **kwargs: False

    buffered_graphs = []

    def fake_buffer(graph_dict):
        if graph_dict:
            buffered_graphs.append(graph_dict)

    pop._buffer_decoder_replay_dict = fake_buffer

    offspring = pop.generate_guided_offspring([], config, n_offspring=1, latent_steps=1)

    assert offspring  # even penalized genomes should be returned
    assert buffered_graphs, "decoder replay buffer did not capture near-miss graphs"


def test_generate_guided_offspring_penalizes_max_nodes(monkeypatch):
    base_factory = make_graph_factory(0.1)

    def capped_factory():
        graph = base_factory()
        graph["_max_nodes_hit"] = True
        graph["_decoder_max_nodes"] = 4
        return graph

    pop, config = configure_stub_population([capped_factory], monkeypatch)
    pop._reset_guided_offspring_stats()

    offspring = pop.generate_guided_offspring([], config, n_offspring=1, latent_steps=1)

    assert offspring
    genome = offspring[0]
    assert getattr(genome, "invalid_graph", False)
    assert genome.invalid_reason == "decoder_max_nodes"
    assert genome.skip_evaluation is True
    assert genome.invalid_penalty_details.get("max_nodes") == 4
    stats = pop._guided_offspring_stats
    assert stats["decoder_max_nodes_hits"] == 1
    assert stats["decoder_max_nodes_invalid"] == 1


def test_generate_guided_offspring_penalizes_max_edges(monkeypatch):
    base_factory = make_graph_factory(0.1)

    def capped_factory():
        graph = base_factory()
        graph["_max_edges_hit"] = True
        graph["_decoder_max_edges"] = 2
        return graph

    pop, config = configure_stub_population([capped_factory], monkeypatch)
    pop._reset_guided_offspring_stats()

    offspring = pop.generate_guided_offspring([], config, n_offspring=1, latent_steps=1)

    assert offspring
    genome = offspring[0]
    assert getattr(genome, "invalid_graph", False)
    assert genome.invalid_reason == "decoder_max_edges"
    assert genome.skip_evaluation is True
    assert genome.invalid_penalty_details.get("max_edges") == 2
    stats = pop._guided_offspring_stats
    assert stats["decoder_max_edges_hits"] == 1
    assert stats["decoder_max_edges_invalid"] == 1


def test_generate_guided_offspring_records_inactive_details(monkeypatch):
    config = make_neat_config()
    config.guided_replay_fraction = 0.0
    config.guided_seed_fraction = 0.0
    config.guided_inactive_detail_limit = 1
    pop = GuidedPopulation(config)
    pop.guide = StubGuide([make_graph_factory(0.1)])
    pop._optimizer_updates_parameters = lambda *args, **kwargs: False
    monkeypatch.setattr(
        population_module,
        "rebuild_and_script",
        lambda *args, **kwargs: DummyStatefulOptimizer(),
    )
    pop._reset_guided_offspring_stats()

    pop.generate_guided_offspring([], config, n_offspring=1, latent_steps=1)

    stats = pop._guided_offspring_stats
    assert stats["inactive_details_total"] >= 1
    assert len(stats["inactive_details"]) == 1
    detail = stats["inactive_details"][0]
    assert detail["invalid_reason"] == "inactive_optimizer"
    assert "latent_norm" in detail
    assert "param_delta" in detail


def test_export_inactive_detail_history_includes_current(monkeypatch):
    pop, config = configure_stub_population([make_graph_factory(0.1)], monkeypatch)
    stats = {
        "generation": 1,
        "requested": 4,
        "accepted": 1,
        "invalid_total": 3,
        "invalid_by_reason": Counter({"inactive_optimizer": 3}),
        "inactive_details_total": 2,
        "inactive_details": [{"child_index": 0}],
        "inactive_repair_salvaged": 1,
        "inactive_repair_salvaged_total": 1,
    }
    pop._record_inactive_detail_history(stats)
    exported = pop.export_inactive_detail_history(include_current=True)
    assert exported
    assert exported[0]["inactive_invalid"] == 3


def test_log_inactive_detail_artifact_writes_json(monkeypatch, tmp_path):
    pop, config = configure_stub_population([make_graph_factory(0.1)], monkeypatch)
    stats = {
        "generation": 2,
        "requested": 5,
        "accepted": 1,
        "invalid_total": 4,
        "invalid_by_reason": Counter({"inactive_optimizer": 4}),
        "inactive_details_total": 1,
        "inactive_details": [{"child_index": 0, "param_delta": 0.0}],
        "inactive_repair_salvaged": 0,
        "inactive_repair_salvaged_total": 0,
    }
    pop._record_inactive_detail_history(stats)

    captured = {}

    class DummyRun:
        def log_artifact(self, path: str):
            payload = json.loads(Path(path).read_text())
            captured.update(payload)

    pop.log_inactive_detail_artifact(DummyRun(), artifact_file="inactive.json")
    assert captured["records"]


def test_generate_guided_offspring_counts_inactive_repair_salvage(monkeypatch):
    pop, config = configure_stub_population([make_graph_factory(0.15)], monkeypatch)
    pop.repair_activity_probe_fraction = 1.0
    pop.repair_activity_probe_max = 5
    monkeypatch.setattr(
        population_module,
        "rebuild_and_script",
        lambda *args, **kwargs: DummyStatefulOptimizer(),
    )
    pop._optimizer_updates_parameters = lambda *args, **kwargs: True

    real_repair = pop._repair_graph_dict

    def forced_repair(graph_dict):
        real_repair(graph_dict)
        graph_dict["_repair_applied"] = True
        return True

    pop._repair_graph_dict = forced_repair
    pop._should_probe_repair_activity = MethodType(lambda self: True, pop)
    probe_results = deque(
        [
            OptimizerValidationResult(outcome=ValidatorOutcome.DELTA_BELOW_THRESHOLD),
            OptimizerValidationResult(outcome=ValidatorOutcome.ACTIVE),
        ]
    )

    def fake_probe(*args, **kwargs):
        return probe_results.popleft()

    pop._probe_graph_activity = fake_probe
    pop._reset_guided_offspring_stats()

    pop.generate_guided_offspring([], config, n_offspring=1, latent_steps=1)

    stats = pop._guided_offspring_stats
    assert stats["inactive_repair_salvaged"] == 1
    assert stats["inactive_repair_salvaged_total"] >= 1


def test_repair_activity_probe_limit_applies_per_generation(monkeypatch):
    pop, config = configure_stub_population(
        [make_graph_factory(0.2), make_graph_factory(0.25)],
        monkeypatch,
    )
    pop.repair_activity_probe_fraction = 1.0
    pop.repair_activity_probe_max = 1
    monkeypatch.setattr(population_module.random, "random", lambda: 0.0)

    probe_calls = {"count": 0}
    pop._probe_graph_activity = lambda *args, **kwargs: probe_calls.update(
        {"count": probe_calls["count"] + 1}
    ) or OptimizerValidationResult(outcome=ValidatorOutcome.DELTA_BELOW_THRESHOLD)

    real_repair = pop._repair_graph_dict

    def forced_repair(graph_dict):
        real_repair(graph_dict)
        graph_dict["_repair_applied"] = True
        return True

    pop._repair_graph_dict = forced_repair
    pop._reset_guided_offspring_stats()

    pop.generate_guided_offspring([], config, n_offspring=1, latent_steps=1)
    pop.generate_guided_offspring([], config, n_offspring=1, latent_steps=1)

    assert probe_calls["count"] == 1


def test_latent_structure_penalty_uses_cached_centers():
    config = make_neat_config()
    config.guided_structure_buffer = 4
    config.guided_structure_weight = 1.0
    config.guided_structure_margin = 0.1
    pop = GuidedPopulation(config)

    pop._record_latent_label(torch.tensor([0.0, 0.0]), valid=False)
    pop._record_latent_label(torch.tensor([1.0, 0.0]), valid=True)

    latents = torch.tensor([[0.05, 0.0], [0.9, 0.0]], dtype=torch.float32)
    penalty = pop._latent_structure_penalty(latents)

    assert penalty.item() > 0


def test_generate_guided_offspring_records_latent_labels(monkeypatch):
    config = make_neat_config()
    config.guided_structure_buffer = 4
    config.guided_structure_weight = 0.0
    config.guided_replay_fraction = 0.0
    config.guided_seed_fraction = 0.0
    pop = GuidedPopulation(config)
    pop.guide = StubGuide([make_invalid_graph_factory(), make_graph_factory(0.3)])
    pop._optimizer_updates_parameters = lambda *args, **kwargs: True
    monkeypatch.setattr(
        population_module,
        "rebuild_and_script",
        lambda *args, **kwargs: DummyStatefulOptimizer(),
    )

    recorded = []

    def fake_record(latent, valid):
        recorded.append(bool(valid))

    pop._record_latent_label = fake_record

    pop.generate_guided_offspring([], config, n_offspring=2, latent_steps=1)

    assert len(recorded) == 2
    assert any(recorded)
    assert any(not flag for flag in recorded)


def test_seed_latent_structure_from_population(monkeypatch):
    config = make_neat_config()
    config.guided_structure_buffer = 4
    pop = GuidedPopulation(config)
    pop.guide = StubGuide([make_graph_factory(0.2)])
    genome = create_simple_genome(0)
    genome.graph_dict = make_graph_factory(0.2)()
    pop.population = {0: genome}

    latents = torch.tensor([[0.5, -0.25]], dtype=torch.float32)

    def fake_encode(data_list):
        return latents[: len(data_list)]

    monkeypatch.setattr(pop, "_encode_graph_batch", fake_encode)

    seeded = pop._seed_latent_structure_from_population()
    assert seeded == 1
    assert len(pop._latent_valid_samples) == 1
    assert torch.allclose(pop._latent_valid_samples[0], latents[0])

    seeded_again = pop._seed_latent_structure_from_population()
    assert seeded_again == 0


def test_generate_guided_offspring_tracks_structure_penalty(monkeypatch):
    config = make_neat_config()
    config.guided_replay_fraction = 0.0
    config.guided_seed_fraction = 0.0
    config.guided_structure_buffer = 4
    config.guided_structure_weight = 1.0
    pop = GuidedPopulation(config)
    pop.guide = StubGuide([make_graph_factory(0.3)])
    pop._optimizer_updates_parameters = lambda *args, **kwargs: True
    monkeypatch.setattr(
        population_module,
        "rebuild_and_script",
        lambda *args, **kwargs: DummyStatefulOptimizer(),
    )

    penalty_values = [0.1, 0.2, 0.0]

    def fake_penalty(latents):
        idx = min(fake_penalty.calls, len(penalty_values) - 1)
        fake_penalty.calls += 1
        return torch.tensor(penalty_values[idx], dtype=torch.float32)

    fake_penalty.calls = 0
    pop._latent_structure_penalty = fake_penalty
    pop._reset_guided_offspring_stats()

    pop.generate_guided_offspring(
        [],
        config,
        n_offspring=1,
        latent_steps=len(penalty_values),
    )

    stats = pop._guided_offspring_stats
    expected_samples = fake_penalty.calls
    assert stats["structure_penalty_samples"] == expected_samples
    expected_values = [penalty_values[min(i, len(penalty_values) - 1)] for i in range(expected_samples)]
    assert stats["structure_penalty_last"] == pytest.approx(expected_values[-1])
    expected_mean = sum(expected_values) / expected_samples
    assert stats["structure_penalty_mean"] == pytest.approx(expected_mean)


def test_latent_decoder_relax_penalty_tracks_stats():
    config = make_neat_config()
    config.guided_decoder_relax_weight = 1.0
    config.guided_decoder_relax_steps = 4
    config.guided_decoder_relax_edge_depth = 2
    pop = GuidedPopulation(config)
    latent_dim = pop.guide.graph_encoder.latent_dim
    decoder_stub = RelaxDecoderStub(latent_dim)
    decoder_stub.stop_head.bias.data.fill_(5.0)
    decoder_stub.edge_head.bias.data.fill_(-5.0)
    pop.guide.decoder = decoder_stub
    latents = torch.zeros((2, latent_dim), dtype=torch.float32, requires_grad=True)

    penalty = pop._latent_decoder_relax_penalty(latents)

    assert penalty.item() > 0
    penalty.backward()
    assert latents.grad is not None
    stats = pop._decoder_relax_stats
    _, _, min_nodes = pop._decoder_pin_requirements()
    assert stats["expected_nodes"] < float(min_nodes)
    assert "expected_edges" in stats


def test_decoder_replay_seed_support_follows_validator_fail_ratio():
    config = make_neat_config()
    config.guided_replay_fraction = 0.0
    config.guided_seed_fraction = 0.5
    config.guided_seed_fail_threshold = 0.25
    pop = GuidedPopulation(config)

    seed_graph = make_graph_factory(0.1)()
    seed_graph["node_attributes"][0]["tag"] = "seed"
    pop._seeded_replay_entries = [{"graph": seed_graph}]

    reservoir_graph = make_graph_factory(0.2)()
    reservoir_graph["node_attributes"][0]["tag"] = "reservoir"
    pop._decoder_replay_reservoir.clear()
    pop._decoder_replay_reservoir.extend([(None, reservoir_graph)])

    pop._reset_guided_offspring_stats()
    pop._guided_offspring_stats["requested"] = 4
    pop._guided_offspring_stats["validator_failures"] = 2

    seeded_draw = pop._sample_decoder_replay_graphs(1)
    assert any(attr.get("tag") == "seed" for g in seeded_draw for attr in g.get("node_attributes", []))

    pop._reset_guided_offspring_stats()
    pop._guided_offspring_stats["requested"] = 4
    pop._guided_offspring_stats["validator_failures"] = 0
    pop._prev_guided_offspring_stats = None

    regular_draw = pop._sample_decoder_replay_graphs(1)
    assert all(attr.get("tag") != "seed" for g in regular_draw for attr in g.get("node_attributes", []))


def test_latent_structure_penalty_uses_cached_centers():
    config = make_neat_config()
    config.guided_structure_buffer = 4
    config.guided_structure_weight = 1.0
    config.guided_structure_margin = 0.1
    pop = GuidedPopulation(config)

    pop._record_latent_label(torch.tensor([0.0, 0.0]), valid=False)
    pop._record_latent_label(torch.tensor([1.0, 0.0]), valid=True)

    latents = torch.tensor([[0.05, 0.0], [0.9, 0.0]], dtype=torch.float32)
    penalty = pop._latent_structure_penalty(latents)

    assert penalty.item() > 0


def test_generate_guided_offspring_tracks_repair_salvage(monkeypatch):
    config = make_neat_config()
    config.guided_replay_fraction = 0.0
    config.guided_seed_fraction = 0.0
    pop = GuidedPopulation(config)
    pop.guide = StubGuide([make_invalid_graph_factory(), make_graph_factory(0.2)])
    pop._optimizer_updates_parameters = lambda *args, **kwargs: True
    monkeypatch.setattr(
        population_module,
        "rebuild_and_script",
        lambda *args, **kwargs: DummyStatefulOptimizer(),
    )

    pop.generate_guided_offspring([], config, n_offspring=2, latent_steps=1)

    assert pop._last_repair_salvaged >= 1
    assert pop._total_repair_salvaged >= pop._last_repair_salvaged


def test_seed_decoder_replay_rebuilds_missing_graph_dicts():
    config = make_neat_config()
    pop = GuidedPopulation(config)

    population_size = 3
    genomes = {idx: create_simple_genome(idx) for idx in range(population_size)}
    for genome in genomes.values():
        genome.graph_dict = {"graph_ir": {"nodes": []}}
    pop.population = genomes
    pop.decoder_replay_max = population_size

    def fake_genome_to_data(self, genome):
        bias = float(genome.key + 1)
        graph = make_decoded_graph(bias)
        cloned = {
            "node_types": graph["node_types"].clone(),
            "edge_index": graph["edge_index"].clone(),
            "node_attributes": [dict(attrs) for attrs in graph["node_attributes"]],
        }
        genome.graph_dict = cloned
        return Data(
            node_types=graph["node_types"],
            edge_index=graph["edge_index"],
            node_attributes=graph["node_attributes"],
        )

    pop.genome_to_data = MethodType(fake_genome_to_data, pop)
    pop._decoder_replay_cache.clear()
    pop._decoder_replay_reservoir.clear()
    pop._decoder_replay_signatures.clear()
    pop._decoder_replay_seeded = False

    seeded = pop._seed_decoder_replay_from_population()

    assert seeded == population_size
    assert len(pop._decoder_replay_cache) == population_size

    payload = pop._consume_decoder_replay_graphs()
    assert len(payload) == population_size
    for graph_dict in payload:
        assert graph_dict.get("node_types") is not None
        assert graph_dict.get("edge_index") is not None
        assert graph_dict.get("node_attributes")


def test_generation_eval_steps_respects_max_cap():
    config = make_neat_config()
    config.max_evaluation_steps = 30
    pop = GuidedPopulation(config)
    pop.generation = 50
    assert pop._generation_eval_steps() == 30

    pop.max_regression_epochs = None
    pop.generation = 5
    assert pop._generation_eval_steps() == 25


def test_test_mode_reduces_trainer_epochs():
    config = make_neat_config()
    config.test_mode = True
    config.test_epoch_scale = 0.1
    pop = GuidedPopulation(config)

    pop.generation = 0
    warmup_schedule = pop._trainer_epoch_schedule()
    assert warmup_schedule["epochs"] < 100
    assert warmup_schedule["warmup_epochs"] <= warmup_schedule["epochs"]

    pop.generation = pop.full_train_resize_generation + 5
    steady_schedule = pop._trainer_epoch_schedule()
    assert steady_schedule["epochs"] <= 1
    assert steady_schedule["loss_threshold"] <= 0.9


def test_generate_guided_offspring():
    config = make_neat_config()
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
    config = make_neat_config()
    pop = GuidedPopulation(config)
    task = RegressionTask.random_init(num_samples=4, silent=True)

    genome = create_simple_genome()
    genome.connections = {}
    genome.skip_evaluation = True
    genome.invalid_reason = "empty_graph"

    pop.task = task
    counter_snapshot = population_module._INVALID_REASON_COUNTER.copy()
    try:
        pop.eval_genomes([(0, genome)], config, steps=1)
    finally:
        population_module._INVALID_REASON_COUNTER.clear()
        population_module._INVALID_REASON_COUNTER.update(counter_snapshot)

    assert getattr(genome, "invalid_graph", False)
    assert genome.fitnesses[AreaUnderTaskMetrics] == GuidedPopulation.INVALID_METRIC_VALUE
    assert genome.fitnesses[TimeCost] == GuidedPopulation.INVALID_METRIC_VALUE
    assert genome.fitnesses[MemoryCost] == GuidedPopulation.INVALID_METRIC_VALUE
    assert genome.fitness == -0.1


def test_eval_genomes_logs_prev_guided_invalid_summary(monkeypatch):
    config = make_neat_config()
    pop = GuidedPopulation(config)
    pop.reporters = RecordingReporters()
    task = RegressionTask.random_init(num_samples=4, silent=True)
    pop.task = task

    valid_genome = create_simple_genome(0)
    valid_genome.optimizer = DummyStatefulOptimizer()

    invalid_genome = create_simple_genome(1)
    invalid_genome.skip_evaluation = True
    invalid_genome.invalid_reason = "inactive_optimizer"

    pop._prev_guided_offspring_stats = {
        "generation": 12,
        "spawn_generation": 13,
        "invalid_total": 7,
        "invalid_by_reason": Counter({"inactive_optimizer": 6, "decoder_max_nodes": 1}),
    }

    def fake_evaluate_optimizer(self, optimizer, model, steps=0):
        return 0.5, {AreaUnderTaskMetrics: 0.4}, 0.1, 0.2

    monkeypatch.setattr(
        pop,
        "evaluate_optimizer",
        MethodType(fake_evaluate_optimizer, pop),
    )

    counter_snapshot = population_module._INVALID_REASON_COUNTER.copy()
    try:
        pop.eval_genomes([(0, valid_genome), (1, invalid_genome)], config, steps=1)
    finally:
        population_module._INVALID_REASON_COUNTER.clear()
        population_module._INVALID_REASON_COUNTER.update(counter_snapshot)

    all_messages = "\n".join(pop.reporters.messages)
    assert "inactive_optimizer=6" in all_messages
    assert "decoder_max_nodes=1" in all_messages
    assert "spawned gen 13" in all_messages


def test_assign_penalty_records_decoder_failure_when_requested():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    recorded = {}

    def fake_recorder(graph_dict, fitnesses, reason=""):
        recorded["graph"] = graph_dict
        recorded["fitnesses"] = fitnesses
        recorded["reason"] = reason

    pop._record_decoder_failure = fake_recorder
    genome = create_simple_genome()
    dummy_graph = {"edge_index": torch.empty((2, 0), dtype=torch.long)}

    metrics = pop._assign_penalty(
        genome,
        reason="inactive_optimizer",
        skip_evaluation=True,
        graph_dict=dummy_graph,
        record_decoder_failure=True,
    )

    assert recorded["graph"] is dummy_graph
    assert recorded["reason"] == "inactive_optimizer"
    assert recorded["fitnesses"] == metrics


def test_guided_invalid_visualization_emits_mermaid_files(tmp_path):
    config = make_neat_config()
    pop = GuidedPopulation(config)
    pop.generation = 2
    pop.guided_invalid_viz_enabled = True
    pop.guided_invalid_viz_dir = tmp_path
    pop.guided_invalid_viz_limit = 4
    pop._guided_invalid_viz_generation = None
    pop._guided_invalid_viz_used = 0

    genome = create_simple_genome()
    graph_dict = {
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "node_attributes": [{"node_type": "input"}, {"node_type": "output"}],
    }

    pop._maybe_visualize_guided_invalid_graph(
        genome,
        graph_dict,
        reason="empty_graph",
        child_index=3,
        num_edges=0,
    )

    repaired_files = list(tmp_path.glob("*_repaired.mmd"))
    assert repaired_files, "expected repaired mermaid visualization"
    text = repaired_files[0].read_text()
    assert "graph LR" in text
    assert "status=invalid" in text
    assert "empty_graph" in text


def test_guided_invalid_visualization_writes_separate_decoded_mermaid(tmp_path):
    config = make_neat_config()
    pop = GuidedPopulation(config)
    pop.generation = 5
    pop.guided_invalid_viz_enabled = True
    pop.guided_invalid_viz_dir = tmp_path
    genome = create_simple_genome()

    decoded_graph = {
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "node_attributes": [{"node_type": "input"}, {"node_type": "output"}],
    }
    repaired_graph = {
        "node_types": torch.tensor([0], dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_attributes": [{"node_type": "input"}],
        population_module.DECODED_GRAPH_DICT_KEY: decoded_graph,
    }

    pop._maybe_visualize_guided_invalid_graph(
        genome,
        repaired_graph,
        reason="missing_output_slots",
        child_index=1,
        num_edges=0,
    )

    decoded_files = list(tmp_path.glob("*_decoded.mmd"))
    assert decoded_files, "decoded snapshot should be visualized"
    text = decoded_files[0].read_text()
    assert "node_0 --> node_1" in text


def test_guided_invalid_visualization_mermaid_only(tmp_path):
    config = make_neat_config()
    pop = GuidedPopulation(config)
    pop.generation = 4
    pop.guided_invalid_viz_enabled = True
    pop.guided_invalid_viz_dir = tmp_path
    pop.guided_invalid_viz_limit = 2

    genome = create_simple_genome()
    graph_dict = {
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "node_attributes": [{"node_type": "input"}, {"node_type": "output"}],
    }

    pop._maybe_visualize_guided_invalid_graph(
        genome,
        graph_dict,
        reason="missing_output_slots",
        child_index=0,
        num_edges=1,
    )

    mermaid_files = list(tmp_path.glob("*_repaired.mmd"))
    assert mermaid_files, "expected repaired Mermaid visualization"
    text = mermaid_files[0].read_text()
    assert "graph LR" in text
    assert "status=invalid" in text


def test_guided_invalid_visualization_mermaid_uses_repaired_graph(tmp_path):
    config = make_neat_config()
    pop = GuidedPopulation(config)
    pop.generation = 6
    pop.guided_invalid_viz_enabled = True
    pop.guided_invalid_viz_dir = tmp_path
    genome = create_simple_genome()
    decoded_graph = {
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "node_attributes": [{"node_type": "input"}, {"node_type": "output"}],
    }
    repaired_graph = {
        "node_types": torch.tensor([0], dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "node_attributes": [{"node_type": "input"}],
        population_module.DECODED_GRAPH_DICT_KEY: decoded_graph,
    }

    pop._maybe_visualize_guided_invalid_graph(
        genome,
        repaired_graph,
        reason="missing_output_slots",
        child_index=2,
        num_edges=0,
    )

    repaired_files = list(tmp_path.glob("*_repaired.mmd"))
    assert repaired_files, "repaired mermaid visualization missing"
    text = repaired_files[0].read_text()
    assert "%% Graph has no edges" in text
    assert "node_0 --> node_1" not in text


def test_generate_guided_offspring_handles_missing_elites():
    config = make_neat_config()
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


def test_repair_connects_hidden_nodes_to_inputs_and_outputs():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    node_attrs = [
        {"pin_role": "input", "pin_slot_index": 0},
        {"pin_role": "input", "pin_slot_index": 1},
        {"pin_role": "input", "pin_slot_index": 2},
        {"pin_role": "hidden"},
        {"pin_role": "hidden"},
        {"pin_role": "output"},
    ]
    graph = _make_empty_graph_dict(len(node_attrs), node_attrs)

    assert pop._repair_graph_dict(graph)

    edges = graph["edge_index"].t().tolist()
    adjacency_out = {idx: [] for idx in range(len(node_attrs))}
    adjacency_in = {idx: [] for idx in range(len(node_attrs))}
    for src, dst in edges:
        adjacency_out[src].append(dst)
        adjacency_in[dst].append(src)

    def reachable_from_inputs():
        seen = {idx for idx, attrs in enumerate(node_attrs) if attrs.get("pin_role") == "input"}
        queue = deque(seen)
        while queue:
            current = queue.popleft()
            for nxt in adjacency_out.get(current, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def reaches_outputs():
        outputs = {idx for idx, attrs in enumerate(node_attrs) if attrs.get("pin_role") == "output"}
        seen = set(outputs)
        queue = deque(outputs)
        while queue:
            current = queue.popleft()
            for src in adjacency_in.get(current, []):
                if src not in seen:
                    seen.add(src)
                    queue.append(src)
        return seen

    forward = reachable_from_inputs()
    backward = reaches_outputs()
    hidden_indices = [idx for idx, attrs in enumerate(node_attrs) if attrs.get("pin_role") == "hidden"]
    for idx in hidden_indices:
        assert idx in forward
        assert idx in backward


def test_prepare_decoded_graph_dict_rebuilds_block_registry_from_payload():
    class LoopTemplate(torch.nn.Module):
        def forward(self, loss, prev_loss, named_parameters):
            total = loss
            i = 0
            while i < 2:
                total = total + prev_loss
                i += 1
            updates = {}
            for name, param in named_parameters:
                updates[name] = param - 0.1 * total
            return updates

    config = make_neat_config()
    pop = GuidedPopulation(config)
    genome = create_initial_genome(config, torch.jit.script(LoopTemplate()))
    graph_dict = pop._clone_graph_dict(genome.graph_dict, include_history=False)
    graph_dict.pop("block_registry", None)
    for attrs in graph_dict.get("node_attributes", []):
        for key in list(attrs.keys()):
            if isinstance(key, str) and key.startswith(BLOCK_REF_ATTR_PREFIX):
                attrs.pop(key)
    node_count, _ = pop._prepare_decoded_graph_dict(graph_dict)
    assert node_count > 0
    registry = graph_dict.get("block_registry")
    assert registry, "Block registry should be reconstructed from payload tensors"
    for attrs in graph_dict.get("node_attributes", []):
        payload_keys = [k for k in attrs if isinstance(k, str) and k.startswith(BLOCK_PAYLOAD_ATTR_PREFIX)]
        if not payload_keys:
            continue
        assert any(
            isinstance(k, str) and k.startswith(BLOCK_REF_ATTR_PREFIX) for k in attrs
        ), "Repair should add block references when payloads exist"


def test_repair_builds_loop_from_payload_without_registry():
    class LoopTemplate(torch.nn.Module):
        def forward(self, loss, prev_loss, named_parameters):
            total = loss * 0.5 + prev_loss * 0.25
            i = 0
            while i < 3:
                total = total + prev_loss
                i += 1
            updates = {}
            for name, param in named_parameters:
                updates[name] = param - 0.05 * total
            return updates

    config = make_neat_config()
    pop = GuidedPopulation(config)
    genome = create_initial_genome(config, torch.jit.script(LoopTemplate()))
    graph_dict = pop._clone_graph_dict(genome.graph_dict, include_history=False)
    graph_dict.pop("block_registry", None)
    for attrs in graph_dict.get("node_attributes", []):
        for key in list(attrs.keys()):
            if isinstance(key, str) and key.startswith(BLOCK_REF_ATTR_PREFIX):
                attrs.pop(key)
    assert pop._repair_graph_dict(graph_dict)
    assert graph_dict.get("block_registry"), "Repair should derive block registry from payload"
    prime_registry(graph_dict.get("block_registry"))
    rebuilt = rebuild_and_script(graph_dict, config.genome_config, key=0)
    assert "prim::Loop" in str(rebuilt.graph)


def test_repair_preserves_seeded_input_pins():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    required_inputs = len(config.genome_config.input_keys)
    if required_inputs < 2:
        pytest.skip("insufficient configured inputs for this test")
    node_attrs = []
    for slot in range(required_inputs):
        node_attrs.append(
            {
                "pin_role": "input",
                "pin_slot_index": slot,
            }
        )
    node_attrs.append({})
    graph = _make_empty_graph_dict(len(node_attrs), node_attrs)

    pop._repair_graph_dict(graph)

    attrs = graph["node_attributes"]
    for slot in range(required_inputs):
        assert attrs[slot]["pin_role"] == "input"
        assert attrs[slot]["pin_slot_index"] == slot


def test_repair_preserves_predicted_edges_for_visualization():
    config = make_neat_config()
    required_inputs = len(config.genome_config.input_keys)
    if required_inputs <= 1:
        pytest.skip("config does not require multiple input slots")
    pop = GuidedPopulation(config)
    node_count = max(1, required_inputs - 1)
    graph = _make_empty_graph_dict(node_count, [{} for _ in range(node_count)])
    predicted = torch.tensor([[0], [0]], dtype=torch.long)
    graph["edge_index"] = predicted.clone()

    assert not pop._repair_graph_dict(graph)
    assert "_decoded_edge_index" not in graph
    assert torch.equal(graph["edge_index"], predicted)

    decoded = graph.get(population_module.DECODED_GRAPH_DICT_KEY)
    assert decoded is not None
    assert torch.equal(decoded["edge_index"], predicted)
    # ensure decoded snapshot did not receive synthesized pin_role attributes
    original_attrs = decoded["node_attributes"]
    assert all("pin_role" not in attrs for attrs in original_attrs)


def test_output_slot_coverage_accepts_metadata_slots():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    node_attrs = [
        {"pin_role": "input", "pin_slot_index": 0},
        {"pin_role": "input", "pin_slot_index": 1},
        {"pin_role": "input", "pin_slot_index": 2},
        {"pin_role": "output", "pin_slot_index": 0},
    ]
    graph = {
        "node_types": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "edge_index": torch.tensor([[1, 2], [3, 3]], dtype=torch.long),
        "node_attributes": node_attrs,
    }

    ok, details = pop._graph_output_slot_coverage(graph)

    assert ok, f"coverage should succeed, details={details}"
    assert details["missing_slots"] == []
    assert details["wrong_type_slots"] == []


def test_output_slot_coverage_requires_incoming_edges():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    node_attrs = [
        {"pin_role": "input", "pin_slot_index": 0},
        {"pin_role": "input", "pin_slot_index": 1},
        {"pin_role": "input", "pin_slot_index": 2},
        {"pin_role": "output", "pin_slot_index": 0},
    ]
    graph = {
        "node_types": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "node_attributes": node_attrs,
    }

    ok, details = pop._graph_output_slot_coverage(graph)

    assert not ok
    assert details["missing_slots"] == [0]


def test_prepare_decoded_graph_strips_invalid_edges():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    graph = {
        "node_types": torch.tensor([0], dtype=torch.long),
        "edge_index": torch.tensor([[0], [5]], dtype=torch.long),
        "node_attributes": [{"node_type": "input"}],
    }

    node_count, edges = pop._prepare_decoded_graph_dict(graph)

    assert node_count == 1
    assert edges == []
    decoded = graph[population_module.DECODED_GRAPH_DICT_KEY]
    assert decoded["edge_index"].numel() == 0


def test_penalty_scale_handles_missing_input_slots():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    details = {"total_slots": 4, "missing_count": 2}
    scale = pop._penalty_scale("missing_input_slots", details)
    assert pop.MISSING_SLOT_PENALTY_MIN_SCALE <= scale <= pop.MISSING_SLOT_PENALTY_MAX_SCALE


def test_penalty_scale_accounts_for_wrong_type_outputs():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    details = {
        "total_slots": 3,
        "missing_slots": [],
        "wrong_type_slots": [0, 2],
    }
    scale = pop._penalty_scale("missing_output_slots", details)
    assert scale > pop.MISSING_SLOT_PENALTY_MIN_SCALE
    assert scale <= pop.MISSING_SLOT_PENALTY_MAX_SCALE


def test_repair_connects_each_input_to_output():
    config = make_neat_config()
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


def test_repair_randomization_changes_fallback_edge_choices():
    base_config = make_neat_config()
    base_config.genome_config.input_keys = [0]
    base_config.genome_config.output_keys = [0]
    deterministic_pop = GuidedPopulation(base_config)

    rand_config = make_neat_config()
    rand_config.genome_config.input_keys = [0]
    rand_config.genome_config.output_keys = [0]
    rand_config.repair_randomize_connections = True
    rand_config.repair_random_seed = 123
    randomized_pop = GuidedPopulation(rand_config)

    node_attrs = [
        {"pin_role": "input", "pin_slot_index": 0},
        {"pin_role": "input", "pin_slot_index": 1},
        {"pin_role": "input", "pin_slot_index": 2},
        {"pin_role": "hidden"},
        {"pin_role": "hidden"},
        {"pin_role": "hidden"},
        {"pin_role": "output"},
    ]

    base_graph = _make_empty_graph_dict(len(node_attrs), copy.deepcopy(node_attrs))
    rand_graph = _make_empty_graph_dict(len(node_attrs), copy.deepcopy(node_attrs))

    assert deterministic_pop._repair_graph_dict(base_graph)
    assert randomized_pop._repair_graph_dict(rand_graph)

    base_edges = sorted(base_graph["edge_index"].t().tolist())
    rand_edges = sorted(rand_graph["edge_index"].t().tolist())
    assert base_edges != rand_edges


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
    config = make_neat_config()
    population = GuidedPopulation(config)
    model = ManyLossMinimaModel(population.task.train_data.num_input_features)
    optimizer = DummyStatefulOptimizer()

    # Should not raise even though optimizer buffers start with mismatched shapes.
    population.evaluate_optimizer(optimizer, model, steps=1)

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

    config = make_neat_config()
    population = GuidedPopulation(config)
    model = ManyLossMinimaModel(population.task.train_data.num_input_features)
    optimizer = TrackingOptimizer()

    population.evaluate_optimizer(optimizer, model, steps=1)
    # Simulate leftover state before the second evaluation.
    optimizer.state = {k: v + 5 for k, v in optimizer.state.items()}
    optimizer.step = 42
    population.evaluate_optimizer(optimizer, model, steps=1)

    assert [step for step, _ in optimizer.observed] == [0, 0]
    assert all(len(state) == 0 for _, state in optimizer.observed)


def test_evaluate_optimizer_marks_nan_outputs_invalid():
    class NaNOptimizer(torch.nn.Module):
        def forward(self, loss, prev_loss, named_parameters):
            return {name: torch.full_like(param, float("nan")) for name, param in named_parameters}

    config = make_neat_config()
    population = GuidedPopulation(config)
    model = ManyLossMinimaModel(population.task.train_data.num_input_features)

    optimizer = NaNOptimizer()
    result = population.evaluate_optimizer(optimizer, model, steps=1)

    assert result is None


def test_optimizer_state_attributes_include_empty_dicts():
    config = make_neat_config()
    pop = GuidedPopulation(config)
    opt = EmptyStateOptimizer()

    attrs = pop._optimizer_state_attributes(opt)

    assert "state" in attrs
