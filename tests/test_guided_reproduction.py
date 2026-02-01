import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from reproduction import GuidedReproduction


class DummyGenome:
    def __init__(self, key, fitness=0.0):
        self.key = key
        self.fitness = fitness
        self.crossover_parents = None
        self.mutated = False
        self.optimizer = None
        self.graph_dict = None
        self.nodes = {}
        self.connections = {}
        self.is_fallback = False

    def configure_crossover(self, p1, p2, config):
        self.crossover_parents = (p1.key, p2.key)

    def mutate(self, config):
        self.mutated = True

    def configure_new(self, genome_config):
        for node_key in getattr(genome_config, "output_keys", []):
            self.nodes[node_key] = {"node_type": "output"}

    def create_connection(self, genome_config, input_id, output_id):
        conn = type("DummyConn", (), {})()
        conn.key = (input_id, output_id)
        conn.enabled = True
        self.connections[conn.key] = conn
        return conn

    def create_node(self, genome_config, node_id):
        node = type("DummyNode", (), {})()
        node.key = node_id
        node.node_type = "hidden"
        node.dynamic_attributes = {}
        return node

    def compile_optimizer(self, genome_config):
        self.optimizer = object()
        self.graph_dict = {}


class DummySpecies:
    def __init__(self, key, members):
        self.key = key
        self.members = {m.key: m for m in members}
        self.adjusted_fitness = None


class DummySpeciesSet:
    def __init__(self, *species_list):
        if len(species_list) == 1 and isinstance(species_list[0], (list, tuple, set)):
            species_list = tuple(species_list[0])
        self.species = {species.key: species for species in species_list}


class DummyStagnation:
    def __init__(self, stagnant_ids=None):
        self.stagnant_ids = set(stagnant_ids or [])

    def update(self, species_set, generation):
        return [(sid, s, sid in self.stagnant_ids) for sid, s in species_set.species.items()]


class DummyReporters:
    def species_stagnant(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class DummyConfig:
    def __init__(self, genome_type):
        self.genome_type = genome_type

        class GC:
            def __init__(self):
                self.input_keys = [-1, -2]
                self.output_keys = [0, 1]

        self.genome_config = GC()


def make_reproduction(elitism=1, survival_threshold=0.5, reporters=None, stagnation=None):
    class RC:
        def __init__(self):
            self.elitism = elitism
            self.survival_threshold = survival_threshold
            self.min_species_size = 1

    return GuidedReproduction(RC(), reporters or DummyReporters(), stagnation or DummyStagnation())


def test_guided_and_standard_offspring():
    g1 = DummyGenome(101, fitness=1.0)
    g2 = DummyGenome(102, fitness=0.5)
    g3 = DummyGenome(103, fitness=0.2)
    species = DummySpecies(1, [g1, g2, g3])
    species_set = DummySpeciesSet(species)

    repro = make_reproduction(elitism=1)
    # force deterministic spawn amounts
    repro.compute_spawn = lambda *a, **k: [3]

    guided_called = {}

    def guide_fn(starting, repro_config, n_offspring):
        guided_called["n"] = n_offspring
        guided_called["starting_size"] = len(starting)
        return [DummyGenome(50)]

    repro.guide_fn = guide_fn

    config = DummyConfig(DummyGenome)

    class DummyTask:
        def name(self):
            return "dummy"

        features = []

    pop = repro.reproduce(config, species_set, 3, 0, DummyTask())

    assert guided_called["n"] == 1
    assert guided_called["starting_size"] == len(species.members)
    # expect 3 individuals: 1 elite, 1 guided, 1 crossover
    assert len(pop) == 3
    assert 101 in pop  # elite preserved
    assert 50 in pop  # guided child
    # crossover child gets key 1 from genome_indexer
    assert any(isinstance(k, int) and k not in (101, 102, 103, 50) for k in pop)


def test_only_elites_when_spawn_too_small():
    g1 = DummyGenome(201, fitness=1.0)
    g2 = DummyGenome(202, fitness=0.5)
    species = DummySpecies(2, [g1, g2])
    species_set = DummySpeciesSet(species)

    repro = make_reproduction(elitism=1)
    repro.compute_spawn = lambda *a, **k: [1]

    repro.guide_fn = lambda *a, **k: pytest.fail("guide_fn should not be called")

    config = DummyConfig(DummyGenome)

    class DummyTask:
        def name(self):
            return "dummy"

        features = []

    pop = repro.reproduce(config, species_set, 2, 0, DummyTask())

    # only elite should remain
    assert len(pop) == 1
    assert 201 in pop


def test_stagnated_species_removed_from_species_set():
    stagnation = DummyStagnation()
    repro = make_reproduction(elitism=1, stagnation=stagnation)
    repro.compute_spawn = lambda *a, **k: [1, 1]
    repro.guide_fn = lambda *a, **k: []
    config = DummyConfig(DummyGenome)

    class DummyTask:
        def name(self):
            return "dummy"

        features = []

    alive_species = DummySpecies(10, [DummyGenome(501, fitness=1.5)])
    stagnant_species = DummySpecies(11, [DummyGenome(601, fitness=0.1)])
    species_set = DummySpeciesSet(alive_species, stagnant_species)

    stagnation.stagnant_ids = {stagnant_species.key}

    repro.reproduce(config, species_set, 2, 0, DummyTask())

    assert alive_species.key in species_set.species
    assert stagnant_species.key not in species_set.species


def test_fallback_min_graph_used_when_validation_fails():
    g1 = DummyGenome(301, fitness=1.0)
    g2 = DummyGenome(302, fitness=0.8)
    species = DummySpecies(4, [g1, g2])
    species_set = DummySpeciesSet(species)

    repro = make_reproduction(elitism=0)
    repro.compute_spawn = lambda *a, **k: [2]
    repro.optimizer_validator = lambda optimizer: False

    config = DummyConfig(DummyGenome)

    class DummyTask:
        def name(self):
            return "dummy"

        features = []

    pop = repro.reproduce(config, species_set, 2, 0, DummyTask())

    assert len(pop) == 2
    fallback_children = [child for child in pop.values() if getattr(child, "_minimum_graph_fallback", False)]
    assert fallback_children


def test_minimum_graph_connects_hidden_and_outputs():
    repro = make_reproduction(elitism=0)
    config = DummyConfig(DummyGenome)

    genome = repro._build_random_minimum_genome(config, key=999)

    hidden_ids = getattr(genome, "hidden_node_ids", [])
    assert len(hidden_ids) >= 2

    output_keys = set(config.genome_config.output_keys)
    input_keys = list(config.genome_config.input_keys)
    conn_keys = set(genome.connections.keys())

    for out_key in output_keys:
        assert any(dst == out_key for _, dst in conn_keys)

    for hid in hidden_ids:
        assert any(dst == hid for _, dst in conn_keys)
        assert any(src == hid for src, _ in conn_keys)

    for in_key in input_keys:
        assert any(src == in_key and dst in hidden_ids for src, dst in conn_keys)
