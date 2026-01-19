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

    def configure_crossover(self, p1, p2, config):
        self.crossover_parents = (p1.key, p2.key)

    def mutate(self, config):
        self.mutated = True


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
        self.genome_config = object()


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

    def guide_fn(name, features, starting, config, n_offspring):
        guided_called["n"] = n_offspring
        return [DummyGenome(50)]

    repro.guide_fn = guide_fn

    config = DummyConfig(DummyGenome)

    class DummyTask:
        def name(self):
            return "dummy"

        features = []

    pop = repro.reproduce(config, species_set, 3, 0, DummyTask())

    assert guided_called["n"] == 1
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
