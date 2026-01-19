import types

from relative_rank_stagnation import RelativeRankStagnation


class DummyMember:
    def __init__(self, fitness):
        self.fitness = fitness


class DummySpecies:
    def __init__(self, key, member_fitnesses):
        self.key = key
        self.members = {i: DummyMember(f) for i, f in enumerate(member_fitnesses)}
        self.fitness_history = []
        self.last_improved = None
        self.created = 0

    def set_member_fitness(self, idx, value):
        self.members[idx].fitness = value


class DummySpeciesSet:
    def __init__(self, species_list):
        self.species = {s.key: s for s in species_list}


class DummyReporters:
    def species_stagnant(self, *args, **kwargs):
        pass


def make_config(**overrides):
    cfg = {
        "species_fitness_func": "max",
        "max_stagnation": 5,
        "species_elitism": 1,
        "rank_tolerance": 1e-6,
        "fitness_tolerance": 1e-9,
    }
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def test_relative_rank_orders_species_by_score():
    config = make_config(max_stagnation=10, species_elitism=1)
    stagnation = RelativeRankStagnation(config, DummyReporters())

    elite = DummySpecies(1, [3.0, 2.5])
    mid = DummySpecies(2, [2.0, 1.0])
    laggard = DummySpecies(3, [0.5])
    species_set = DummySpeciesSet([elite, mid, laggard])

    result = stagnation.update(species_set, generation=5)

    assert [sid for sid, *_ in result] == [1, 2, 3]
    assert elite._best_rank == 1
    assert laggard._best_rank == 3
    assert not result[0][2]


def test_species_flags_stagnant_after_rank_plateau():
    config = make_config(max_stagnation=2, species_elitism=1)
    stagnation = RelativeRankStagnation(config, DummyReporters())

    leader = DummySpecies(10, [5.0])
    trailer = DummySpecies(11, [4.0])
    species_set = DummySpeciesSet([leader, trailer])

    for generation in range(3):
        result = stagnation.update(species_set, generation)

    stagnant_entry = next(item for item in result if item[0] == trailer.key)
    assert stagnant_entry[2] is True
    leader_entry = next(item for item in result if item[0] == leader.key)
    assert leader_entry[2] is False


def test_normalized_fitness_improvement_resets_stagnation_even_when_rank_constant():
    config = make_config(max_stagnation=3, species_elitism=0)
    stagnation = RelativeRankStagnation(config, DummyReporters())

    leader = DummySpecies(21, [6.0])
    trailer = DummySpecies(22, [1.0])
    species_set = DummySpeciesSet([leader, trailer])

    stagnation.update(species_set, generation=0)

    # trailer improves absolute fitness but remains rank 2
    trailer.set_member_fitness(0, 3.0)
    leader.set_member_fitness(0, 6.0)
    stagnation.update(species_set, generation=1)

    # additional generations without improvement should eventually stagnate
    leader.set_member_fitness(0, 6.0)
    trailer.set_member_fitness(0, 2.9)
    result = stagnation.update(species_set, generation=3)

    trailer_entry = next(item for item in result if item[0] == trailer.key)
    assert trailer_entry[2] is False  # improvement at gen1 reset the counter


def test_rank_improvement_resets_when_relative_gap_constant():
    config = make_config(max_stagnation=3, species_elitism=0)
    stagnation = RelativeRankStagnation(config, DummyReporters())

    leader = DummySpecies(31, [6.0])
    blocker = DummySpecies(32, [5.0])
    climber = DummySpecies(33, [4.0])
    species_set = DummySpeciesSet([leader, blocker, climber])

    stagnation.update(species_set, generation=0)

    # Climber and leader both gain the same absolute amount, so their relative gap stays constant.
    leader.set_member_fitness(0, 8.0)
    blocker.set_member_fitness(0, 4.0)
    climber.set_member_fitness(0, 6.0)
    result = stagnation.update(species_set, generation=1)

    climber_entry = next(item for item in result if item[0] == climber.key)
    assert climber_entry[2] is False  # rank improved from 3 -> 2 despite identical leader gap
