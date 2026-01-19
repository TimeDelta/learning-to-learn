from typing import Dict

from neat.config import ConfigParameter, DefaultClassConfig
from neat.stagnation import DefaultStagnation


class RelativeRankStagnation(DefaultStagnation):
    """Marks species as stagnant based on percentile rank within the current generation."""

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("species_fitness_func", str, "max"),
                ConfigParameter("max_stagnation", int, 15),
                ConfigParameter("species_elitism", int, 2),
                ConfigParameter("rank_tolerance", float, 1e-6),
                ConfigParameter("fitness_tolerance", float, 1e-9),
            ],
        )

    def __init__(self, config, reporters):
        super().__init__(config, reporters)
        self.rank_tolerance = getattr(config, "rank_tolerance", 1e-6)
        # DefaultStagnation may not set attributes when custom config objects are used, so cache them here.
        self.max_stagnation = getattr(config, "max_stagnation", 15)
        self.species_elitism = getattr(config, "species_elitism", 2)
        self.fitness_tolerance = getattr(config, "fitness_tolerance", 1e-9)

    def _compute_best_member_fitness(self, species) -> float:
        if not species.members:
            return float("-inf")
        return max(member.fitness for member in species.members.values())

    def _compute_dense_ranks(self, raw_scores: Dict[int, float]) -> Dict[int, int]:
        ordered = sorted(raw_scores.items(), key=lambda item: (-item[1], item[0]))
        ranks = {}
        current_rank = 1
        prev_score = None
        for idx, (sid, score) in enumerate(ordered):
            if prev_score is None:
                current_rank = 1
            elif score > prev_score + self.rank_tolerance:
                # should not happen due to ordering
                current_rank = idx + 1
            elif score < prev_score - self.rank_tolerance:
                current_rank = idx + 1
            ranks[sid] = current_rank
            prev_score = score
        return ranks

    def update(self, species_set, generation):
        if not species_set.species:
            return []

        raw_scores = {sid: self._compute_best_member_fitness(species) for sid, species in species_set.species.items()}
        ranks = self._compute_dense_ranks(raw_scores)
        max_score = max(raw_scores.values())

        species_data = []
        for sid, species in species_set.species.items():
            current_rank = ranks[sid]
            current_score = raw_scores[sid]
            species.fitness = current_score
            species.fitness_history.append(current_score)

            best_rank = getattr(species, "_best_rank", None)
            rank_improved = False
            if best_rank is None or current_rank < best_rank:
                species._best_rank = current_rank
                species.last_improved = generation
                rank_improved = True

            relative_score = current_score - max_score

            best_relative = getattr(species, "_best_relative", None)
            normalized_improved = False
            if best_relative is None or relative_score > best_relative + self.fitness_tolerance:
                species._best_relative = relative_score
                normalized_improved = True

            if normalized_improved and not rank_improved:
                species.last_improved = generation

            if species.last_improved is None:
                species.last_improved = generation

            stagnant = generation - species.last_improved >= self.max_stagnation
            species_data.append((sid, species, stagnant))

        species_data.sort(key=lambda item: item[1].fitness, reverse=True)
        for i in range(min(self.species_elitism, len(species_data))):
            sid, species, _ = species_data[i]
            species_data[i] = (sid, species, False)

        return species_data
