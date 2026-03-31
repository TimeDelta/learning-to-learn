"""Novelty archive utilities for structural diversity tracking."""
from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Deque, Dict, Iterable, List, Mapping, Sequence, Tuple

from metrics import Metric

if False:  # pragma: no cover - import guard for type checking without runtime cycle
    from genome import OptimizerGenome  # noqa: F401


class NoveltyMetric(Metric):
    name = "Novelty Score"
    objective = "max"
    best_value = 0.0
    guidance_weight = 0.0


def graph_behavior_descriptor(genome, *, top_type_bins: int = 4) -> Tuple[float, ...]:
    """Return a structural descriptor summarizing the genome's optimizer graph."""

    node_count = len(getattr(genome, "nodes", {}))
    connection_values = list(getattr(genome, "connections", {}).values())
    connection_count = len(connection_values)
    enabled_connections = sum(1 for conn in connection_values if getattr(conn, "enabled", True))
    attr_count = 0
    type_counter: Counter[str] = Counter()
    for node in getattr(genome, "nodes", {}).values():
        attrs = getattr(node, "dynamic_attributes", None) or {}
        attr_count += len(attrs)
        node_type = getattr(node, "node_type", None)
        if node_type:
            type_counter[node_type] += 1

    total_typed = sum(type_counter.values()) or 1
    top_type_bins = max(1, int(top_type_bins))
    most_common = type_counter.most_common(top_type_bins)
    type_ratios = [count / total_typed for _, count in most_common]
    while len(type_ratios) < top_type_bins:
        type_ratios.append(0.0)

    # Normalize counts into smooth ranges so Euclidean distances behave nicely.
    descriptor = [
        math.tanh(node_count / 50.0),
        math.tanh(connection_count / 100.0),
        enabled_connections / max(connection_count, 1),
        math.tanh(attr_count / 50.0),
    ]
    descriptor.extend(type_ratios)
    return tuple(descriptor)


@dataclass
class NoveltyArchive:
    """Maintains a novelty archive and computes per-generation novelty scores."""

    k: int = 15
    max_size: int = 512
    min_fill: int = 16
    insertion_probability: float = 0.2
    score_threshold: float | None = None
    entries: Deque[Tuple[float, ...]] = field(default_factory=deque, init=False)

    def __post_init__(self):
        self.k = max(1, int(self.k))
        self.max_size = max(1, int(self.max_size))
        self.min_fill = max(0, int(self.min_fill))
        prob = float(self.insertion_probability)
        self.insertion_probability = max(0.0, min(1.0, prob))
        if self.score_threshold is not None:
            try:
                self.score_threshold = float(self.score_threshold)
            except (TypeError, ValueError):
                self.score_threshold = None
        self.entries = deque(maxlen=self.max_size)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.entries)

    def score_population(self, descriptors: Mapping[int, Tuple[float, ...]]) -> Dict[int, float]:
        """Compute novelty scores for the current generation's descriptors."""

        if not descriptors:
            return {}
        items = list(descriptors.items())
        scores: Dict[int, float] = {}
        for idx, (genome_id, descriptor) in enumerate(items):
            neighbors = [other for jdx, (_, other) in enumerate(items) if jdx != idx]
            distances = self._neighbor_distances(descriptor, neighbors)
            if distances:
                limit = min(self.k, len(distances))
                scores[genome_id] = sum(distances[:limit]) / float(limit)
            else:
                scores[genome_id] = 0.0
        return scores

    def update(
        self, descriptors: Mapping[int, Tuple[float, ...]], scores: Mapping[int, float], valid_ids: Iterable[int]
    ):
        """Maybe add valid individuals to the archive based on novelty."""

        valid_set = set(valid_ids)
        if not valid_set:
            return
        for genome_id in valid_set:
            descriptor = descriptors.get(genome_id)
            if descriptor is None:
                continue
            novelty = scores.get(genome_id, 0.0)
            should_add = len(self.entries) < self.min_fill
            if not should_add and self.score_threshold is not None:
                should_add = novelty >= self.score_threshold
            if not should_add and random.random() < self.insertion_probability:
                should_add = True
            if should_add:
                self.entries.append(tuple(descriptor))

    def _neighbor_distances(
        self, descriptor: Tuple[float, ...], population_neighbors: Sequence[Tuple[float, ...]]
    ) -> List[float]:
        pool = list(self.entries)
        pool.extend(population_neighbors)
        distances: List[float] = []
        for other in pool:
            if other is None:
                continue
            distances.append(self._euclidean(descriptor, other))
        distances.sort()
        return distances

    @staticmethod
    def _euclidean(v1: Tuple[float, ...], v2: Tuple[float, ...]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip_longest(v1, v2, fillvalue=0.0)))
