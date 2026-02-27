"""Shared helpers for constructing reusable test fixtures."""

from __future__ import annotations

import pathlib
from typing import Type

import neat

from genome import OptimizerGenome
from relative_rank_stagnation import RelativeRankStagnation
from reproduction import GuidedReproduction

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_NEAT_CONFIG_PATH = REPO_ROOT / "neat-config"


def make_neat_config(
    *,
    config_path: str | pathlib.Path | None = None,
    genome_type: Type[OptimizerGenome] = OptimizerGenome,
    reproduction_type: Type[GuidedReproduction] = GuidedReproduction,
    species_set_type: Type[neat.DefaultSpeciesSet] = neat.DefaultSpeciesSet,
    stagnation_type: Type[RelativeRankStagnation] = RelativeRankStagnation,
) -> neat.Config:
    """Return a NEAT config wired to the repo's default settings for optimizer evolution tests."""

    resolved_path = pathlib.Path(config_path) if config_path is not None else DEFAULT_NEAT_CONFIG_PATH
    return neat.Config(
        genome_type,
        reproduction_type,
        species_set_type,
        stagnation_type,
        str(resolved_path),
    )


__all__ = ["make_neat_config", "DEFAULT_NEAT_CONFIG_PATH", "REPO_ROOT"]
