import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

from pareto import dominates, nondominated_sort
from metrics import MSELoss, TimeCost


class MaxMetric:
    """Simple metric with maximization objective for testing."""

    name = "MaxMetric"
    objective = "max"


def test_dominates_all_min_objectives():
    metrics = {
        1: {MSELoss: 0.1, TimeCost: 1.0},
        2: {MSELoss: 0.2, TimeCost: 2.0},
    }
    assert dominates(metrics, 1, 2)
    assert not dominates(metrics, 2, 1)


def test_dominates_mixed_objectives():
    metrics = {
        1: {MSELoss: 0.1, MaxMetric: 0.5},
        2: {MSELoss: 0.2, MaxMetric: 0.4},
        3: {MSELoss: 0.1, MaxMetric: 0.5},
    }
    # Genome 1 dominates 2 because it is better in both metrics
    assert dominates(metrics, 1, 2)
    # Genome 1 does not dominate 3 because all metrics are equal
    assert not dominates(metrics, 1, 3)
    # Genome 2 does not dominate 1 due to worse metrics
    assert not dominates(metrics, 2, 1)


def test_nondominated_sort():
    metrics = {
        1: {MSELoss: 1.0, MaxMetric: 2.0},
        2: {MSELoss: 2.0, MaxMetric: 1.0},
        3: {MSELoss: 1.0, MaxMetric: 1.0},
    }
    fronts = nondominated_sort(metrics)
    # Expected order: [1] dominates 3, which dominates 2
    assert fronts == [[1], [3], [2]]

