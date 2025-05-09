import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from typing import List, Dict

from data import generate_complex_loss_landscape_data
from complexity import SERIES_STATS
from metrics import *


class Task:
    feature_functions = []

    def __init__(self, metrics: List[Metric], num_samples:int, train_ratio:float):
        self.metrics = metrics
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.data = [[], []] # training, validation

    def evaluate_metrics(self, model: nn.Module, data) -> Dict[str, float]:
        """ This also inverts minimization objectives """
        values = {m.name: m(model(torch.tensor(data[0], dtype=torch.float32)), data[1]) for m in self.metrics}
        norm = []
        for metric in self.metrics:
            if metric.objective == 'max':
                norm.append(values[metric.name])
            else: # 'min': invert
                norm.append(-values[metric.name])
        return torch.tensor(norm, dtype=torch.float32)


class RegressionTask(Task):
    feature_functions = [*SERIES_STATS,
        lambda _: self.true_dims,
        lambda _: self.observed_dims,
    ]

    def __init__(self, true_dims:int, observed_dims:int, metrics:List, num_samples:int, train_ratio:float):
        super().__init__(metrics, num_samples, train_ratio)
        self.true_dims = true_dims
        self.observed_dims = observed_dims
        inputs, outputs = generate_complex_loss_landscape_data(num_samples, self.true_dims, self.observed_dims)
        cut_index = int(num_samples*train_ratio)
        self.train_data = (inputs[:cut_index], outputs[:cut_index])
        self.valid_data = (inputs[cut_index:], outputs[cut_index:])
