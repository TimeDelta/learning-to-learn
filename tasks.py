import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import random
from typing import List, Dict

from data import generate_complex_loss_landscape_data
from complexity import SERIES_STATS
from metrics import *


class Task:
    feature_functions = []
    name = str(__name__).replace('tasks.', '')

    @staticmethod
    def random_init():
        pass

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

    def get_features(self):
        return [np.mean([func(series) for series in self.data[0][0]]) for func in self.feature_functions],\
               [np.mean([func(series) for series in self.data[1][0]]) for func in self.feature_functions]

    def check_partition_similarity(self):
        train_feats, valid_feats = self.get_features()
        cos_sim = F.cosine_similarity(train_feats, valid_feats, dim=0)
        if cos_sim < 0.85:
            warn(f"Cosine similarity of task data partitions low (cos={cos_sim:.3f})")

class RegressionTask(Task):
    feature_functions = [*SERIES_STATS,
        lambda _: self.true_dims,
        lambda _: self.observed_dims,
    ]

    @staticmethod
    def random_init():
        return RegressionTask(
            true_dims=random.randint(1, 10),
            observed_dims=random.randint(1,10),
            metrics=[MSELoss()],
            num_samples=1000,
            train_ratio=2.0/3.0
        )

    def __init__(self, true_dims:int, observed_dims:int, metrics:List, num_samples:int, train_ratio:float):
        super().__init__(metrics, num_samples, train_ratio)
        self.true_dims = true_dims
        self.observed_dims = observed_dims
        inputs, outputs = generate_complex_loss_landscape_data(num_samples, self.true_dims, self.observed_dims)
        cut_index = int(num_samples*train_ratio)
        self.train_data = self.data[0] = (inputs[:cut_index,], outputs[:cut_index,])
        self.valid_data = self.data[1] = (inputs[cut_index:,], outputs[cut_index:,])
        print(self.train_data[0].shape)
        self.check_partition_similarity()

TASK_FEATURE_DIMS = {
    RegressionTask.name: len(RegressionTask.feature_functions),
}
TASK_TYPE_TO_INDEX = {k: i for i, k in enumerate(TASK_FEATURE_DIMS.keys())}
TASK_TYPE_TO_CLASS = {
    RegressionTask.name: RegressionTask,
}
