import numpy as np
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
from typing import List, Dict

from data import generate_complex_regression_data, generate_fbm_sequence
from complexity import SERIES_STATS
from metrics import *


class TaskDataset(Dataset):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.outputs[i]

    @property
    def num_input_features(self):
        return self.inputs[0].shape[-1]

    @property
    def num_output_features(self):
        return self.inputs[0].shape[-1]


class Task:
    feature_functions = [
    ]

    @staticmethod
    def random_init():
        raise NotImplementedError()

    @classmethod
    def name(cls):
        return cls.__name__.replace('tasks.', '')

    def __init__(self, metrics:List[Metric], num_samples:int, train_ratio:float, inputs:np.ndarray, outputs:np.ndarray):
        self.metrics = metrics
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        cut_index = int(num_samples*train_ratio)
        self.train_data = TaskDataset(inputs[:cut_index,], outputs[:cut_index,])
        self.valid_data = TaskDataset(inputs[cut_index:,], outputs[cut_index:,])
        self.feature_functions.append(lambda _: self.num_samples)
        self.feature_functions.append(lambda _: self.train_ratio)
        self.feature_functions.append(lambda _: self.train_data.num_input_features)
        self.feature_functions.append(lambda _: self.train_data.num_output_features)
        # TODO: include metrics in task feature_functions for the task embedding to have access
        print('  Calculating Task Features')
        self.features = []
        print('    Training Data Inputs')
        self.features.append([np.mean(func(self.train_data.inputs)) for func in self.feature_functions])
        print('    Training Data Outputs')
        self.features.append([np.mean(func(self.train_data.outputs)) for func in self.feature_functions])
        print('    Validation Data Inputs')
        self.features.append([np.mean(func(self.valid_data.inputs)) for func in self.feature_functions])
        print('    Validation Data Outputs')
        self.features.append([np.mean(func(self.valid_data.outputs)) for func in self.feature_functions])
        self.check_partition_similarity()

    def evaluate_metrics(self, model: nn.Module, dataset:TaskDataset) -> Dict[str, float]:
        """ This also inverts minimization objectives """
        values = {m.name: m(model(torch.tensor(dataset.inputs, dtype=torch.float32)), dataset.outputs) for m in self.metrics}
        final_values = []
        for metric in self.metrics:
            if metric.objective == 'max':
                final_values.append(values[metric.name])
            else: # 'min': invert
                final_values.append(-values[metric.name])
        return torch.tensor(final_values, dtype=torch.float32)

    def check_partition_similarity(self):
        train_in_feats, train_out_feats, valid_in_feats, valid_out_feats = self.features
        cos_sim = F.cosine_similarity(torch.tensor(train_in_feats), torch.tensor(valid_in_feats), dim=0)
        if cos_sim < .85:
            warn(f"Cosine similarity of task data partitions' inputs low (cos={cos_sim:.3f})")
        if cos_sim == 1.0:
            warn(f"Exact match for cosine similarity of task data partitions' inputs")
        cos_sim = F.cosine_similarity(torch.tensor(train_out_feats), torch.tensor(valid_out_feats), dim=0)
        if cos_sim < .85:
            warn(f"Cosine similarity of task data partitions' outputs low (cos={cos_sim:.3f})")
        if cos_sim == 1.0:
            warn(f"Exact match for cosine similarity of task data partitions' outputs")


class RegressionTask(Task):
    feature_functions = [
        lambda samples: np.mean(samples, axis=0),
        lambda samples: np.std(samples, axis=0),
        lambda samples: np.min(samples, axis=0),
        lambda samples: np.max(samples, axis=0),
        lambda samples: np.median(samples, axis=0),

        lambda samples: np.percentile(samples, 25),
        lambda samples: np.percentile(samples, 75),

        lambda samples: np.linalg.norm(samples, 1),
        lambda samples: np.linalg.norm(samples, 2),
        lambda samples: np.sum(np.abs(samples)),

        # skew,
        # kurtosis,
    ]

    @staticmethod
    def random_init():
        return RegressionTask(
            true_dims=random.randint(1, 10),
            observed_dims=random.randint(1, 10),
            metrics=[MSELoss()],
            num_samples=random.randint(100, 1000),
            train_ratio=max(.5, min(random.random(), .7))
        )

    def __init__(self, true_dims:int, observed_dims:int, metrics:List, num_samples:int, train_ratio:float):
        self.true_dims = true_dims
        self.observed_dims = observed_dims
        inputs, outputs = generate_complex_regression_data(num_samples, true_dims, observed_dims)
        self.feature_functions.append(lambda _: self.true_dims)
        self.feature_functions.append(lambda _: self.observed_dims)
        super().__init__(metrics, num_samples, train_ratio, inputs, outputs)

class HurstTargetTimeSeriesTransformTask(Task):
    feature_functions = [
        *SERIES_STATS,
    ]

    @staticmethod
    def random_init():
        num_features = random.randint(1, 10)
        return HurstTargetTimeSeriesTransformTask(
            mean=np.random.randn(num_features)*100,
            stdev=np.random.randn(num_features)*100 + 1,
            hurst_target=random.random(),
            fbm_length=random.random()*10,
            series_length=random.randint(100, 200),
            num_features=num_features,
            metrics=[MSELoss()],
            num_samples=random.randint(100, 500),
            train_ratio=max(.5, min(random.random(), .7))
        )

    def __init__(self,
        mean:float,
        stdev:float,
        hurst_target:float,
        fbm_length:float,
        series_length:int,
        num_features:int,
        metrics:List[Metric],
        num_samples:int,
        train_ratio:float
    ):
        self.mean = mean
        self.stdev = stdev
        self.hurst_target = hurst_target
        self.series_length = series_length
        self.fbm_length = fbm_length
        self.num_features = num_features
        sequences = [generate_fbm_sequence(mean, stdev, hurst_target, fbm_length, num_features, series_length) for _ in range(num_samples)]
        outputs = [generate_fbm_sequence(mean, stdev, hurst_target, fbm_length, num_features, series_length) for _ in range(num_samples)]
        self.feature_functions.append(lambda _: self.mean)
        self.feature_functions.append(lambda _: self.stdev)
        self.feature_functions.append(lambda _: self.hurst_target)
        self.feature_functions.append(lambda _: self.fbm_length)
        self.feature_functions.append(lambda _: self.series_length)
        self.feature_functions.append(lambda _: self.num_features)
        super().__init__(metrics, num_samples, train_ratio, np.array(sequences), np.array(outputs))

TASK_TYPE_TO_CLASS = {
    RegressionTask.name(): RegressionTask,
    HurstTargetTimeSeriesTransformTask.name(): HurstTargetTimeSeriesTransformTask,
}
TASK_FEATURE_DIMS = {n: len(c.feature_functions) for n, c in TASK_TYPE_TO_CLASS.items()}
TASK_TYPE_TO_INDEX = {k: i for i, k in enumerate(TASK_FEATURE_DIMS.keys())}
