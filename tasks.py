import random
from typing import Dict, List
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kurtosis, skew
from torch.utils.data import DataLoader, Dataset

from complexity import SERIES_STATS
from data import generate_complex_regression_data, generate_fbm_sequence
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
    @staticmethod
    def random_init():
        raise NotImplementedError()

    @classmethod
    def name(cls):
        return cls.__name__.replace("tasks.", "")

    def __init__(
        self,
        metrics: List[Metric],
        num_samples: int,
        train_ratio: float,
        inputs: np.ndarray,
        outputs: np.ndarray,
        silent=False,
    ):
        self.metrics = metrics
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        cut_index = int(num_samples * train_ratio)
        self.train_data = TaskDataset(inputs[:cut_index,], outputs[:cut_index,])
        self.valid_data = TaskDataset(inputs[cut_index:,], outputs[cut_index:,])
        self.feature_functions.append(lambda _: self.num_samples)
        self.feature_functions.append(lambda _: self.train_ratio)
        self.feature_functions.append(lambda _: self.train_data.num_input_features)
        self.feature_functions.append(lambda _: self.train_data.num_output_features)
        # TODO: include metrics in task feature_functions for the task embedding to have access
        if not silent:
            print("  Calculating Task Features")
        self.features = []
        if not silent:
            print("    Training Data Inputs")
        self.features.append([np.mean(func(self.train_data.inputs)) for func in self.feature_functions])
        if not silent:
            print("    Training Data Outputs")
        self.features.append([np.mean(func(self.train_data.outputs)) for func in self.feature_functions])
        if not silent:
            print("    Validation Data Inputs")
        self.features.append([np.mean(func(self.valid_data.inputs)) for func in self.feature_functions])
        if not silent:
            print("    Validation Data Outputs")
        self.features.append([np.mean(func(self.valid_data.outputs)) for func in self.feature_functions])
        self.check_partition_similarity()

    def evaluate_metrics(self, model: nn.Module, dataset: TaskDataset) -> Dict[str, float]:
        values = {
            m.name: m(model(torch.tensor(dataset.inputs, dtype=torch.float32)), dataset.outputs) for m in self.metrics
        }
        return torch.tensor([v for _, v in sorted(values.items(), key=lambda i: i[0])], dtype=torch.float32)

    def check_partition_similarity(self):
        train_in_feats, train_out_feats, valid_in_feats, valid_out_feats = self.features
        cos_sim = F.cosine_similarity(torch.tensor(train_in_feats), torch.tensor(valid_in_feats), dim=0)
        if cos_sim < 0.85:
            warn(f"Cosine similarity of task data partitions' inputs low (cos={cos_sim:.3f})")
        elif cos_sim == 1.0:
            warn(f"Exact match for cosine similarity of task data partitions' inputs")
        cos_sim = F.cosine_similarity(torch.tensor(train_out_feats), torch.tensor(valid_out_feats), dim=0)
        if cos_sim < 0.85:
            warn(f"Cosine similarity of task data partitions' outputs low (cos={cos_sim:.3f})")
        elif cos_sim == 1.0:
            warn(f"Exact match for cosine similarity of task data partitions' outputs")


class RegressionTask(Task):
    @staticmethod
    def random_init(num_samples=None, silent=False):
        return RegressionTask(
            true_dims=random.randint(1, 10),
            observed_dims=random.randint(1, 10),
            metrics=[MSELoss()],
            num_samples=num_samples if num_samples is not None else random.randint(100, 1000),
            train_ratio=max(0.5, min(random.random(), 0.7)),
            silent=silent,
        )

    def __init__(
        self, true_dims: int, observed_dims: int, metrics: List, num_samples: int, train_ratio: float, silent=False
    ):
        self.true_dims = true_dims
        self.observed_dims = observed_dims
        inputs, outputs = generate_complex_regression_data(num_samples, true_dims, observed_dims)
        self.feature_functions = [
            lambda _: self.true_dims,
            lambda _: self.observed_dims,
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
            skew,
            kurtosis,
        ]
        super().__init__(metrics, num_samples, train_ratio, inputs, outputs, silent)


class HurstTargetTimeSeriesTransformTask(Task):
    @staticmethod
    def random_init(num_samples=None, silent=False):
        num_features = random.randint(1, 10)
        return HurstTargetTimeSeriesTransformTask(
            mean=np.random.randn(num_features) * 100,
            stdev=np.random.randn(num_features) * 100 + 1,
            hurst_target=random.random(),
            fbm_length=random.random() * 10,
            series_length=random.randint(100, 200),
            num_features=num_features,
            metrics=[MSELoss()],
            num_samples=num_samples if num_samples is not None else random.randint(100, 500),
            train_ratio=max(0.5, min(random.random(), 0.7)),
            silent=silent,
        )

    def __init__(
        self,
        mean: float,
        stdev: float,
        hurst_target: float,
        fbm_length: float,
        series_length: int,
        num_features: int,
        metrics: List[Metric],
        num_samples: int,
        train_ratio: float,
        silent=False,
    ):
        self.mean = mean
        self.stdev = stdev
        self.hurst_target = hurst_target
        self.series_length = series_length
        self.fbm_length = fbm_length
        self.num_features = num_features
        sequences = [
            generate_fbm_sequence(mean, stdev, hurst_target, fbm_length, num_features, series_length)
            for _ in range(num_samples)
        ]
        outputs = [
            generate_fbm_sequence(mean, stdev, hurst_target, fbm_length, num_features, series_length)
            for _ in range(num_samples)
        ]
        self.feature_functions = [
            *SERIES_STATS,
            lambda _: self.mean,
            lambda _: self.stdev,
            lambda _: self.hurst_target,
            lambda _: self.fbm_length,
            lambda _: self.series_length,
            lambda _: self.num_features,
        ]
        super().__init__(metrics, num_samples, train_ratio, np.array(sequences), np.array(outputs), silent)


TASK_TYPE_TO_CLASS = {
    RegressionTask.name(): RegressionTask,
    # HurstTargetTimeSeriesTransformTask.name(): HurstTargetTimeSeriesTransformTask,
}
TASK_FEATURE_DIMS = {
    n: 4 * len(c.random_init(num_samples=4, silent=True).feature_functions) for n, c in TASK_TYPE_TO_CLASS.items()
}
TASK_TYPE_TO_INDEX = {k: i for i, k in enumerate(TASK_FEATURE_DIMS.keys())}
