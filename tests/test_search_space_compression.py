import pytest
import torch

from metrics import AreaUnderTaskMetrics
from search_space_compression import (
    OnlineTrainer,
    SharedAttributeVocab,
    StagedBetaSchedule,
)


class MinimalGuide(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_attr_vocab = SharedAttributeVocab([], embedding_dim=4)
        self.dummy = torch.nn.Parameter(torch.zeros(1))


def test_staged_beta_schedule_phases():
    schedule = StagedBetaSchedule(
        start_beta=0.0,
        target_beta=0.2,
        warmup_epochs=2,
        ramp_epochs=2,
        hold_epochs=1,
        cycle_length=3,
        cycle_floor=0.05,
    )
    values = [schedule.value(i) for i in range(10)]
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.1)
    assert values[3] == pytest.approx(0.2)
    assert values[4] == pytest.approx(0.2)
    assert values[5] == pytest.approx(0.05)
    assert values[6] > values[5]
    assert values[7] > values[6]


def test_online_trainer_resolves_dynamic_kl_weight():
    model = MinimalGuide()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model, optimizer, metric_keys=[AreaUnderTaskMetrics])
    schedule = StagedBetaSchedule(start_beta=0.0, target_beta=0.5, warmup_epochs=1, ramp_epochs=1)
    trainer.configure_kl_scheduler(schedule, reset_state=True)
    assert trainer._resolve_kl_weight(0.1) == pytest.approx(0.0)
    trainer._kl_global_epoch = 1
    assert trainer._resolve_kl_weight(0.1) == pytest.approx(0.5)
    trainer._kl_global_epoch = 5
    assert trainer._resolve_kl_weight(0.1) == pytest.approx(0.5)
    trainer.configure_kl_scheduler(schedule, reset_state=True)
    assert trainer._kl_global_epoch == 0
