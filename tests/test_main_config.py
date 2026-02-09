import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import GUIDED_POPULATION_SECTION, load_guided_population_overrides


def test_load_guided_population_overrides_reads_values(tmp_path):
    config_path = tmp_path / "neat-config"
    config_path.write_text(
        textwrap.dedent(
            f"""
            [NEAT]
            pop_size = 10

            [{GUIDED_POPULATION_SECTION}]
            kl_partial_slice_ratio = 0.25
            kl_partial_slice_dims = 12
            kl_partial_slice_start = 3
            wl_kernel_loss_weight = 0.75
            wl_kernel_iterations = 4
            trainer_freeze_cycle = decoder+predictor,encoder,all
            trainer_freeze_verbose = true
            """
        ).strip()
    )
    overrides = load_guided_population_overrides(str(config_path))
    assert overrides["kl_partial_slice_ratio"] == pytest.approx(0.25)
    assert overrides["kl_partial_slice_dims"] == 12
    assert overrides["kl_partial_slice_start"] == 3
    assert overrides["wl_kernel_loss_weight"] == pytest.approx(0.75)
    assert overrides["wl_kernel_iterations"] == 4
    assert overrides["trainer_freeze_cycle"] == "decoder+predictor,encoder,all"
    assert overrides["trainer_freeze_verbose"] is True


def test_load_guided_population_overrides_rejects_bad_values(tmp_path):
    config_path = tmp_path / "neat-config"
    config_path.write_text(
        textwrap.dedent(
            f"""
            [{GUIDED_POPULATION_SECTION}]
            kl_partial_slice_ratio = not-a-number
            """
        ).strip()
    )
    with pytest.raises(ValueError):
        load_guided_population_overrides(str(config_path))
