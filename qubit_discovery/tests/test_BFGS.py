"""Test Module for BFGS algorithm."""
import pytest

import torch
import numpy as np

import SQcircuit as sq

from SQcircuit import Circuit

from qubit_discovery.optimization import run_BFGS
from qubit_discovery.losses.loss import calculate_loss_metrics
from qubit_discovery.tests.conftest import (
    get_fluxonium,
    get_bounds,
    are_loss_dicts_close
)


def test_bfgs_run() -> None:
    """Test one step of BFGS algorithm."""

    target_params = torch.tensor(
        [4.6898e-12, 1.6346e-07, 6.2832e+09],
        dtype=torch.float64
    )

    target_loss_record = {
        'frequency_loss': [np.array(1e-13)],
        'anharmonicity_loss': [np.array(1.42174492)],
        't1_loss': [np.array(0)],
        't_phi_loss': [np.array(0)],
        'flux_sensitivity_loss': [np.array(1.e-13)],
        'charge_sensitivity_loss': [np.array(1e-13)],
        'total_loss': [np.array(1.42174492)],
    }

    def my_loss_function(cr: Circuit):
        return calculate_loss_metrics(
            cr,
            use_losses={
                "frequency": 1.0,
                "anharmonicity": 1.0,
                "flux_sensitivity": 1.0,
                "charge_sensitivity": 1.0,
                "t1": 1.0,
                "t_phi": 1.0
            },
            use_metrics=[]
        )

    sq.set_optim_mode(True)

    total_trunc_num: int = 120
    baseline_trunc_num = total_trunc_num

    circuit = get_fluxonium()
    circuit.set_trunc_nums([total_trunc_num])

    params, loss_record = run_BFGS(
        circuit=circuit,
        circuit_code="JL",
        loss_metric_function=my_loss_function,
        name="BFGS_test",
        num_eigenvalues=10,
        total_trunc_num=total_trunc_num,
        baseline_trunc_nums=[baseline_trunc_num],
        bounds=get_bounds(),
        lr=1.0,
        max_iter=1,
        tolerance=1e-7,
        verbose=False,
        save_intermediate_circuits=False,
    )

    del loss_record['circuit_code']

    assert params.detach() == pytest.approx(target_params, rel=1e-2)
    assert are_loss_dicts_close(loss_record, target_loss_record, rel=1e-2)
