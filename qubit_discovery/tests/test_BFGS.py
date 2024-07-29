"""Test Module for BFGS algorithm."""
import pytest

import matplotlib.pyplot as plt
import numpy as np
import torch

import SQcircuit as sq

from SQcircuit import Circuit

from qubit_discovery.optimization import run_BFGS
from qubit_discovery.losses import build_loss_function
from qubit_discovery.tests.conftest import (
    get_fluxonium,
    get_bounds,
    are_loss_dicts_close
)


def test_bfgs_run() -> None:
    """Test two steps of the BFGS algorithm."""

    target_params = [1.7817e-14, 3.4925e-07, 6.2715e+09]

    target_loss_record = {
        'frequency_loss': [np.array(1.e-13),np.array(1.e-13)],
        'number_of_gates_loss': [np.array(2.2558016e-05), np.array(1.19195123e-05)],
        'flux_sensitivity_loss': [np.array(1.e-13), np.array(1.e-13)],
        'charge_sensitivity_loss': [np.array(1.e-13), np.array(1.e-13)],
        'total_loss': [np.array(2.25580163e-05), np.array(1.19195126e-05)]
    }

    my_loss_function = build_loss_function(
        use_losses={
                "frequency": 1.0,
                "number_of_gates": 1.0,
                "flux_sensitivity": 1.0,
                "charge_sensitivity": 1.0,
            },
        use_metrics=[]
    )

    sq.set_optim_mode(True)

    total_trunc_num = 120

    circuit = get_fluxonium()
    circuit.set_trunc_nums([total_trunc_num])

    circuit, loss_record, _ = run_BFGS(
        circuit=circuit,
        loss_metric_function=my_loss_function,
        max_iter=2,
        total_trunc_num=total_trunc_num,
        bounds = get_bounds(),
        num_eigenvalues=10,
        lr=1.0,
        tolerance=1e-7,
        verbose=False
    )

    print(circuit.parameters)
    assert torch.stack(circuit.parameters).detach() == pytest.approx(target_params, rel=1e-2)
    assert are_loss_dicts_close(loss_record, target_loss_record, rel=1e-2)
