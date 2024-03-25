"""Test Module for BFGS algorithm."""
import pytest

import torch
import numpy as np

import SQcircuit as sq

from SQcircuit import Circuit

from qubit_discovery.optimization import run_BFGS
from qubit_discovery.losses import loss_functions
from qubit_discovery.tests.conftest import (
    get_fluxonium,
    get_bounds,
    are_loss_dicts_close
)


def test_BFGS() -> None:
    """Test one step of BFGS algorithm."""

    target_params = torch.tensor(
        [2.7302e-12, 1.6346e-07, 6.2832e+09],
        dtype=torch.float64
    )

    target_loss_record = {
        'frequency_loss': [np.array(5.48357335)],
        'anharmonicity_loss': [np.array(1.42174492)],
        'T1_loss': [np.array(42581151.89052253)],
        'flux_sensitivity_loss': [np.array(1.e-14)],
        'charge_sensitivity_loss': [np.array(0.)],
        'total_loss': [np.array(42581158.79584081)],
    }

    def my_loss_function(cr: Circuit):
        return loss_functions['default'](
            cr,
            use_frequency_loss=True,
            use_anharmonicity_loss=True,
            use_flux_sensitivity_loss=True,
            use_charge_sensitivity_loss=True,
            use_T1_loss=True,
        )

    sq.set_optim_mode(True)

    circuit = get_fluxonium()
    circuit.set_trunc_nums([200])

    params, loss_record = run_BFGS(
        circuit=circuit,
        circuit_code="JL",
        loss_metric_function=my_loss_function,
        name="BFGS_test",
        num_eigenvalues=10,
        total_trunc_num=200,
        save_loc="./",
        bounds=get_bounds(),
        lr=1.0,
        max_iter=1,
        tolerance=1e-7,
        verbose=False,
        save_intermediate_circuits=True,
    )

    del loss_record['circuit_code']
    del loss_record['experimental_sensitivity_loss']

    assert params.detach() == pytest.approx(target_params, rel=1e-2)
    assert are_loss_dicts_close(loss_record, target_loss_record, rel=1e-2)
