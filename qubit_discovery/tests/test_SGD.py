"""Test Module for BFGS algorithm."""

import torch

import SQcircuit as sq

from SQcircuit import Circuit

from qubit_discovery.optimization import run_SGD
from qubit_discovery.losses import loss_functions
from qubit_discovery.tests.conftest import (
    get_fluxonium,
    get_bounds,
)


def test_sgd_run() -> None:
    """Test one step of SGD algorithm."""

    target_params = torch.tensor(
        [1.9758e-14, 1.6673e-07, 6.4088e+09],
        dtype=torch.float64
    )

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
    total_trunc_num = 120

    circuit = get_fluxonium()
    circuit.set_trunc_nums([total_trunc_num])

    run_SGD(
        circuit=circuit,
        circuit_code="JL",
        loss_metric_function=my_loss_function,
        name="SGD_test",
        num_eigenvalues=10,
        baseline_trunc_nums=[total_trunc_num],
        total_trunc_num=total_trunc_num,
        num_epochs=1,
        bounds=get_bounds(),
        save_loc="./",
        save_intermediate_circuits=False
    )

    assert torch.isclose(torch.stack(circuit.parameters), target_params).all()
