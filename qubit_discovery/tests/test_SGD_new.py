"""Test Module for BFGS algorithm."""

import torch

import SQcircuit as sq

from SQcircuit import Circuit

from qubit_discovery.optimization import run_SGD
from qubit_discovery.losses.loss import calculate_loss_metrics
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
        return calculate_loss_metrics(
            cr,
            use_losses={
                "frequency": 1.0,
                "anharmonicity": 1.0,
                "flux_sensitivity": 1.0,
                "charge_sensitivity": 1.0,
                "T1": 1.0,
            },
            use_metrics=[],
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
        save_intermediate_circuits=False,
    )

    assert torch.isclose(torch.stack(circuit.parameters), target_params).all()
