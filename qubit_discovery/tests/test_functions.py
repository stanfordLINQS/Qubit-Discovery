"""Test Module for loss helper functions."""

import numpy as np
import torch

import SQcircuit as sq

from qubit_discovery.losses.functions import (
    fastest_gate_speed,
)
from qubit_discovery.tests.conftest import (
    get_fluxonium, 
)


def test_fastest_gate_speed() -> None:
    """Test fastest_gate_speed function."""

    # Test settings
    target_omega = torch.tensor(
        [0.55255205],
        dtype=torch.float64
    )
    total_trunc_num = 120
    n_eigs = 10

    def get_circuit_based_op_optim(optimization_mode: bool) -> sq.Circuit:

        sq.set_optim_mode(optimization_mode)

        cr = get_fluxonium()
        cr.set_trunc_nums([total_trunc_num])
        cr.diag(n_eigs)

        return cr

    for optim_mode in [True, False]:

        circuit = get_circuit_based_op_optim(optimization_mode=optim_mode)
        omega = fastest_gate_speed(circuit)

        if optim_mode:
            torch.isclose(omega, target_omega, rtol=1e-2)
        else:
            np.isclose(omega, target_omega.item(), rtol=1e-2)
