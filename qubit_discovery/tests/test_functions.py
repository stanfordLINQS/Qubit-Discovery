"""Test Module for loss helper functions."""

import numpy as np
import torch

import SQcircuit as sq
from SQcircuit.settings import set_optim_mode

from qubit_discovery.losses.functions import (
    element_sensitivity,
    fastest_gate_speed,
    flux_sensitivity,
    number_of_gates
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

        set_optim_mode(optimization_mode)

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

def test_flux_sensitivity() -> None:
    target_sensitivity = torch.tensor(0.001179668, dtype=torch.float64)

    set_optim_mode(True)
    cr = get_fluxonium()
    cr.set_trunc_nums([120])
    cr.diag(20)
    sens = flux_sensitivity(cr)
    assert torch.isclose(sens, target_sensitivity)

def test_element_sensitivity() -> None:
    target_sensitivity = torch.tensor(0.0178, dtype=torch.float32)

    set_optim_mode(True)
    cr = get_fluxonium()
    cr.set_trunc_nums([120])
    cr.diag(20)
    sens = element_sensitivity(cr, number_of_gates)
    print(sens.dtype)
    assert torch.isclose(sens, target_sensitivity, rtol=0.5)
