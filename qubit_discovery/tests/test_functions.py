"""Test Module for loss helper functions."""

import numpy as np
import qutip as qt
import torch

from SQcircuit import Circuit
import SQcircuit as sq
import SQcircuit.functions as sqf

from qubit_discovery.losses.functions import (
    # Todo: add more functions here
    fastest_gate_speed,
    partial_deriv_approx_flux,
)
from qubit_discovery.tests.conftest import get_fluxonium


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

def test_flux_deriv_approx() -> None:
    circuit_generators = [(get_fluxonium, [120])]
    n_eigs = 10


    sq.set_optim_mode(True)
    for circ_gen, total_trunc_nums in circuit_generators:
        cr = circ_gen()
        cr.set_trunc_nums(total_trunc_nums)
        cr.diag(n_eigs)

        deriv_approx = partial_deriv_approx_flux(cr, 
                                                 cr.loops[0]) * 2 * np.pi
        deriv_exact = cr._get_partial_omega_mn(cr.loops[0], states=(0, 1))

        print(deriv_exact, deriv_approx)
        assert(torch.isclose(deriv_approx, deriv_exact, rtol=1e-2))

def exact_charge_deriv(cr: Circuit, charge_mode: int):
    state1 = cr._evecs[0]
    state2 = cr._evecs[1]
    
    charge_island_idx = charge_mode - 1
    op = qt.Qobj()
    for j in range(cr.n):
        op += (cr.cInvTrans[charge_island_idx, j] * cr._memory_ops["Q"][j] / np.sqrt(sq.units.hbar))
        
    return sqf.abs(sqf.operator_inner_product(state2, op, state2) - sqf.operator_inner_product(state1, op, state1))

def test_charge_deriv_approx() -> None:
    pass

def test_EJ_deriv_approx() -> None:
    pass

def test_flux_decoherence_grad() -> None:
    pass

def test_charge_decoherence_grad() -> None:
    pass

def test_EJ_decoherence_grad() -> None:
    pass