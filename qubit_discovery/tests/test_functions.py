"""Test Module for loss helper functions."""

import numpy as np
import qutip as qt
import torch

from SQcircuit import Circuit, Capacitor, Junction, Inductor
import SQcircuit as sq
import SQcircuit.functions as sqf
import SQcircuit.units as unt

from qubit_discovery.losses.functions import (
    # Todo: add more functions here
    fastest_gate_speed,
    partial_deriv_approx_flux,
    flux_decoherence_approx,
    partial_deriv_approx_charge,
    charge_decoherence_approx,
    find_elem,
    partial_deriv_approx_elem,
    cc_decoherence_approx,
)
from qubit_discovery.tests.conftest import (
    get_fluxonium, 
    get_fluxonium_random,
    get_cpb,
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

def test_flux_deriv_approx() -> None:
    circuit_generators = [(get_fluxonium_random, [120])]
    n_eigs = 10

    sq.set_optim_mode(True)
    for circ_gen, total_trunc_nums in circuit_generators:
        cr = circ_gen()
        cr.loops[0].set_flux(0.5 - 1e-2)
        cr.set_trunc_nums(total_trunc_nums)
        cr.diag(n_eigs)

        deriv_approx = partial_deriv_approx_flux(cr, 
                                                 cr.loops[0]) * 2 * np.pi
        deriv_exact = cr._get_partial_omega_mn(cr.loops[0], states=(0, 1))

        print(deriv_exact, deriv_approx)
        assert(torch.isclose(deriv_approx, deriv_exact, rtol=1e-2))

def test_flux_decoherence_approx() -> None:
    circuit_generators = [(get_fluxonium_random, [120])]
    n_eigs = 10

    flux_points = [0.1, 0.25, 0.5 - 1e-2]

    sq.set_optim_mode(True)
    for circ_gen, total_trunc_nums in circuit_generators:
        cr = circ_gen()
        cr.set_trunc_nums(total_trunc_nums)

        for flux in flux_points:
            cr.loops[0].set_flux(flux)
            cr.diag(n_eigs)

            decoherence_approx = flux_decoherence_approx(cr).item()
            decoherence_exact = cr.dec_rate('flux', (0, 1)).item()

            print(decoherence_approx, decoherence_exact)
            assert(np.isclose(decoherence_approx, decoherence_exact, rtol=1e-2))

def exact_charge_deriv(cr: Circuit, charge_mode: int):
    state1 = cr._evecs[0]
    state2 = cr._evecs[1]
    
    charge_island_idx = charge_mode - 1
    op = qt.Qobj()
    for j in range(cr.n):
        op += (cr.cInvTrans[charge_island_idx, j] \
               *  cr._memory_ops["Q"][j] \
                / np.sqrt(sq.units.hbar))
        
    return sqf.abs(sqf.operator_inner_product(state2, op, state2) \
                   - sqf.operator_inner_product(state1, op, state1))

def test_charge_deriv_approx() -> None:
    circuit_generators = [(get_cpb, [60])]
    n_eigs = 10

    charge_points = [1e-2, 0.25, 0.5-1e-1]

    sq.set_optim_mode(True)
    for circ_gen, total_trunc_nums in circuit_generators:
        cr = circ_gen()
        cr.set_trunc_nums(total_trunc_nums)

        for charge_pt in charge_points:
            for charge_island_idx in cr.charge_islands.keys():
                charge_mode = charge_island_idx + 1
                cr.set_charge_offset(charge_mode, charge_pt)
                cr.diag(n_eigs)

                deriv_approx = partial_deriv_approx_charge(cr, charge_mode) \
                    * 2 * np.pi
                deriv_exact = exact_charge_deriv(cr, charge_mode)

                print(deriv_approx, deriv_exact)
                assert(torch.isclose(deriv_approx, deriv_exact, rtol=1e-1))

def test_charge_decoherence_approx() -> None:
    circuit_generators = [(get_cpb, [60])]
    n_eigs = 10

    charge_points = [1e-2, 0.25, 0.5-1e-1]

    sq.set_optim_mode(True)
    for circ_gen, total_trunc_nums in circuit_generators:
        cr = circ_gen()
        cr.set_trunc_nums(total_trunc_nums)

        for charge_pt in charge_points:
            for charge_island_idx in cr.charge_islands.keys():
                charge_mode = charge_island_idx + 1
                cr.set_charge_offset(charge_mode, charge_pt)
            cr.diag(n_eigs)

            dec_approx = charge_decoherence_approx(cr).item()
            dec_exact = cr.dec_rate('charge', (0, 1)).item()

            print(dec_approx, dec_exact)
            assert(np.isclose(dec_approx, dec_exact, rtol=1e-1))

def test_EJ_deriv_approx() -> None:
    circuit_generators = [(get_fluxonium_random, [120])]
                        #   (get_cpb, [60])]
    n_eigs = 10

    sq.set_optim_mode(True)
    for circ_gen, total_trunc_nums in circuit_generators:
        cr = circ_gen()
        cr.set_trunc_nums(total_trunc_nums)
        cr.diag(n_eigs)

        for el, B_idx in cr._memory_ops['cos']:
            edge, el_idx = find_elem(cr, el)
            deriv_approx = partial_deriv_approx_elem(cr, edge, el_idx, symmetric=True) \
                * 2 * np.pi
            deriv_exact = cr._get_partial_omega_mn(el, states=(0, 1), _B_idx=B_idx)

            print(deriv_approx, deriv_exact)
            assert(torch.isclose(deriv_approx, deriv_exact, rtol=1e-2))

def test_cc_decoherence_approx() -> None:
    circuit_generators = [(get_fluxonium_random, [120])]
    n_eigs = 10


    sq.set_optim_mode(True)
    for circ_gen, total_trunc_nums in circuit_generators:
        cr = circ_gen()
        cr.set_trunc_nums(total_trunc_nums)
        cr.diag(n_eigs)

        decoherence_approx = cc_decoherence_approx(cr).item()
        decoherence_exact = cr.dec_rate('cc', (0, 1)).item()

        print(decoherence_approx, decoherence_exact)
        assert(np.isclose(decoherence_approx, decoherence_exact, rtol=1e-2))

all_units = unt.farad_list | unt.freq_list | unt.henry_list

def max_ratio(a, b):
    return np.max([np.abs(b / a), np.abs(a / b)])

def function_grad_test(circuit_numpy,
                       function_numpy,
                       circuit_torch,
                       function_torch, 
                       delta=1e-4):
    """General test function for comparing linear approximation with gradient 
    computed with PyTorch backpropagation.

    Parameters
    ----------
        circuit_numpy:
            Numpy circuit for which linear approximation will be calculated.
        function_numpy:
            Function to call on the numpy circuit. This should match the 
            expected output of `function_torch`.
        circuit_torch:
            Equivalent circuit to `circuit_numpy`, but constructed in PyTorch.
        function_torch:
            Equivalent function to `function_numpy`, but written in PyTorch.
        delta:
            Perturbation dx to each parameter value in `circuit_numpy` to 
            compute linear gradient df/dx~(f(x+dx)-f(x)/dx).
    """
    eigen_count = 20
    tolerance = 2e-1
    
    sq.set_optim_mode(False)
    circuit_numpy.diag(eigen_count)
    
    sq.set_optim_mode(True)
    circuit_torch.diag(eigen_count)
    
    tensor_val = function_torch(circuit_torch)
    optimizer = torch.optim.SGD(circuit_torch.parameters, lr=1)
    tensor_val.backward()
    
    for edge_idx, elements_by_edge in enumerate(circuit_numpy.elements.values()):
        for element_idx, element_numpy in enumerate(elements_by_edge):
            sq.set_optim_mode(False)
            scale_factor = (1 / (2 * np.pi) if type(element_numpy) is Junction else 1)
            
            # Calculate f(x+delta)
            element_numpy.set_value(scale_factor * element_numpy.get_value(
                u=element_numpy.unit) + delta,
                element_numpy.unit
            )
            circuit_numpy.update()
            circuit_numpy.diag(eigen_count)
            val_plus = function_numpy(circuit_numpy)
            
            # Calculate f(x-delta)
            element_numpy.set_value(scale_factor * element_numpy.get_value(
                u=element_numpy.unit) - 2 * delta,
                element_numpy.unit
            )
            circuit_numpy.update()
            circuit_numpy.diag(eigen_count)
            val_minus = function_numpy(circuit_numpy)
            
            
            grad_numpy = (val_plus - val_minus) / (2 * delta * all_units[element_numpy.unit])
            element_numpy.set_value(scale_factor * element_numpy.get_value(
                u=element_numpy.unit) + delta,
                element_numpy.unit
            )

            edge_elements_torch = list(circuit_torch.elements.values())[0]
            for edge_element in edge_elements_torch:
                print(f"edge element: {edge_element}")
                print(f"value: {edge_element._value}")
                print(f"value grad: {edge_element._value.grad}")
            grad_torch = edge_elements_torch[element_idx]._value.grad.detach().numpy()
            
            # Scale gradients
            if type(element_numpy) is Capacitor and element_numpy.unit in unt.freq_list:
                grad_factor = -unt.e**2/2/element_numpy._value**2/(2*np.pi*unt.hbar)
                grad_torch /= grad_factor
            elif type(element_numpy) is Inductor and element_numpy.unit in unt.freq_list:
                grad_factor = -(unt.Phi0/2/np.pi)**2/element_numpy._value**2/(2*np.pi*unt.hbar)
                grad_torch /= grad_factor
            if type(element_numpy) is Junction:
                grad_torch *= (2 * np.pi)
            print(f"grad torch: {grad_torch}, grad numpy: {grad_numpy}")
            
            assert max_ratio(grad_torch, grad_numpy) <= 1 + tolerance
            
    optimizer.zero_grad()

def test_flux_decoherence_grad() -> None:
    sq.set_optim_mode(False)
    fl_numpy = get_fluxonium()
    fl_numpy.loops[0].set_flux(0.5 - 1e-2)
    fl_numpy.set_trunc_nums([120])
    sq.set_optim_mode(True)
    fl_torch = get_fluxonium()
    fl_torch.loops[0].set_flux(0.5 - 1e-2)
    fl_torch.set_trunc_nums([120])

    func_numpy = lambda cr: cr.dec_rate('flux', (0, 1))
    func_torch = lambda cr: flux_decoherence_approx(cr)

    function_grad_test(fl_numpy, func_numpy,
                       fl_torch, func_torch)

def test_charge_decoherence_grad() -> None: # 
    sq.set_optim_mode(False)
    trans_numpy = get_cpb()
    trans_numpy.set_charge_offset(1, 0.5 - 1e-1)
    trans_numpy.set_trunc_nums([60])

    sq.set_optim_mode(True)
    trans_torch = get_cpb()
    trans_torch.set_charge_offset(1, 0.5 - 1e-1)
    trans_torch.set_trunc_nums([60])

    func_numpy = lambda cr: cr.dec_rate('charge', (0, 1))
    func_torch = lambda cr: charge_decoherence_approx(cr)

    function_grad_test(trans_numpy, func_numpy,
                       trans_torch, func_torch)
    
def test_EJ_decoherence_grad() -> None:
    sq.set_optim_mode(False)
    fl_numpy = get_fluxonium()
    fl_numpy.loops[0].set_flux(0.5 - 1e-2)
    fl_numpy.set_trunc_nums([120])

    sq.set_optim_mode(True)
    fl_torch = get_fluxonium()
    fl_torch.loops[0].set_flux(0.5 - 1e-2)
    fl_torch.set_trunc_nums([120])

    func_numpy = lambda cr: cr.dec_rate('cc', (0, 1))
    func_torch = lambda cr: cc_decoherence_approx(cr)

    function_grad_test(fl_numpy, func_numpy,
                       fl_torch, func_torch)