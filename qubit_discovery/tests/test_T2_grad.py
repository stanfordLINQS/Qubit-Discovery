import numpy as np

import SQcircuit as sq
from SQcircuit import  Circuit
from SQcircuit import functions as sqf
from SQcircuit import units as unt
from qubit_discovery.losses.T2_functions import (
    partial_cc_dec,
    partial_charge_dec,
    partial_flux_dec,
    dec_rate_flux_torch,
)

###############################################################################
# Circuit helper functions
###############################################################################

def build_trans_elements(EC, EJ, flux, flux_dist):
    loop1 = sq.Loop(flux)
    # define the circuit elements
    C = sq.Capacitor(EC, 'GHz', requires_grad=sq.get_optim_mode())
    JJ = sq.Junction(EJ, 'GHz', loops=[loop1], requires_grad=sq.get_optim_mode())

    # define the circuit
    elements = {
        (0, 1): [JJ, C]
    }

    cr = sq.Circuit(elements, flux_dist=flux_dist)
    return cr


def build_fl_elements(EC, EL, EJ, flux, flux_dist='junctions'):
    loop1 = sq.Loop(flux)
    # define the circuit elements
    C = sq.Capacitor(EC, 'GHz', requires_grad=sq.get_optim_mode())
    L = sq.Inductor(EL, 'GHz', loops=[loop1], requires_grad=sq.get_optim_mode())
    JJ = sq.Junction(EJ, 'GHz', loops=[loop1], requires_grad=sq.get_optim_mode())

    # define the circuit
    elements = {
        (0, 1): [L, JJ, C]
    }

    cr = sq.Circuit(elements, flux_dist=flux_dist)
    return cr

###############################################################################
# Gradient check helper functions
###############################################################################

def test_grad_func(cr: Circuit, 
                   cr_delta: Circuit,
                   delta,
                   unit_factor,
                   np_func,
                   func_grad) -> None:
    """
    Check the gradient of `np_func` numerically at two points `cr` and
    `cr_delta` against the outuput of `func_grad` at `cr`.
    """
    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([300])
        _ = circuit.diag(100)

    grad_numeric = sqf.numpy(np_func(cr_delta) - np_func(cr)) / delta * unit_factor
    grad_analytic = func_grad(cr)

    print('Numeric:', grad_numeric, 'Analytic:', grad_analytic)
    assert np.isclose(grad_numeric, grad_analytic, rtol=1e-2)

###############################################################################
# Dec rate derivative tests 
###############################################################################

def test_cc_grad() -> None:
    EC, EL, EJ = 3.6, 0.46, 50
    delta = 1e-8
    cr = build_fl_elements(EC, EL, EJ, 0.5, 'junctions')
    cr_delta = build_fl_elements(EC, EL, EJ + delta, 0.5, 'junctions')

    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([300])
        _ = circuit.diag(150)

    unit_factor = 1 / 2 / np.pi / 1e9
    grad_numeric = (
        cr_delta.dec_rate('cc', states=(0, 1))
        - cr.dec_rate('cc', states=(0, 1))
    ) / delta * unit_factor

    grad_analytic = partial_cc_dec(cr, cr.elements[(0, 1)][1], (0, 1))

    print('Numeric:', grad_numeric, 'Analytic:', grad_analytic)
    assert np.isclose(grad_numeric, grad_analytic, rtol=1e-2)


def test_flux_grad_EJ() -> None:
    EC, EL, EJ = 3.6, 0.46, 20.5
    delta = 1e-6
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    for optim_mode in [True, False]:
        sq.set_optim_mode(optim_mode)
        for flux_point in flux_points:
            print('Optim mode=', optim_mode, 'Flux =', flux_point)
            cr = build_fl_elements(EC, EL, EJ, flux_point, 'junctions')
            cr_delta = build_fl_elements(EC, EL, EJ + delta, flux_point, 'junctions')

            unit_factor = 1 / 2 / np.pi / 1e9
            test_grad_func(
                cr, cr_delta, delta, unit_factor,
                lambda x: x.dec_rate('flux', states=(0, 1)),
                lambda x: partial_flux_dec(x, x.elements[(0, 1)][1], (0, 1))
            )
            print('\n')
        print('~'*10)

def test_flux_grad_EL() -> None:
    EC, EL, EJ = 3.6, 0.46, 10.2
    delta = 1e-6
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    for optim_mode in [False, True]:
        sq.set_optim_mode(optim_mode)
        for flux_point in flux_points:
            print('Optim mode=', optim_mode, 'Flux =', flux_point)
            cr = build_fl_elements(EC, EL, EJ, flux_point, 'junctions')
            cr_delta = build_fl_elements(EC, EL+delta, EJ, flux_point, 'junctions')

            unit_factor = sqf.numpy(-(unt.Phi0/2/np.pi)**2/cr.elements[(0, 1)][0]._value**2/(2*np.pi*unt.hbar)/1e9)
            test_grad_func(
                cr, cr_delta, delta, unit_factor,
                lambda x: x.dec_rate('flux', states=(0, 1)),
                lambda x: partial_flux_dec(x, x.elements[(0, 1)][0], (0, 1))
            )

def test_charge_grad() -> None:
    EC, EJ = 3.6, 10
    delta = 1e-4
    flux_point = 0.5
    charge_points= [0.1] #[1e-2, 0.2, 0.4, 0.5 - 1e-2, 0.5 + 1e-2, 0.6, 0.8, 1-1e-2]

    for charge_offset in charge_points:
        cr = build_trans_elements(EC, EJ, flux_point, 'junctions')
        cr_delta = build_trans_elements(EC + delta, EJ, flux_point, 'junctions')

        for circuit in [cr, cr_delta]:
            circuit.set_trunc_nums([200])
            # Needs to be offset from 0.5, I think
            circuit.set_charge_offset(1, charge_offset)
            _ = circuit.diag(100)

        unit_factor = sqf.numpy(unt.e**2/2/sq.Capacitor(EC, 'GHz')._value**2/(2*np.pi*unt.hbar) / 1e9)
        test_grad_func(
            cr, cr_delta, delta, unit_factor,
            lambda x: x.dec_rate('charge', states=(0, 1)),
            lambda x: partial_charge_dec(x, x.elements[(0, 1)][1], (0, 1))
        )

###############################################################################
# Torch test functions
###############################################################################


def test_torch_flux_grad() -> None:
    EC, EL, EJ = 3.6, 0.46, 20.5
    delta = 1e-6
    flux_point = 0.5 - 1e-2

    sq.set_optim_mode(True)
    cr = build_fl_elements(EC, EL, EJ, flux_point, 'junctions')
    cr_delta = build_fl_elements(EC, EL, EJ + delta, flux_point, 'junctions')

    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([300])
        _ = circuit.diag(100)

    unit_factor = 1 / 2 / np.pi / 1e9
    grad_numeric = sqf.numpy(
        cr_delta.dec_rate('flux', states=(0, 1))
        - cr.dec_rate('flux', states=(0, 1))
    ) / delta * unit_factor

    flux_dec = dec_rate_flux_torch(cr, (0, 1))
    flux_dec.backward()
    grad_torch = cr.elements[(0, 1)][1]._value.grad

    print('Flux =', flux_point, 'Numeric:', grad_numeric, 'Torch:', grad_torch)
    assert np.isclose(grad_numeric, grad_torch, rtol=1e-2)