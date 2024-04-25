import numpy as np

import SQcircuit as sq
from SQcircuit import Capacitor, Circuit, Inductor, Junction
from SQcircuit import functions as sqf
from SQcircuit import units as unt
from qubit_discovery.losses.T2_functions import (
    partial_cc_dec,
    partial_charge_dec,
    partial_flux_dec,
    dec_rate_cc_torch,
    dec_rate_charge_torch,
    dec_rate_flux_torch,
)
from qubit_discovery.optimization.utils import set_grad_zero


###############################################################################
# Circuit helper functions
###############################################################################

def build_trans_elements(EC, EJ, flux, flux_dist):
    loop1 = sq.Loop(flux)
    # define the circuit elements
    C = sq.Capacitor(
        EC, 'GHz', requires_grad=sq.get_optim_mode()
    )
    JJ = sq.Junction(
        EJ, 'GHz', loops=[loop1], requires_grad=sq.get_optim_mode()
    )

    # define the circuit
    elements = {
        (0, 1): [JJ, C]
    }

    cr = sq.Circuit(elements, flux_dist=flux_dist)
    return cr


def build_fl_elements(EC, EL, EJ, flux, flux_dist='junctions'):
    loop1 = sq.Loop(flux)
    # define the circuit elements
    C = sq.Capacitor(
        EC, 'GHz', requires_grad=sq.get_optim_mode()
    )
    L = sq.Inductor(
        EL, 'GHz', loops=[loop1], requires_grad=sq.get_optim_mode()
    )
    JJ = sq.Junction(
        EJ, 'GHz', loops=[loop1], requires_grad=sq.get_optim_mode()
    )

    # define the circuit
    elements = {
        (0, 1): [L, JJ, C]
    }

    cr = sq.Circuit(elements, flux_dist=flux_dist)
    return cr


###############################################################################
# Gradient check helper functions
###############################################################################

UNIT_FACTORS = {
    Capacitor: lambda EC: sqf.numpy(
        -unt.e**2/2/sq.Capacitor(EC, 'GHz')._value**2/(2*np.pi*unt.hbar) / 1e9
    ),
    Inductor: lambda EL: sqf.numpy(
        -(unt.Phi0/2/np.pi)**2/sq.Inductor(EL, 'GHz')._value**2
        / (2*np.pi*unt.hbar)/1e9
    ),
    Junction: 1 / 2 / np.pi / 1e9
}


def do_grad_func_test(
    cr: Circuit,
    cr_delta: Circuit,
    delta,
    unit_factor,
    np_func,
    func_grad
) -> None:
    """
    Check the gradient of `np_func` numerically at two points `cr` and
    `cr_delta` against the outuput of `func_grad` at `cr`.
    """
    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([300])
        _ = circuit.diag(100)

    grad_numeric = (
        sqf.numpy(np_func(cr_delta) - np_func(cr))
        / delta
        * unit_factor
    )
    grad_analytic = func_grad(cr)

    print('Numeric:', grad_numeric, 'Analytic:', grad_analytic)
    assert np.isclose(grad_numeric, grad_analytic, rtol=1e-2)


def do_test_dec_rate_fluxonium(
    dec_rate_func,
    dec_grad,
    EC=3.6,
    EL=0.46,
    EJ=20.5,
    delta=1e-6,
) -> None:
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    for optim_mode in [True, False]:
        sq.set_optim_mode(optim_mode)
        for flux_point in flux_points:
            print('Optim mode=', optim_mode, 'Flux =', flux_point)
            cr = build_fl_elements(EC, EL, EJ, flux_point, 'junctions')
            cr_delta_J  = build_fl_elements(
                EC, EL, EJ + delta, flux_point, 'junctions'
            )
            cr_delta_L = build_fl_elements(
                EC, EL + delta, EJ, flux_point, 'junctions'
            )
            cr_delta_C = build_fl_elements(
                EC + delta, EL, EJ, flux_point, 'junctions'
            )

            do_grad_func_test(
                cr, cr_delta_J, delta, UNIT_FACTORS[Junction],
                lambda x: dec_rate_func(x, states=(0, 1)),
                lambda x: dec_grad(x, x.elements[(0, 1)][1], (0, 1))
            )
            do_grad_func_test(
                cr, cr_delta_L, delta, UNIT_FACTORS[Inductor](EL),
                lambda x: dec_rate_func(x, states=(0, 1)),
                lambda x: dec_grad(x, x.elements[(0, 1)][0], (0, 1))
            )
            do_grad_func_test(
                cr, cr_delta_C, delta, UNIT_FACTORS[Capacitor](EC),
                lambda x: dec_rate_func(x, states=(0, 1)),
                lambda x: dec_grad(x, x.elements[(0, 1)][2], (0, 1))
            )
            print('\n')
        print('~'*10)


def do_test_torch_func(
    cr: Circuit,
    cr_delta: Circuit,
    delta,
    unit_factor,
    np_func,
    torch_func,
    elem
) -> None:
    """
    Check the gradient of `np_func` numerically at two points `cr` and
    `cr_delta` against the output of `func_grad` at `cr`.
    """
    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([300])
        _ = circuit.diag(100)

    grad_numeric = (
            sqf.numpy(np_func(cr_delta) - np_func(cr)) / delta * unit_factor
    )

    dec_rate = torch_func(cr)
    dec_rate.backward()
    grad_torch = elem._value.grad

    print('Numeric:', grad_numeric, 'Torch:', grad_torch)
    assert np.isclose(grad_numeric, grad_torch, rtol=1e-2)


def do_test_dec_torch_fluxonium(
    dec_rate_np,
    dec_rate_torch,
    EC=3.6,
    EL=0.46,
    EJ=20.5,
    delta=1e-6
) -> None:
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    sq.set_optim_mode(True)
    for flux_point in flux_points:
        print('phi_ext =', flux_point)
        cr = build_fl_elements(EC, EL, EJ, flux_point, 'junctions')
        cr_delta_J  = build_fl_elements(
            EC, EL, EJ + delta, flux_point, 'junctions'
        )
        cr_delta_L = build_fl_elements(
            EC, EL+delta, EJ, flux_point, 'junctions'
        )
        cr_delta_C = build_fl_elements(
            EC + delta, EL, EJ, flux_point, 'junctions'
        )

        do_test_torch_func(
            cr, cr_delta_J, delta, UNIT_FACTORS[Junction],
            lambda x: dec_rate_np(x, states=(0, 1)),
            lambda x: dec_rate_torch(x, (0,1)),
            cr.elements[(0, 1)][1]
        )
        set_grad_zero(cr) # don't forget this!
        do_test_torch_func(
            cr, cr_delta_L, delta, UNIT_FACTORS[Inductor](EL),
            lambda x: dec_rate_np(x, states=(0, 1)),
            lambda x: dec_rate_torch(x, (0,1)),
            cr.elements[(0, 1)][0]
        )
        set_grad_zero(cr)
        do_test_torch_func(
            cr, cr_delta_C, delta, UNIT_FACTORS[Capacitor](EC),
            lambda x: dec_rate_np(x, states=(0, 1)),
            lambda x: dec_rate_torch(x, (0,1)),
            cr.elements[(0, 1)][2]
        )
        print('\n')


def do_test_dec_torch_trans(
    dec_rate_np,
    dec_rate_torch,
    EC=3.6,
    EJ=10.2,
    delta=1e-6
) -> None:

    flux_point = 0.5
    charge_points = [1e-2, 0.2, 0.4, 0.5 - 1e-2, 0.5 + 1e-2, 0.6, 0.8, 1-1e-2]

    sq.set_optim_mode(True)
    for charge_offset in charge_points:
        print('ng =', charge_offset)
        cr = build_trans_elements(EC, EJ, flux_point, 'junctions')
        cr_delta_C = build_trans_elements(
            EC + delta, EJ, flux_point, 'junctions'
        )
        cr_delta_J = build_trans_elements(
            EC, EJ + delta, flux_point, 'junctions'
        )

        for circ in [cr, cr_delta_C, cr_delta_J]:
            circ.set_charge_offset(1, charge_offset)

        do_test_torch_func(
            cr, cr_delta_J, delta, UNIT_FACTORS[Junction],
            lambda x: dec_rate_np(x, states=(0, 1)),
            lambda x: dec_rate_torch(x, (0,1)),
            cr.elements[(0, 1)][0]
        )
        set_grad_zero(cr)  # don't forget this!
        do_test_torch_func(
            cr, cr_delta_C, delta, UNIT_FACTORS[Capacitor](EC),
            lambda x: dec_rate_np(x, states=(0, 1)),
            lambda x: dec_rate_torch(x, (0,1)),
            cr.elements[(0, 1)][1]
        )
        print('\n')


###############################################################################
# Dec rate derivative tests 
###############################################################################

def test_cc_grad() -> None:
    do_test_dec_rate_fluxonium(
        lambda cr, states: cr.dec_rate('cc', states=states),
        lambda cr, elem, states: partial_cc_dec(cr, elem, states)
    )


def test_charge_grad() -> None:
    EC, EJ = 3.6, 20
    delta = 1e-4
    flux_point = 0.5
    charge_points= [1e-2, 0.2, 0.4, 0.5 - 1e-2, 0.5 + 1e-2, 0.6, 0.8, 1-1e-2]

    for charge_offset in charge_points:
        cr = build_trans_elements(EC, EJ, flux_point, 'junctions')
        cr_delta_C = build_trans_elements(
            EC + delta, EJ, flux_point, 'junctions'
        )
        cr_delta_J = build_trans_elements(
            EC, EJ + delta, flux_point, 'junctions'
        )

        for circ in [cr, cr_delta_C, cr_delta_J]:
            circ.set_charge_offset(1, charge_offset)

        do_grad_func_test(
            cr, cr_delta_C, delta, UNIT_FACTORS[Capacitor](EC),
            lambda x: x.dec_rate('charge', states=(0, 1)),
            lambda x: partial_charge_dec(x, x.elements[(0, 1)][1], (0, 1))
        )
        do_grad_func_test(
            cr, cr_delta_J, delta, UNIT_FACTORS[Junction],
            lambda x: x.dec_rate('charge', states=(0, 1)),
            lambda x: partial_charge_dec(x, x.elements[(0, 1)][0], (0, 1))
        )


def test_flux_grad() -> None:
    do_test_dec_rate_fluxonium(
        lambda cr, states: cr.dec_rate('flux', states=states),
        lambda cr, elem, states: partial_flux_dec(cr, elem, states)
    )


###############################################################################
# Torch test functions
###############################################################################

def test_torch_flux_grad() -> None:
    do_test_dec_torch_fluxonium(
        lambda cr, states: cr.dec_rate('flux', states=states),
        lambda cr, states: dec_rate_flux_torch(cr, states)
    )


def test_torch_cc_grad() -> None:
    do_test_dec_torch_fluxonium(
        lambda cr, states: cr.dec_rate('cc', states=states),
        lambda cr, states: dec_rate_cc_torch(cr, states)
    )


def test_torch_charge_grad() -> None:
    do_test_dec_torch_trans(
        lambda cr, states: cr.dec_rate('charge', states=states),
        lambda cr, states: dec_rate_charge_torch(cr, states)
    )
