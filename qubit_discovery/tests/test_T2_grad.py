import numpy as np

import SQcircuit as sq
from SQcircuit import units as unt
from qubit_discovery.losses.T2_functions import (
    partial_cc_dec,
    partial_charge_dec,
    partial_flux_dec
)


def build_trans_elements(EC, EJ, flux, flux_dist):
    loop1 = sq.Loop(flux)
    # define the circuit elements
    C = sq.Capacitor(EC, 'GHz', requires_grad=sq.get_optim_mode())
    JJ = sq.Junction(EJ,'GHz', loops=[loop1], requires_grad=sq.get_optim_mode())

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
    L = sq.Inductor(EL,'GHz', loops=[loop1], requires_grad=sq.get_optim_mode())
    JJ = sq.Junction(EJ,'GHz', loops=[loop1], requires_grad=sq.get_optim_mode())

    # define the circuit
    elements = {
        (0, 1): [L, JJ, C]
    }

    cr = sq.Circuit(elements, flux_dist=flux_dist)
    return cr


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
    grad_numeric = (cr_delta.dec_rate('cc', states=(0, 1)) - cr.dec_rate('cc', states=(0, 1))) / delta * unit_factor
    grad_analytic = partial_cc_dec(cr, cr.elements[(0, 1)][1], (0, 1))

    print('Numeric:', grad_numeric, 'Analytic:', grad_analytic)
    assert(np.isclose(grad_numeric, grad_analytic, rtol=1e-2))


def test_flux_grad_EJ() -> None:
    EC, EL, EJ = 3.6, 0.46, 10.2
    delta = 1e-6
    flux_point = 0.5 - 1e-2
    cr = build_fl_elements(EC, EL, EJ, flux_point, 'junctions')
    cr_delta = build_fl_elements(EC, EL, EJ + delta, flux_point, 'junctions')

    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([300])
        _ = circuit.diag(100)

    unit_factor = 1 / 2 / np.pi / 1e9
    grad_numeric = (cr_delta.dec_rate('flux', states=(0, 1)) - cr.dec_rate('flux', states=(0, 1))) / delta * unit_factor
    grad_analytic = partial_flux_dec(cr, cr.elements[(0, 1)][1], (0, 1))

    print('Numeric:', grad_numeric, 'Analytic:', grad_analytic)
    assert(np.isclose(grad_numeric, grad_analytic, rtol=1e-2))


def test_flux_grad_EL() -> None:
    EC, EL, EJ = 3.6, 0.46, 10.2
    delta = 1e-6
    flux_point = 0.5 - 1e-2
    cr = build_fl_elements(EC, EL, EJ, flux_point, 'junctions')
    cr_delta = build_fl_elements(EC, EL+delta, EJ, flux_point, 'junctions')

    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([300])
        _ = circuit.diag(100)

    unit_factor = -(unt.Phi0/2/np.pi)**2/cr.elements[(0, 1)][0]._value**2/(2*np.pi*unt.hbar)/1e9
    grad_numeric = (cr_delta.dec_rate('flux', states=(0, 1)) - cr.dec_rate('flux', states=(0, 1))) / delta * unit_factor
    grad_analytic = partial_flux_dec(cr, cr.elements[(0, 1)][0], (0, 1))

    print('Numeric:', grad_numeric, 'Analytic:', grad_analytic)
    assert(np.isclose(grad_numeric, grad_analytic, rtol=1e-2))


def test_charge_grad() -> None:
    EC, EJ = 3.6, 10
    delta = 1e-4
    cr = build_trans_elements(EC + delta, EJ, 0.1, 'junctions')
    cr_delta = build_trans_elements(EC, EJ, 0.1, 'junctions')

    circuits = [cr, cr_delta]
    for circuit in circuits:
        circuit.set_trunc_nums([200])
        circuit.set_charge_offset(1, 0.5 - 1e-1) # Needs to be offset from 0.5, I think
        _ = circuit.diag(100)

    unit_factor = unt.e**2/2/sq.Capacitor(EC, 'GHz')._value**2/(2*np.pi*unt.hbar) / 1e9
    grad_numeric = (cr_delta.dec_rate('charge', states=(0, 1)) - cr.dec_rate('charge', states=(0, 1))) / delta * unit_factor
    grad_analytic = partial_charge_dec(cr, cr.elements[(0, 1)][1], (0, 1))

    print('Numeric:', grad_numeric, 'Analytic:', grad_analytic)
    assert(np.isclose(grad_numeric, grad_analytic, rtol=1e-2))