import numpy as np

import SQcircuit as sq
from qubit_discovery.losses.T2_functions import (
    partial_cc_dec,
)

def build_fl_elements(EC, EL, EJ, flux, flux_dist='junctions'):
    loop1 = sq.Loop(flux)
    # define the circuit elements
    C = sq.Capacitor(EC, 'GHz', requires_grad=sq.get_optim_mode())
    L = sq.Inductor(EL,'GHz', loops=[loop1], requires_grad=sq.get_optim_mode())
    JJ = sq.Junction(EJ,'GHz', loops=[loop1], requires_grad=sq.get_optim_mode(),)

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