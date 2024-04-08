"""Contains helper functions used in remainder of code."""

from copy import copy, deepcopy
from typing import Union, Tuple

import numpy as np
import torch

from SQcircuit import Circuit, Loop, Element
from SQcircuit.settings import get_optim_mode
import SQcircuit.functions as sqf
import SQcircuit.units as unt

SQValType = Union[float, torch.Tensor]

# Helper functions
# NOTE: Ensure all functions treat the input `circuit` as const, at least
# in effect.


def zero() -> SQValType:

    if get_optim_mode():
        return torch.tensor(0.0)

    return 0.0


def first_resonant_frequency(circuit: Circuit) -> SQValType:
    """Calculates resonant frequency of first excited eigenstate in circuit."""
    omega = circuit.efreqs[1] - circuit.efreqs[0]
    return omega


def calculate_anharmonicity(circuit: Circuit) -> SQValType:
    """Calculates anharmonicity (ratio between first and second energy
    eigenvalue spacings)."""
    return (circuit.efreqs[2] - circuit.efreqs[1]) / \
           (circuit.efreqs[1] - circuit.efreqs[0])


def charge_sensitivity(
    circuit: Circuit,
    code=1,
    epsilon=1e-14,
) -> SQValType:
    """Returns the charge sensitivity of the circuit for all charge islands.
    Designed to account for entire charge spectrum, to account for charge drift
    (as opposed to e.g. flux sensitivity, which considers perturbations around
    flux operation point)."""

    # This assumes two charge modes. TODO: Generalize to n charge modes.
    # Note: This will scale exponentially for circuits with more charge modes.
    if code == 1 or code == 2:
        charge_offsets = [(0, 0), (0.25, 0.25), (0.5, 0.5)]
    elif code == 3 or code == 4:
        charge_offsets = [
            (0, 0), (0, 0.25), (0, 0.5), (0.25, 0.5),
            (0.5, 0.5), (0.5, 0.25), (0.5, 0), (0.25, 0)
        ]
    omega_values = []

    # Edge case: For circuit with no charge modes, assign zero sensitivity
    if np.all(circuit.omega != 0):
        if get_optim_mode():
            return torch.as_tensor(epsilon) + 0 * circuit.efreqs[0]
        else:
            return epsilon

    # Assume all charge modes set to 0 initially
    c_0 = circuit.efreqs[1] - circuit.efreqs[0]
    omega_values.append(c_0)

    # Copy circuit to create new container for perturbed eigenstates
    perturb_circ = copy(circuit)
    # For each mode, if charge mode exists then set gate charge to obtain
    # minimum frequency
    for charge_offset in charge_offsets:
        for charge_island_idx in perturb_circ.charge_islands.keys():
            charge_mode = charge_island_idx + 1
            perturb_circ.set_charge_offset(
                charge_mode,
                charge_offset[charge_island_idx]
            )

        perturb_circ.diag(len(circuit.efreqs))
        c_delta = perturb_circ.efreqs[1] - perturb_circ.efreqs[0]
        omega_values.append(c_delta)

    # Reset charge modes to 0; this is necessary because peturb_circ and
    # circ are basically the same
    for charge_island_idx in circuit.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        # Reset charge modes in test circuit
        perturb_circ.set_charge_offset(charge_mode, 0.)

    if get_optim_mode():
        if code == 1 or code == 3:
            c_max = torch.max(torch.stack(omega_values))
            c_min = torch.min(torch.stack(omega_values))
            return c_max - c_min

        if code == 2 or code == 4:
            c_avg = torch.mean(torch.stack(omega_values))
            c_var = torch.var(torch.stack(omega_values) - c_avg)
            return c_var
    else:
        if code == 1 or code == 3:
            c_max = np.max(np.stack(omega_values))
            c_min = np.min(np.stack(omega_values))
            return c_max - c_min

        if code == 2 or code == 4:
            c_avg = np.mean(np.stack(omega_values))
            c_var = np.var(np.stack(omega_values) - c_avg)
            return c_var


def flux_sensitivity(
    circuit: Circuit,
    flux_point=0.5,
    delta=0.01
) -> SQValType:
    """Return the flux sensitivity of the circuit around half flux quantum."""
    f_0 = circuit.efreqs[1] - circuit.efreqs[0]

    # Copy circuit to create new container for perturbed eigenstates
    perturb_circ = copy(circuit)
    loop = perturb_circ.loops[0]
    org_flux = loop.value() / (2 * np.pi)  # should be `flux_point`

    # Change the flux and get the eigen-frequencies
    loop.set_flux(flux_point + delta)
    perturb_circ.diag(len(circuit.efreqs))
    f_delta = perturb_circ.efreqs[1] - perturb_circ.efreqs[0]

    # Return loop back to original flux
    loop.set_flux(org_flux)

    if get_optim_mode():
        S = torch.abs((f_delta - f_0) / f_0)
    else:
        S = np.abs((f_delta - f_0) / f_0)

    return S


def reset_charge_modes(circuit: Circuit) -> None:
    """Sets gate charge of all charge degrees of freedom to zero."""
    default_n_g = 0.0
    if sum(circuit.omega == 0) == 0:
        return
    else:
        for charge_island_idx in circuit.charge_islands.keys():
            charge_mode = charge_island_idx + 1
            circuit.set_charge_offset(charge_mode, default_n_g)


def decoherence_time(circuit: Circuit, t_type: str, dec_type: str) -> SQValType:
    """Return the decoherence time for a given circuit and its decoherence type.

    Parameters
    ----------
        circuit:
            A Circuit object of SQcircuit specifying the qubit.
        t_type:
            A string specifying the type of decoherence. It must be either 't1'
            or 't2.
        dec_type:
            A string specifying the channel of the decoherence time. It must be
            either 'capacitive', 'inductive', 'quasiparticle', 'charge', 'cc',
            'flux', or 'total'.
    """

    gamma = 0.0

    if t_type == "t1":
        all_t1_channels = ['capacitive', 'inductive', 'quasiparticle']
        if dec_type == 'total':
            dec_type_list = all_t1_channels
        else:
            assert dec_type in all_t1_channels, (
                f"dec_type with 't1' mode should be {all_t1_channels}, or "
                "'total'"
            )
            dec_type_list = [dec_type]

    elif t_type == "t2":
        all_t2_channels = ['charge', 'cc', 'flux']
        if dec_type == 'total':
            dec_type_list = all_t2_channels
        else:
            assert dec_type in all_t2_channels, (
                f"dec_type with 't2' mode should be {all_t2_channels}, or "
                "'total'"
            )
            dec_type_list = [dec_type]

    else:
        raise ValueError("t_type must be either 't1' or 't2'")

    for dec_type in dec_type_list:
        gamma = gamma + circuit.dec_rate(dec_type, (0, 1))

    return 1 / gamma


def fastest_gate_speed(circuit: Circuit) -> SQValType:
    """Calculates the upper bound for the speed of the single qubit gate of the
    qubit. The upper bound is:
    min{f_i0 - f_10, |f_i0 - 2f_10|};  for i>1

    Parameters
    ----------
        circuit:
            A Circuit object of SQcircuit specifying the qubit.
    """

    omega = circuit.efreqs[2] - circuit.efreqs[1]

    for i in range(2, len(circuit.efreqs)):

        f_i1 = circuit.efreqs[i] - circuit.efreqs[1]
        anharm_i = abs(
            (circuit.efreqs[i] - circuit.efreqs[0]) -
            2*(circuit.efreqs[1] - circuit.efreqs[0])
        )

        if f_i1 < omega:
            omega = f_i1
        elif anharm_i < omega:
            omega = anharm_i

    return omega

def partial_deriv_approx_flux(circuit: Circuit, 
                              loop: Loop, 
                              delta=0.001,
                              symmetric=True) -> SQValType:
    """ Calculates an approximation to the derivative of the first 
    eigenfrequency of `circuit` with respect to the external flux through
    `loop`.

    Parameters
    ----------
        circuit:
            A `Circuit` object of SQcircuit
        loop:
            The loop in `circuit` to calculate the derivative of external
            flux with respect to
        delta:
            The perturbation to use to calculate the finite difference
        symmetric:
            Whether to calculate the symmetric difference quotient or not
    """

    # Create a copy circuit, to ensure old eigenfreqs/values aren't overwritten
    perturb_circ = copy(circuit)
    org_flux = loop.value() / (2 * np.pi)
    omega10 = (circuit.efreqs[1] - circuit.efreqs[0]) * 1e9

    # Eigenfrequencies at perturbed phi_ext + delta
    loop.set_flux(org_flux + (delta / 2 / np.pi))
    perturb_circ.diag(len(circuit.efreqs))
    omega10_plus = (perturb_circ.efreqs[1] - perturb_circ.efreqs[0]) * 1e9

    # If doing symmetric difference, also calculate at phi_ext - delta
    if symmetric:
        loop.set_flux(org_flux - (delta / 2 / np.pi))
        perturb_circ.diag(len(circuit.efreqs))
        omega10_minus = (perturb_circ.efreqs[1] - perturb_circ.efreqs[0]) * 1e9

    # Reset the external circuit flux
    loop.set_flux(org_flux)

    if symmetric:
        return sqf.abs((omega10_plus - omega10_minus) / (2 * delta))
    else:
        return sqf.abs((omega10_plus - omega10) / (delta))
    
def flux_decoherence_approx(cr: Circuit):
    """ Calculates the decoherence due to flux noise for the 0-1 transition,
    using the value calculated by `partial_deriv_approx_flux` to approximate
    the partial derivative of the eigenfrequencies.

    Parameters
    ----------
        circuit:
            A `Circuit` object of SQcircuit
    """
    decay = sqf.array(0.0)
    for loop in cr.loops:
        # Need to convert from Hz to rad to match `._dephasing` use.
        partial_omega = partial_deriv_approx_flux(cr, loop) * 2 * np.pi
        A = loop.A
        decay += cr._dephasing(A, partial_omega)
    return decay

def partial_deriv_approx_charge(
        circuit: Circuit,
        charge_mode: int,
        charge_point=0,
        delta=0.01,
        symmetric=True,
) -> SQValType:
    """ Calculates an approximation to the derivative of the first 
    eigenfrequency of `circuit` with respect to the gate charge of
    the `charge_mode`th charge_mode.

    Parameters
    ----------
        circuit:
            A `Circuit` object of SQcircuit
        charge_mode:
            The number of the charge mode to calculate the derivative of
        charge_point:
            The current gate charge of `charge_mode`
        delta:
            The perturbation of ng to use to approximate the derivative.
        symmetric:
            Whether to calculate the symmetric difference quotient or not
    """
    perturb_circ = copy(circuit)
    omega10 = (circuit.efreqs[1] - circuit.efreqs[0]) * 1e9
    
    perturb_circ.set_charge_offset(charge_mode, charge_point+delta)
    perturb_circ.diag(len(circuit.efreqs))
    omega10_plus = (perturb_circ.efreqs[1] - perturb_circ.efreqs[0]) * 1e9
    
    if symmetric:
        perturb_circ.set_charge_offset(charge_mode, charge_point-delta)
        perturb_circ.diag(len(circuit.efreqs))
        omega10_minus = (perturb_circ.efreqs[1] - perturb_circ.efreqs[0]) * 1e9
    
    perturb_circ.set_charge_offset(charge_mode, charge_point)
    
    if symmetric:
        return sqf.abs((omega10_plus - omega10_minus)/(2 * delta))
    else:
        return sqf.abs((omega10_plus - omega10)/(delta))

def charge_decoherence_approx(cr: Circuit) -> SQValType:
    decay = sqf.array(0.0)
    for charge_island_idx in cr.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        
        partial_omega = partial_deriv_approx_charge(cr, charge_mode)
        A = cr.charge_islands[charge_island_idx].A * 2 * unt.e
        decay = decay + cr._dephasing(A, partial_omega)
    return decay

def set_elem_value(elem, val):
    elem._value = val

def partial_deriv_approx_elem(circuit: Circuit, 
                            edge: Tuple[int, int], 
                            el_idx: int, 
                            delta=0.001, 
                            symmetric=True) -> SQValType:
    """ Calculates an approximation to the derivative of the first 
    eigenfrequency of `circuit` with respect to the element at
    `circuit.elements[edge][el_idx]`.

    Parameters
    ----------
        circuit:
            A `Circuit` object of SQcircuit
        loop:
            The loop in `circuit` to calculate the derivative of external
            flux with respect to
        delta:
            The perturbation to use to calculate the finite difference
        symmetric:
            Whether to calculate the symmetric difference quotient or not
    """
    omega10 = (circuit.efreqs[1] - circuit.efreqs[0]) * 1e9
    
    new_elements = deepcopy(circuit.elements)
    
    set_elem_value(new_elements[edge][el_idx], 
                   circuit.elements[edge][el_idx]._value + delta)
    perturb_circ = Circuit(new_elements)
    perturb_circ.set_trunc_nums(circuit.trunc_nums)
    perturb_circ.diag(len(circuit.efreqs))
    omega10_plus = (perturb_circ.efreqs[1] - perturb_circ.efreqs[0]) * 1e9
    
    if symmetric:
        set_elem_value(new_elements[edge][el_idx],
                       circuit.elements[edge][el_idx]._value - delta)
        perturb_circ.update()
        perturb_circ.diag(len(circuit.efreqs))
        omega10_minus = (perturb_circ.efreqs[1] - perturb_circ.efreqs[0]) * 1e9
    
    if symmetric:
        return sqf.abs((omega10_plus - omega10_minus)/(2 * delta))
    else:
        return sqf.abs((omega10_plus - omega10)/(delta))

def find_elem(cr: Circuit, el: Element) -> Tuple[Tuple[int, int], int]:
    """ Finds the index of `el` in the edge graph of `circuit`.

    Parameters
    ----------
        circuit:
            A `Circuit` object of SQcircuit
        element:
            The element to locate
    """
    for edge in cr.elements.keys():
        for i in range(len(cr.elements[edge])):
            if cr.elements[edge][i] is el:
                return edge, i
    return None

def cc_decoherence_approx(cr: Circuit) -> SQValType:
    """ Calculates the decoherence due to critical current noise for the 
    0-1 transition, using the value calculated by `partial_deriv_approx_elem` 
    to approximate to calculate the partial derivative with respect to the
    Josephson energy.

    Parameters
    ----------
        circuit:
            A `Circuit` object of SQcircuit
    """
    decay = sqf.array(0.0)
    for el, B_idx in cr._memory_ops['cos']:
        edge, el_idx = find_elem(cr, el)
        partial_omega = partial_deriv_approx_elem(cr, edge, el_idx) * 2 * np.pi
        A = el.A * el.get_value()
        decay = decay + cr._dephasing(A, partial_omega)
    return decay