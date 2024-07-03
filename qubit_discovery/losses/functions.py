"""Contains helper functions used in remainder of code."""
from copy import copy
from typing import Tuple

import numpy as np
import torch
from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode
from SQcircuit.units import get_unit_freq

from .utils import (
    construct_perturbed_elements,
    zero,
    SQValType
)

# Helper functions
# NOTE: Ensure all functions treat the input `circuit` as const, at least
# in effect.

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
    n_samples: int = 4,
    epsilon: float = 1e-14,
) -> SQValType:
    """Returns the charge sensitivity of the circuit for all charge islands.
    Designed to account for entire charge spectrum, to account for charge drift
    (as opposed to e.g. flux sensitivity, which considers perturbations around
    flux operation point)."""

    charge_offsets = np.linspace(0.0, 0.5, n_samples)
    efreq_values = []

    # Edge case: For circuit with no charge modes, assign zero sensitivity
    if np.all(circuit.omega != 0):
        if get_optim_mode():
            return torch.as_tensor(epsilon) + 0 * circuit.efreqs[0]
        else:
            return epsilon

    # Assume all charge modes set to 0 initially
    freq_q = circuit.efreqs[1] - circuit.efreqs[0]
    efreq_values.append(freq_q)

    # Copy circuit to create new container for perturbed eigenstates
    perturb_circ = copy(circuit)
    # For each mode, if charge mode exists, then set gate charge to obtain
    # minimum frequency

    for charge_offset in charge_offsets:
        # we have already added this to the efreqs_values
        if charge_offset == 0.0:
            continue

        for charge_island_idx in perturb_circ.charge_islands.keys():
            perturb_circ.set_charge_offset(
                charge_island_idx+1,
                charge_offset,
            )

        perturb_circ.diag(len(circuit.efreqs))
        freq_q = perturb_circ.efreqs[1] - perturb_circ.efreqs[0]
        efreq_values.append(freq_q)

    # Reset charge modes to 0; this is necessary because perturb_circ and
    # circ are basically the same
    for charge_island_idx in circuit.charge_islands.keys():
        perturb_circ.set_charge_offset(charge_island_idx + 1, 0.)

    if get_optim_mode():
        c_max = torch.max(torch.stack(efreq_values))
        c_min = torch.min(torch.stack(efreq_values))
        return 2 * (c_max - c_min) / (c_max + c_min)

    else:
        c_max = np.max(np.stack(efreq_values))
        c_min = np.min(np.stack(efreq_values))
        return 2 * (c_max - c_min) / (c_max + c_min)


def flux_sensitivity(
    circuit: Circuit,
    delta=0.01
) -> SQValType:
    """Return the flux sensitivity of the circuit around half flux quantum."""
    f_0 = circuit.efreqs[1] - circuit.efreqs[0]

    # Copy circuit to create new container for perturbed eigenstates
    perturb_circ = copy(circuit)
    loop = perturb_circ.loops[0]
    org_flux = loop.internal_value

    # Change the flux and get the eigen-frequencies
    loop.internal_value = org_flux + delta * 2 * np.pi
    perturb_circ.diag(len(circuit.efreqs))
    f_delta = perturb_circ.efreqs[1] - perturb_circ.efreqs[0]

    # Return loop back to original flux
    loop.internal_value = org_flux

    if get_optim_mode():
        S = torch.abs((f_delta - f_0) / f_0)
    else:
        S = np.abs((f_delta - f_0) / f_0)

    return S


def element_sensitivity(
    circuit: Circuit,
    n_samples=25,
    error=0.01
) -> Tuple[SQValType, SQValType]:
    """"Returns an estimate of parameter sensitivity, as determined by variation
    of gate # in Gaussian probability distribution about internal element values.
    
    Only works with optim_mode = True (assumes all element values are Tensors)
    """
    dist = torch.distributions.MultivariateNormal(
        torch.stack(circuit.parameters),
        torch.diag((error * torch.stack(circuit.parameters)) ** 2)
    )
    # re-parameterization trick
    new_params = dist.rsample((n_samples, ))
    vals = torch.zeros((n_samples,))

    for i in range(n_samples):
        elements_sampled = construct_perturbed_elements(circuit,
                                                        new_params[i,:])
        cr_sampled = Circuit(elements_sampled)
        cr_sampled.set_trunc_nums(circuit.trunc_nums)
        cr_sampled.diag(len(circuit.efreqs))
        vals[i] = number_of_gates(cr_sampled)

    sensitivity = torch.std(vals) / torch.mean(vals)

    return sensitivity


def reset_charge_modes(circuit: Circuit) -> None:
    """Sets gate charge of all charge degrees of freedom to zero."""
    default_n_g = 0.0
    if sum(circuit.omega == 0) == 0:
        return
    else:
        for charge_island_idx in circuit.charge_islands.keys():
            charge_mode = charge_island_idx + 1
            circuit.set_charge_offset(charge_mode, default_n_g)


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


def decoherence_time(circuit: Circuit, t_type: str, dec_type: str) -> SQValType:
    """Return the decoherence time for a given circuit and its decoherence type.

    Parameters
    ----------
        circuit:
            A Circuit object of SQcircuit specifying the qubit.
        t_type:
            A string specifying the type of decoherence. It must be either
            't1' or 't2'.
        dec_type:
            A string specifying the channel of the decoherence time. It must be
            either 'capacitive', 'inductive', 'quasiparticle', 'charge', 'cc',
            'flux', or 'total'.
    """

    gamma = zero()

    if t_type == 't1':
        all_t1_channels = ['capacitive', 'inductive', 'quasiparticle']
        if dec_type == 'total':
            dec_type_list = all_t1_channels
        else:
            assert dec_type in all_t1_channels, (
                f"dec_type with 't1' mode should be {all_t1_channels}, or "
                "'total'"
            )
            dec_type_list = [dec_type]
    elif t_type == 't2':
        all_t2_channels = ['charge', 'cc', 'flux']
        if dec_type == 'total':
            dec_type_list = all_t2_channels
        else:
            assert dec_type in all_t2_channels, (
                f"dec_type with 't2' mode should be in {all_t2_channels}, or "
                "'total'"
            )
            dec_type_list = [dec_type]
    else:
        raise ValueError("t_type must be either 't1' or 't2'")

    for dec_type in dec_type_list:
        gamma = gamma + circuit.dec_rate(dec_type, (0, 1))

    return 1 / gamma

def total_dec_time(circuit: Circuit) -> SQValType:
    """
    Return the T2 (total decohence) time
    """
    t1 = decoherence_time(
        circuit=circuit,
        t_type='t1',
        dec_type='total'
    )

    t2 = decoherence_time(
        circuit=circuit,
        t_type='t2',
        dec_type='total'
    )

    return 2*t1*t2 / (2*t1+t2)

def number_of_gates(circuit: Circuit) -> SQValType:
    """Return the number of single qubit gate"""

    # don't forget the units!
    gate_speed = fastest_gate_speed(circuit) * get_unit_freq()

    t = total_dec_time(circuit)

    return gate_speed * t
