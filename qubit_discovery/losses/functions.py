"""Contains helper functions used in remainder of code."""

from copy import copy
from typing import Union

import numpy as np
import torch

from SQcircuit import Circuit, CircuitSampler
from SQcircuit.elements import Loop
from SQcircuit.settings import get_optim_mode

SQArrType = Union[np.ndarray, torch.Tensor]
SQValType = Union[float, torch.Tensor]

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


def charge_sensitivity(circuit: Circuit, 
                       epsilon=1e-14) -> SQValType:
    """Returns the charge sensitivity of the circuit for all charge islands.
    Designed to account for entire charge spectrum, to account for charge drift
    (as opposed to e.g. flux sensitivity, which considers perturbations around
    flux operation point)."""

    # Edge case: For circuit with no charge modes, assign zero sensitivity
    if np.all(circuit.omega != 0):
        if get_optim_mode():
            return torch.as_tensor(epsilon)
        else:
            return epsilon

    c_0 = circuit.efreqs[1] - circuit.efreqs[0]

    # Copy circuit to create new container for perturbed eigenstates
    perturb_circ = copy(circuit)
    # For each mode, if charge mode exists then set gate charge to obtain
    # minimum frequency
    for charge_island_idx in perturb_circ.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        # Set gate charge to 0.5 in each mode
        # (to extremize relative to n_g=0)
        perturb_circ.set_charge_offset(charge_mode, 0.5)

    perturb_circ.diag(len(circuit.efreqs))
    c_delta = perturb_circ.efreqs[1] - perturb_circ.efreqs[0]

    # Reset charge modes; this is necessary because peturb_circ and
    # circ are basically the same
    for charge_island_idx in circuit.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        # Reset charge modes in test circuit
        perturb_circ.set_charge_offset(charge_mode, 0.)

    if get_optim_mode():
        return torch.abs((c_delta - c_0) / ((c_delta + c_0) / 2))
    else:
        return np.abs((c_delta - c_0) / ((c_delta + c_0) / 2))


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
    org_flux = loop.value() / (2 * np.pi) # should be `flux_point`

    # Change the flux and get the eigenfrequencies
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

def flux_sensitivity_constantnorm(
        circuit: Circuit,
        OMEGA_TARGET,
        flux_point=0.5,
        delta=0.01
) -> SQValType:
    """Return the flux sensitivity of the circuit around half flux quantum."""
    f_0 = circuit.efreqs[1] - circuit.efreqs[0]

    # Copy circuit to create new container for perturbed eigenstates
    perturb_circ = copy(circuit)
    loop = perturb_circ.loops[0]
    org_flux = loop.value() / (2 * np.pi) # should be `flux_point`

    # Change the flux and get the eigenfrequencies
    loop.set_flux(flux_point + delta)
    perturb_circ.diag(len(circuit.efreqs))
    f_delta = perturb_circ.efreqs[1] - perturb_circ.efreqs[0]

    # Return loop back to original flux
    loop.set_flux(org_flux)

    if get_optim_mode():
        S = torch.abs((f_delta - f_0) / OMEGA_TARGET)
    else:
        S = np.abs((f_delta - f_0) / OMEGA_TARGET)

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