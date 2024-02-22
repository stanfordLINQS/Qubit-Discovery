"""Contains helper functions used in remainder of code."""

from copy import copy
from typing import Union

import numpy as np
import torch

from SQcircuit import Circuit, CircuitSampler
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
                       code=1,
                       epsilon=1e-14, loss_type='diag') -> SQValType:
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
            return torch.as_tensor(epsilon)
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
            # Set gate charge to 0.5 in each mode
            # (to extremize relative to n_g=0)
            perturb_circ.set_charge_offset(charge_mode, charge_offset[charge_island_idx])

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
            # c_var = torch.sum(torch.abs(torch.stack(omega_values) - c_avg) / len(omega_values))
            c_var = torch.var(torch.stack(omega_values) - c_avg)
            return c_var
    else:
        if code == 1 or code == 3:
            c_max = np.max(np.stack(omega_values))
            c_min = np.min(np.stack(omega_values))
            return c_max - c_min

        if code == 2 or code == 4:
            c_avg = np.mean(np.stack(omega_values))
            # c_var = np.sum(np.abs(np.stack(omega_values) - c_avg)) / len(omega_values)
            c_var = np.var(np.stack(omega_values) - c_avg)
            return c_var

   # if get_optim_mode():
   #      return torch.abs((c_delta - c_0) / ((c_delta + c_0) / 2))
   #  else:
   #      return np.abs((c_delta - c_0) / ((c_delta + c_0) / 2))


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