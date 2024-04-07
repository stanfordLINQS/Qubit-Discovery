"""Contains helper functions used in remainder of code."""

from copy import copy
from typing import Union

import numpy as np
import torch

from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode

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

