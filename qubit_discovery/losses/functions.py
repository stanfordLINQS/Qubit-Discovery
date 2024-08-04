"""Contains helper functions used in remainder of code."""
from copy import copy
import logging
from typing import Callable, Tuple

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

logger = logging.getLogger(__name__)

# NOTE: All functions treat the input `circuit` as const, at least
# in effect.

def first_resonant_frequency(circuit: Circuit) -> SQValType:
    """Calculates resonant frequency of first excited eigenstate in circuit.
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.

    Returns
    ----------
        The frequency f_{10}, in the frequency units of ``SQcircuit``.
    """
    omega = circuit.efreqs[1] - circuit.efreqs[0]
    return omega


def calculate_anharmonicity(circuit: Circuit) -> SQValType:
    """Calculates anharmonicity (ratio between first and second energy
    eigenvalue spacings).
    
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.

    Returns
    ----------
        The anharmonicity of ``circuit``.
    """
    return (circuit.efreqs[2] - circuit.efreqs[1]) / \
           (circuit.efreqs[1] - circuit.efreqs[0])


def fastest_gate_speed(circuit: Circuit) -> SQValType:
    """Calculates the upper bound for the speed of the single-qubit gate.
    The upper bound is:
    min{f_i0 - f_10, |f_i0 - 2f_10|}; for i>1

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.

    Returns
    ----------
        The uppper bound on a single-qubit gate, in the frequency units of
        ``SQcircuit``.
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
    """Return the decoherence time for a given circuit and decoherence type,
    between the 0th and 1st eigenstates.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to calculate the decoherence time of.
        t_type:
            A string specifying the type of decoherence. It must be either
            ``'t1'`` or ``'t_phi'``.
        dec_type:
            A string specifying the channel of the decoherence time. It must be
            either ``'capacitive'``, ``'inductive'``, ``'quasiparticle'``,
            ``'charge'``, ``'cc'``, ``'flux'``, or ``'total'``.

    Returns
    ----------
        The decoherence time of ``circuit`` of type ``t_type`` via channel
        ``dec_type``.
    """

    gamma = zero()

    if t_type == 't1':
        all_t1_channels = ['capacitive', 'inductive', 'quasiparticle']
        if dec_type == 'total':
            dec_type_list = all_t1_channels
        else:
            if dec_type not in all_t1_channels:
                raise ValueError(
                    "dec_type with 't1' mode should be {all_t1_channels}, or "
                    "'total'"
                )
            dec_type_list = [dec_type]
    elif t_type == 't_phi':
        all_tp_channels = ['charge', 'cc', 'flux']
        if dec_type == 'total':
            dec_type_list = all_tp_channels
        else:
            if dec_type not in all_tp_channels:
                raise ValueError(
                    "dec_type with 't_phi' mode should be in {all_tp_channels}"
                    ", or 'total'"
                )
            dec_type_list = [dec_type]
    else:
        raise ValueError("t_type must be either 't1' or 't_phi'")

    for dec_type in dec_type_list:
        gamma = gamma + circuit.dec_rate(dec_type, (0, 1))

    return 1 / gamma


def total_dec_time(circuit: Circuit) -> SQValType:
    """Return the T2 (total decohence) time.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to calculat the T2 of.

    Returns
    ----------
        The T2 time of ``circuit``, between the 0th and 1st eigenstates,
        via the capacitive, inductive, quasiparticle, charge, critical current,
        and flux channels.
    """
    t1 = decoherence_time(
        circuit=circuit,
        t_type='t1',
        dec_type='total'
    )

    tp = decoherence_time(
        circuit=circuit,
        t_type='t_phi',
        dec_type='total'
    )

    return 2*t1*tp / (2*t1+tp)


def number_of_gates(circuit: Circuit) -> SQValType:
    """Return the number of single-qubit gates, by calculating the gate speed
    (loosely, an anharmonicity) and total decoherence time.
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object.
        
    Returns
    ----------
        The maximum number of sequential single-qubit gates which can be
        executed.
    """

    # don't forget the units!
    gate_speed = fastest_gate_speed(circuit) * get_unit_freq()

    t = total_dec_time(circuit)

    return gate_speed * t


def charge_sensitivity(
    circuit: Circuit,
    n_samples: int = 4,
    epsilon: float = 1e-14,
) -> SQValType:
    """Compute the sensitivity in the circuit's qubit frequency to drift in
    gate charge in all charge islands. To compute this, samples the qubit
    frequency 'diagonally' in charge space: evenly samples ``n_samples`` points
    along the diagonal line n_{g1} = n_{g2} = ..., computes the qubit frequency
    at each point, and returns a normalized range of variation (max frequency
    sampled minus min, over the average frequency).
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object to calculate the sensitivity of.
        n_samples:
            The number of gate charge points to sample.
        epsilon:
            The value to return if there are no charge modes (so the
            sensitivity is formally zero).

    Returns
    ----------
        The normalized range of f_{10} across all gate charge points sampled.
    """

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
    delta=0.01,
    loop_idx=0,
) -> SQValType:
    """Compute the sensitivity in the circuit's qubit frequency to variations
    in external flux through one of its loops. The flux sensitivity is defined
    as the normalized change in qubit frequency for a perturbation of ``delta``
    about the current external flux, i.e.
        ``|f_{10}(phi_ext + delta) - f_{10}(phi_ext)|/f_{10}(phi_ext)``. 

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to calculate the sensitivity of.
        delta:
            The perturbation in internal flux.
        loop_idx:
            The index of the loop in ``circuit.loops`` to perturb.

    Returns
    ----------
        The normalized variation of f_{10} to a ``delta`` perturbation in
        external flux through the ``loop_idx``th loop.
    """
    f_0 = circuit.efreqs[1] - circuit.efreqs[0]

    # Copy circuit to create new container for perturbed eigenstates
    perturb_circ = copy(circuit)
    loop = perturb_circ.loops[loop_idx]
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
    circuit_metric: Callable[[Circuit], SQValType],
    n_samples=25,
    fabrication_error=0.01,
) -> Tuple[SQValType, SQValType]:
    """Compute an estimate of circuit sensitivity, as determined by the
    standard deviation in a circuit metric calculated for circuits with a
    normally-distributed fabrication error.

    In particular, constructs ``n_samples`` circuits with elements which are
    sampled according to a normal distribution about the values in ``circuit``,
    assuming a percentage fabrication error of ``fabrication_error``. Computes
    ``circuit_metric`` for each sampled circuit and returns the normalized
    standard deviation (std over mean) of values calculated.
    
    Only works when ``SQcircuit``'s optimization mode is on (assumes all
    element values are Tensors).

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to use as the center of the distribution.
        circuit_metric:
            A metric to calculate the sensitivity of ``circuit`` to. 
        n_samples:
            The number of randomly sampled circuits to calculate. More samples
            provide more reproducible results.
        fabrication_error:
            The percentage fabrication error to simulate. The elements for the
            new circuit are sampled from a normal distribution with standard
            deviation ``fabrication_error * element_value``.

    Returns
    ----------
        The normalized standard deviation of ``circuit_metric`` over
        ``n_samples`` randomly sampled circuits.
    """
    dist = torch.distributions.MultivariateNormal(
        torch.stack(circuit.parameters),
        torch.diag((fabrication_error * torch.stack(circuit.parameters)) ** 2)
    )
    # re-parameterization trick
    new_params = dist.rsample((n_samples, ))
    vals = torch.zeros((n_samples,))

    logger.info('Calculating element sensitivity...')
    for i in range(n_samples):
        logger.info('%i/%i', i+1, n_samples)
        elements_sampled = construct_perturbed_elements(circuit,
                                                        new_params[i,:])
        cr_sampled = Circuit(elements_sampled)
        cr_sampled.set_trunc_nums(circuit.trunc_nums)
        cr_sampled.diag(len(circuit.efreqs))
        vals[i] = circuit_metric(cr_sampled)

    sensitivity = torch.std(vals) / torch.mean(vals)

    return sensitivity


def reset_charge_modes(circuit: Circuit) -> None:
    """Sets gate charge of all charge degrees of freedom to zero.
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object to reset.
    """
    default_n_g = 0.0

    if sum(circuit.omega == 0) == 0:
        return
    else:
        for charge_island_idx in circuit.charge_islands.keys():
            charge_mode = charge_island_idx + 1
            circuit.set_charge_offset(charge_mode, default_n_g)
