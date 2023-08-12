"""Contains helper functions used in remainder of code."""

from copy import copy
from typing import Iterable, List, TypeAlias, TypeVar, Union

import dill as pickle
import numpy as np
import torch

from SQcircuit import Element, Circuit, CircuitSampler
from SQcircuit.elements import Capacitor, Inductor, Junction, Loop
from SQcircuit.settings import get_optim_mode
from typing import Tuple, Optional

from settings import RESULTS_DIR

SQArrType = Union[np.ndarray, torch.Tensor]
SQValType = Union[float, torch.Tensor]

# Helper functions

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
                       test_circuit: Circuit, 
                       epsilon=1e-14) -> SQValType:
    """Returns the charge sensitivity of the circuit for all charge islands.
    Designed to account for entire charge spectrum, to account for charge drift
    (as opposed to e.g. flux sensitivity, which considers perturbations around
    flux operation point)."""

    # Edge case: For circuit with no charge modes, assign zero sensitivity
    if sum(circuit.omega == 0) == 0:
        if get_optim_mode():
            return torch.as_tensor(epsilon)
        else:
            return epsilon

    c_0 = circuit.efreqs[1] - circuit.efreqs[0]
    # new_circuit = copy(circuit)
    # For each mode, if charge mode exists then set gate charge to obtain
    # minimum frequency
    for charge_island_idx in test_circuit.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        # Set gate charge to 0.5 in each mode
        # (to extremize relative to n_g=0)
        test_circuit.set_charge_offset(charge_mode, 0.5)

    test_circuit.diag(len(circuit.efreqs))
    c_delta = test_circuit.efreqs[1] - test_circuit.efreqs[0]

    for charge_island_idx in test_circuit.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        # Reset charge modes in test circuit
        test_circuit.set_charge_offset(charge_mode, 0.)

    if get_optim_mode():
        return torch.abs((c_delta - c_0) / ((c_delta + c_0) / 2))
    else:
        return np.abs((c_delta - c_0) / ((c_delta + c_0) / 2))


def flux_sensitivity(
        circuit: Circuit,
        test_circuit: Circuit,
        flux_point=0.5,
        delta=0.01
) -> SQValType:
    """Return the flux sensitivity of the circuit around half flux quantum."""
    # NOTE: Instead of passing in `test_circuit`, originally tried to call
    # `new_circuit = copy(circuit)`. However, had an issue with PyTorch
    # retaining the intermediate gradient and leading to RAM accumulation.

    # Issue seems to disappear when using a single circuit copy for all
    # subsequent perturbations (ex. testing different charge/flux values).
    f_0 = circuit.efreqs[1] - circuit.efreqs[0]

    # new_circuit = copy(circuit)
    new_loop = Loop()
    new_loop.set_flux(flux_point + delta)
    test_circuit.loops[0] = new_loop
    test_circuit.diag(len(circuit.efreqs))
    f_delta = test_circuit.efreqs[1] - test_circuit.efreqs[0]
    test_circuit.loops[0].set_flux(flux_point)

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

def save_results(loss_record, metric_record, circuit_code, run_id, prefix="") -> None:
    save_records = {"loss": loss_record, "metrics": metric_record}
    if prefix != "":
        prefix += '_'
    for record_type, record in save_records.items():
        save_url = f'{RESULTS_DIR}/{prefix}{record_type}_record_{circuit_code}_{run_id}.pickle'
        save_file = open(save_url, 'wb')
        pickle.dump(record, save_file)
        save_file.close()