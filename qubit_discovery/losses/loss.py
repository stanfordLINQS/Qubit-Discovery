"""Contains code for defining loss functions used in circuit optimization."""
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Union

import numpy as np
import torch

from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode

from .functions import (
    calculate_anharmonicity,
    charge_sensitivity,
    flux_sensitivity,
    element_sensitivity,
    first_resonant_frequency,
    reset_charge_modes,
    zero,
    decoherence_time,
    total_dec_time,
    fastest_gate_speed,
    number_of_gates
)
from .utils import (
    hinge_loss
)

# when the loss are close to zero, but we want to reserve the zero value for
# the metric that do not have loss functions.
EPSILON = 1e-13
SQValType = Union[float, torch.Tensor]

###############################################################################
# Only metric loss functions
###############################################################################


def anharmonicity_loss(
    circuit: Circuit,
    alpha=1,
    epsilon=1e-14
) -> Tuple[SQValType, SQValType]:
    """Designed to penalize energy level occupancy in the vicinity of
    ground state or twice resonant frequency"""
    message = "Anharmonicity is only defined for at least three energy levels."
    assert len(circuit.efreqs) > 2, message
    omega_10 = circuit.efreqs[1] - circuit.efreqs[0]
    omega_i0 = circuit.efreqs[2:] - circuit.efreqs[0]
    x1 = alpha * (omega_i0 - 2 * omega_10) / omega_10
    x2 = alpha * (omega_i0 - omega_10) / omega_10

    anharmonicity = calculate_anharmonicity(circuit)
    if get_optim_mode():
        loss = torch.sum(
            torch.exp(-torch.abs(x1)) + torch.exp(-torch.abs(x2))
        ) + epsilon
    else:
        loss = np.sum(np.exp(-np.abs(x1)) + np.exp(-np.abs(x2))) + epsilon
    return loss, anharmonicity


def t1_loss(
    circuit: Circuit,
    dec_type: str = 'total',
) -> Tuple[SQValType, SQValType]:

    t1 = decoherence_time(
        circuit=circuit,
        t_type='t1',
        dec_type=dec_type
    )

    return zero(), t1


def t2_loss(circuit: Circuit, dec_type='total') -> Tuple[SQValType, SQValType]:

    t2 = decoherence_time(
        circuit=circuit,
        t_type='t2',
        dec_type=dec_type
    )

    return zero(), t2


def t_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    t = total_dec_time(circuit)

    return zero(), t


def element_sensitivity_loss(
    circuit: Circuit,
    n_samples=10,
    error=0.01,
) -> Tuple[SQValType, SQValType]:
    
    sens = element_sensitivity(circuit, n_samples, error)

    return zero(), sens


def gate_speed_loss(circuit: Circuit):

    gate_speed = fastest_gate_speed(circuit)

    return zero(), gate_speed


###############################################################################
# In Optimization loss functions
###############################################################################

def frequency_loss(
        circuit: Circuit,
        freq_threshold: float = 100.0,
) -> Tuple[SQValType, SQValType]:
    """Loss function for frequency that penalizes the qubit that has frequencies
     larger than the freq_threshold.
     """
    freq = first_resonant_frequency(circuit)
    if freq > freq_threshold:
        loss = (freq - freq_threshold)**2
    else:
        loss = zero()
    return loss + EPSILON, freq


def flux_sensitivity_loss(
    circuit: Circuit,
    a=0.1,
    b=1,
) -> Tuple[SQValType, SQValType]:
    """Return the flux sensitivity of the circuit around flux operation point
    (typically half flux quantum)."""

    sens = flux_sensitivity(circuit)

    return hinge_loss(sens, a, b) + EPSILON, sens


def charge_sensitivity_loss(
    circuit: Circuit,
    a=0.02,
    b=1,
) -> Tuple[SQValType, SQValType]:
    """Assigns a hinge loss to charge sensitivity of circuit."""

    sens = charge_sensitivity(circuit)

    reset_charge_modes(circuit)
    return hinge_loss(sens, a, b) + EPSILON, sens


def number_of_gates_loss(
    circuit: Circuit,
) -> Tuple[SQValType, SQValType]:
    """Return the number of single qubit gate of the qubit as well as the loss
    associated with the metric."""

    N = number_of_gates(circuit)

    loss = 1 / N

    return loss, N


###############################################################################
# Incorporating all losses into one loss function
###############################################################################


ALL_FUNCTIONS = {
    ###########################################################################
    'frequency': frequency_loss,
    'flux_sensitivity': flux_sensitivity_loss,
    'charge_sensitivity': charge_sensitivity_loss,
    'number_of_gates': number_of_gates_loss,
    ###########################################################################
    'anharmonicity': anharmonicity_loss,
    'element_sensitivity': element_sensitivity_loss,
    'gate_speed': gate_speed_loss,
    't': t_loss,
    't1': t1_loss,
    't1_capacitive': lambda cr: t1_loss(cr, dec_type='capacitive'),
    't1_inductive': lambda cr: t1_loss(cr, dec_type='inductive'),
    't1_quasiparticle': lambda cr: t1_loss(cr, dec_type='quasiparticle'),
    't2': t2_loss,
    't2_charge': lambda cr: t2_loss(cr, dec_type='charge'),
    't2_cc': lambda cr: t2_loss(cr, dec_type='cc'),
    't2_flux': lambda cr: t2_loss(cr, dec_type='flux'),
}


def detach_if_optim(value: SQValType) -> SQValType:
    """Detach the value if is in torch. Otherwise, return the value itself."""

    if get_optim_mode():
        return value.detach()

    return value


def get_all_metrics() -> List[str]:
    """Returns a list of all metrics."""

    return list(ALL_FUNCTIONS.keys())


def calculate_loss_metrics(
    circuit: Circuit,
    use_losses: Dict[str, float],
    use_metrics: List[str],
    master_use_grad: bool = True,
) -> Tuple[SQValType, Dict[str, SQValType], Dict[str, SQValType]]:

    if get_optim_mode():
        loss = torch.zeros((), requires_grad=master_use_grad)
    else:
        loss = 0.0

    loss_values: Dict[str, SQValType] = {}
    metrics: Dict[str, SQValType] = {}

    for key in get_all_metrics():

        if key in use_losses:
            with torch.set_grad_enabled(master_use_grad):
                specific_loss, specific_metric = ALL_FUNCTIONS[key](circuit)
                loss = loss + use_losses[key] * specific_loss

            with torch.no_grad():
                loss_values[key + '_loss'] = detach_if_optim(specific_loss)
                metrics[key] = detach_if_optim(specific_metric)

        elif key not in use_losses and key in use_metrics:
            with torch.no_grad():
                specific_loss, specific_metric = ALL_FUNCTIONS[key](circuit)

                loss_values[key + '_loss'] = detach_if_optim(specific_loss)
                metrics[key] = detach_if_optim(specific_metric)

    loss_values['total_loss'] = detach_if_optim(loss)

    return loss, loss_values, metrics
