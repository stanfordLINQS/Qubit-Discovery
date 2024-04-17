"""Contains code for defining loss functions used in circuit optimization."""
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Union

import numpy as np
import torch

from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode
from SQcircuit.units import get_unit_freq

from .functions import (
    calculate_anharmonicity,
    charge_sensitivity,
    flux_sensitivity,
    first_resonant_frequency,
    reset_charge_modes,
    zero,
    decoherence_time,
    fastest_gate_speed,
)

# when the loss are close to zero, but we want to reserve the zero value for
# the metric that do not have loss functions.
EPSILON = 1e-13
SQValType = Union[float, torch.Tensor]

################################################################################
# Only metric loss functions
################################################################################


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
        t_type="t1",
        dec_type=dec_type
    )

    return zero(), t1


def t2_loss(circuit: Circuit, dec_type='total') -> Tuple[SQValType, SQValType]:

    t2 = decoherence_time(
        circuit=circuit,
        t_type="t2",
        dec_type=dec_type
    )

    return zero(), t2


def t2_proxy_loss(
    circuit: Circuit,
    dec_type='total'
) -> Tuple[SQValType, SQValType]:
    """Not implemented yet. """

    return zero(), zero()


def element_sensitivity_loss(
    circuit: Circuit,
    n_samples=10,
    p_error=1,
    n_eig=10
) -> Tuple[SQValType, SQValType]:
    """"Returns an estimate of parameter sensitivity, as determined by variation
    of T1 value in Gaussian probability distribution about element values"""
    def set_elem_value(elem, val):
        elem._value = val

    elements_to_update = defaultdict(list)
    for edge in circuit.elements:
        for i, el in enumerate(circuit.elements[edge]):
            if el in list(circuit._parameters):
                elements_to_update[edge].append(
                    (i, list(circuit._parameters).index(el))
                )

    dist = torch.distributions.MultivariateNormal(
        torch.stack(circuit.parameters),
        torch.diag(((p_error * torch.stack(circuit.parameters)) / 100) ** 2)
    )
    # re-parameterization trick
    new_params = dist.rsample((n_samples, ))
    vals = torch.zeros((n_samples,))
    for i in range(n_samples):
        # assumes all leaf tensors in .elements
        elements_sampled = deepcopy(circuit.elements)
        for edge in elements_to_update:
            for el_idx, param_idx in elements_to_update[edge]:
                set_elem_value(
                    elements_sampled[edge][el_idx],
                    new_params[i, param_idx]
                )

        cr_sampled = Circuit(elements_sampled)
        cr_sampled.set_trunc_nums(circuit.trunc_nums)
        cr_sampled.diag(n_eig)
        _, vals[i] = t1_loss(cr_sampled)
    sensitivity = torch.std(vals) / torch.mean(vals)

    return zero(), sensitivity


def gate_speed_loss(circuit: Circuit):

    gate_speed = fastest_gate_speed(circuit)

    return zero(), gate_speed


################################################################################
# In Optimization loss functions
################################################################################

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

    # Apply hinge loss
    if sens < a:
        loss = 0.0 * sens
    else:
        loss = b * (sens - a)

    return loss + EPSILON, sens


def charge_sensitivity_loss(
    circuit: Circuit,
    code=1,
    a=0.02,
) -> Tuple[SQValType, SQValType]:
    """Assigns a hinge loss to charge sensitivity of circuit."""

    sens = charge_sensitivity(circuit, code)

    # Hinge loss transform
    if sens < a:
        loss = 0.0 * sens
    else:
        loss = sens * (sens - a)

    reset_charge_modes(circuit)
    return loss + EPSILON, sens


def number_of_gates_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    """Return an upper bound on the number of single qubit gates that can be applied to the qubit as well as the loss
    associated with the metric."""

    # we should not forget the units
    gate_speed = fastest_gate_speed(circuit) * get_unit_freq()

    t1 = decoherence_time(
        circuit=circuit,
        t_type="t1",
        dec_type='total'
    )

    with torch.set_grad_enabled(False):
        t2 = decoherence_time(
            circuit=circuit,
            t_type="t2",
            dec_type='total'
        )

    number_of_gates_t1 = t1 * gate_speed
    number_of_gates = 1 / (1/t1 + 1/t2) * gate_speed

    if get_optim_mode():
        # loss = -torch.log(number_of_gates)
        loss = 1 / number_of_gates_t1 * 1e3
    else:
        # loss = -np.log(number_of_gates)
        loss = 1 / number_of_gates_t1 * 1e3

    return loss, number_of_gates


################################################################################
# Incorporating all losses into one loss function
################################################################################


ALL_FUNCTIONS = {
    ############################################################################
    'frequency': frequency_loss,
    'flux_sensitivity': flux_sensitivity_loss,
    'charge_sensitivity': charge_sensitivity_loss,
    'number_of_gates': number_of_gates_loss,
    ############################################################################
    'anharmonicity': anharmonicity_loss,
    'gate_speed': gate_speed_loss,
    't1': t1_loss,
    't1_capacitive': lambda cr: t1_loss(cr, dec_type='capacitive'),
    't1_inductive': lambda cr: t1_loss(cr, dec_type='inductive'),
    't1_quasiparticle': lambda cr: t1_loss(cr, dec_type='quasiparticle'),
    't2': t2_loss,
    't2_proxy': t2_proxy_loss,
    't2_charge': lambda cr: t2_loss(cr, dec_type='charge'),
    't2_proxy_charge': lambda cr: t2_proxy_loss(cr, dec_type='charge'),
    't2_cc': lambda cr: t2_loss(cr, dec_type='cc'),
    't2_proxy_cc': lambda cr: t2_proxy_loss(cr, dec_type='cc'),
    't2_flux': lambda cr: t2_loss(cr, dec_type='flux'),
    't2_proxy_flux': lambda cr: t2_proxy_loss(cr, dec_type='flux'),
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