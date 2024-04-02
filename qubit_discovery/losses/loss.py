"""Contains code for defining loss functions used in circuit optimization."""
from collections import defaultdict
from copy import deepcopy
from typing import List, TypedDict, Tuple, Dict

import numpy as np
import torch

from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode

from .functions import (
    calculate_anharmonicity,
    charge_sensitivity,
    flux_sensitivity,
    first_resonant_frequency,
    reset_charge_modes,
    SQValType,
)


# Loss function settings
OMEGA_TARGET = 0.64  # GHz


def frequency_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    omega = first_resonant_frequency(circuit)
    loss = (omega - OMEGA_TARGET) ** 2 / OMEGA_TARGET ** 2
    return loss, omega


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


def T1_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    gamma_1 = circuit.dec_rate('capacitive', (0, 1))
    gamma_2 = circuit.dec_rate('inductive', (0, 1))
    gamma_3 = circuit.dec_rate('quasiparticle', (0, 1))
    gamma = gamma_1 + gamma_2 + gamma_3
    T1 = 1 / gamma

    loss = gamma ** 2
    if not get_optim_mode():
        loss = loss.item()

    return loss, T1


def T2_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    gamma_1 = circuit.dec_rate('charge', (0, 1))
    gamma_2 = circuit.dec_rate('cc', (0, 1))
    gamma_3 = circuit.dec_rate('flux', (0, 1))
    gamma = gamma_1 + gamma_2 + gamma_3
    T2 = 1 / gamma

    loss = gamma ** 2
    if not get_optim_mode():
        loss = loss.item()

    return loss, T2


def flux_sensitivity_loss(
    circuit: Circuit,
    a=0.1,
    b=1,
    epsilon=1e-14
) -> Tuple[SQValType, SQValType]:
    """Return the flux sensitivity of the circuit around flux operation point
    (typically half flux quantum)."""

    S = flux_sensitivity(circuit)

    # Apply hinge loss
    if S < a:
        loss = 0.0 * S + epsilon
    else:
        loss = b * (S - a) + epsilon

    return loss, S


def charge_sensitivity_loss(
    circuit: Circuit,
    code=1,
    a=0.02,
    b=1
) -> Tuple[SQValType, SQValType]:
    """Assigns a hinge loss to charge sensitivity of circuit."""

    S = charge_sensitivity(circuit, code)

    # Hinge loss transform
    if S < a:
        loss = 0.0 * S
    else:
        loss = b * (S - a)

    reset_charge_modes(circuit)
    return loss, S


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
        _, vals[i] = T1_loss(cr_sampled)
    sensitivity = torch.std(vals) / torch.mean(vals)
    loss = sensitivity
    return loss, sensitivity


class LossOut(TypedDict):
    frequency_loss: SQValType
    anharmonicity_loss: SQValType
    T1_loss: SQValType
    flux_sensitivity_loss: SQValType
    charge_sensitivity_loss: SQValType
    total_loss: SQValType


class MetricOut(TypedDict):
    omega: SQValType
    A: SQValType
    T1: SQValType
    flux_sensitivity: SQValType
    charge_sensitivity: SQValType
    T2: SQValType


default_functions = {
    'omega': frequency_loss,
    'aharm': anharmonicity_loss,
    'T1': T1_loss,
    'flux': flux_sensitivity_loss,
    'charge': charge_sensitivity_loss,
    'T2': T2_loss,
}


def detach_if_optim(value: SQValType) -> SQValType:
    """Detach the value if is in torch. Otherwise, return the value itself."""

    if get_optim_mode():
        return value.detach()

    return value


def calculate_loss_metrics(
    circuit: Circuit,
    function_dict=default_functions,
    use_frequency_loss=True,
    use_anharmonicity_loss=True,
    use_flux_sensitivity_loss=True,
    use_charge_sensitivity_loss=1,
    use_T1_loss=False,
    log_loss=False,
    master_use_grad=True
) -> Tuple[SQValType, LossOut, MetricOut]:

    if get_optim_mode():    
        loss = torch.zeros((), requires_grad=master_use_grad)
    else:
        loss = 0

    frequency_loss = function_dict['omega']
    anharmonicity_loss = function_dict['aharm']
    T1_loss = function_dict['T1']
    flux_sensitivity_loss = function_dict['flux']
    charge_sensitivity_loss = function_dict['charge']
    # experimental_sensitivity_loss = function_dict['experiment']

    # Calculate frequency
    with torch.set_grad_enabled(use_frequency_loss and master_use_grad):
        loss_frequency, frequency = frequency_loss(circuit)
        if use_frequency_loss:
            loss = loss + loss_frequency

    # Calculate anharmonicity
    with torch.set_grad_enabled(use_anharmonicity_loss and master_use_grad):
        loss_anharmonicity, anharmonicity = anharmonicity_loss(circuit)
        if use_anharmonicity_loss:
            loss = loss + loss_anharmonicity

    # Calculate T1
    with torch.set_grad_enabled(use_T1_loss and master_use_grad):
        loss_T1, T1_time = T1_loss(circuit)
        # loss_T1 = 0
        if use_T1_loss:
            loss = loss + loss_T1

    # Calculate T2
    # Note: Gradient for T2 currently not computable, coming soon ;)
    with torch.no_grad():
        loss_T2, T2_time = T2_loss(circuit)
        # if use_T2_loss:
        #     loss = loss + loss_T2

    # Calculate flux sensitivity loss
    with torch.set_grad_enabled(use_flux_sensitivity_loss and master_use_grad):
        loss_flux_sensitivity, flux_sensitivity_value = (
            flux_sensitivity_loss(circuit)
        )
        if use_flux_sensitivity_loss:
            loss = loss + loss_flux_sensitivity

    # Calculate charge sensitivity loss
    charge_sensitivity_loss_bool = (
        True if use_charge_sensitivity_loss != 0 else False
    )
    with torch.set_grad_enabled(
            charge_sensitivity_loss_bool and master_use_grad
    ):
        loss_charge_sensitivity, charge_sensitivity_value = (
            charge_sensitivity_loss(circuit, code=use_charge_sensitivity_loss)
        )

        if charge_sensitivity_loss_bool:
            loss = loss + loss_charge_sensitivity

    if log_loss:
        if get_optim_mode():
            loss = torch.log(1 + loss)
        else:
            loss = np.log(1 + loss)

    with torch.no_grad():
        loss_values: LossOut = {
            'frequency_loss': detach_if_optim(loss_frequency),
            'anharmonicity_loss': detach_if_optim(loss_anharmonicity),
            'T1_loss': detach_if_optim(loss_T1),
            'flux_sensitivity_loss': detach_if_optim(loss_flux_sensitivity),
            'charge_sensitivity_loss': detach_if_optim(loss_charge_sensitivity),
            'total_loss': detach_if_optim(loss)
        }
        metrics: MetricOut = {
            'omega': detach_if_optim(frequency),
            'A': detach_if_optim(anharmonicity),
            'T1': detach_if_optim(T1_time),
            'flux_sensitivity': detach_if_optim(flux_sensitivity_value),
            'charge_sensitivity': detach_if_optim(charge_sensitivity_value),
            'T2': detach_if_optim(T2_time),
        }

    return loss, loss_values, metrics


def get_all_functions() -> Dict[str, callable]:
    """Returns a dictionary of all loss function with their proper keys."""

    all_functions = {
        'frequency': frequency_loss,
        'anharmonicity': anharmonicity_loss,
        'flux_sensitivity': flux_sensitivity_loss,
        'charge_sensitivity': charge_sensitivity_loss,
        'element_sensitivity': element_sensitivity_loss,
        'T1': T1_loss,
        'T2': T2_loss,
    }

    return all_functions


use_losses_default = {
    'anharmonicity': 1.0,
    'flux_sensitivity': 1.0,
    'charge_sensitivity': 1.0,
}

use_metrics_default = [
    'T1', 'T2',
]


def calculate_loss_metrics_new(
    circuit: Circuit,
    use_losses: Dict[str, float] = None,
    use_metrics: List[str] = None,
    master_use_grad: bool = True,
) -> Tuple[SQValType, Dict[str, SQValType], Dict[str, SQValType]]:

    if use_metrics is None:
        use_metrics = use_metrics_default

    if use_losses is None:
        use_losses = use_losses_default

    function_dict = get_all_functions()

    if get_optim_mode():
        loss = torch.zeros((), requires_grad=master_use_grad)
    else:
        loss = 0.0

    loss_values: Dict[str, SQValType] = {}
    metrics: Dict[str, SQValType] = {}

    for key in function_dict.keys():

        if key in use_losses:
            with torch.set_grad_enabled(master_use_grad):
                specific_loss, specific_metric = function_dict[key](circuit)
                loss = loss + use_losses[key] * specific_loss

            with torch.no_grad():
                loss_values[key + '_loss'] = detach_if_optim(specific_loss)
                metrics[key] = detach_if_optim(specific_metric)

        elif key not in use_losses and key in use_metrics:
            with torch.no_grad():
                specific_loss, specific_metric = function_dict[key](circuit)

                loss_values[key + '_loss'] = detach_if_optim(specific_loss)
                metrics[key] = detach_if_optim(specific_metric)

    loss_values['total_loss'] = detach_if_optim(loss)

    return loss, loss_values, metrics
