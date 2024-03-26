"""Contains code for defining loss functions used in circuit optimization."""
from collections import defaultdict
from copy import deepcopy
from typing import TypedDict, Tuple

from .functions import (
    calculate_anharmonicity,
    calculate_T1_rate,
    charge_sensitivity,
    flux_sensitivity,
    flux_sensitivity_constantnorm,
    first_resonant_frequency,
    gate_metric,
    reset_charge_modes,
    SQValType,
)
from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode
from SQcircuit import functions as sqf

import numpy as np
import torch

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
        loss = torch.sum(torch.exp(-torch.abs(x1)) + torch.exp(-torch.abs(x2))) + epsilon
    else:
        loss = np.sum(np.exp(-np.abs(x1)) + np.exp(-np.abs(x2))) + epsilon
    return loss, anharmonicity


def anharmonicity_loss_constantnorm(
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
    x1 = alpha * (omega_i0 - 2 * omega_10) / OMEGA_TARGET
    x2 = alpha * (omega_i0 - omega_10) / OMEGA_TARGET
    
    anharmonicity = calculate_anharmonicity(circuit)
    if get_optim_mode():
        loss = torch.sum(torch.exp(-torch.abs(x1)) + torch.exp(-torch.abs(x2))) + epsilon
    else:
        loss = np.sum(np.exp(-np.abs(x1)) + np.exp(-np.abs(x2))) + epsilon
    return loss, anharmonicity


def T1_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    Gamma = calculate_T1_rate(circuit)
    T1 = 1 / Gamma

    loss = Gamma ** 2
    if not get_optim_mode():
        loss = loss.item()

    return loss, T1


def T2_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    Gamma_1 = circuit.dec_rate('charge', (0, 1))
    Gamma_2 = circuit.dec_rate('cc', (0, 1))
    Gamma_3 = circuit.dec_rate('flux', (0, 1))
    Gamma = Gamma_1 + Gamma_2 + Gamma_3
    T2 = 1 / Gamma

    loss = Gamma ** 2
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


def flux_sensitivity_loss_constantnorm(
    circuit: Circuit,
    a=0.1,
    b=1,
    epsilon=1e-14
) -> Tuple[SQValType, SQValType]:
    """Return the flux sensitivity of the circuit around flux operation point
    (typically half flux quantum)."""

    S = flux_sensitivity_constantnorm(circuit, OMEGA_TARGET)

    # Apply hinge loss
    if S < a:
        loss = 0.0 * S + epsilon
    else:
        loss = b * (S - a) + epsilon

    return 100 * loss, S


def charge_sensitivity_loss(circuit: Circuit,
                            code=1,
                            a=0.02,
                            b=1) -> Tuple[SQValType, SQValType]:
    """Assigns a hinge loss to charge sensitivity of circuit."""

    S = charge_sensitivity(circuit, code)

    # Hinge loss transform
    if S < a:
        loss = 0.0 * S
    else:
        loss = b * (S - a)

    reset_charge_modes(circuit)
    return loss, S


def experimental_sensitivity_loss(circuit: Circuit,
                             N_SAMPLES=10,
                             P_ERROR=1,
                             n_eig=10) -> Tuple[SQValType, SQValType]:
    """"Returns an estimate of parameter sensitivity, as determined by variation of
    T1 value in Gaussian probability distribution about element values"""
    def set_elem_value(elem, val):
        elem._value = val

    elements_to_update = defaultdict(list)
    for edge in circuit.elements:
        for i, el in enumerate(circuit.elements[edge]):
            if el in list(circuit._parameters):
                elements_to_update[edge].append((i, list(circuit._parameters).index(el)))

    dist = torch.distributions.MultivariateNormal(torch.stack(circuit.parameters),
                                                  torch.diag(((P_ERROR * torch.stack(circuit.parameters)) / 100) ** 2))
    new_params = dist.rsample((N_SAMPLES, ))  # reparameterization trick
    vals = torch.zeros((N_SAMPLES,))
    for i in range(N_SAMPLES):
        elements_sampled = deepcopy(circuit.elements) # assumes all leaf tensors in .elements
        for edge in elements_to_update:
            for el_idx, param_idx in elements_to_update[edge]:
                set_elem_value(elements_sampled[edge][el_idx], new_params[i, param_idx])

        cr_sampled = Circuit(elements_sampled)
        cr_sampled.set_trunc_nums(circuit.trunc_nums)
        cr_sampled.diag(n_eig)
        _, vals[i] = T1_loss(cr_sampled)
    E = torch.std(vals) / torch.mean(vals)
    loss = E
    return loss, E

def gate_loss(circuit: Circuit):
    """"Returns loss corresponding to the estimated bound on number of gates that circuit can perform"""
    G = gate_metric(circuit)
    loss = -sqf.log(G)
    return loss, G


class LossOut(TypedDict):
    frequency_loss: SQValType
    anharmonicity_loss: SQValType
    T1_loss: SQValType
    flux_sensitivity_loss: SQValType
    charge_sensitivity_loss: SQValType
    gate_loss: SQValType
    total_loss: SQValType


class MetricOut(TypedDict):
    omega: SQValType
    A: SQValType
    T1: SQValType
    flux_sensitivity: SQValType
    charge_sensitivity: SQValType
    gate: SQValType
    T2: SQValType


default_functions = {
    'omega':frequency_loss,
    'aharm': anharmonicity_loss,
    'T1': T1_loss,
    'flux': flux_sensitivity_loss,
    'charge': charge_sensitivity_loss,
    'gate': gate_loss,
    'T2': T2_loss,
}


def calculate_loss_metrics(
    circuit: Circuit,
    function_dict=default_functions,
    use_frequency_loss=True,
    use_anharmonicity_loss=True,
    use_flux_sensitivity_loss=True,
    use_charge_sensitivity_loss=1,
    use_experimental_sensitivity_loss=False,
    use_T1_loss=False,
    use_T2_loss=False,
    use_gate_loss=False,
    log_loss=False,
    loss_normalization=False,
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
    experimental_sensitivity_loss = function_dict['experiment']
    gate_loss = function_dict['gate']

    if loss_normalization:
        if get_optim_mode():
            loss_frequency_init = frequency_loss(circuit)[0].detach()
            loss_anharmonicity_init = anharmonicity_loss(circuit)[0].detach()
            loss_T1_init = T1_loss(circuit)[0].detach()
            loss_T2_init = T2_loss(circuit)[0].detach()
            loss_flux_sensitivity_init = flux_sensitivity_loss(circuit)[0].detach()
            loss_charge_sensitivity_init = charge_sensitivity_loss(circuit)[0].detach()
            loss_experimental_sensitivity_init = experimental_sensitivity_loss(circuit)[0].detach()
            loss_gate_init = gate_loss(circuit)[0].detach()
        else:
            loss_frequency_init, _ = frequency_loss(circuit)
            loss_anharmonicity_init, _ = anharmonicity_loss(circuit)
            loss_T1_init, _ = T1_loss(circuit)
            loss_T2_init, _ = T2_loss(circuit)
            loss_flux_sensitivity_init, _ = flux_sensitivity_loss(circuit)
            loss_charge_sensitivity_init, _ = charge_sensitivity_loss(circuit)
            loss_experimental_sensitivity_init, _ = experimental_sensitivity_loss(circuit)
            loss_gate_init, _ = gate_loss(circuit)

    # Calculate frequency
    with torch.set_grad_enabled(use_frequency_loss and master_use_grad):
        loss_frequency, frequency = frequency_loss(circuit)
        if loss_normalization:
            loss_frequency /= loss_frequency_init
        if use_frequency_loss:
            loss = loss + loss_frequency

    # Calculate anharmonicity
    with torch.set_grad_enabled(use_anharmonicity_loss and master_use_grad):
        loss_anharmonicity, anharmonicity = anharmonicity_loss(circuit)
        if loss_normalization:
            loss_anharmonicity /= loss_anharmonicity_init
        if use_anharmonicity_loss:
            loss = loss + loss_anharmonicity

    # Calculate T1
    with torch.set_grad_enabled(use_T1_loss and master_use_grad):
    # with torch.no_grad():
        loss_T1, T1_time = T1_loss(circuit)
        # loss_T1 = 0
        if loss_normalization:
            loss_T1 /= loss_T1_init
        if use_T1_loss:
            loss = loss + loss_T1

    # Calculate T2
    # Note: Gradient for T2 currently not computable, coming soon ;)
    with torch.no_grad():
        loss_T2, T2_time = T2_loss(circuit)
        '''if loss_normalization:
            loss_T2 /= loss_T2_init
        if use_T2_loss:
            loss = loss + loss_T2'''

    # Calculate flux sensitivity loss
    with torch.set_grad_enabled(use_flux_sensitivity_loss and master_use_grad):
        loss_flux_sensitivity, flux_sensitivity_value = flux_sensitivity_loss(circuit)
        if loss_normalization:
            loss_flux_sensitivity /= loss_flux_sensitivity_init
        if use_flux_sensitivity_loss:
            loss = loss + loss_flux_sensitivity

    # Calculate charge sensitivity loss
    charge_sensitivity_loss_bool = (True if use_charge_sensitivity_loss != 0 else False)
    with torch.set_grad_enabled(charge_sensitivity_loss_bool and master_use_grad):
        loss_charge_sensitivity, charge_sensitivity_value = charge_sensitivity_loss(circuit, code=use_charge_sensitivity_loss)
        if loss_normalization:
            loss_charge_sensitivity /= loss_charge_sensitivity_init
        if charge_sensitivity_loss_bool:
            loss = loss + loss_charge_sensitivity

    # Calculate gate loss
    with torch.set_grad_enabled(use_gate_loss and master_use_grad):
        loss_gate, gate_metric_value = gate_loss(circuit)
        if loss_normalization:
            loss_gate /= loss_gate_init
        if use_gate_loss:
            loss = loss + loss_gate

    with torch.set_grad_enabled(use_experimental_sensitivity_loss and master_use_grad):
        loss_experimental_sensitivity, experimental_sensitivity_value = experimental_sensitivity_loss(circuit)
        if loss_normalization:
            loss_experimental_sensitivity /= loss_experimental_sensitivity_init
        if use_experimental_sensitivity_loss:
            loss = loss + loss_experimental_sensitivity

    if log_loss:
        if get_optim_mode():
            loss = torch.log(1 + loss)
        else:
            loss = np.log(1 + loss)

    with torch.no_grad():
        loss_values: LossOut = {
            'frequency_loss': loss_frequency.detach() if get_optim_mode() else loss_frequency,
            'anharmonicity_loss': loss_anharmonicity.detach() if get_optim_mode() else loss_anharmonicity,
            'T1_loss': loss_T1.detach() if get_optim_mode() else loss_T1,
            'flux_sensitivity_loss': loss_flux_sensitivity.detach() if get_optim_mode() else loss_flux_sensitivity,
            'charge_sensitivity_loss': loss_charge_sensitivity.detach() if get_optim_mode() else loss_charge_sensitivity,
            'experimental_sensitivity_loss': loss_experimental_sensitivity.detach() if get_optim_mode() else loss_experimental_sensitivity,
            'gate_loss': loss_gate.detach() if get_optim_mode() else loss_gate,
            'total_loss': loss.detach() if get_optim_mode() else loss
        }
        metrics: MetricOut = {
            'omega': frequency.detach() if get_optim_mode() else frequency,
            'A': anharmonicity.detach() if get_optim_mode() else anharmonicity,
            'T1': T1_time.detach() if get_optim_mode() else T1_time,
            'flux_sensitivity': flux_sensitivity_value.detach() if get_optim_mode() else flux_sensitivity_value,
            'charge_sensitivity': charge_sensitivity_value.detach() if get_optim_mode() else charge_sensitivity_value,
            'gate': gate_metric_value.detach() if get_optim_mode() else gate_metric_value,
            'T2': T2_time,
        }

    return loss, loss_values, metrics