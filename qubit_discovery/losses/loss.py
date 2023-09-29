"""Contains code for defining loss functions used in circuit optimization."""
from typing import TypedDict, Tuple

from .functions import (
    calculate_anharmonicity,
    charge_sensitivity,
    flux_sensitivity,
    flux_sensitivity_constantnorm,
    first_resonant_frequency,
    reset_charge_modes,
    SQValType,
)
from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode

import numpy as np
import torch

# Loss function settings
OMEGA_TARGET = 0.64 # GHz

def frequency_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    omega = first_resonant_frequency(circuit)
    return (omega - OMEGA_TARGET) ** 2 / OMEGA_TARGET ** 2, omega


def anharmonicity_loss(circuit: Circuit, 
                       alpha=1, 
                       epsilon=1e-14) -> Tuple[SQValType, SQValType]:
    """Designed to penalize energy level occupancy in the vicinity of ground state
    or twice resonant frequency"""
    assert len(circuit.efreqs) > 2, "Anharmonicity is only defined for at least three energy levels."
    omega_10 = circuit.efreqs[1] - circuit.efreqs[0]
    omega_i0 = circuit.efreqs[2:] - circuit.efreqs[0]
    x1 = alpha * (omega_i0 - 2 * omega_10) / omega_10
    x2 = alpha * (omega_i0 - omega_10) / omega_10
    
    anharmonicity = calculate_anharmonicity(circuit)
    if get_optim_mode():
        loss = 2 * torch.sum(torch.exp(-torch.abs(x1)) + torch.exp(-torch.abs(x2))) + epsilon
    else:
        loss = 2 * np.sum(np.exp(-np.abs(x1)) + np.exp(-np.abs(x2))) + epsilon
    return loss, anharmonicity

def anharmonicity_loss_constantnorm(circuit: Circuit, 
                       alpha=1, 
                       epsilon=1e-14) -> Tuple[SQValType, SQValType]:
    """Designed to penalize energy level occupancy in the vicinity of ground state
    or twice resonant frequency"""
    assert len(circuit.efreqs) > 2, "Anharmonicity is only defined for at least three energy levels."
    omega_10 = circuit.efreqs[1] - circuit.efreqs[0]
    omega_i0 = circuit.efreqs[2:] - circuit.efreqs[0]
    x1 = alpha * (omega_i0 - 2 * omega_10) / OMEGA_TARGET
    x2 = alpha * (omega_i0 - omega_10) / OMEGA_TARGET
    
    anharmonicity = calculate_anharmonicity(circuit)
    if get_optim_mode():
        loss = 2 * torch.sum(torch.exp(-torch.abs(x1)) + torch.exp(-torch.abs(x2))) + epsilon
    else:
        loss = 2 * np.sum(np.exp(-np.abs(x1)) + np.exp(-np.abs(x2))) + epsilon
    return loss, anharmonicity


def T1_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    Gamma_1 = circuit.dec_rate('capacitive', (0, 1))
    Gamma_2 = circuit.dec_rate('inductive', (0, 1))
    Gamma_3 = circuit.dec_rate('quasiparticle', (0, 1))
    Gamma = Gamma_1 + Gamma_2 + Gamma_3

    loss = Gamma ** 2
    if not get_optim_mode():
        loss = loss.item()

    return loss, 1 / loss


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

    return loss, S


def charge_sensitivity_loss(circuit: Circuit, 
                            a=0.1, 
                            b=1) -> Tuple[SQValType, SQValType]:
    """Assigns a hinge loss to charge sensitivity of circuit."""

    S = charge_sensitivity(circuit)

    # Hinge loss transform
    if S < a:
        loss = 0.0 * S
    else:
        loss = b * (S - a)

    reset_charge_modes(circuit)
    return loss, S

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
    all_loss: SQValType
default_functions = {
    'omega':frequency_loss,
    'aharm': anharmonicity_loss,
    'T1': T1_loss,
    'flux': flux_sensitivity_loss,
    'charge': charge_sensitivity_loss
}
def calculate_loss_metrics(circuit: Circuit, 
                           function_dict=default_functions,
                           use_frequency_loss=True, 
                           use_anharmonicity_loss=True,
                           use_flux_sensitivity_loss=True, 
                           use_charge_sensitivity_loss=True,
                           use_T1_loss=False, log_loss=False,
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

    if loss_normalization:
        if get_optim_mode():
            loss_frequency_init = frequency_loss(circuit)[0].detach()
            loss_anharmonicity_init = anharmonicity_loss(circuit)[0].detach()
            loss_T1_init = T1_loss(circuit)[0].detach()
            loss_flux_sensitivity_init = flux_sensitivity_loss(circuit)[0].detach()
            loss_charge_sensitivity_init = charge_sensitivity_loss(circuit)[0].detach()
        else:
            loss_frequency_init, _ = frequency_loss(circuit)
            loss_anharmonicity_init, _ = anharmonicity_loss(circuit)
            loss_T1_init, _ = T1_loss(circuit)
            loss_flux_sensitivity_init, _ = flux_sensitivity_loss(circuit)
            loss_charge_sensitivity_init, _ = charge_sensitivity_loss(circuit)

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

    # Calculate flux sensitivity loss
    with torch.set_grad_enabled(use_flux_sensitivity_loss and master_use_grad):
        loss_flux_sensitivity, flux_sensitivity_value = flux_sensitivity_loss(circuit)
        if loss_normalization:
            loss_flux_sensitivity /= loss_flux_sensitivity_init
        if use_flux_sensitivity_loss:
            loss = loss + loss_flux_sensitivity

    # Calculate charge sensitivity loss
    with torch.set_grad_enabled(use_charge_sensitivity_loss and master_use_grad):
        loss_charge_sensitivity, charge_sensitivity_value = charge_sensitivity_loss(circuit)
        if loss_normalization:
            loss_charge_sensitivity /= loss_charge_sensitivity_init
        if use_charge_sensitivity_loss:
            loss = loss + loss_charge_sensitivity

    if log_loss:
        if get_optim_mode():
            loss = torch.log(1 + loss)
        else:
            loss = np.log(1 + loss)

    with torch.no_grad():
        all_loss = loss_frequency + loss_anharmonicity + loss_flux_sensitivity + loss_charge_sensitivity
        loss_values: LossOut = {
            'frequency_loss': loss_frequency.detach() if get_optim_mode() else loss_frequency,
            'anharmonicity_loss': loss_anharmonicity.detach() if get_optim_mode() else loss_anharmonicity,
            'T1_loss': loss_T1.detach() if get_optim_mode() else loss_T1,
            'flux_sensitivity_loss': loss_flux_sensitivity.detach() if get_optim_mode() else loss_flux_sensitivity,
            'charge_sensitivity_loss': loss_charge_sensitivity.detach() if get_optim_mode() else loss_charge_sensitivity,
            'total_loss': loss.detach() if get_optim_mode() else loss
        }
        metrics: MetricOut = {
            'omega': frequency.detach() if get_optim_mode() else frequency,
            'A': anharmonicity.detach() if get_optim_mode() else anharmonicity,
            'T1': T1_time.detach() if get_optim_mode() else T1_time,
            'flux_sensitivity': flux_sensitivity_value.detach() if get_optim_mode() else flux_sensitivity_value,
            'charge_sensitivity': charge_sensitivity_value.detach() if get_optim_mode() else charge_sensitivity_value,
            'all_loss': all_loss.detach() if get_optim_mode() else all_loss
        }

    return loss, loss_values, metrics