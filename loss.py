"""Contains code for defining loss functions used in circuit optimization."""

from functions import (
    calculate_anharmonicity,
    charge_sensitivity,
    flux_sensitivity,
    first_resonant_frequency,
    reset_charge_modes,
)

import torch

# Loss function settings
OMEGA_TARGET = 0.64 # GHz


def frequency_loss(circuit):
    omega = first_resonant_frequency(circuit)
    return (omega - OMEGA_TARGET) ** 2 / OMEGA_TARGET ** 2


def anharmonicity_loss(circuit, alpha=1, epsilon=1e-9):
    """Designed to penalize energy level occupancy in the vicinity of ground state
    or twice resonant frequency"""
    assert len(circuit.efreqs) > 2, "Anharmonicity is only defined for at least three energy levels."
    omega_10 = circuit.efreqs[1] - circuit.efreqs[0]
    omega_i0 = circuit.efreqs[2:] - circuit.efreqs[0]
    x1 = alpha * (omega_i0 - 2 * omega_10) / omega_10
    x2 = alpha * (omega_i0 - omega_10) / omega_10
    return 2 * torch.sum(torch.exp(-torch.abs(x1)) + torch.exp(-torch.abs(x2))) + epsilon


def T1_loss(circuit):
    Gamma_1 = circuit.dec_rate('capacitive', (0, 1))
    Gamma_2 = circuit.dec_rate('inductive', (0, 1))
    Gamma_3 = circuit.dec_rate('quasiparticle', (0, 1))
    Gamma = Gamma_1 + Gamma_2 + Gamma_3

    return Gamma ** 2


def flux_sensitivity_loss(
        circuit,
        a=0.1,
        b=1,
        epsilon=1e-14
):
    """Return the flux sensitivity of the circuit around flux operation point
    (typically half flux quantum)."""

    S = flux_sensitivity(circuit)

    # Apply hinge loss
    if S < a:
        loss = 0.0 * S + epsilon
    else:
        loss = b * (S - a) + epsilon

    return loss, S


def charge_sensitivity_loss(circuit, a=0.1, b=1):
    """Assigns a hinge loss to charge sensitivity of circuit."""

    S = charge_sensitivity(circuit)

    # Hinge loss transform
    if S < a:
        loss = 0.0 * S
    else:
        loss = b * (S - a)

    reset_charge_modes(circuit)
    return loss, S


def calculate_loss(circuit, use_frequency_loss=True, use_anharmonicity_loss=True,
                         use_flux_sensitivity_loss=True, use_charge_sensitivity_loss=True,
                         use_T1_loss=False, log_loss=False,
                         loss_normalization = False):
    loss = torch.zeros((), requires_grad=True)

    if loss_normalization:
        loss_frequency_init = frequency_loss(circuit).detach()
        loss_anharmonicity_init = anharmonicity_loss(circuit).detach()
        loss_T1_init = T1_loss(circuit).detach()
        loss_flux_sensitivity_init = flux_sensitivity_loss(circuit)[0].detach()
        loss_charge_sensitivity_init = charge_sensitivity_loss(circuit)[
            0].detach()

    # Calculate frequency
    loss_frequency = frequency_loss(circuit)
    if loss_normalization:
        loss_frequency /= loss_frequency_init
    if use_frequency_loss:
        loss = loss + loss_frequency
    # Calculate anharmonicity
    loss_anharmonicity = anharmonicity_loss(circuit)
    if loss_normalization:
        loss_anharmonicity /= loss_anharmonicity_init
    if use_anharmonicity_loss:
        loss = loss + loss_anharmonicity
    # Calculate T1
    loss_T1 = T1_loss(circuit)
    if loss_normalization:
        loss_T1 /= loss_T1_init
    if use_T1_loss:
        loss = loss + loss_T1
    # Calculate flux sensitivity loss
    loss_flux_sensitivity, _ = flux_sensitivity_loss(
        circuit)
    if loss_normalization:
        loss_flux_sensitivity /= loss_flux_sensitivity_init
    if use_flux_sensitivity_loss:
        loss = loss + loss_flux_sensitivity
    # Calculate charge sensitivity loss
    loss_charge_sensitivity, _ = charge_sensitivity_loss(circuit)
    if loss_normalization:
        loss_charge_sensitivity /= loss_charge_sensitivity_init
    if use_charge_sensitivity_loss:
        loss = loss + loss_charge_sensitivity

    if log_loss:
        loss = torch.log(1 + loss)

    all_loss = loss_frequency + loss_anharmonicity + loss_flux_sensitivity + loss_charge_sensitivity
    loss_values = (loss_frequency, loss_anharmonicity, loss_T1, loss_flux_sensitivity, loss_charge_sensitivity, all_loss)
    return loss, loss_values

def calculate_metrics(circuit):
    frequency = first_resonant_frequency(circuit)
    anharmonicity = calculate_anharmonicity(circuit)
    T1_time = 1 / T1_loss(circuit)
    flux_sensitivity_value = flux_sensitivity(circuit)
    charge_sensitivity_value = charge_sensitivity(circuit)
    metrics = (frequency, anharmonicity, T1_time, flux_sensitivity_value,
               charge_sensitivity_value)
    return metrics
