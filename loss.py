"""Contains code for defining loss functions used in circuit optimization."""

from functions import (
    charge_sensitivity,
    flux_sensitivity,
    first_resonant_frequency,
    reset_charge_modes,
)

import torch


use_frequency_loss = True
use_anharmonicity_loss = True
use_T1_loss = False
use_flux_sensitivity_loss = True
use_charge_sensitivity_loss = True


def frequency_loss(circuit, omega_target=1):  # GHz
    omega = first_resonant_frequency(circuit)
    return (omega - omega_target) ** 2 / omega_target ** 2


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
        loss = 0.0 * S
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


def calculate_total_loss(frequency_loss, anharmonicity_loss, T1_loss,
                         flux_sensitivity_loss, charge_sensitivity_loss,
                         log_loss=False):
    loss = torch.zeros((), requires_grad=True)
    if use_frequency_loss:
        loss = loss + frequency_loss
    if use_anharmonicity_loss:
        loss = loss + anharmonicity_loss
    if use_T1_loss:
        loss = loss + T1_loss
    if use_flux_sensitivity_loss:
        loss = loss + flux_sensitivity_loss
    if use_charge_sensitivity_loss:
        loss = loss + charge_sensitivity_loss
    if log_loss:
        loss = torch.log(1 + loss)
    return loss
