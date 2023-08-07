"""Contains code for defining loss functions used in circuit optimization."""
import time

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

'''import psutil
def check_memory():
    print(
        f"Total RAM usage (in MB): {psutil.Process().memory_info().rss / (1024 * 1024)}")'''


def frequency_loss(circuit):
    omega = first_resonant_frequency(circuit)
    return (omega - OMEGA_TARGET) ** 2 / OMEGA_TARGET ** 2, omega


def anharmonicity_loss(circuit, alpha=1, epsilon=1e-14):
    """Designed to penalize energy level occupancy in the vicinity of ground state
    or twice resonant frequency"""
    assert len(circuit.efreqs) > 2, "Anharmonicity is only defined for at least three energy levels."
    omega_10 = circuit.efreqs[1] - circuit.efreqs[0]
    omega_i0 = circuit.efreqs[2:] - circuit.efreqs[0]
    x1 = alpha * (omega_i0 - 2 * omega_10) / omega_10
    x2 = alpha * (omega_i0 - omega_10) / omega_10
    x3 = alpha * omega_i0 / omega_10
    anharmonicity = calculate_anharmonicity(circuit)
    return 2 * torch.sum(torch.exp(-torch.abs(x1)) + torch.exp(-torch.abs(x2)) + torch.exp(-torch.abs(x3))) + epsilon, anharmonicity


def T1_loss(circuit):
    Gamma_1 = circuit.dec_rate('capacitive', (0, 1))
    Gamma_2 = circuit.dec_rate('inductive', (0, 1))
    Gamma_3 = circuit.dec_rate('quasiparticle', (0, 1))
    Gamma = Gamma_1 + Gamma_2 + Gamma_3

    return Gamma ** 2, 1 / (Gamma ** 2)


def flux_sensitivity_loss(
        circuit,
        test_circuit,
        a=0.1,
        b=1,
        epsilon=1e-14
):
    """Return the flux sensitivity of the circuit around flux operation point
    (typically half flux quantum)."""

    S = flux_sensitivity(circuit, test_circuit)

    # Apply hinge loss
    if S < a:
        loss = 0.0 * S + epsilon
    else:
        loss = b * (S - a) + epsilon

    return loss, S


def charge_sensitivity_loss(circuit, test_circuit, a=0.1, b=1):
    """Assigns a hinge loss to charge sensitivity of circuit."""

    S = charge_sensitivity(circuit, test_circuit)

    # Hinge loss transform
    if S < a:
        loss = 0.0 * S
    else:
        loss = b * (S - a)

    reset_charge_modes(circuit)
    return loss, S


def calculate_loss_metrics(circuit, test_circuit, use_frequency_loss=True, use_anharmonicity_loss=True,
                         use_flux_sensitivity_loss=True, use_charge_sensitivity_loss=True,
                         use_T1_loss=False, log_loss=False,
                         loss_normalization = False):
    loss = torch.zeros((), requires_grad=True)

    if loss_normalization:
        loss_frequency_init, _ = frequency_loss(circuit).detach()
        loss_anharmonicity_init, _ = anharmonicity_loss(circuit).detach()
        loss_T1_init, _ = T1_loss(circuit).detach()
        loss_flux_sensitivity_init, _ = flux_sensitivity_loss(circuit)
        loss_charge_sensitivity_init, _ = charge_sensitivity_loss(circuit)

    # Calculate frequency
    with torch.set_grad_enabled(use_frequency_loss):
        loss_frequency, frequency = frequency_loss(circuit)
        if loss_normalization:
            loss_frequency /= loss_frequency_init
        if use_frequency_loss:
            loss = loss + loss_frequency

    # Calculate anharmonicity
    with torch.set_grad_enabled(use_anharmonicity_loss):
        loss_anharmonicity, anharmonicity = anharmonicity_loss(circuit)
        if loss_normalization:
            loss_anharmonicity /= loss_anharmonicity_init
        if use_anharmonicity_loss:
            loss = loss + loss_anharmonicity

    # Calculate T1
    with torch.set_grad_enabled(use_T1_loss):
    # with torch.no_grad():
        loss_T1, T1_time = T1_loss(circuit)
        # loss_T1 = 0
        if loss_normalization:
            loss_T1 /= loss_T1_init
        if use_T1_loss:
            loss = loss + loss_T1

    # Calculate flux sensitivity loss
    with torch.set_grad_enabled(use_flux_sensitivity_loss):
        loss_flux_sensitivity, flux_sensitivity_value = flux_sensitivity_loss(circuit, test_circuit)
        if loss_normalization:
            loss_flux_sensitivity /= loss_flux_sensitivity_init
        if use_flux_sensitivity_loss:
            loss = loss + loss_flux_sensitivity

    # Calculate charge sensitivity loss
    with torch.set_grad_enabled(use_charge_sensitivity_loss):
        loss_charge_sensitivity, charge_sensitivity_value = charge_sensitivity_loss(circuit, test_circuit)
        if loss_normalization:
            loss_charge_sensitivity /= loss_charge_sensitivity_init
        if use_charge_sensitivity_loss:
            loss = loss + loss_charge_sensitivity

    if log_loss:
        loss = torch.log(1 + loss)

    with torch.no_grad():
        all_loss = loss_frequency + loss_anharmonicity + loss_flux_sensitivity + loss_charge_sensitivity
        loss_values = (loss_frequency.detach(),
                       loss_anharmonicity.detach(),
                       loss_T1.detach(),
                       loss_flux_sensitivity.detach(),
                       loss_charge_sensitivity.detach(),
                       all_loss.detach())

        metrics = (frequency.detach(),
                   anharmonicity.detach(),
                   T1_time.detach(),
                   flux_sensitivity_value.detach(),
                   charge_sensitivity_value.detach(),
                   loss.detach())

    return loss, loss_values, metrics

'''@torch.no_grad()
def calculate_metrics(circuit, test_circuit):
    frequency = first_resonant_frequency(circuit)
    anharmonicity = calculate_anharmonicity(circuit)
    T1_time = 1 / T1_loss(circuit)
    flux_sensitivity_value = flux_sensitivity(circuit, test_circuit)
    charge_sensitivity_value = charge_sensitivity(circuit, test_circuit)
    metrics = (frequency, anharmonicity, T1_time, flux_sensitivity_value,
               charge_sensitivity_value)
    return metrics'''

@torch.no_grad()
def init_records(circuit, test_circuit, circuit_code):
    # Init loss record
    loss_record = {(circuit, circuit_code, 'frequency_loss'): [],
            (circuit, circuit_code, 'anharmonicity_loss'): [],
            (circuit, circuit_code, 'T1_loss'): [],
            (circuit, circuit_code, 'flux_sensitivity_loss'): [],
            (circuit, circuit_code, 'charge_sensitivity_loss'): [],
            (circuit, circuit_code, 'total_loss'): []}
    total_loss, loss_values, metrics = calculate_loss_metrics(circuit, test_circuit)
    update_loss_record(circuit, circuit_code, loss_record, loss_values)

    # Init metric record
    metric_record = {(circuit, circuit_code, 'T1'): [],
                     (circuit, circuit_code, 'total_loss'): [],
                     (circuit, circuit_code, 'A'): [],
                     (circuit, circuit_code, 'omega'): [],
                     (circuit, circuit_code, 'flux_sensitivity'): [],
                     (circuit, circuit_code, 'charge_sensitivity'): []}
    update_metric_record(circuit, circuit_code, metric_record, metrics)
    return loss_record, metric_record

'''@torch.no_grad()
def init_metric_record(circuit, test_circuit, circuit_code):
    metric_record = {(circuit, circuit_code, 'T1'): [],
                   (circuit, circuit_code, 'total_loss'): [],
                   (circuit, circuit_code, 'A'): [],
                   (circuit, circuit_code, 'omega'): [],
                   (circuit, circuit_code, 'flux_sensitivity'): [],
                   (circuit, circuit_code, 'charge_sensitivity'): []}
    _, _, metrics = calculate_loss_metrics(circuit, test_circuit)
    update_metric_record(circuit, circuit_code, metric_record, metrics)
    return metric_record'''

@torch.no_grad()
def update_loss_record(circuit, codename, loss_record, loss_values):
    """Updates loss record based on next iteration of optimization."""
    frequency_loss, anharmonicity_loss, T1_loss, flux_sensitivity_loss, \
    charge_sensitivity_loss, total_loss = loss_values
    loss_record[(circuit, codename, 'frequency_loss')].append(
        frequency_loss.detach().numpy())
    loss_record[(circuit, codename, 'anharmonicity_loss')].append(
        anharmonicity_loss.detach().numpy())
    loss_record[(circuit, codename, 'T1_loss')].append(T1_loss.detach().numpy())
    loss_record[(circuit, codename, 'flux_sensitivity_loss')].append(
        flux_sensitivity_loss.detach().numpy())
    loss_record[(circuit, codename, 'charge_sensitivity_loss')].append(
        charge_sensitivity_loss.detach().numpy())
    loss_record[(circuit, codename, 'total_loss')].append(
        total_loss.detach().numpy())

@torch.no_grad()
def update_metric_record(circuit, codename, metric_record, metrics):
    """Updates metric record with information from new iteration of optimization."""
    omega_10, A, T1, flux_sensitivity, charge_sensitivity, total_loss = metrics
    metric_record[(circuit, codename, 'T1')].append(T1.detach().numpy())
    metric_record[(circuit, codename, 'A')].append(A.detach().numpy())
    metric_record[(circuit, codename, 'omega')].append(
        omega_10.detach().numpy())
    metric_record[(circuit, codename, 'flux_sensitivity')].append(
        flux_sensitivity.detach().numpy())
    metric_record[(circuit, codename, 'charge_sensitivity')].append(
        charge_sensitivity.detach().numpy())
    metric_record[(circuit, codename, 'total_loss')].append(
        total_loss.detach().numpy())
