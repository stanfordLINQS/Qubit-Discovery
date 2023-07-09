"""Contains helper functions used in remainder of code."""

from copy import copy
import numpy as np
import torch

from numpy import ndarray
from SQcircuit import Circuit, CircuitSampler
from SQcircuit.elements import Capacitor, Inductor, Junction, Loop
from typing import Tuple


# Helper functions

def first_resonant_frequency(circuit):
    omega = circuit.efreqs[1] - circuit.efreqs[0]
    return omega


def calculate_anharmonicity(circuit):
    return (circuit.efreqs[2] - circuit.efreqs[1]) / (circuit.efreqs[1] - circuit.efreqs[0])


def charge_sensitivity(circuit, epsilon=1e-14):
  """Returns the charge sensitivity of the circuit for all charge islands.
  Designed to account for entire charge spectrum, to account for charge drift
  (as opposed to e.g. flux sensitivity, which considers perturbations around
  flux operation point)."""
  f_0 = circuit.efreqs[1] - circuit.efreqs[0]
  new_circuit = copy(circuit)

  # Edge case: For circuit with no charge modes, assigne zero sensitivity
  if sum(circuit.omega == 0) == 0:
    return epsilon

  else:
    # For each mode, if charge mode exists then set gate charge to obtain
    # minimum frequency
    for charge_island_idx in new_circuit.charge_islands.keys():
      charge_mode = charge_island_idx + 1
      # set gate charge to 0.5 in each mode
      # (to extremize relative to n_g=0)
      new_circuit.set_charge_offset(charge_mode, 0.5)

    new_circuit.diag(len(circuit.efreqs))
    f_delta = new_circuit.efreqs[1] - new_circuit.efreqs[0]

    S = torch.abs((f_delta - f_0) / ((f_delta + f_0) / 2))

    print(f"functions: {type(S)}")
    return S

def flux_sensitivity(
        circuit,
        flux_point=0.5,
        delta=0.01
):
    """Return the flux sensitivity of the circuit around half flux quantum."""

    f_0 = circuit.efreqs[1] - circuit.efreqs[0]
    new_circuit = copy(circuit)
    new_loop = Loop()
    new_loop.set_flux(flux_point + delta)
    new_circuit.loops[0] = new_loop
    _, _ = new_circuit.diag(len(circuit.efreqs))
    f_delta = new_circuit.efreqs[1] - new_circuit.efreqs[0]

    S = torch.abs((f_delta - f_0) / f_0)

    return S

def get_reshaped_eigvec(
        circuit: Circuit,
        eig_vec_idx: int,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Return the eigenvec, index1_eigenvec and index2_eigenvec part of
  the eigenvectors."""

    assert len(circuit._efreqs) != 0, "circuit should be diagonalizecd first."

    # Reshape eigenvector dimensions to correspond to individual modes
    eigenvector = np.array(circuit._evecs[eig_vec_idx].detach().numpy())
    eigenvector_reshaped = np.reshape(eigenvector, circuit.m)

    # Extract maximum magnitudes of eigenvector entries along each mode axis
    mode_2_magnitudes = np.amax(np.abs(eigenvector_reshaped) ** 2, axis=1)
    offset_idx = np.argmax(mode_2_magnitudes)
    mode_1_magnitudes = np.abs(eigenvector_reshaped[offset_idx, :]) ** 2
    eigvec_mag = np.abs(eigenvector) ** 2

    return eigvec_mag, mode_1_magnitudes, mode_2_magnitudes

def reset_charge_modes(circuit):
    default_n_g = 0.0
    if sum(circuit.omega == 0) == 0:
        return
    else:
        for charge_island_idx in circuit.charge_islands.keys():
            charge_mode = charge_island_idx + 1
            circuit.set_charge_offset(charge_mode, default_n_g)


def create_sampler(N, capacitor_range, inductor_range, junction_range):
  circuit_sampler = CircuitSampler(N)
  circuit_sampler.capacitor_range = capacitor_range
  circuit_sampler.inductor_range = inductor_range
  circuit_sampler.junction_range = junction_range
  return circuit_sampler

def print_new_circuit_sampled_message(total_l= 131):
  message = "NEW CIRCUIT SAMPLED"
  print(total_l*"*")
  print(total_l*"*")
  print("*" + (total_l-2)*" " + "*")
  print("*"+ int((total_l-len(message)-2)/2)*" "+  message
        + int((total_l-len(message)-2)/2)*" " + "*"
  )
  print("*" + (total_l-2)*" " + "*")
  print(+total_l*"*" )
  print(total_l*"*")

def flatten(l):
  return [item for sublist in l for item in sublist]

def get_element_counts(circuit):
  inductor_count = sum([type(xi) is Inductor for xi in
                        flatten(list(circuit.elements.values()))])
  junction_count = sum([type(xi) is Junction for xi in
                        flatten(list(circuit.elements.values()))])
  capacitor_count = sum([type(xi) is Capacitor for xi in
                         flatten(list(circuit.elements.values()))])
  return junction_count, inductor_count, capacitor_count

'''loss_record[(circuit, codename, 'frequency_loss')] = []
    loss_record[(circuit, codename, 'anharmonicity_loss')] = []
    loss_record[(circuit, codename, 'T1_loss')] = []
    loss_record[(circuit, codename, 'flux_sensitivity_loss')] = []
    loss_record[(circuit, codename, 'charge_sensitivity_loss')] = []
    loss_record[(circuit, codename, 'total_loss')] = []'''

def update_loss_record(circuit, codename, loss_record, loss_values):
  frequency_loss, anharmonicity_loss, T1_loss, flux_sensitivity_loss,\
  charge_sensitivity_loss, total_loss = loss_values
  loss_record[(circuit, codename, 'frequency_loss')].append(frequency_loss.detach().numpy())
  loss_record[(circuit, codename, 'anharmonicity_loss')].append(anharmonicity_loss.detach().numpy())
  loss_record[(circuit, codename, 'T1_loss')].append(T1_loss.detach().numpy())
  loss_record[(circuit, codename, 'flux_sensitivity_loss')].append(flux_sensitivity_loss.detach().numpy())
  loss_record[(circuit, codename, 'charge_sensitivity_loss')].append(charge_sensitivity_loss.detach().numpy())
  loss_record[(circuit, codename, 'total_loss')].append(total_loss.detach().numpy())

def update_metric_record(circuit, codename, metric_record, metrics):
  omega_10, A, T1, flux_sensitivity, charge_sensitivity, total_loss = metrics
  metric_record[(circuit, codename, 'T1')].append(T1.detach().numpy())
  metric_record[(circuit, codename, 'A')].append(A.detach().numpy())
  metric_record[(circuit, codename, 'omega')].append(omega_10.detach().numpy())
  metric_record[(circuit, codename, 'flux_sensitivity')].append(flux_sensitivity.detach().numpy())
  print(f"CS type: {type(charge_sensitivity)}")
  metric_record[(circuit, codename, 'charge_sensitivity')].append(charge_sensitivity.detach().numpy())
  metric_record[(circuit, codename, 'total_loss')].append(total_loss.detach().numpy())

# TODO: Generalize codename to account for element ordering
# (ex. for N=4, JJJL and JJLJ should be distinct)
def lookup_codename(num_junctions, num_inductors):
  if num_inductors == 0 and num_junctions == 3:
    return "JJJ"
  if num_inductors == 1 and num_junctions == 2:
    return "JJL"
  if num_inductors == 2 and num_junctions == 1:
    return "JLL"
  if num_inductors == 1 and num_junctions == 1:
    return "Fluxonium"
  if num_inductors == 0 and num_junctions == 2:
    return "Transmon"

def init_loss_record(circuit, codename):
    loss_record = {}
    loss_record[(circuit, codename, 'frequency_loss')] = []
    loss_record[(circuit, codename, 'anharmonicity_loss')] = []
    loss_record[(circuit, codename, 'T1_loss')] = []
    loss_record[(circuit, codename, 'flux_sensitivity_loss')] = []
    loss_record[(circuit, codename, 'charge_sensitivity_loss')] = []
    loss_record[(circuit, codename, 'total_loss')] = []
    return loss_record

def init_metric_record(circuit, codename):
    loss_record = {}
    loss_record[(circuit, codename, 'T1')] = []
    loss_record[(circuit, codename, 'total_loss')] = []
    loss_record[(circuit, codename, 'A')] = []
    loss_record[(circuit, codename, 'omega')] = []
    loss_record[(circuit, codename, 'flux_sensitivity')] = []
    loss_record[(circuit, codename, 'charge_sensitivity')] = []
    return loss_record
