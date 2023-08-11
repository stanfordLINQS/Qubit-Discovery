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

T = TypeVar('T')
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


def get_reshaped_eigvec(
        circuit: Circuit,
        eig_vec_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def reset_charge_modes(circuit: Circuit) -> None:
    """Sets gate charge of all charge degrees of freedom to zero."""
    default_n_g = 0.0
    if sum(circuit.omega == 0) == 0:
        return
    else:
        for charge_island_idx in circuit.charge_islands.keys():
            charge_mode = charge_island_idx + 1
            circuit.set_charge_offset(charge_mode, default_n_g)


def create_sampler(N: int, 
                   capacitor_range, 
                   inductor_range, 
                   junction_range) -> CircuitSampler:
    """Initializes circuit sampler object within specified parameter range."""
    circuit_sampler = CircuitSampler(N)
    circuit_sampler.capacitor_range = capacitor_range
    circuit_sampler.inductor_range = inductor_range
    circuit_sampler.junction_range = junction_range
    return circuit_sampler


def print_new_circuit_sampled_message(total_l=131) -> None:
    message = "NEW CIRCUIT SAMPLED"
    print(total_l * "*")
    print(total_l * "*")
    print("*" + (total_l - 2) * " " + "*")
    print("*" + int((total_l - len(message) - 2) / 2) * " " + message
          + int((total_l - len(message) - 2) / 2) * " " + "*")
    print("*" + (total_l - 2) * " " + "*")
    print(+ total_l * "*")
    print(total_l * "*")


def flatten(l: Iterable[Iterable[T]]) -> List[T]:
    """Converts array of arrays into single contiguous one-dimensional array."""
    return [item for sublist in l for item in sublist]


def get_element_counts(circuit: Circuit) -> Tuple[int, int, int]:
    """Gets counts of each type of circuit element."""
    inductor_count = sum([type(xi) is Inductor for xi in
                          flatten(list(circuit.elements.values()))])
    junction_count = sum([type(xi) is Junction for xi in
                          flatten(list(circuit.elements.values()))])
    capacitor_count = sum([type(xi) is Capacitor for xi in
                           flatten(list(circuit.elements.values()))])
    return junction_count, inductor_count, capacitor_count


# TODO: Generalize codename to account for element ordering
# (ex. for N=4, JJJL and JJLJ should be distinct)
def lookup_codename(num_junctions: int, num_inductors: int) -> Optional[str]:
    if num_inductors == 0 and num_junctions == 2:
        return "JJ"
    if num_inductors == 1 and num_junctions == 1:
        return "JL"
    if num_inductors == 0 and num_junctions == 3:
        return "JJJ"
    if num_inductors == 1 and num_junctions == 2:
        return "JJL"
    if num_inductors == 2 and num_junctions == 1:
        return "JLL"
    return None


def code_to_codename(circuit_code: str) -> str:
    if circuit_code == "JJ":
        return "Transmon"
    if circuit_code == "JL":
        return "Fluxonium"
    return circuit_code


def clamp_gradient(element: Element, epsilon: float) -> None:
  max = torch.squeeze(torch.Tensor([epsilon, ]))
  max = max.double()
  element._value.grad = torch.minimum(max, element._value.grad)
  element._value.grad = torch.maximum(-max, element._value.grad)


def save_results(loss_record, metric_record, circuit_code, run_id, prefix="") -> None:
    save_records = {"loss": loss_record, "metrics": metric_record}
    if prefix != "":
        prefix += '_'
    for record_type, record in save_records.items():
        save_url = f'{RESULTS_DIR}/{prefix}{record_type}_record_{circuit_code}_{run_id}.pickle'
        save_file = open(save_url, 'wb')
        pickle.dump(record, save_file)
        save_file.close()

def set_params(circuit: Circuit, params: torch.Tensor) -> None:
    for i, element in enumerate(circuit._parameters.keys()):
        element._value = params[i].clone().detach().requires_grad_(True)

    circuit.update()

def get_grad(circuit: Circuit) -> torch.Tensor:

    grad_list = []

    for val in circuit._parameters.values():
        grad_list.append(val.grad)

    if None in grad_list:
        return grad_list

    return torch.stack(grad_list)

def set_grad_zero(circuit: Circuit) -> None:
    for key in circuit._parameters.keys():
        circuit._parameters[key].grad = None

def get_optimal_key(loss_record, code: Optional[str]=None):
  optimal_loss = 1e100
  optimal_key = None

  for circuit, circuit_code, l in loss_record.keys():
    key = (circuit, circuit_code, 'total_loss')
    if len(loss_record[key]) == 0:
        continue
    loss = loss_record[key][-1]
    if loss < optimal_loss and (code in key or code is None):
        optimal_loss = loss
        optimal_key = key

  return optimal_key
