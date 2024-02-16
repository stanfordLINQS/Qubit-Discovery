import os
from typing import (Callable, Dict, Iterable, List,
                    Tuple, TypeVar, Union)

import dill as pickle
import numpy as np
from SQcircuit import (Circuit, CircuitSampler, 
                       Element, Inductor, Junction, Capacitor)
import torch

## General utilities
SQValType = Union[float, torch.Tensor]
LossFunctionType = Callable[[Circuit],
                            Tuple[torch.Tensor,
                                  Dict[str, List[torch.tensor]], 
                                  Dict[str, List[torch.tensor]]]]

T = TypeVar('T')

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


## Utilities for gradient updates

def set_params(circuit: Circuit, params: torch.Tensor) -> None:
    """
    Set the parameters of a circuit to new values.
    """
    for i, element in enumerate(circuit._parameters.keys()):
        element._value = params[i].clone().detach().requires_grad_(True)

    circuit.update()

def get_grad(circuit: Circuit) -> torch.Tensor:

    grad_list = []

    for val in circuit._parameters.values():
        grad_list.append(val.grad)

    if None in grad_list:
        return grad_list

    return torch.stack(grad_list).detach().clone()

def set_grad_zero(circuit: Circuit) -> None:
    for key in circuit._parameters.keys():
        circuit._parameters[key].grad = None

def clamp_gradient(element: Element, epsilon: float) -> None:
  max = torch.squeeze(torch.Tensor([epsilon, ]))
  max = max.double()
  element._value.grad = torch.minimum(max, element._value.grad)
  element._value.grad = torch.maximum(-max, element._value.grad)

def reset_charge_modes(circuit: Circuit) -> None:
    """Sets gate charge of all charge degrees of freedom to zero."""
    default_n_g = 0.0
    if sum(circuit.omega == 0) == 0:
        return
    else:
        for charge_island_idx in circuit.charge_islands.keys():
            charge_mode = charge_island_idx + 1
            circuit.set_charge_offset(charge_mode, default_n_g)


## Sampling utilities

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

## Loss record utilities

RecordType = Dict[str, Union[str,
                             List[torch.Tensor],
                             List[Circuit]]]
@torch.no_grad()
def init_records(circuit_code: str,
                 loss_values: Dict[str, List[torch.tensor]],
                 metric_values: Dict[str, List[torch.tensor]]
) -> Tuple[RecordType, RecordType]:
    # Init loss record
    loss_record: RecordType = {
        loss_type: [] for loss_type in loss_values.keys()
    }
    loss_record['circuit_code'] = circuit_code

    # Init metric record
    metric_record: RecordType = {
        metric_type: [] for metric_type in metric_values.keys()
    }
    metric_record['circuit_code'] = circuit_code
    

    return loss_record, metric_record

@torch.no_grad()
def update_record(circuit: Circuit,  
                  record: RecordType, 
                  values: Dict[str, List[torch.Tensor]]) -> None:
    """Updates record based on next iteration of optimization."""
    for key in values.keys():
        record[key].append(values[key].detach().numpy())

def save_results(loss_record: RecordType, 
                 metric_record: RecordType, 
                 circuit: Circuit,
                 circuit_code: str,
                 name: str, 
                 save_loc: str,
                 prefix="",
                 save_intermediate_circuits=True,
                 ) -> None:
    save_records = {"loss": loss_record, "metrics": metric_record}
    if prefix != "":
        prefix += '_'

    # Extract folder name (remove seed)
    experiment_name = '_'.join(name.split('_')[:-1])

    # Create folder if it doesn't exist
    base_folder = f'{save_loc}/{prefix}{experiment_name}'
    record_folder = os.path.join(base_folder, "records")
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(record_folder, exist_ok=True)

    for record_type, record in save_records.items():
        save_url = f'{record_folder}/{prefix}{record_type}_record_{circuit_code}_{name}.pickle'
        print(f"Saving in {save_url}")
        with open(save_url, 'wb') as f:
            pickle.dump(record, f)
    
    if save_intermediate_circuits:
        write_mode = 'ab+'
    else:
        write_mode = 'wb'
    circuit_save_url = f'{record_folder}/{prefix}circuit_record_{circuit_code}_{name}.pickle'
    with open(circuit_save_url, write_mode) as f:
        pickle.dump(circuit.picklecopy(), f)
    
