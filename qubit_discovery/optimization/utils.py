from collections import defaultdict
import os
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    TypeVar,
    Union
)

import dill as pickle
from SQcircuit import (
    Circuit,
    Loop,
    Element,
    Inductor,
    Junction,
    Capacitor
)
import torch


# General utilities
SQValType = Union[float, torch.Tensor]
LossFunctionType = Callable[
    [Circuit],
    Tuple[torch.Tensor, Dict[str, SQValType], Dict[str, SQValType]]
]

T = TypeVar('T')


# Utilities for gradient updates
def clamp_gradient(val: torch.Tensor, epsilon: float) -> None:
    max = torch.squeeze(torch.Tensor([epsilon, ]))
    max = max.double()
    val.grad = torch.minimum(max, val.grad)
    val.grad = torch.maximum(-max, val.grad)

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


# Loss record utilities
RecordType = Dict[
    str, Union[str, List[torch.Tensor], List[Circuit]]
]


@torch.no_grad()
def init_records(
    circuit_code: str,
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
def update_record(
    circuit: Circuit,
    record: RecordType,
    values: Dict[str, List[torch.Tensor]]
) -> None:
    """Updates record based on next iteration of optimization."""
    for key in values.keys():
        record[key].append(values[key].detach().numpy())


def save_results(
    loss_record: RecordType,
    metric_record: RecordType,
    circuit: Circuit,
    circuit_code: str,
    name: str,
    save_loc: str,
    optim_type: str,
    save_intermediate_circuits=True,
) -> None:
    save_records = {"loss": loss_record, "metrics": metric_record}

    for record_type, record in save_records.items():
        save_url = os.path.join(
            save_loc,
            f'{optim_type}_{record_type}_record_{circuit_code}_{name}.pickle'
        )
        print(f"Saving in {save_url}")
        with open(save_url, 'wb') as f:
            pickle.dump(record, f)
    
    if save_intermediate_circuits:
        write_mode = 'ab+'
    else:
        write_mode = 'wb'

    circuit_save_url = os.path.join(
        save_loc,
        f'{optim_type}_circuit_record_{circuit_code}_{name}.pickle'
    )
    with open(circuit_save_url, write_mode) as f:
        pickle.dump(circuit.picklecopy(), f)


def element_code_to_class(code):
    if code == 'J':
        return Junction
    if code == 'L':
        return Inductor
    if code == 'C':
        return Capacitor
    return None


def build_circuit(element_dictionary):
    # Element dictionary should be of the form {(0,1): ['J', 3.0, 'GHz], ...}
    default_flux = 0
    loop = Loop()
    loop.set_flux(default_flux)
    elements = defaultdict(list)
    for edge, edge_element_details in element_dictionary.items():
        for (circuit_type, value, unit) in edge_element_details:
            if circuit_type in ['J', 'L']:
                element = element_code_to_class(circuit_type)(
                    value,
                    unit,
                    loops=[loop, ],
                    min_value=0,
                    max_value=1e20,
                    requires_grad=True
                )
            else:  # 'C'
                element = element_code_to_class(circuit_type)(
                    value,
                    unit,
                    min_value=0,
                    max_value=1e20,
                    requires_grad=True
                )
            elements[edge].append(element)
    circuit = Circuit(elements)
    return circuit
