from typing import (Dict, Iterable, List, 
                    Tuple, TypeAlias, TypeVar, Union)

from SQcircuit import (Circuit, CircuitSampler, 
                       Element, Inductor, Junction, Capacitor)
import torch

SQValType = Union[float, torch.Tensor]

## General utilities

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

LossRecordType: TypeAlias = Dict[Tuple[Circuit, str, str], List[np.ndarray]]
MetricRecordType: TypeAlias = Dict[Tuple[Circuit, str, str], List[np.ndarray]]
@torch.no_grad()
def init_records(circuit: Circuit, 
                 test_circuit: Circuit, 
                 circuit_code: str,
                 loss_names: List[str],
                 metric_names: List[str]
) -> Tuple[LossRecordType, MetricRecordType]:
    # Init loss record
    loss_record: LossRecordType = {(circuit, circuit_code, name): []
                                   for name in loss_names}
    metric_record: MetricRecordType = {(circuit, circuit_code, name): []
                                       for name in metric_names}
    total_loss, loss_values, metrics = calculate_loss_metrics(circuit, test_circuit) #!
    update_loss_record(circuit, circuit_code, loss_record, loss_values)

    # Init metric record
    metric_record: MetricRecordType = {(circuit, circuit_code, 'T1'): [],
                     (circuit, circuit_code, 'total_loss'): [],
                     (circuit, circuit_code, 'A'): [],
                     (circuit, circuit_code, 'omega'): [],
                     (circuit, circuit_code, 'flux_sensitivity'): [],
                     (circuit, circuit_code, 'charge_sensitivity'): []}
    update_metric_record(circuit, circuit_code, metric_record, metrics) #!
    return loss_record, metric_record

@torch.no_grad()
def update_loss_record(circuit: Circuit, 
                       codename: str, 
                       loss_record: LossRecordType, 
                       loss_values: Tuple[torch.Tensor, ...],
                       loss_names: List[str]) -> None:
    """Updates loss record based on next iteration of optimization."""
    for idx, name in enumerate(loss_names):
        loss_record[(circuit, codename, name)].append(
            loss_values[idx].detach().numpy()
        )

@torch.no_grad()
def update_metric_record(circuit: Circuit, 
                         codename: str, 
                         metric_record: MetricRecordType, 
                         metrics: Tuple[torch.Tensor, ...],
                         metric_names: List[str]) -> None:
    """Updates metric record with information from new iteration of optimization."""
    for idx, name in enumerate(metric_names):
        metric_record[(circuit, codename, name)].append(
            metrics[idx].detach.numpy()
        )