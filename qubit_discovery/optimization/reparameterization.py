from SQcircuit import Circuit, Loop
import torch
from torch import Tensor

def get_alpha_param_from_circuit_param(
    circuit_param: Tensor,
    bounds: dict,
    elem_type=None
) -> Tensor:
    """
    Get the circuit parameter from the alpha parameterization.
    """
    l_bound, u_bound = bounds[elem_type]

    if elem_type == Loop:
        var = (u_bound - l_bound) / 2
        mean = (u_bound + l_bound) / 2

        return torch.acos((circuit_param - mean) / var)
    else:
        var = (torch.log(u_bound) - torch.log(l_bound))/2
        mean = (torch.log(u_bound) + torch.log(l_bound))/2

        return torch.acos((torch.log(circuit_param) - mean) / var)


def get_alpha_params_from_circuit_params(circuit: Circuit, bounds) -> Tensor:
    alpha_params = torch.zeros(torch.stack(circuit.parameters).shape)

    for param_idx, circuit_param in enumerate(circuit.parameters):
        alpha_params[param_idx] = get_alpha_param_from_circuit_param(
            circuit_param,
            bounds,
            circuit.get_params_type()[param_idx]
        )
    return alpha_params.detach()


def get_circuit_param_from_alpha_param(
    alpha_param: Tensor,
    bounds: dict,
    elem_type=None,
) -> Tensor:
    l_bound, u_bound = bounds[elem_type]

    if elem_type == Loop:
        var = (u_bound - l_bound) / 2
        mean = (u_bound + l_bound) / 2

        return mean + var * torch.cos(alpha_param)
    else:
        var = (torch.log(u_bound) - torch.log(l_bound))/2
        mean = (torch.log(u_bound) + torch.log(l_bound))/2

        return torch.exp(mean + var * torch.cos(alpha_param))


def get_circuit_params_from_alpha_params(
    alpha_params: Tensor,
    circuit: Circuit,
    bounds
) -> Tensor:
    circuit_params = torch.zeros(alpha_params.shape)

    for param_idx, alpha_param in enumerate(alpha_params):
        circuit_params[param_idx] = get_circuit_param_from_alpha_param(
            alpha_param,
            bounds,
            circuit.get_params_type()[param_idx]
        )
    return circuit_params
