"""set of utility functions to apply a reparameterization to circuit element
values. We call the reparameterization the "alpha" parameters.
"""

from typing import Dict, Type

from SQcircuit import Circuit, CircuitComponent, Loop
import torch
from torch import Tensor

def get_alpha_param_from_circuit_param(
    circuit_param: Tensor,
    bounds: Dict[CircuitComponent, Tensor],
    elem_type: Type[CircuitComponent]
) -> Tensor:
    """Compute the alpha parameter corresponding to a circuit parameter.

    Parameters
    ----------
        circuit_param:
            A circuit parameter (element value).
        bounds:
            A dictionary of the element bounds to use to compute the alpha
            parameter.
        elem_type:
            The element type that ``circuit_param`` corresponds to.

    Returns
    ----------
        The alpha parameter corresponding to ``circuit_param``.
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


def get_alpha_params_from_circuit_params(
    circuit: Circuit,
    bounds: Dict[CircuitComponent, Tensor]
) -> Tensor:
    """Compute the alpha parameters for a circuit.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object.
        bounds:
            A dictionary of the element bounds to use to compute the alpha
            parameters.

    Returns
    ----------
        The alpha parameters corresponding to ``circuit.parameters``.
    """
    alpha_params = torch.zeros(torch.stack(circuit.parameters).shape,
                               dtype=torch.float64)

    for param_idx, circuit_param in enumerate(circuit.parameters):
        alpha_params[param_idx] = get_alpha_param_from_circuit_param(
            circuit_param,
            bounds,
            circuit.get_params_type()[param_idx]
        )
    return alpha_params


def get_circuit_param_from_alpha_param(
    alpha_param: Tensor,
    bounds: Dict[CircuitComponent, Tensor],
    elem_type: Type[CircuitComponent],
) -> Tensor:
    """Compute the circuit parameter corresponding to an alpha parameter.

    Parameters
    ----------
        alpha_param:
            The alpha parameter.
        bounds:
            A dictionary of the element bounds used when computing
            ``alpha_param``.
        elem_type:
            The type of element corresponding to ``alpha_param``.

    Returns
    ----------
        The circuit parameter (element value) corresponding to ``alpha_param``.
    """
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
    bounds: Dict[CircuitComponent, Tensor]
) -> Tensor:
    """Compute the parameters for a circuit corresponding to alpha 
    parameters.
    
    Parameters
    ----------
        alpha_params:
            A Tensor of alpha_params, corresponding to the parameters of
            ``circuit``.
        circuit:
            A ``Circuit`` object.
        bounds:
            A dictionary of the element bounds used when computing
            ``alpha_params``.

    Returns
    ----------
        A Tensor of circuit parameters corresponding to ``alpha_params``.
    """
    circuit_params = torch.zeros(alpha_params.shape, dtype=torch.float64)

    for param_idx, alpha_param in enumerate(alpha_params):
        circuit_params[param_idx] = get_circuit_param_from_alpha_param(
            alpha_param,
            bounds,
            circuit.get_params_type()[param_idx]
        )
    return circuit_params
