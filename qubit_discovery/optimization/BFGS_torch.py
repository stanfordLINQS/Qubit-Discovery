from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

import SQcircuit as sq
from SQcircuit import Circuit, Element, Loop

from .truncation import assign_trunc_nums, test_convergence
from .utils import (
    print_loss_records,
    init_records,
    update_record,
    save_results,
    RecordType,
    ConvergenceError
)

from ..losses.loss import LossFunctionType

SQValType = Union[float, Tensor]


def get_alpha_param_from_circuit_param(
    circuit_param: Tensor,
    bounds: dict,
    elem_type=None
) -> Tensor:
    """
    Get the circuit parameter from the alpha parameterization.
    """
    l_bound, u_bound = bounds[elem_type]

    if elem_type == sq.Loop:
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

    if elem_type == sq.Loop:
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


def diag_with_convergence(
        circuit: Circuit,
        num_eigenvalues: int,
        total_trunc_num: int
) -> bool:
    """
    Diagonalize the circuit, and, if the circuit has not converged, try
    re-allocating the truncation numbers. If this fails, then we give up and
    say the circuit has not converged.
    """

    # Reset to even split of truncation numbers
    circuit.truncate_circuit(total_trunc_num)
    circuit.diag(num_eigenvalues)
    # Check if converges with even split
    converged, _ = test_convergence(circuit, eig_vec_idx=1)
    # Otherwise try re-allocating with the heuristic function
    if not converged:
        assign_trunc_nums(circuit, total_trunc_num)
        circuit.diag(num_eigenvalues)

        converged, eps = test_convergence(circuit, eig_vec_idx=1, t=10)
        if not converged:
            raise ConvergenceError(eps)

    return True


def run_BFGS(
    circuit: Circuit,
    loss_metric_function: LossFunctionType,
    max_iter: int,
    total_trunc_num: int,
    bounds: Dict[Union[Element, Loop], Tensor],
    num_eigenvalues: int = 10,
    lr: float = 100,
    tolerance: float = 1e-15,
    save_loc: Optional[str] = None,
    identifier: Optional[str] = None,
    save_intermediate_circuits: bool = False,
    verbose: bool = False
) -> Tuple[Circuit, RecordType, RecordType]:
    """Runs BFGS for a maximum of ``max_iter`` beginning with ``circuit`` using
    ``loss_metric_function``.

    Parameters
    ----------
        circuit:
            A circuit to optimize.
        loss_metric_function:
            Loss function to optimize.
        max_iter:
            Maximum number of iterations.
        total_trunc_num:
            Maximum total truncation number to allocate.
        bounds:
            Dictionary giving bounds for each element type.
        num_eigenvalues:
            Number of eigenvalues to calculate when diagonalizing.
        lr:
            Learning rate
        tolerance:
            Minimum change each step must achieve to not terminate.
        save_loc:
            Folder to save results in, or None.
        identifier:
            String identifying this run (e.g. seed, name, circuit code, …)
            to use when saving.
        save_intermediate_circuits:
            Whether to save the circuit at each iteration.
        verbose:
            Whether to print out progress.
        
    """

    # Set up initial reparameterization
    alpha_params = get_alpha_params_from_circuit_params(circuit, bounds).detach().clone().requires_grad_(True)

    # Set initial truncation numbers
    circuit.truncate_circuit(total_trunc_num)

    lbfgs = torch.optim.LBFGS([alpha_params],
                              history_size=10,
                              max_iter=1,
                              line_search_fn="strong_wolfe")

    last_total_loss = None
    last_loss_values = None
    last_metric_values = None

    def objective_closure():
        """
        Objective function for optimization with reparameterization. It 
            1. Rebuilds circuit from alpha_params
            2. Diagonalizes it and checks convergence
            3. Computes losses and metrics
        """

        # Hacky way to extract history
        nonlocal last_total_loss
        nonlocal last_loss_values
        nonlocal last_metric_values

        # Zero the gradient
        lbfgs.zero_grad()

        # Rebuild the circuit from the alpha parameters
        circuit_params = get_circuit_params_from_alpha_params(
            alpha_params, circuit, bounds
        )
        circuit.parameters = circuit_params
        diag_with_convergence(circuit, num_eigenvalues, total_trunc_num)

        # Compute the loss…
        last_total_loss, last_loss_values, last_metric_values = loss_metric_function(circuit)
        # …and the gradient
        last_total_loss.backward()

        return last_total_loss

    # Get starting values to save
    objective_closure()

    loss_record, metric_record = init_records(
        last_loss_values,
        last_metric_values
    )

    for it in range(max_iter):
        print(it)
        lbfgs.step(objective_closure)
        update_record(
            loss_record,
            last_loss_values
        )
        update_record(
            metric_record,
            last_metric_values
        )
        if save_loc:
            save_results(
                loss_record,
                metric_record,
                circuit,
                identifier,
                save_loc,
                'BFGS',
                save_intermediate_circuits=save_intermediate_circuits
            )

    return circuit, loss_record, metric_record
