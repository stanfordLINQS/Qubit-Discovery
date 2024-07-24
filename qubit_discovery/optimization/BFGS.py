from typing import Callable, Dict, Optional, Tuple, Union, List

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
    RecordType
)

from ..losses.loss import LossFunctionType

SQValType = Union[float, Tensor]


def get_alpha_param_from_circuit_param(
    circuit_param: Tensor,
    bounds: dict,
    elem_type=None
) -> Tensor:

    u_bound, l_bound = bounds[elem_type]

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
    return alpha_params


def get_circuit_param_from_alpha_param(
    alpha_param: Tensor,
    bounds: dict,
    elem_type=None,
) -> Tensor:

    u_bound, l_bound = bounds[elem_type]

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


def get_gradients(loss: Tensor, circuit: Circuit, bounds) -> Tensor:
    loss.backward()
    partial_loss_partial_elem = circuit.parameters_grad
    # print(f"partial_loss_partial_elem: {partial_loss_partial_elem}")
    circuit.zero_parameters_grad()

    alpha_params = get_alpha_params_from_circuit_params(circuit, bounds)
    alpha_params.backward(torch.tensor(len(alpha_params) * [1.0]))
    partial_alpha_partial_elem = circuit.parameters_grad
    # print(f"partial_alpha_partial_elem: {partial_alpha_partial_elem}")
    circuit.zero_parameters_grad()

    gradient = partial_loss_partial_elem / partial_alpha_partial_elem

    return gradient.to(torch.float64)


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
) -> Tuple[Circuit, Tensor, RecordType]:
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

    params = get_alpha_params_from_circuit_params(circuit, bounds).detach().clone()
    identity = torch.eye(params.size(0), dtype=torch.float64)
    H = identity

    circuit.truncate_circuit(total_trunc_num)
    circuit.diag(num_eigenvalues)
    
    # Get gradient and loss values to start with
    loss, loss_values, metric_values = loss_metric_function(circuit)
    loss_record, metric_record = init_records(
        loss_values,
        metric_values
    )

    def objective_func(cr: Circuit, x: Tensor):
        circ_params = get_circuit_params_from_alpha_params(
            x, cr, bounds
        )
        cr.parameters = circ_params
        cr.diag(num_eigenvalues)
        t_loss, _, _ = loss_metric_function(cr)

        return t_loss

    for iteration in range(max_iter):
        print(f"Iteration {iteration}")

        assign_trunc_nums(circuit, total_trunc_num)
        circuit.diag(num_eigenvalues)
        converged, _ = test_convergence(circuit, eig_vec_idx=1)

        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            break

        total_loss, loss_values, metric_values = loss_metric_function(circuit)
        update_record(circuit, metric_record, metric_values)
        update_record(circuit, loss_record, loss_values)

        circ_params = get_circuit_params_from_alpha_params(
            params, circuit, bounds
        )
        print(
            "Optimization Progress:\n",
            90 * "-" + "\n",
            f"params: {params.detach().numpy()}\n",
            f"circuit params: {circ_params.detach().numpy()}\n",
            f"i:{iteration}",
            f"loss: {loss.detach().numpy()}",
        )
        print_loss_records(loss_record)
        print(90 * "-")

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

        loss = objective_func(circuit, params)
        gradient = get_gradients(loss, circuit, bounds)

        p = -torch.matmul(H, gradient)

        alpha = backtracking_line_search(
            circuit,
            objective_func,
            params,
            gradient,
            p,
            bounds,
            lr=lr
        )
        delta_params = alpha * p

        params_next = (
                params + delta_params
        ).clone().detach().requires_grad_(True)

        loss_next = objective_func(circuit, params_next)
        next_gradient = get_gradients(loss_next, circuit, bounds)

        loss_diff = loss_next - loss
        loss_diff_ratio = torch.abs(loss_diff / (loss + 1e-30))

        if verbose:
            if iteration % 1 == 0:
                print(
                    f"i:{iteration}",
                    f"loss: {loss.detach().numpy()}",
                    f"loss_diff_ratio={loss_diff_ratio.detach().numpy()}",
                    f"alpha={alpha}"
                )

                print_loss_records(loss_record)

        if loss_diff_ratio < tolerance:
            break

        s = delta_params

        y = next_gradient - gradient

        rho = 1 / torch.dot(y, s)

        if rho.item() < 0:
            H = identity
        else:
            A = identity - rho * torch.matmul(s.unsqueeze(1), y.unsqueeze(0))
            B = identity - rho * torch.matmul(y.unsqueeze(1), s.unsqueeze(0))
            H = (
                    torch.matmul(A, torch.matmul(H, B))
                    + rho * torch.matmul(s.unsqueeze(1), s.unsqueeze(0))
            )

        params = params_next
        # circuit.update()

    return circuit, params, loss_record


def backtracking_line_search(
    circuit: Circuit,
    objective_func: Callable[[Circuit, Tensor], Tensor],
    params: torch.tensor,  # params at starting point
    gradient: torch.tensor,  # gradient at starting point
    p: torch.tensor,  # search direction,
    bounds=None,
    lr=0.01,
    c=1e-4,
    rho=0.5
) -> float:
    """At the end of line search, `circuit` will have its internal parameters
    set to ``params + alpha * p``.
    """
    print(50*"=" + "Line search called." + 50*"=")
    alpha = lr

    baseline_loss = objective_func(circuit, params)
    print(f"params:{params}")
    print(f"p: {p}, alpha: {alpha}")
    counter = 0
    while (
            objective_func(circuit, params + alpha * p)
            > baseline_loss + c * alpha * torch.dot(p, gradient)
    ):
        alpha *= rho
        counter += 1
        if rho**counter < 1e-8:
            print("The line search broke")
            break

    print(60 * "=" + "The end." + 60 * "=")
    return alpha
