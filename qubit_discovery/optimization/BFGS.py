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
    Get the circuit parameter from the alpha in [0, π] parameterization.
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
    circuit.diag(num_eigenvalues)

    # Check if converges with old truncation numbers
    converged, _ = test_convergence(circuit, eig_vec_idx=1)
    # Otherwise try re-allocating
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

    # Define our objective function
    def objective_func(cr: Circuit, x: Tensor):
        """
        Objective function for optimization with reparameterization. It 
            1. Rebuilds circuit from alpha_params
            2. Diagonalizes it and checks convergence
            3. Computes losses and metrics
        """
        circuit_params = get_circuit_params_from_alpha_params(
            x, cr, bounds
        )
        cr.parameters = circuit_params
        diag_with_convergence(circuit, num_eigenvalues, total_trunc_num)

        return loss_metric_function(cr)

    # Set up initial reparameterization
    alpha_params = get_alpha_params_from_circuit_params(circuit, bounds).detach().clone().requires_grad_(True)
    
    # Set initial truncation numbers
    circuit.truncate_circuit(total_trunc_num)

    # Get gradient and loss values to start with
    loss, loss_values, metric_values = objective_func(circuit, alpha_params)
    loss_record, metric_record = init_records(
        loss_values,
        metric_values
    )
    loss.backward()
    gradient = alpha_params.grad.type(torch.float64) # TODO: why double?

    # Initialize BFGS algorithm
    def identity():
        return torch.eye(alpha_params.size(0), dtype=torch.float64)
    H = identity()

    for iteration in range(max_iter):
        print(f"Iteration {iteration}")

        # 0. Save values
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

        # 1. Compute search direction
        p = -torch.matmul(H, gradient)

        # 2. Compute step size
        alpha = backtracking_line_search(
            circuit,
            objective_func,
            alpha_params,
            loss,
            gradient,
            p,
            lr=lr
        )
        delta_params = alpha * p

        # 3. Compute next parameters and zero their gradient
        alpha_params_next = (
                alpha_params + delta_params
        )
        alpha_params_next = alpha_params_next.clone().detach().requires_grad_(True)

        # 4. Step the circuit, and compute the loss + gradient at the new
        # parameter values.
        loss_next, loss_values, metric_values = objective_func(
            circuit,
            alpha_params_next
        )
        update_record(circuit, metric_record, metric_values)
        update_record(circuit, loss_record, loss_values)

        loss_next.backward()
        gradient_next = alpha_params_next.grad


        # 5. Check whether to break
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

        # 6. Compute next Hessian approximation
        s = delta_params

        y = gradient_next - gradient

        rho = 1 / torch.dot(y, s)

        if rho.item() < 0:
            H = identity()
        else:
            A = identity() - rho * torch.matmul(s.unsqueeze(1), y.unsqueeze(0))
            B = identity() - rho * torch.matmul(y.unsqueeze(1), s.unsqueeze(0))
            H = (
                    torch.matmul(A, torch.matmul(H, B))
                    + rho * torch.matmul(s.unsqueeze(1), s.unsqueeze(0))
            )

        # 7. "Re-index"
        loss = loss_next
        alpha_params = alpha_params_next
        gradient = gradient_next

    return circuit, loss_record, metric_record


def backtracking_line_search(
    circuit: Circuit,
    objective_func: Callable[[Circuit, Tensor], Tensor],
    params: torch.tensor,       # params at starting point
    initial_loss: torch.tensor, # loss at starting point
    gradient: torch.tensor,     # gradient at starting point
    p: torch.tensor,            # search direction
    lr=0.01,
    c=1e-4,
    rho=0.5
) -> float:
    """At the end of line search, `circuit` will have its internal parameters
    set to ``params + alpha * p``.
    """
    print(50*"=" + "Line search called." + 50*"=")

    alpha = lr

    print(f"params: {params}")
    print(f"circuit params: {circuit.parameters}")
    print(f"p: {p}, alpha: {alpha}")

    counter = 0
    # with torch.no_grad(): # messes up adding things to parameters. Need to fix
    while (
            objective_func(circuit, params + alpha * p)[0]
            > initial_loss + c * alpha * torch.dot(p, gradient)
    ):
        alpha *= rho
        counter += 1
        if rho**counter < 1e-8:
            print("The line search broke")
            break

    print(60 * "=" + "The end." + 60 * "=")
    return alpha
