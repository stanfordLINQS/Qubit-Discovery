from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from SQcircuit import Circuit

from .truncation import assign_trunc_nums, test_convergence
from .utils import (
    set_grad_zero,
    get_grad,
    set_params,
    init_records,
    update_record,
    save_results,
    RecordType
)


SQValType = Union[float, Tensor]
LossFunctionType = Callable[
    [Circuit],
    Tuple[Tensor, Dict[str, SQValType], Dict[str, SQValType]]
]


def run_BFGS(
    circuit: Circuit,
    circuit_code: str,
    loss_metric_function: LossFunctionType,
    name: str,
    num_eigenvalues: int,
    total_trunc_num: int,
    save_loc: Optional[str] = None,
    bounds: Optional = None,
    lr: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-7,
    verbose: bool = False,
    save_intermediate_circuits: bool = True
) -> Tuple[Tensor, RecordType]:
    """Runs BFGS for a maximum of ``max_iter`` beginning with ``circuit`` using
    ``loss_metric_function``.

    Parameters
    ----------
        circuit:
            A circuit which has preliminary truncation numbers assigned, but
            not necessarily diagonalized.
        circuit_code:
            A string giving the type of the circuit.
        loss_metric_function:
            Loss function to optimize.
        name:
            Name identifying this run (e.g. seed, etc.)
        num_eigenvalues:
            Number of eigenvalues to calculate when diagonalizing.
        total_trunc_num:
            Maximum total truncation number to allocate.
        save_loc:
            Folder to save results in.
        bounds:
            Dictionary giving bounds for each element type.
        lr:
            Learning rate
        max_iter:
            Maximum number of iterations.
        tolerance:
            Minimum change each step must achieve to not terminate.
        verbose:
            Whether to print out progress.
        save_intermediate_circuits:
            Whether to save the circuit at each iteration.
    """
    params = torch.stack(circuit.parameters).clone()
    identity = torch.eye(params.size(0), dtype=torch.float64)
    H = identity

    circuit.diag(num_eigenvalues)
    # Get gradient and loss values to start with
    loss, loss_values, metric_values = loss_metric_function(circuit)
    loss_record, metric_record = init_records(
        circuit_code,
        loss_values,
        metric_values
    )

    def objective_func(cr: Circuit, x: Tensor, n_eigs: int):
        set_params(cr, x)
        cr.diag(n_eigs)
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

        if save_loc:
            save_results(
                loss_record,
                metric_record,
                circuit,
                circuit_code,
                name,
                save_loc,
                'BFGS',
                save_intermediate_circuits=save_intermediate_circuits
            )

        loss = objective_func(circuit, params, num_eigenvalues)
        loss.backward()
        gradient = get_grad(circuit)
        set_grad_zero(circuit)

        p = -torch.matmul(H, gradient)

        alpha = backtracking_line_search(
            circuit,
            objective_func,
            params,
            gradient,
            p,
            num_eigenvalues,
            bounds,
            lr=lr
        )
        delta_params = alpha * p

        params_next = (
                params + delta_params
        ).clone().detach().requires_grad_(True)

        loss_next = objective_func(circuit, params_next, num_eigenvalues)
        loss_next.backward()
        next_gradient = get_grad(circuit)
        set_grad_zero(circuit)

        loss_diff = loss_next - loss

        if verbose:
            if iteration % 1 == 0:
                print(
                    f"i:{iteration}",
                    f"loss: {loss.detach().numpy()}",
                    f"loss_diff={loss_diff.detach().numpy()}",
                    f"alpha={alpha}"
                )

        if torch.abs(loss_diff) < tolerance:
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
        circuit.update()
        
    return params, loss_record


def not_param_in_bounds(params, bounds, circuit_element_types) -> bool:
    """
    Check whether a given set of parameters are within bounds.
    """
    for param_idx, param in enumerate(params):
        circuit_element_type = circuit_element_types[param_idx]
        lower_bound, upper_bound = bounds[circuit_element_type]
        if param < lower_bound or param > upper_bound:
            return True

    return False


def backtracking_line_search(
    circuit: Circuit,
    objective_func: Callable[[Circuit, Tensor, int], Tensor],
    params: torch.tensor,  # params at starting point
    gradient: torch.tensor,  # gradient at starting point
    p: torch.tensor,  # search direction,
    num_eigenvalues: int,
    bounds=None,
    lr=1.0,
    c=1e-45,
    rho=0.1
) -> float:
    """At end of line search, `circuit` will have its internal parameters set
    to ``params + alpha * p``.
    """
    alpha = lr
    circuit_elements = circuit.get_all_circuit_elements()
    circuit_element_types = [type(element) for element in circuit_elements]

    if bounds is not None:
        while (not_param_in_bounds(
            params + alpha * p,
            bounds,
            circuit_element_types
        )):
            alpha *= rho

    baseline_loss = objective_func(circuit, params, num_eigenvalues)
    while (
        objective_func(circuit, params + alpha * p, num_eigenvalues)
        > baseline_loss + c * alpha * torch.dot(p, gradient)
    ):
        alpha *= rho
    return alpha
