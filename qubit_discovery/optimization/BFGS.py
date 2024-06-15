from typing import Callable, Dict, Optional, Tuple, Union, List

import torch
from torch import Tensor

from SQcircuit import Circuit

from .truncation import assign_trunc_nums, test_convergence
from .utils import (
    print_loss_records,
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


def get_alpha_param_from_circuit_param(
    circuit_param: Tensor,
    u_bound: Tensor,
    l_bound: Tensor
) -> Tensor:

    var = (torch.log(u_bound) - torch.log(l_bound))/2
    mean = (torch.log(u_bound) + torch.log(l_bound))/2

    return torch.acos((torch.log(circuit_param) - mean) / var)


def get_alpha_params_from_circuit_params(circuit: Circuit, bounds) -> Tensor:

    alpha_params = torch.zeros(torch.stack(circuit.parameters).shape)

    for param_idx, circuit_param in enumerate(circuit.parameters):
        circuit_element_type = circuit.get_params_type()[param_idx]
        lower_bound, upper_bound = bounds[circuit_element_type]
        alpha_params[param_idx] = get_alpha_param_from_circuit_param(
            circuit_param,
            upper_bound,
            lower_bound
        )
    return alpha_params


def get_circuit_param_from_alpha_param(
    alpha_param: Tensor,
    u_bound: Tensor,
    l_bound: Tensor
) -> Tensor:

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
        circuit_element_type = circuit.get_params_type()[param_idx]
        lower_bound, upper_bound = bounds[circuit_element_type]
        circuit_params[param_idx] = get_circuit_param_from_alpha_param(
            alpha_param,
            upper_bound,
            lower_bound
        )
    return circuit_params


def get_gradients(loss: Tensor, circuit: Circuit, bounds) -> Tensor:
    loss.backward()
    partial_loss_partial_elem = circuit.parameters_grad
    print(f"partial_loss_partial_elem.dtype: {partial_loss_partial_elem.dtype}")
    circuit.zero_parameters_grad()

    alpha_params = get_alpha_params_from_circuit_params(circuit, bounds)
    alpha_params.backward(torch.tensor(len(alpha_params) * [1.0]))
    partial_alpha_partial_elem = circuit.parameters_grad
    circuit.zero_parameters_grad()
    print(f"partial_alpha_partial_elem.dtype: {partial_alpha_partial_elem.dtype}")

    gradient = (partial_loss_partial_elem / partial_alpha_partial_elem)

    return gradient.to(torch.float64)


def run_BFGS(
    circuit: Circuit,
    circuit_code: str,
    loss_metric_function: LossFunctionType,
    name: str,
    num_eigenvalues: int,
    baseline_trunc_nums: List[int],
    total_trunc_num: int,
    save_loc: Optional[str] = None,
    bounds: Optional = None,
    lr: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-15,
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
        baseline_trunc_nums:
            Number of trunc nums to allocate for heuristic function.
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

    alpha_params = get_alpha_params_from_circuit_params(circuit, bounds)
    alpha_params.backward(torch.tensor(len(alpha_params) * [1.0]))

    circuit_params = get_circuit_params_from_alpha_params(
        alpha_params, circuit, bounds
    )

    print(f"circuit_params: {torch.stack(circuit.parameters)}")
    print(f"alpha_params: {alpha_params}")
    print(f"reconstructed_circuit_params: {circuit_params}")
    print(f"alpha_params.grad: {circuit.parameters_grad}")

    print(130 * "=")

    # params = torch.stack(circuit.parameters).clone()
    params = get_alpha_params_from_circuit_params(circuit, bounds).clone()
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

    def objective_func(cr: Circuit, x: Tensor):
        print("Objective function called.")
        print(f"alpha_params: {x}")
        circ_params = get_circuit_params_from_alpha_params(
            x, cr, bounds
        )
        print(f"circ_params: {circ_params}")
        cr.parameters = circ_params
        cr.diag(num_eigenvalues)
        t_loss, _, _ = loss_metric_function(cr)

        return t_loss

    for iteration in range(max_iter):
        print(f"Iteration {iteration}")

        # Test if circuit has at least one harmonic mode
        if sum(circuit.omega != 0) > 0:
            circuit.set_trunc_nums(baseline_trunc_nums)
            circuit.diag(num_eigenvalues)
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

        loss = objective_func(circuit, params)
        # loss.backward()
        # gradient = circuit.parameters_grad
        # circuit.zero_parameters_grad()
        gradient = get_gradients(loss, circuit, bounds)
        print(f"gradient: {gradient}")

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
        # loss_next.backward()
        # next_gradient = circuit.parameters_grad
        # circuit.zero_parameters_grad()
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

    return params, loss_record


# def not_param_in_bounds(params, bounds, params_type) -> bool:
#     """
#     Check whether a given set of parameters are within bounds.
#     """
#     for param_idx, param in enumerate(params):
#         circuit_element_type = params_type[param_idx]
#         lower_bound, upper_bound = bounds[circuit_element_type]
#         if param < lower_bound or param > upper_bound:
#             return True
#
#     return False


def backtracking_line_search(
    circuit: Circuit,
    objective_func: Callable[[Circuit, Tensor], Tensor],
    params: torch.tensor,  # params at starting point
    gradient: torch.tensor,  # gradient at starting point
    p: torch.tensor,  # search direction,
    bounds=None,
    lr=0.01,
    c=1e-45,
    rho=0.8
) -> float:
    """At the end of line search, `circuit` will have its internal parameters
    set to ``params + alpha * p``.
    """
    print(50*"=" + "Line search called." + 50*"=")
    alpha = lr
    params_type = circuit.get_params_type()

    # if bounds is not None:
    #     while (not_param_in_bounds(
    #             params + alpha * p,
    #             bounds,
    #             params_type
    #     )):
    #         # print(f"alpha: {alpha}")
    #         # print(f"params + alpha * p: {params + alpha * p}")
    #
    #         alpha *= rho
    # print(130 * "=")
    # print(alpha)
    # for i in range(len(params)):
    #     print(70*"-")
    #     print(f"element_type {i}: {params_type[i]}")
    #     print(f"params {i}: {params[i].detach().numpy()}")
    #     print(f"p {i}: {p[i].detach().numpy()}")
    #     print(f"params + alpha * p {i}: {(params + alpha * p)[i].detach().numpy()}")
    # print(130 * "=")

    baseline_loss = objective_func(circuit, params)
    print(f"p: {p}, alpha: {alpha}")
    while (
            objective_func(circuit, params + alpha * p)
            > baseline_loss + c * alpha * torch.dot(p, gradient)
    ):
        alpha *= rho

    print(50 * "=" + "The end." + 50 * "=")
    return alpha
