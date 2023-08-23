from copy import copy
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from SQcircuit import Circuit

from .utils import (
    set_grad_zero,
    get_grad,
    set_params,
)
from qubit_discovery.losses.loss import (
    calculate_loss_metrics,
)
from .utils import (
    init_records,
    update_record,
    save_results,
    LossFunctionType,
    RecordType
)
from .truncation import assign_trunc_nums, test_convergence


def run_BFGS(
    circuit: Circuit,
    circuit_code: str,
    loss_function: LossFunctionType,
    seed: Optional[int],
    num_eigenvalues: int,
    total_trunc_num: int,
    save_loc: str,
    bounds=None,
    lr=1.0,
    max_iter=100,
    tolerance=1e-7,
    verbose=False
    ) -> Tuple[Tensor, RecordType]: 
    params = torch.stack(circuit.parameters).clone()
    identity = torch.eye(params.size(0), dtype=torch.float64)
    H = identity

    loss_record, metric_record = None, None
    for iteration in range(max_iter):
        print(f"Iteration {iteration}")

        assign_trunc_nums(circuit, total_trunc_num)
        circuit.diag(num_eigenvalues)
        converged, _ = test_convergence(circuit, eig_vec_idx=1)

        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            break

        # Compute loss and metrics, update records
        loss, loss_values, metric_values = calculate_loss_metrics(circuit,
                                                                  use_frequency_loss=True,
                                                                  use_anharmonicity_loss=True,
                                                                  use_flux_sensitivity_loss=False,
                                                                  use_charge_sensitivity_loss=False)
        if loss_record is None: 
            loss_record, metric_record = init_records(circuit_code, 
                                                      loss_values, 
                                                      metric_values)
        update_record(circuit, metric_record, metric_values)
        update_record(circuit, loss_record, loss_values)
        save_results(loss_record, metric_record, circuit_code, seed, 
                     save_loc, prefix='BFGS')


        loss = objective_func(circuit, params, num_eigenvalues)
        loss.backward()
        gradient = get_grad(circuit)
        set_grad_zero(circuit)

        p = -torch.matmul(H, gradient)

        alpha = line_search(circuit, objective_func, params, gradient, p, num_eigenvalues, bounds, lr=lr)
        delta_params = alpha * p

        params_next = (params + delta_params).clone().detach().requires_grad_(True)

        loss_next = objective_func(circuit, params_next, num_eigenvalues)
        loss_next.backward()
        next_gradient = get_grad(circuit)
        set_grad_zero(circuit)

        loss_diff = loss_next - loss

        if verbose:
            if iteration % 1 == 0:
                print(f"i:{iteration}",
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
            H = torch.matmul(A, torch.matmul(H, B)) + rho * torch.matmul(s.unsqueeze(1), s.unsqueeze(0))

        params = params_next

    return params, loss_record


def not_param_in_bounds(params, bounds, circuit_element_types) -> bool:
    for param_idx, param in enumerate(params):
        circuit_element_type = circuit_element_types[param_idx]
        lower_bound, upper_bound = bounds[circuit_element_type]
        if param < lower_bound or param > upper_bound:
            return True

    return False

def line_search(
    circuit: Circuit,
    objective_func: Callable[[Circuit, Circuit, Tensor, int], Tensor],
    params,
    gradient,
    p,
    num_eigenvalues: int,
    bounds=None,
    lr=1.0,
    c=1e-14,
    rho=0.1
) -> float:
    alpha = lr
    circuit_elements = circuit.get_all_circuit_elements()
    circuit_element_types = [type(element) for element in circuit_elements]

    if bounds is not None:
        while (
            not_param_in_bounds(params + alpha * p, bounds, circuit_element_types)
        ):
            alpha *= rho

    base_value = objective_func(circuit, test_circuit, params, num_eigenvalues)
    while (
        objective_func(circuit, test_circuit, params + alpha * p, num_eigenvalues)
        > base_value + c * alpha * torch.dot(p, gradient)
    ):
        alpha *= rho
    return alpha

def objective_func(circuit: Circuit, 
                   test_circuit: Circuit, 
                   x, 
                   num_eigenvalues: int):

    set_params(circuit, x)
    circuit.diag(num_eigenvalues)
    return calculate_loss_metrics(circuit,
                                  test_circuit,
                                  use_frequency_loss=True,
                                  use_anharmonicity_loss=True,
                                  use_flux_sensitivity_loss=False,
                                  use_charge_sensitivity_loss=False)
