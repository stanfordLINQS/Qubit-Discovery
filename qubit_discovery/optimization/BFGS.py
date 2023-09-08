from copy import copy
from typing import Callable, Optional, Tuple
import sys

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
    LossFunctionType,
    RecordType
)

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
    verbose=False,
    save_circuit=True
    ) -> Tuple[Tensor, RecordType]: 
    params = torch.stack(circuit.parameters).clone()
    identity = torch.eye(params.size(0), dtype=torch.float64)
    H = identity

    # Get gradient and loss values to start with
    do_truncation(circuit, num_eigenvalues, total_trunc_num)
    loss, loss_values, metric_values = loss_function(circuit)
    loss.backward()
    loss_record, metric_record = init_records(circuit_code, 
                                              loss_values, 
                                              metric_values)
    update_record(circuit, metric_record, metric_values)
    update_record(circuit, loss_record, loss_values)
    save_results(loss_record, metric_record, circuit, circuit_code, seed, 
                    save_loc, prefix='BFGS', save_circuit=save_circuit)
    gradient = get_grad(circuit)
    set_grad_zero(circuit)
    
    
    def objective_func(circuit):
        return loss_without_gradient(circuit, loss_function)

    for iteration in range(max_iter):
        # Compute search direction
        # (we know the gradient since we calculated it in order to get H_k)
        p = -torch.matmul(H, gradient)

        # Compute next step by line search
        alpha = backtracking_line_search(circuit, objective_func, params,
                                         gradient, loss, p, num_eigenvalues,
                                         total_trunc_num, bounds=bounds, lr=lr)
        
        delta_params = alpha * p

        # Update parameters
        params_next = (params + delta_params).clone().detach().requires_grad_(True)

        # Calculate next inverse Hessian approximation
        ## Get gradient at next step
        gradient_next, loss_next = compute_and_save_gradient(circuit, 
                                                             circuit_code, 
                                                             seed,
                                                             loss_function, 
                                                             metric_record, loss_record,
                                                             save_loc, save_circuit=save_circuit)

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

        ## Calculate H_{k+1}
        s = delta_params

        y = gradient_next - gradient

        rho = 1 / torch.dot(y, s)

        if rho.item() < 0:
            H = identity
        else:
            A = identity - rho * torch.matmul(s.unsqueeze(1), y.unsqueeze(0))
            B = identity - rho * torch.matmul(y.unsqueeze(1), s.unsqueeze(0))
            H = torch.matmul(A, torch.matmul(H, B)) + rho * torch.matmul(s.unsqueeze(1), s.unsqueeze(0))

        # Replace current values with next values
        params = params_next
        loss = loss_next
        gradient = gradient_next

    return params, loss_record


def not_param_in_bounds(params, bounds, circuit_element_types) -> bool:
    for param_idx, param in enumerate(params):
        circuit_element_type = circuit_element_types[param_idx]
        lower_bound, upper_bound = bounds[circuit_element_type]
        if param < lower_bound or param > upper_bound:
            return True

    return False

def test_truncation(circuit: Circuit,
                    num_eigenvalues: int) -> bool:
    circuit.diag(num_eigenvalues)
    # Check if converged with old truncation numbers
    converged, _  = test_convergence(circuit, eig_vec_idx=1)
    return converged 

def do_truncation(circuit: Circuit,
                  num_eigenvalues: int,
                  total_trunc_num: int) -> None:
    # Check if converged with old truncation numbers
    converged = test_truncation(circuit, num_eigenvalues)
    if not converged:
        # Attempt to re-allocate using our heuristic
        assign_trunc_nums(circuit, total_trunc_num)
        converged = test_truncation(circuit, num_eigenvalues)
        # If it still hasn't converged after re-allocating, give up
        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            sys.exit(1)

def compute_and_save_gradient(circuit: Circuit,
                              circuit_code: str,
                              seed: int,
                              loss_function: LossFunctionType,
                              metric_record: RecordType,
                              loss_record: RecordType,
                              save_loc: str,
                              save_circuit=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the loss and gradient of circuit. Save it. 
    
    Clears gradient afterwards.
    """
    loss, loss_values, metric_values = loss_function(circuit)
    loss.backward()

    update_record(circuit, metric_record, metric_values)
    update_record(circuit, loss_record, loss_values)
    save_results(loss_record, metric_record, circuit, circuit_code, seed, 
                 save_loc, prefix='BFGS', save_circuit=save_circuit)
    gradient = get_grad(circuit)
    set_grad_zero(circuit)
    return gradient, loss

def loss_without_gradient(circuit: Circuit,
                          loss_function: LossFunctionType) -> torch.Tensor:
    loss, _, _ = loss_function(circuit, master_use_grad=False)
    return loss

def backtracking_line_search(
    circuit: Circuit,
    objective_func: Callable[[Circuit, Tensor], Tensor],
    params: torch.tensor, # params at starting pioint
    gradient: torch.tensor, # gradient at starting point
    base_loss: torch.tensor, # loss at starting point
    p: torch.tensor, # search direction,
    num_eigenvalues: int,
    total_trunc_num: int,
    bounds=None,
    lr=1.0,
    c=1e-14,
    rho=0.1
) -> float:
    """"
    At end of line search, `circuit` will have its internal parameters set to 
    `params + alpha * p`.
    """
    alpha = lr
    circuit_elements = circuit._parameters.keys()
    circuit_element_types = [type(element) for element in circuit_elements]

    if bounds is not None:
        while (
            not_param_in_bounds(params + alpha * p, bounds, circuit_element_types)
        ):
            alpha *= rho

    while True:
        set_params(circuit, params + alpha * p)

        # See if new parameters need new truncation numbers
        if not test_truncation(circuit, num_eigenvalues):
            while True:
                # If they do, re-allocate
                assign_trunc_nums(circuit, total_trunc_num)
                converged = test_truncation(circuit, num_eigenvalues)
                if not converged:
                    # backtrack further
                    alpha *= rho
                    set_params(circuit, params + alpha * p)
                    continue
                # and calculate loss
                curr_loss = objective_func(circuit)

                # Need consistent truncation numbers to compare points along
                # line search, so  put circuit back and see if it works 
                # with original params
                set_params(circuit, params)
                start_converged = test_truncation(circuit, num_eigenvalues)
                if not start_converged:
                    # If not, backtrack further 
                    alpha *= rho
                    set_params(circuit, params + alpha * p)
                    continue
                else:
                    base_loss = objective_func(circuit)
                    break
        else:
            # Otherwise, can just calculate the loss immediately
            curr_loss = objective_func(circuit)

        # If we are at a better position, line search is over
        if curr_loss <= (base_loss + c * alpha * torch.dot(p, gradient)):
            break
        # Otherwise, backtrack
        alpha *= rho
    return alpha
