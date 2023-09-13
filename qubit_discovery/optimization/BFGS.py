from copy import copy
from typing import Callable, Dict, Optional, Tuple
import sys

import torch
from torch import Tensor

import SQcircuit
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
    name: str,
    num_eigenvalues: int,
    total_trunc_num: int,
    save_loc: str,
    bounds: Dict[SQcircuit.Element, Tensor] = None,
    lr=1.0,
    max_iter=100,
    tolerance=1e-7,
    verbose=False,
    save_circuit=True
    ) -> Tuple[Tensor, RecordType]: 
    """
    Runs BFGS for a maximum of `max_iter` beginning with `circuit` using 
    `loss_function`.

    Parameters
    ----------
        circuit:
            A circuit which has preliminary truncation numbers assigned, but
            not necessarily diagonalized.
        circuit_code:
            A string giving the type of the circuit.
        loss_function:
            Loss function to optimize.
        name:
            Name identifying this run (e.g. seed, etc.)
        num_eigenvalues:
            Number of eigenvalues to calculate when diagonalizing.
        total_trunc_num:
            Maximum total runcation number to allocate.
        save_loc:
            Folder to save results in.
        bounds:
            Dictionary giving bounds for each element type.
        lr:
            Learning rate
        max_iter:
            Maximum number of iterations.
        tolerance:
            Minimum change each step must achieve to not termiante.
        verbose:
            Whether to print out progress.
        save_circuit:
            Whether to save the circuit at each iteration.
    """
    params = torch.stack(circuit.parameters).clone()
    identity = torch.eye(params.size(0), dtype=torch.float64)
    H = identity

    
    circuit.diag(num_eigenvalues)
    # Force re-allocation up to total_trunc_num
    assign_trunc_nums(circuit, total_trunc_num)
    converged = test_truncation(circuit, num_eigenvalues)
    # If hasn't converged after re-allocating, give up
    if not converged:
        print("Warning: Circuit did not converge")
        # TODO: ArXiv circuits that do not converge
        return None

    # Get gradient and loss values to start with
    loss, loss_values, metric_values = loss_function(circuit)
    loss.backward()
    loss_record, metric_record = init_records(circuit_code, 
                                              loss_values, 
                                              metric_values)
    update_record(circuit, metric_record, metric_values)
    update_record(circuit, loss_record, loss_values)
    save_results(loss_record, metric_record, circuit, circuit_code, name, 
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
        # set_params(circuit, params)
        # circuit.diag(num_eigenvalues)
        # print('L0', objective_func(circuit))
        print('P0', torch.stack(circuit.parameters).clone() - params)
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
                                                             name,
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
    """
    Check whether a given set of parameters are within bounds.
    """
    for param_idx, param in enumerate(params):
        circuit_element_type = circuit_element_types[param_idx]
        lower_bound, upper_bound = bounds[circuit_element_type]
        if param < lower_bound or param > upper_bound:
            return True

    return False

def test_truncation(circuit: Circuit,
                    num_eigenvalues: int) -> bool:
    """
    Test the currently assigned truncation numbers, by diagonalizing and then
    running the convergence test.
    """
    circuit.diag(num_eigenvalues)
    # Check if converged with old truncation numbers
    converged, _  = test_convergence(circuit, eig_vec_idx=1)
    return converged 

def compute_and_save_gradient(circuit: Circuit,
                              circuit_code: str,
                              name: str,
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
    save_results(loss_record, metric_record, circuit, circuit_code, name, 
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
    c=1e-5,
    rho=0.1,
    machine_tol=1e-14
) -> float:
    """"
    At end of line search, `circuit` will have its internal parameters set to 
    `params + alpha * p`.
    """
    # print('P0', circuit.parameters, params, )
    # set_params(circuit, params)
    # circuit.diag(num_eigenvalues)
    # base_loss_2 = objective_func(circuit)
    # print('With or w/o grad', (base_loss_2- base_loss).detach().numpy())
    # print('L4', base_loss_2.detach().numpy())
    # base_loss = base_loss_2

    alpha = lr
    circuit_elements = circuit._parameters.keys()
    circuit_element_types = [type(element) for element in circuit_elements]

    if bounds is not None:
        while (
            not_param_in_bounds(params + alpha * p, bounds, circuit_element_types)
        ):
            alpha *= rho

    print('P', params)
    while True:
        # set_params(circuit, params + alpha * p)
        # circuit.diag(num_eigenvalues)
        # See if new parameters need new truncation numbers
        if not test_truncation(circuit, num_eigenvalues):
            while True:
                # If they do, re-allocate
                print('realloc')
                assign_trunc_nums(circuit, total_trunc_num)
                converged = test_truncation(circuit, num_eigenvalues)
                if not converged:
                    # backtrack further
                    print('backtrack further')
                    alpha *= rho
                    set_params(circuit, params + alpha * p)
                    circuit.diag(num_eigenvalues)
                    continue
                # and calculate loss
                curr_loss = objective_func(circuit)

                # Need consistent truncation numbers to compare points along
                # line search, so put circuit back and see if it works 
                # with original params
                set_params(circuit, params)
                start_converged = test_truncation(circuit, num_eigenvalues)
                if not start_converged:
                    print('backtrack OG further')
                    # If not, backtrack further 
                    alpha *= rho
                    set_params(circuit, params + alpha * p)
                    circuit.diag(num_eigenvalues)
                    continue
                else:
                    print('recalculated')
                    base_loss = objective_func(circuit)
                    break
        else:
            # Otherwise, can just calculate the loss immediately
            curr_loss = objective_func(circuit)

        # If we are at a better position, line search is over
        # print('alpha', alpha)
        # print('PP', alpha * p)
        # print('L', curr_loss.detach().numpy(),(base_loss + c * alpha * torch.dot(p, gradient)).detach().numpy(),
        #       (base_loss + c * alpha * torch.dot(p, gradient)).detach().numpy() - curr_loss.detach().numpy(),
        #       curr_loss <= (base_loss + c * alpha * torch.dot(p, gradient)))
        if curr_loss <= (base_loss + c * alpha * torch.dot(p, gradient) + machine_tol):
            break
        if torch.all((1 - ((params + alpha * p)/params)) < 1e-15):
            break
        # Otherwise, backtrack
        alpha *= rho
    return alpha
