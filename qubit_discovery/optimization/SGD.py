from copy import copy
from typing import Optional

import torch

from SQcircuit import Circuit

from .utils import (
    clamp_gradient,
    save_results,
    init_records,
    update_record, 
    LossFunctionType
)
from .truncation import assign_trunc_nums, test_convergence

# Global settings
log_loss = False
nesterov_momentum = False
momentum_value = 0.9
gradient_clipping = True
loss_normalization = False
learning_rate_scheduler = True
scheduler_decay_rate = 0.99
gradient_clipping_threshold = 2
gc_norm_type = 'inf'
learning_rate = 1e-1


def run_SGD(circuit: Circuit, 
            circuit_code: str, 
            loss_metric_function: LossFunctionType,
            seed: Optional[int], 
            num_eigenvalues: int,
            total_trunc_num: int,
            num_epochs: int,
            save_loc: str,
            save_circuit=True) -> None:
    """"
    Runs SGD for `num_epochs` beginning with `circuit` using
    `loss_metric_function`.

    `circuit` should have truncation numbers allocated, but need not have 
    been diagonalised. Diagonalisation is attempted with max total truncation
    number of `total_trunc_nums`, and with `num_eigenvalues`.

    `seed` and `circuit_code` are just used to add metadata to file saved at
    `save_loc` (but should be accurate).
    """
    
    loss_record, metric_record = None, None
    # Initialise optimiser
    optimizer = torch.optim.SGD(
        circuit.parameters,
        nesterov=nesterov_momentum,
        momentum=momentum_value if nesterov_momentum else 0.0,
        lr=learning_rate
    )
    # Circuit optimisation loop
    for iteration in range(num_epochs):       
        # Calculate circuit
        print(f"Iteration {iteration}")
        optimizer.zero_grad()

        circuit.diag(num_eigenvalues)
        # Check if converged with old truncation numbers
        converged, _  = test_convergence(circuit, eig_vec_idx=1)
        if not converged:
            # Attempt to re-allocate using our heuristic
            assign_trunc_nums(circuit, total_trunc_num)
            circuit.diag(num_eigenvalues)
            converged, _ = test_convergence(circuit, eig_vec_idx=1)
            # If it still hasn't converged after re-allocating, give up
            if not converged:
                print("Warning: Circuit did not converge")
                # TODO: ArXiv circuits that do not converge
                break

        # Calculate loss, backprop
        total_loss, loss_values, metric_values = loss_metric_function(circuit)
        total_loss.backward()

        # Store history
        if loss_record is None: 
            loss_record, metric_record = init_records(circuit_code, 
                                                      loss_values, 
                                                      metric_values)
        update_record(circuit, metric_record, metric_values)
        update_record(circuit, loss_record, loss_values)
        save_results(loss_record, metric_record, circuit, circuit_code, seed, 
                     save_loc, prefix='SGD', save_circuit=save_circuit)

        # Clamp gradients, if desired
        with torch.no_grad():
            # without .no_grad() the element._value.grads track grad themselves
            for element in list(circuit._parameters.keys()):
                element._value.grad *= element._value
                if gradient_clipping:
                    # torch.nn.utils.clip_grad_norm_(element._value,
                    #                                max_norm=gradient_clipping_threshold,
                    #                                norm_type=gc_norm_type)
                    clamp_gradient(element, gradient_clipping_threshold)
                element._value.grad *= element._value
                if learning_rate_scheduler:
                    element._value.grad *= (scheduler_decay_rate ** iteration)

        # Step (to truly step, need to update the circuit as well)
        optimizer.step()
        circuit.update()