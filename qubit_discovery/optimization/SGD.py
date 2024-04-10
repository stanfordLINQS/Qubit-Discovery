from copy import copy
from typing import List, Optional

import numpy as np
import torch

from SQcircuit import Capacitor, Circuit, Element, Inductor, Junction

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
learning_rate = 1e-2


def run_SGD(
    circuit: Circuit,
    circuit_code: str,
    loss_metric_function: LossFunctionType,
    name: str,
    num_eigenvalues: int,
    baseline_trunc_nums: List[int],
    total_trunc_num: int,
    num_epochs: int,
    bounds: List[Element],
    save_loc: str,
    save_intermediate_circuits=False
) -> None:
    """"
    Runs SGD for `num_epochs` beginning with `circuit` using
    `loss_metric_function`.

    `circuit` should have truncation numbers allocated, but need not have 
    been diagonalised. Diagonalisation is attempted with max total truncation
    number of `total_trunc_nums`, and with `num_eigenvalues`.

    `name` and `circuit_code` are just used to add metadata to file saved at
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

    # Circuit optimization loop
    for iteration in range(num_epochs):       
        # Calculate circuit
        print(f"Iteration {iteration}")
        optimizer.zero_grad()

        # Check if converged
        circuit.set_trunc_nums(baseline_trunc_nums)
        circuit.diag(num_eigenvalues)
        assign_trunc_nums(circuit, total_trunc_num)
        circuit.diag(num_eigenvalues)
        converged, _  = test_convergence(circuit, eig_vec_idx=1)
        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            break

        # Calculate loss, backpropagation
        total_loss, loss_values, metric_values = loss_metric_function(circuit)
        total_loss.backward()

        # Store history
        if loss_record is None:
            loss_record, metric_record = init_records(
                circuit_code,
                loss_values,
                metric_values
            )
        update_record(circuit, metric_record, metric_values)
        update_record(circuit, loss_record, loss_values)
        save_results(
            loss_record,
            metric_record,
            circuit,
            circuit_code,
            name,
            save_loc,
            'SGD',
            save_intermediate_circuits=True
        )

        # Clamp gradients, if desired
        with torch.no_grad():
            # without .no_grad() the element._value.grads track grad themselves
            for element in list(circuit._parameters.keys()):
                # norm_factor = 1
                norm_factor = element._value
                # for element_type in bounds.keys():
                #     if type(element) is element_type:
                #         norm_factor = bounds[T][1] - bounds[element_type][0]
                #         print(element._value.item(), norm_factor)
                #         break
                # for T in bounds.keys():
                #     if type(element) is T:
                #         norm_factor = np.log(bounds[T][1]) - np.log(bounds[T][0])

                # element._value.grad = torch.as_tensor((1/learning_rate) * ((np.exp(- (norm_factor)**2 * learning_rate * element._value.grad.item() * element._value.item())) - 1) * element._value.item())
                element._value.grad *= norm_factor
                if gradient_clipping:
                    clamp_gradient(element, gradient_clipping_threshold)
                element._value.grad *= norm_factor
                if learning_rate_scheduler:
                    element._value.grad *= (scheduler_decay_rate ** iteration)
            print('\n')

        # Step (to truly step, need to update the circuit as well)
        optimizer.step()
        circuit.update()