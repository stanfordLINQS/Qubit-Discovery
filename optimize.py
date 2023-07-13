"""Contains code for optimization of single circuit instance."""

import random

from functions import (
    create_sampler,
    get_element_counts,
    clamp_gradient,
    save_results
)
from loss import (
    calculate_loss,
    calculate_metrics,
    init_loss_record,
    init_metric_record,
    update_loss_record,
    update_metric_record
)

from truncation import trunc_num_heuristic, test_convergence

import argparse
import numpy as np
import SQcircuit as sq
import torch

# Assign keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument("code")
parser.add_argument("id")
args = parser.parse_args()

# Optimization settings

seed = int(args.id) + 240
random.seed(seed)
np.random.seed(seed)
num_epochs = 50  # number of training iterations
lr = 1e-1  # learning rate
num_eigenvalues = 10
total_trunc_num = 140

# Target parameter range
capacitor_range = [1e-15, 12e-12]  # F
inductor_range = [2e-8, 5e-6]  # H
junction_range = [1e9 * 2 * np.pi, 100e9 * 2 * np.pi]  # Hz
# capacitor_range = [8e-15, 12e-14] # F
# inductor_range = [2e-7, 5e-6] # H
# junction_range = [1e9 * 2 * np.pi, 12e9 * 2 * np.pi] # Hz

# Settings
gradient_clipping = True
loss_normalization = False
learning_rate_scheduler = True
scheduler_decay_rate = 0.99
gradient_clipping_threshold = 2
gc_norm_type = 'inf'

log_loss = False
nesterov_momentum = False
momentum_value = 0.9

element_verbose = False

def main():
    sq.set_optim_mode(True)

    circuit_code = args.code
    run_id = int(args.id)
    N = len(circuit_code)

    sampler = create_sampler(N, capacitor_range, inductor_range, junction_range)
    circuit = sampler.sample_circuit_code(circuit_code)
    print("Circuit sampled!")
    trunc_nums = circuit.truncate_circuit(total_trunc_num)
    print("Circuit truncated...")

    junction_count, inductor_count, _ = get_element_counts(circuit)

    circuit.diag(num_eigenvalues)
    print("Circuit diagonalized")

    metric_record = init_metric_record(circuit, circuit_code)
    loss_record = init_loss_record(circuit, circuit_code)

    converged = True
    # Circuit optimization loop
    for iteration in range(num_epochs):
        save_results(loss_record, metric_record, args.code, run_id)
        print(f"Iteration {iteration}")
        optimizer = torch.optim.SGD(
            circuit.parameters,
            nesterov=nesterov_momentum,
            momentum=momentum_value if nesterov_momentum else 0.0,
            lr=lr,
        )

        circuit.diag(num_eigenvalues)
        if len(circuit.m) == 1:
            converged = circuit.test_convergence(trunc_nums)
        elif len(circuit.m) == 2:
            trunc_nums = trunc_num_heuristic(circuit,
                                             K=4000,
                                             eig_vec_idx=1,
                                             axes=None)
            circuit.set_trunc_nums(trunc_nums)
            circuit.diag(num_eigenvalues)

            # converged = circuit.test_convergence(trunc_nums)
            converged, _, _ = test_convergence(circuit, eig_vec_idx=1)

        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            break

        # Calculate loss, backprop
        total_loss, loss_values = calculate_loss(circuit)
        metrics = calculate_metrics(circuit) + (total_loss, )
        update_metric_record(circuit, circuit_code, metric_record, metrics)
        update_loss_record(circuit, circuit_code, loss_record, loss_values)
        total_loss.backward()

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
        optimizer.step()
        optimizer.zero_grad()
        circuit.update()


if __name__ == "__main__":
    main()
