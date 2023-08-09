from copy import copy

import torch

from functions import (
    get_element_counts,
    clamp_gradient,
    save_results
)
from loss import (
    calculate_loss_metrics,
    init_records,
    update_loss_record,
    update_metric_record
)
from truncation import assign_trunc_nums, test_convergence

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


def run_SGD(circuit, circuit_code, seed, num_eigenvalues, total_trunc_num, num_epochs):

    junction_count, inductor_count, _ = get_element_counts(circuit)

    test_circuit = copy(circuit)
    loss_record, metric_record = init_records(circuit, test_circuit, circuit_code)

    # Circuit optimization loop
    for iteration in range(num_epochs):
        save_results(loss_record, metric_record, circuit_code, seed, prefix='SGD')
        print(f"Iteration {iteration}")

        optimizer = torch.optim.SGD(
            circuit.parameters,
            nesterov=nesterov_momentum,
            momentum=momentum_value if nesterov_momentum else 0.0,
            lr=learning_rate,
        )

        assign_trunc_nums(circuit, total_trunc_num)
        circuit.diag(num_eigenvalues)
        converged, _ = test_convergence(circuit, eig_vec_idx=1)

        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            break

        # Calculate loss, backprop
        optimizer.zero_grad()
        total_loss, loss_values, metrics = calculate_loss_metrics(circuit, test_circuit,
                                                                  use_frequency_loss=True,
                                                                  use_anharmonicity_loss=True,
                                                                  use_flux_sensitivity_loss=False,
                                                                  use_charge_sensitivity_loss=False)
        total_loss.backward()
        update_metric_record(circuit, circuit_code, metric_record, metrics)
        update_loss_record(circuit, circuit_code, loss_record, loss_values)

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
        circuit.update()