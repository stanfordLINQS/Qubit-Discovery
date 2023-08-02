import torch

from functions import (
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

from truncation import verify_convergence

from pympler import summary, muppy, asizeof
import pympler

def check_memory(circuit):
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)
    check_objects = {
        'circuit': circuit,
        'memory_ops': circuit._memory_ops,
        'LC_hamil': circuit._LC_hamil,
        'hamil': circuit.hamiltonian()
    }
    for key, val in check_objects.items():
        total_size = pympler.asizeof.asizeof(val)
        print(f"Total {key} size: {total_size}")


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

def run_SGD(circuit, circuit_code, seed, num_eigenvalues, trunc_nums, num_epochs):

    junction_count, inductor_count, _ = get_element_counts(circuit)

    metric_record = init_metric_record(circuit, circuit_code)
    loss_record = init_loss_record(circuit, circuit_code)

    # Circuit optimization loop
    for iteration in range(num_epochs):
        save_results(loss_record, metric_record, circuit_code, seed, prefix='SGD')
        print(f"Iteration {iteration}")
        check_memory(circuit)
        optimizer = torch.optim.SGD(
            circuit.parameters,
            nesterov=nesterov_momentum,
            momentum=momentum_value if nesterov_momentum else 0.0,
            lr=learning_rate,
        )

        circuit.diag(num_eigenvalues)
        converged = verify_convergence(circuit, trunc_nums, num_eigenvalues)

        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            break

        # Calculate loss, backprop
        total_loss, loss_values = calculate_loss(circuit)
        metrics = calculate_metrics(circuit) + (total_loss,)
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