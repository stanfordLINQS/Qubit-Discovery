from functions import (
    set_grad_zero,
    get_grad,
    set_params,
    save_results
)

from loss import (
    calculate_loss,
    calculate_metrics,
    init_loss_record,
    init_metric_record,
    update_metric_record,
    update_loss_record
)

from truncation import verify_convergence

import torch

def run_BFGS(
    circuit,
    circuit_code,
    seed,
    num_eigenvalues,
    trunc_nums,
    bounds=None,
    lr=1.0,
    max_iter=100,
    tolerance=1e-7,
    verbose=False
):
    params = torch.stack(circuit.parameters).clone()
    identity = torch.eye(params.size(0), dtype=torch.float64)
    H = identity

    metric_record = init_metric_record(circuit, circuit_code)
    loss_record = init_loss_record(circuit, circuit_code)

    for iteration in range(max_iter):
        save_results(loss_record, metric_record, circuit_code, seed, prefix='BFGS')
        print(f"Iteration {iteration}")

        circuit.diag(num_eigenvalues)
        converged = verify_convergence(circuit, trunc_nums, num_eigenvalues)

        if not converged:
            print("Warning: Circuit did not converge")
            # TODO: ArXiv circuits that do not converge
            break

        loss = objective_func(circuit, params, num_eigenvalues)

        # Compute loss and metrics, update records
        total_loss, loss_values = calculate_loss(circuit)
        metrics = calculate_metrics(circuit) + (total_loss,)
        update_metric_record(circuit, circuit_code, metric_record, metrics)
        update_loss_record(circuit, circuit_code, loss_record, loss_values)

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


def not_param_in_bounds(param, bounds) -> bool:

    lower_bound = bounds[0]
    upper_bound = bounds[1]

    return not ((param>lower_bound).all() and (param<upper_bound).all())

def line_search(
    circuit,
    objective_func,
    params,
    gradient,
    p,
    num_eigenvalues,
    bounds=None,
    lr=1.0,
    c=1e-4,
    rho=0.1
):
    alpha = lr

    if bounds is not None:
        while (
            not_param_in_bounds(params + alpha * p, bounds)
        ):
            alpha *= rho

    while (
        objective_func(circuit, params + alpha * p, num_eigenvalues)
        > objective_func(circuit, params, num_eigenvalues) + c * alpha * torch.dot(p, gradient)
    ):
        alpha *= rho
    return alpha

def objective_func(circuit, x, num_eigenvalues):

    set_params(circuit, x)
    circuit.diag(num_eigenvalues)
    total_loss, _ = calculate_loss(circuit)

    return total_loss