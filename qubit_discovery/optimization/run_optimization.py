import logging
from typing import Dict, Optional, Tuple, Union

from SQcircuit import Circuit, Element, Loop
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import Tensor
from tqdm import trange

from .reparameterization import (
    get_alpha_params_from_circuit_params,
    get_circuit_params_from_alpha_params
)

from .truncation import assign_trunc_nums, test_convergence
from .utils import (
    ConvergenceError,
    init_records,
    LossFunctionType,
    RecordType,
    save_results,
    update_record,
)

logger = logging.getLogger(__name__)


def diag_with_convergence(
        circuit: Circuit,
        num_eigenvalues: int,
        total_trunc_num: int
) -> bool:
    """
    Diagonalize the circuit, and, if the circuit has not converged, try
    re-allocating the truncation numbers. If this fails, then we give up and
    say the circuit has not converged.
    """

    # Reset to even split of truncation numbers
    circuit.truncate_circuit(total_trunc_num)
    circuit.diag(num_eigenvalues)
    # Check if converges with even split
    converged, _ = test_convergence(circuit, eig_vec_idx=1)
    # Otherwise try re-allocating with the heuristic function
    if not converged:
        assign_trunc_nums(circuit, total_trunc_num)
        circuit.diag(num_eigenvalues)

        converged, eps = test_convergence(circuit, eig_vec_idx=1, t=10)
        if not converged:
            raise ConvergenceError(eps)

    return True


def run_optimization(
    circuit: Circuit,
    loss_metric_function: LossFunctionType,
    max_iter: int,
    total_trunc_num: int,
    bounds: Dict[Union[Element, Loop], Tensor],
    optimizer: Optimizer,
    uses_closure: bool,
    optimizer_kwargs: Optional[Dict] = None,
    scheduler : Optional[LRScheduler] = None,
    scheduler_kwargs: Optional[Dict] = None,
    num_eigenvalues: int = 10,
    save_loc: Optional[str] = None,
    identifier: Optional[str] = None,
    save_intermediate_circuits: bool = False,
) -> Tuple[Circuit, RecordType, RecordType]:
    # Set initial truncation numbers
    circuit.truncate_circuit(total_trunc_num)

    # Set up initial reparameterization
    alpha_params = get_alpha_params_from_circuit_params(
        circuit, bounds
    ).detach().clone().requires_grad_(True)

    last_total_loss = None
    last_loss_values = None
    last_metric_values = None

    optim = optimizer(
        [alpha_params],
        **optimizer_kwargs
    )

    if scheduler is not None:
        sched = scheduler(optim, **scheduler_kwargs)

    def update_circuit() -> None:
        circuit_params = get_circuit_params_from_alpha_params(
            alpha_params, circuit, bounds
        )
        circuit.parameters = circuit_params

    def objective_closure() -> Tensor:
        """Objective function for optimization with reparameterization. It 
            1. Rebuilds circuit from alpha_params
            2. Diagonalizes it and checks convergence
            3. Computes losses and metrics
        """

        # Hacky way to extract history
        nonlocal last_total_loss
        nonlocal last_loss_values
        nonlocal last_metric_values

        # Zero the gradient
        optim.zero_grad()

        # Rebuild the circuit from the alpha parameters
        update_circuit()
        diag_with_convergence(circuit, num_eigenvalues, total_trunc_num)

        # Compute the loss…
        last_total_loss, last_loss_values, last_metric_values = loss_metric_function(circuit)
        # …and the gradient
        last_total_loss.backward()

        return last_total_loss

    # Get starting values to save
    objective_closure()

    loss_record, metric_record = init_records(
        last_loss_values,
        last_metric_values
    )

    with trange(max_iter) as t:
        for iteration in t:
            # Update tqdm
            t.set_description(f'Iteration {iteration}')
            t.set_postfix(loss=f'{last_total_loss.detach().numpy():.3e}')
            # Log detailed information
            logger.info(
                '%s\n'
                + 'Optimization progress\n'
                + 'iteration: %s\n'
                + 'params: %s\n'
                + 'circuit params: %s\n'
                + 'loss: %s',
                90 * '-', iteration, alpha_params.detach().numpy(),
                [p.detach().numpy() for p in circuit.parameters],
                last_total_loss.detach().numpy()
            )
            for key in loss_record:
                logger.info('\t%s: %s', key, loss_record[key][-1])

            # Hold old parameter values
            old_alpha_params = alpha_params.detach().clone()

            if uses_closure:
                optim.step(objective_closure)
            else:
                objective_closure()
                optim.step()
            update_circuit()

            if scheduler is not None:
                sched.step()

            # Update records
            update_record(
                loss_record,
                last_loss_values
            )
            update_record(
                metric_record,
                last_metric_values
            )
            if save_loc:
                save_results(
                    loss_record,
                    metric_record,
                    circuit,
                    identifier,
                    save_loc,
                    save_intermediate_circuits=save_intermediate_circuits
                )

            # Check if the optimizer has finished (params did not move)
            if torch.all(old_alpha_params == alpha_params):
                break

    return circuit, loss_record, metric_record
