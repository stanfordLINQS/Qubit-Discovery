from typing import Dict, Optional

from SQcircuit import Circuit, CircuitComponent
from torch.optim import LBFGS
from torch import Tensor

from .optim import OptimizationRecord, run_optimization
from .utils import LossFunctionType

# We set `max_iter = 1` and iterate manually to control print-out.
# This incurs 1 extra function evaluation per step vs. using  the `max_iter`
# in LBFGS itself.
DEFAULT_OPTIM_KWARGS = {
    'lr': 10,
    'history_size': None,
    'max_iter': 1,
    'line_search_fn': 'strong_wolfe',
}

def run_BFGS(
    circuit: Circuit,
    loss_metric_function: LossFunctionType,
    max_iter: int,
    total_trunc_num: int,
    bounds: Dict[CircuitComponent, Tensor],
    optimizer_kwargs: Optional[Dict] = None,
    num_eigenvalues: int = 10,
    save_loc: Optional[str] = None,
    identifier: Optional[str] = None,
    save_intermediate_circuits: bool = False,
) -> OptimizationRecord:
    """Runs BFGS for a maximum of ``max_iter`` beginning with ``circuit`` using
    ``loss_metric_function``.

    Parameters
    ----------
        circuit:
            A circuit to optimize.
        loss_metric_function:
            Loss function to optimize.
        max_iter:
            Maximum number of iterations.
        total_trunc_num:
            Maximum total truncation number to allocate.
        bounds:
            Dictionary giving bounds for each element type.
        num_eigenvalues:
            Number of eigenvalues to calculate when diagonalizing.
        save_loc:
            Folder to save results in, or None.
        identifier:
            String identifying this run (e.g. seed, name, circuit code, …)
            to use when saving.
        save_intermediate_circuits:
            Whether to save the circuit at each iteration.  

    Returns
    ----------
        An ``OptimizationRecord`` containing the optimized circuit, the final
        loss, and a record of the component loss and metric values at each
        epoch.
    """

    # Set up optimizer keyword arguments
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer_kwargs = DEFAULT_OPTIM_KWARGS | optimizer_kwargs

    if optimizer_kwargs['history_size'] is None:
        optimizer_kwargs['history_size'] = max_iter


    # Run it!
    return run_optimization(
        circuit = circuit,
        loss_metric_function = loss_metric_function,
        max_iter = max_iter,
        total_trunc_num = total_trunc_num,
        bounds = bounds,
        optimizer = LBFGS,
        uses_closure=True,
        optimizer_kwargs = optimizer_kwargs,
        num_eigenvalues = num_eigenvalues,
        save_loc = save_loc,
        identifier = identifier,
        save_intermediate_circuits = save_intermediate_circuits
    )
