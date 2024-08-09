"""This module implements the SGD optimizer."""

from typing import Dict, Optional, Tuple, Union

from SQcircuit import Circuit, Element, Loop
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch import Tensor

from .optim import run_optimization
from .utils import LossFunctionType, RecordType

DEFAULT_OPTIM_KWARGS = {
    'nesterov': False,
    'momentum': 0.0,
    'lr': 1e-2
}

SCHEDULER_KWARGS = {
    'gamma': 0.99
}

def run_SGD(
    circuit: Circuit,
    loss_metric_function: LossFunctionType,
    max_iter: int,
    total_trunc_num: int,
    bounds: Dict[Union[Element, Loop], Tensor],
    optimizer_kwargs: Optional[Dict] = None,
    num_eigenvalues: int = 10,
    save_loc: Optional[str] = None,
    identifier: Optional[str] = None,
    save_intermediate_circuits: bool = False,
) -> Tuple[Circuit, RecordType, RecordType]:
    """Runs SGD for a maximum of ``max_iter`` beginning with ``circuit`` using
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
    """

    # Set up optimizer kwargs
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer_kwargs = DEFAULT_OPTIM_KWARGS | optimizer_kwargs

    return run_optimization(
        circuit = circuit,
        loss_metric_function = loss_metric_function,
        max_iter = max_iter,
        total_trunc_num = total_trunc_num,
        bounds = bounds,
        optimizer = SGD,
        uses_closure = False,
        optimizer_kwargs = optimizer_kwargs,
        scheduler=ExponentialLR,
        scheduler_kwargs=SCHEDULER_KWARGS,
        num_eigenvalues = num_eigenvalues,
        save_loc = save_loc,
        identifier = identifier,
        save_intermediate_circuits = save_intermediate_circuits
    )
