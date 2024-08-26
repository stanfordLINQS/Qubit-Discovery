"""utils.py """

import logging
import os
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Union
)

import dill as pickle
from SQcircuit import Circuit
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


################################################################################
# Helper functions.
################################################################################

def float_list(ls: List[str]) -> List[float]:
    """Evaluates a list of strings and returns a list of floats.
    """
    return [float(i) for i in ls]


# Typing
SQValType = Union[float, Tensor]
LossFunctionType = Callable[
    [Circuit],
    Tuple[Tensor, Dict[str, SQValType], Dict[str, SQValType]]
]

RecordType = Dict[
    str, Union[str, List[Tensor], List[Circuit]]
]


################################################################################
# Functions for records.
################################################################################

@torch.no_grad()
def init_records(
    loss_values: Dict[str, Tensor],
    metric_values: Dict[str, Tensor]
) -> Tuple[RecordType, RecordType]:
    """Initialize a loss and metric record, starting with a set of initial
    values.
    
    Parameters
    ----------
        loss_values:
            A dictionary of initial loss values.
        metric_values:
            A dictionary of initital metric values.

    Returns
    ----------
        loss_record:
            A dictionary of loss_name: [list of values for each epoch].
        metric_record
            A dictionary of metric_name: [list of values for each epoch].
    """
    # Init loss record
    loss_record: RecordType = {
        loss_type: [loss_value.detach().numpy()]
        for loss_type, loss_value in loss_values.items()
    }

    # Init metric record
    metric_record: RecordType = {
        metric_type: [metric_value.detach().numpy()]
        for metric_type, metric_value in metric_values.items()
    }

    return loss_record, metric_record


@torch.no_grad()
def update_record(
    record: RecordType,
    values: Dict[str, List[Tensor]]
) -> None:
    """Updates a record based on next epoch of optimization.
    
    Parameters
    ----------
        record
            A dictionary of name: [values for each epoch].
        metric_values:
            A dictionary of values for the current epoch.
    """
    for key in values.keys():
        record[key].append(values[key].detach().numpy())


def save_results(
    loss_record: RecordType,
    metric_record: RecordType,
    circuit: Circuit,
    identifier: str,
    save_loc: str,
    save_intermediate_circuits=True,
) -> None:
    """Save results from optimization to ``save_loc``.

    Parameters
    ----------
        loss_record
            The loss record from optimization.
        metric_record:
            The metric record from optimization.
        circuit:
            The current circuit.
        identifier:
            A string to use to label the saved results with.
        save_loc:
            A path to save the results at.
        save_intermediate_circuits:
            Whether to save intermediate circuits. If ``True``, the saved
            circuit record is appended to with the new ``circuit``; otherwise
            it is overwritten by ``circuit``.
    """
    save_records = {'loss': loss_record, 'metrics': metric_record}

    for record_type, record in save_records.items():
        save_url = os.path.join(
            save_loc,
            f'{record_type}_record_{identifier}.pickle'
        )
        logger.info('Saving to %s', save_url)
        with open(save_url, 'wb') as f:
            pickle.dump(record, f)

    if save_intermediate_circuits:
        write_mode = 'ab+'
    else:
        write_mode = 'wb'

    circuit_save_url = os.path.join(
        save_loc,
        f'circuit_record_{identifier}.pickle'
    )
    with open(circuit_save_url, write_mode) as f:
        pickle.dump(circuit.picklecopy(), f)
