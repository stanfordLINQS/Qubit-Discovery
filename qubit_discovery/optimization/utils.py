"""utils.py """

from collections import defaultdict
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
from SQcircuit import (
    Circuit,
    Loop,
    Element,
    Inductor,
    Junction,
    Capacitor
)
import torch

logger = logging.getLogger(__name__)


################################################################################
# Helper functions.
################################################################################

def float_list(ls: List[str]) -> List[float]:
    """Evaluates a list of strings and returns a list of floats.
    """
    return [float(i) for i in ls]


# Typing
SQValType = Union[float, torch.Tensor]
LossFunctionType = Callable[
    [Circuit],
    Tuple[torch.Tensor, Dict[str, SQValType], Dict[str, SQValType]]
]

RecordType = Dict[
    str, Union[str, List[torch.Tensor], List[Circuit]]
]

@torch.no_grad()
def init_records(
    loss_values: Dict[str, List[torch.tensor]],
    metric_values: Dict[str, List[torch.tensor]]
) -> Tuple[RecordType, RecordType]:
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
    values: Dict[str, List[torch.Tensor]]
) -> None:
    """Updates record based on next iteration of optimization."""
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


class ConvergenceError(Exception):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def __str__(self):
        return f'Your circuit did not converge. The computed epsilon was {self.epsilon}.'
