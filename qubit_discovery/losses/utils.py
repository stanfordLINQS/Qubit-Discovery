from collections import defaultdict
from copy import copy
from typing import Union

from SQcircuit import Circuit, get_optim_mode
import torch

SQValType = Union[float, torch.Tensor]

def construct_perturbed_elements(
        circuit: Circuit,
        new_values: torch.Tensor,
):
    # Construct replacement loops with updated values, where necessary
    replacement_loops = {}
    for loop in circuit.loops:
        if loop in circuit.parameters:
            new_loop = copy(loop)
            new_loop.internal_value = new_values[circuit.parameters_elems.index(loop)]
        else:
            replacement_loops[loop] = loop

    new_elements = defaultdict(list)
    for edge in circuit.elements:
        for el in circuit.elements[edge]:
            new_el = copy(el)

            # Update value, if necessary
            if el in circuit.parameters_elems:
                new_el.internal_value = new_values[circuit.parameters_elems.index(el)]

            # Update loop values, if necessary
            if hasattr(el, 'loops'):
                new_loops = []
                for loop in el.loops:
                    new_loops.append(replacement_loops[loop])
                new_el.loops = new_loops

            new_elements[edge].append(new_el)

    return new_elements


def hinge_loss(val, cutoff, slope) -> SQValType:
    if val < cutoff:
        return 0.0 * val
    else:
        return slope * (val - cutoff)


def zero() -> SQValType:

    if get_optim_mode():
        return torch.tensor(0.0)

    return 0.0
