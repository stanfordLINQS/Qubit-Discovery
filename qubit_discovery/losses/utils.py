from collections import defaultdict
from copy import copy
from typing import Dict, List, Tuple, Union

from SQcircuit import Circuit, Element, get_optim_mode
import torch

SQValType = Union[float, torch.Tensor]


def construct_perturbed_elements(
        circuit: Circuit,
        new_values: torch.Tensor,
) -> Dict[Tuple[int, int], List[Element]]:
    """
    Construct a dictionary of elements based on ``circuit``, where the elements
    present in `circuit.parameters`` have their values updated by those
    provided in ``new_values``.

    Parameters
    ----------
        circuit:
            A circuit to use as the initial elements.
        new_values:
            The values for the perturbed elements to take. This is a tensor of
            length ``circuit.parameters``, and the values are assigned to the 
            elements according to the order given by
            ``circuit.parameters_elems``.
    
    Returns
    ----------
        A dictionary of elements, with perturbed values.
    """
    # Construct replacement loops with updated values, where necessary
    replacement_loops = {}
    for loop in circuit.loops:
        if loop in circuit.parameters:
            new_loop = copy(loop)
            new_loop.internal_value = new_values[
                circuit.parameters_elems.index(loop)
            ]
        else:
            replacement_loops[loop] = loop

    new_elements = defaultdict(list)
    for edge in circuit.elements:
        for el in circuit.elements[edge]:
            new_el = copy(el)

            # Update value, if necessary
            if el in circuit.parameters_elems:
                new_el.internal_value = new_values[
                    circuit.parameters_elems.index(el)
                ]

            # Update loop values, if necessary
            if hasattr(el, 'loops'):
                new_loops = []
                for loop in el.loops:
                    new_loops.append(replacement_loops[loop])
                new_el.loops = new_loops

            new_elements[edge].append(new_el)

    return new_elements


def hinge_loss(val: SQValType, cutoff: float, slope: float) -> SQValType:
    """
    Compute a linear hinge loss.
    
    Parameters
    ----------
        val:
            The input value.
        cutoff:
            The cutoff above which to apply hinge loss.
        slope:
            The slope of the loss above ``cutoff``.
    
    Returns
    ----------
        Zero if ``val < cutoff``, otherwise ``slope * (val - cutoff)``.      
    """
    if val < cutoff:
        return 0.0 * val
    else:
        return slope * (val - cutoff)


def zero() -> SQValType:
    """
    Return the value of 0 in the appropriate datatype for the current
    optimization mode.

    Returns
    ----------
        The float 0.0 if not get_optim_mode(), otherwise the tensor value of 0.
    """
    if get_optim_mode():
        return torch.tensor(0.0)

    return 0.0

def detach_if_optim(value: SQValType) -> SQValType:
    """Detach the value if is in torch. Otherwise, return the value itself."""

    if get_optim_mode():
        return value.detach()

    return value
