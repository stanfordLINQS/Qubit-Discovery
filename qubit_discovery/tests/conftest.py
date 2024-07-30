"""Module contains general configuration and functions for
qubit_discovery tests.
"""

from typing import Dict

import torch
import numpy as np

from torch import Tensor

import SQcircuit as sq
from SQcircuit.settings import get_optim_mode


def get_fluxonium() -> sq.Circuit:
    """Returns a Fluxonium qubit for test purposes."""

    loop = sq.Loop(0.5)
    cap = sq.Capacitor(
        1.0, requires_grad=get_optim_mode()
    )
    ind = sq.Inductor(
        1.0, loops=[loop], requires_grad=get_optim_mode()
    )
    junc = sq.Junction(
        1.0, loops=[loop], requires_grad=get_optim_mode()
    )

    circuit = sq.Circuit(
        elements={
            (0, 1): [cap, ind, junc]
        }
    )

    return circuit

def get_fluxonium_random() -> sq.Circuit:
    """Returns a Fluxonium qubit for test purposes."""

    loop = sq.Loop(0.5)
    cap = sq.Capacitor(
        np.random.normal(1.0, 0.1), requires_grad=get_optim_mode()
    )
    ind = sq.Inductor(
        np.random.normal(1.0, 0.1), loops=[loop], requires_grad=get_optim_mode()
    )
    junc = sq.Junction(
        np.random.normal(1.0, 0.1), loops=[loop], requires_grad=get_optim_mode()
    )

    circuit = sq.Circuit(
        elements={
            (0, 1): [cap, ind, junc]
        }
    )

    return circuit

def get_cpb() -> sq.Circuit:
    """Returns a Transmon qubit for test purposes."""

    loop = sq.Loop(0.5)
    cap = sq.Capacitor(
        1.2, 'GHz', requires_grad=get_optim_mode()
    )
    junc = sq.Junction(
        0.2, 'GHz', loops=[loop], requires_grad=get_optim_mode()
    )

    circuit = sq.Circuit(
        elements={
            (0, 1): [cap, junc]
        }
    )

    return circuit


def get_bounds() -> Dict[sq.Element, Tensor]:
    """Returns element bounds for test purposes."""

    bounds = {
        sq.Capacitor: torch.tensor([1e-15, 12e-12]),
        sq.Inductor: torch.tensor([1e-15, 5e-6]),
        sq.Junction: torch.tensor([1e9, 100e9]),
        sq.Loop: torch.tensor([0, 1])
    }

    return bounds


def are_loss_dicts_close(dict1, dict2, rel: float = 1e-2) -> bool:
    """Check if two loss dictionaries are close in values
     within a given tolerance."""

    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        for i in range(len(dict1[key])):
            if not np.isclose(dict1[key][i], dict2[key][i], rtol=rel):
                return False

    return True
