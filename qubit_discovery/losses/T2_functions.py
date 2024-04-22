from typing import Tuple, Union

import numpy as np
import qutip as qt
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from SQcircuit import Capacitor, Circuit, Element, Inductor, Junction
from SQcircuit.noise import ENV
from SQcircuit import functions as sqf
from SQcircuit import units as unt

def partial_squared_omega_mn_EJ(
    cr: Circuit,
    EJ_el: Junction,
    grad_el: Element,
    B_idx: int,
    states: Tuple[int, int]
):
    partial_H = cr._get_partial_H(EJ_el, _B_idx = B_idx)

    m, n = states
    state_m = sqf.qutip(cr._evecs[m], dims=cr._get_state_dims())
    partial_state_m = cr.get_partial_vec(grad_el, m)
    state_n = sqf.qutip(cr._evecs[n], dims=cr._get_state_dims())
    partial_state_n = cr.get_partial_vec(grad_el, n)

    return 2 * np.real(
        partial_state_m.dag() * (partial_H * state_m) \
        - partial_state_n.dag() * (partial_H * state_n)
    )[0][0]


def partial_cc_dec(
    cr: Circuit,
    grad_el: Element,
    states: Tuple[int, int]
):
    dec_rate_grad = 0
    for EJ_el, B_idx in cr._memory_ops['cos']:
        partial_omega_mn = cr._get_partial_omega_mn(EJ_el, states=states, _B_idx=B_idx)

        partial_squared_omega_mn = partial_squared_omega_mn_EJ(cr, EJ_el, grad_el, B_idx, states)

        partial_A = EJ_el.A if grad_el is EJ_el else 0
        dec_rate_grad += (np.sign(partial_omega_mn) 
                          * np.sqrt(2 * np.abs(np.log(ENV["omega_low"] * ENV["t_exp"]))) 
                          * (partial_A * partial_omega_mn
                             + EJ_el.A * EJ_el.get_value() * partial_squared_omega_mn
                            )
                         )

    return dec_rate_grad


def DecRateFlux(Function):
    @staticmethod
    def forward(ctx, 
                element_tensors: Tensor,
                circuit: 'Circuit',
                n_eig: int) -> Tensor:
        pass

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor]:
        pass

def DecRateCharge(Function):
    @staticmethod
    def forward(ctx, 
                element_tensors: Tensor,
                circuit: 'Circuit',
                state1: np.ndarray,
                state2: np.ndarray) -> Tensor:
        pass

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor]:
        pass

def DecRateCC(Function):
    @staticmethod
    def forward(ctx, 
                element_tensors: Tensor,
                circuit: 'Circuit',
                n_eig: int) -> Tensor:
        pass

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor]:
        pass