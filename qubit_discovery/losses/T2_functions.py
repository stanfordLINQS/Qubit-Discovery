from typing import Tuple, Union

import numpy as np
import qutip as qt
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from SQcircuit import Capacitor, Circuit, Element, Inductor, Junction, Loop
from SQcircuit.noise import ENV
from SQcircuit import functions as sqf
from SQcircuit import units as unt

################################################################################
# Charge Noise
################################################################################


def partial_H_ng(
    cr: Circuit,
    charge_idx: int
):
    op = qt.Qobj()
    for j in range(cr.n):
        op += (
            cr.cInvTrans[charge_idx, j]
            * cr._memory_ops["Q"][j]
            / np.sqrt(unt.hbar)
        )
    return -op


def partial_omega_ng(
    cr: Circuit,
    charge_idx: int,
    states: Tuple[int, int]
):

    state1 = cr._evecs[states[0]]
    state2 = cr._evecs[states[1]]
    op = partial_H_ng(cr, charge_idx)
    return (
        sqf.operator_inner_product(state2, op, state2)
        - sqf.operator_inner_product(state1, op, state1)
    )


def partial_squared_H_ng(
    cr: Circuit,
    charge_idx: int,
    grad_el: Union[Capacitor, Inductor, Junction]
):
    if type(grad_el) is not Capacitor:
        return 0

    cInv = np.linalg.inv(sqf.numpy(cr.C))
    A = cInv @ cr.partial_mats[grad_el] @ cInv

    op = qt.Qobj()
    for j in range(cr.n):
        op += A[charge_idx, j] * cr._memory_ops["Q"][j] / np.sqrt(unt.hbar)
    return op


def partial_squared_omega_mn_ng(
    cr: Circuit,
    charge_idx: int,
    grad_el: Union[Capacitor, Inductor, Junction],
    states: Tuple[int, int]
):
    partial_H = partial_H_ng(cr, charge_idx)
    partial_H_squared = partial_squared_H_ng(cr, charge_idx, grad_el)

    m, n = states
    state_m = sqf.qutip(cr._evecs[m], dims=cr._get_state_dims())
    partial_state_m = cr.get_partial_vec(grad_el, m)
    state_n = sqf.qutip(cr._evecs[n], dims=cr._get_state_dims())
    partial_state_n = cr.get_partial_vec(grad_el, n)

    p2_omega_1 = 2 * np.real(
        partial_state_m.dag() * (partial_H * state_m)
        - partial_state_n.dag() * (partial_H * state_n)
    )[0][0]
    p2_omega_2 = (
        state_m.dag() * (partial_H_squared * state_m)
        - state_n.dag() * (partial_H_squared * state_n)
    )[0][0]

    p2_omega = p2_omega_1 + p2_omega_2
    
    assert np.imag(p2_omega)/np.real(p2_omega) < 1e-6
    return np.real(p2_omega)


def partial_charge_dec(
    cr: Circuit,
    grad_el: Element,
    states: Tuple[int, int]
):
    """
    The A is independent of all elements
    """
    dec_rate_grad = 0
    for i in range(cr.n):
        if cr._is_charge_mode(i):
            partial_omega_mn = partial_omega_ng(cr, i, states)

            partial_squared_omega_mn = partial_squared_omega_mn_ng(
                cr=cr,
                charge_idx=i,
                grad_el=grad_el,
                states=states
            )

            A = cr.charge_islands[i].A * 2 * unt.e
            dec_rate_grad += (
                np.sign(partial_omega_mn)
                * np.sqrt(2 * np.abs(np.log(ENV["omega_low"] * ENV["t_exp"])))
                * A * partial_squared_omega_mn
            )

    return dec_rate_grad

################################################################################
# Critical Current Noise
################################################################################


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
        partial_state_m.dag() * (partial_H * state_m)
        - partial_state_n.dag() * (partial_H * state_n)
    )[0][0]


def partial_cc_dec(
    cr: Circuit,
    grad_el: Element,
    states: Tuple[int, int]
):
    dec_rate_grad = 0
    for EJ_el, B_idx in cr._memory_ops['cos']:

        partial_omega_mn = cr._get_partial_omega_mn(
            EJ_el,
            states=states,
            _B_idx=B_idx
        )

        partial_squared_omega_mn = partial_squared_omega_mn_EJ(
            cr,
            EJ_el,
            grad_el,
            B_idx,
            states
        )

        partial_A = EJ_el.A if grad_el is EJ_el else 0
        dec_rate_grad += (
            np.sign(partial_omega_mn)
            * np.sqrt(2 * np.abs(np.log(ENV["omega_low"] * ENV["t_exp"])))
            * (
                partial_A * partial_omega_mn
                + EJ_el.A * EJ_el.get_value() * partial_squared_omega_mn
            )
        )

    return dec_rate_grad

################################################################################
# Flux Noise
################################################################################


def get_B_idx(
    cr: Circuit,
    el: Union[Junction, Inductor]
):
    if type(el) is Junction:
        for edge, el_JJ, B_idx, W_idx in cr.elem_keys[Junction]:
            if el_JJ is el:
                return B_idx
    elif type(el) is Inductor:
        for edge, el_ind, B_idx in cr.elem_keys[Inductor]:
            if el_ind is el:
                return B_idx

    return None


def partial_squared_H_phi(
    cr: Circuit,
    loop: Loop,
    grad_el: Union[Capacitor, Inductor, Junction]
):
    if type(grad_el) is Capacitor:
        return 0

    loop_idx = cr.loops.index(loop)
    B_idx = get_B_idx(cr, grad_el)

    if type(grad_el) is Junction:
        return cr.B[B_idx, loop_idx] * cr._memory_ops['sin'][(grad_el, B_idx)]
    elif type(grad_el) is Inductor:
        return (
            cr.B[B_idx, loop_idx]
            / -sqf.numpy(grad_el.get_value()**2)
            * unt.Phi0 / np.sqrt(unt.hbar) / 2 / np.pi
            * cr._memory_ops["ind_hamil"][(grad_el, B_idx)]
        )


def partial_squared_omega_mn_phi(
    cr: Circuit,
    loop: Loop,
    grad_el: Union[Capacitor, Inductor, Junction],
    states: Tuple[int, int]
):
    partial_H = cr._get_partial_H(loop)
    partial_H_squared = partial_squared_H_phi(cr, loop, grad_el)

    m, n = states
    state_m = sqf.qutip(cr._evecs[m], dims=cr._get_state_dims())
    partial_state_m = cr.get_partial_vec(grad_el, m)
    state_n = sqf.qutip(cr._evecs[n], dims=cr._get_state_dims())
    partial_state_n = cr.get_partial_vec(grad_el, n)

    p2_omega_1 = 2 * np.real(
        partial_state_m.dag() * (partial_H * state_m)
        - partial_state_n.dag() * (partial_H * state_n)
    )[0][0]
    p2_omega_2 = (
        state_m.dag() * (partial_H_squared * state_m)
        - state_n.dag() * (partial_H_squared * state_n)
    )[0][0]

    p2_omega = p2_omega_1 + p2_omega_2
    assert np.imag(p2_omega)/np.real(p2_omega) < 1e-6

    return np.real(p2_omega)


def partial_flux_dec(
    cr: Circuit,
    grad_el: Element,
    states: Tuple[int, int]
):
    """The A is independent of all elements
    """
    dec_rate_grad = 0
    for loop in cr.loops:
        partial_omega_mn = cr._get_partial_omega_mn(loop, states=states)

        partial_squared_omega_mn = partial_squared_omega_mn_phi(
            cr,
            loop,
            grad_el,
            states
        )

        A = loop.A
        dec_rate_grad += (
            np.sign(partial_omega_mn)
            * np.sqrt(2 * np.abs(np.log(ENV["omega_low"] * ENV["t_exp"])))
            * A * partial_squared_omega_mn
        )

    return dec_rate_grad


################################################################################
# Torch Nodes
################################################################################

def DecRateFlux(Function):
    @staticmethod
    def forward(
        ctx, 
        element_tensors: Tensor,
        circuit: 'Circuit',
        n_eig: int
    ) -> Tensor:
        pass

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor]:
        pass

def DecRateCharge(Function):
    @staticmethod
    def forward(
        ctx, 
        element_tensors: Tensor,
        circuit: 'Circuit',
        state1: np.ndarray,
        state2: np.ndarray
    ) -> Tensor:
        pass

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor]:
        pass

def DecRateCC(Function):
    @staticmethod
    def forward(
        ctx, 
        element_tensors: Tensor,
        circuit: 'Circuit'
    ) -> Tensor:
        pass

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor]:
        pass