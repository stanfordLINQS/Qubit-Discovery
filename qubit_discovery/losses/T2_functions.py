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


###############################################################################
# Decoherence rate helper functions
###############################################################################

def partial_squared_omega(
    cr: Circuit,
    grad_el: Element,
    partial_H: qt.Qobj,
    partial_H_squared: qt.Qobj,
    states: Tuple[int, int]
) -> float:
    """ Calculates the second derivative of the difference between the `m`th 
    and `n`th eigenfrequencies with respect to an arbitrary parameter `x` and
    `grad_el`.

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the eigenfrequencies of
        grad_el:
            The element to take the gradient of the frequencies with respect to
        partial_H:
            The derivative of the circuit Hamiltonian with respect to the
            arbitrary parameter.
        partial_H_squared:
            The second derivative of the circuit Hamiltonian with respect to 
            the arbitrary parameter and `grad_el`.
        states:
            The numbers `(m, n)` of the eigenfrequencies to differentiate.
    """

    m, n = states
    state_m = sqf.qutip(cr._evecs[m], dims=cr._get_state_dims())
    partial_state_m = cr.get_partial_vec(grad_el, m)
    state_n = sqf.qutip(cr._evecs[n], dims=cr._get_state_dims())
    partial_state_n = cr.get_partial_vec(grad_el, n)

    p2_omega_1 = 2 * np.real(
        partial_state_m.dag() * (partial_H * state_m)
        - partial_state_n.dag() * (partial_H * state_n)
    )[0, 0]
    p2_omega_2 = (
        state_m.dag() * (partial_H_squared * state_m)
        - state_n.dag() * (partial_H_squared * state_n)
    )[0, 0]

    p2_omega = p2_omega_1 + p2_omega_2
    assert np.imag(p2_omega)/np.real(p2_omega) < 1e-6

    return np.real(p2_omega)


def partial_dephasing_rate(
    A,
    partial_A,
    partial_omega_mn,
    partial_squared_omega_mn
):
    """
    Calculate the derivative of the dephasing rate with noise amplitude `A`
    that depends on `partial_omega_mn`.

    Parameters
    ----------
        A:
            The noise amplitude of the dephasing
        partial_A:
            The derivative of `A` with respect to an external parameter
        partial_omega_mn:
            The derivative of the difference between eigenfrequencies 
            used to calculate the dephasing rate
        partial_squared_omega_mn:
            The derivative of `partial_omega_mn` with respect to an external
            parameter
    """
    return (
        np.sign(partial_omega_mn)
        * np.sqrt(2 * np.abs(np.log(ENV["omega_low"] * ENV["t_exp"])))
        * (partial_A * partial_omega_mn + A * partial_squared_omega_mn)
    )


def get_B_idx(
    cr: Circuit,
    el: Union[Junction, Inductor]
):
    if isinstance(el, Junction):
        for _, el_JJ, B_idx, _ in cr.elem_keys[Junction]:
            if el_JJ is el:
                return B_idx
    elif isinstance(el, Inductor):
        for _, el_ind, B_idx in cr.elem_keys[Inductor]:
            if el_ind is el:
                return B_idx

    return None


###############################################################################
# Charge noise
###############################################################################

def partial_H_ng(
    cr: Circuit,
    charge_idx: int
):
    """ Calculates the  derivative of the Hamiltonian of `cr` with 
    respect to the gate charge on the `charge_idx` charge mode.

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the Hamiltonian of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to
    """
    op = qt.Qobj()
    for j in range(cr.n):
        op += (
            cr.cInvTrans[charge_idx, j]
            * cr._memory_ops["Q"][j]
            / np.sqrt(unt.hbar)
        )
    return op


def partial_squared_H_ng(
    cr: Circuit,
    charge_idx: int,
    grad_el: Union[Capacitor, Inductor, Junction]
):
    """ Calculates the second derivative of the Hamiltonian of `cr` with 
    respect to the gate charge on the `charge_idx` charge mode and `grad_el`

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the Hamiltonian of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to
        grad_el:
            The circuit element to differentiate with respect to
    """
    if not isinstance(grad_el, Capacitor):
        return 0

    cInv = np.linalg.inv(sqf.numpy(cr.C))
    A = cInv @ cr.partial_mats[grad_el] @ cInv

    op = qt.Qobj()
    for j in range(cr.n):
        op += A[charge_idx, j] * cr._memory_ops["Q"][j] / np.sqrt(unt.hbar)
    return -op


def partial_omega_ng(
    cr: Circuit,
    charge_idx: int,
    states: Tuple[int, int]
):
    """ Calculates the derivative of the difference between the `m`th
    and `n`th eigenfrequencies of `cr` with respect to the gate charge of the
    `charge_idx` charge mode.

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the eigenfrequencies of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to
        states:
            The numbers `(m, n)` of the eigenfrequencies to differentiate.
    """
    state_m = sqf.qutip(cr._evecs[states[0]], dims=cr._get_state_dims())
    state_n = sqf.qutip(cr._evecs[states[1]], dims=cr._get_state_dims())
    op = partial_H_ng(cr, charge_idx)

    partial_omega_mn = (
        state_m.dag() * (op * state_m)
        - state_n.dag() * (op * state_n)
    )[0, 0]
    assert np.imag(partial_omega_mn)/np.real(partial_omega_mn) < 1e-5

    return np.real(partial_omega_mn)


def partial_squared_omega_mn_ng(
    cr: Circuit,
    charge_idx: int,
    grad_el: Union[Capacitor, Inductor, Junction],
    states: Tuple[int, int]
):
    """ Calculates the second derivative of the difference between the `m`th 
    and `n`th eigenfrequencies with respect to the gate charge on the
    `charge_idx` mode and `grad_el`.

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the eigenfrequencies of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to
        grad_el:
            The circuit element to differentiate with respect to
        states:
            The numbers `(m, n)` of the eigenfrequencies to differentiate.
    """
    partial_H = partial_H_ng(cr, charge_idx)
    partial_H_squared = partial_squared_H_ng(cr, charge_idx, grad_el)

    return partial_squared_omega(
        cr,
        grad_el,
        partial_H,
        partial_H_squared,
        states
    )


def partial_charge_dec(
    cr: Circuit,
    grad_el: Element,
    states: Tuple[int, int]
):
    """
    Calculate the derivative of the charge decoherence between the states given 
    in `states` with respect to `grad_el`.

    Parameters
    ----------
        cr:
            The `Circuit` object.
        grad_el:
            The circuit element to differentiate with respect to
        states:
            A tuple `(m, n)` of states to consider charge decoherence between.
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
            partial_A = 0

            dec_rate_grad += partial_dephasing_rate(
                A,
                partial_A,
                partial_omega_mn,
                partial_squared_omega_mn
            )

    return dec_rate_grad

###############################################################################
# Critical current noise
###############################################################################


def partial_squared_omega_mn_EJ(
    cr: Circuit,
    EJ_el: Junction,
    B_idx: int,
    grad_el: Union[Capacitor, Inductor, Junction],
    states: Tuple[int, int]
):
    """ Calculates the second derivative of the difference between the `m`th 
    and `n`th eigenfrequencies with respect to `EJ_el` and `grad_el`.

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the eigenfrequencies of
        EJ_el:
            A Josephson junction to differentiate with respect to
        B_idx:
            A number
        grad_el:
            A circuit element to differentiate with respect to
        states:
            The numbers `(m, n)` of the eigenfrequencies to differentiate.
    """
    partial_H = cr._get_partial_H(EJ_el, _B_idx = B_idx)
    partial_H_squared = 0

    return partial_squared_omega(
        cr,
        grad_el,
        partial_H,
        partial_H_squared,
        states
    )


def partial_cc_dec(
    cr: Circuit,
    grad_el: Element,
    states: Tuple[int, int]
):
    """ Calculate the derivative of the critical current decoherence between the 
    states given  in `states` with respect to `grad_el`.

    Parameters
    ----------
        cr:
            The `Circuit` object.
        states:
            A tuple `(m, n)` of states to consider charge decoherence between.
        grad_el:
            The circuit element to differentiate with respect to
    """
    dec_rate_grad = 0
    for EJ_el, B_idx in cr._memory_ops['cos']:
        partial_omega_mn = sqf.numpy(cr._get_partial_omega_mn(
            EJ_el,
            states=states,
            _B_idx=B_idx
        ))
        partial_squared_omega_mn = partial_squared_omega_mn_EJ(
            cr,
            EJ_el,
            B_idx,
            grad_el,
            states
        )
        A = sqf.numpy(EJ_el.A * EJ_el.get_value())
        partial_A = EJ_el.A if grad_el is EJ_el else 0

        dec_rate_grad += partial_dephasing_rate(
            A,
            partial_A,
            partial_omega_mn,
            partial_squared_omega_mn
        )

    return dec_rate_grad


###############################################################################
# Flux noise
###############################################################################

def partial_squared_H_phi(
    cr: Circuit,
    loop: Loop,
    grad_el: Union[Capacitor, Inductor, Junction]
):
    """ Calculates the second derivative of the Hamiltonian of `cr` with 
    respect to the external flux through `loop` and `grad_el`

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the Hamiltonian of
        loop:
            The loop with external flux to differentiate with respect to
        grad_el:
            The circuit element to differentiate with respect to
    """
    if isinstance(grad_el, Capacitor):
        return 0

    loop_idx = cr.loops.index(loop)
    B_idx = get_B_idx(cr, grad_el)

    if isinstance(grad_el, Junction):
        return cr.B[B_idx, loop_idx] * cr._memory_ops['sin'][(grad_el, B_idx)]
    elif isinstance(grad_el, Inductor):
        return (
            cr.B[B_idx, loop_idx]
            / -sqf.numpy(grad_el.get_value()**2)
            * unt.Phi0 / np.sqrt(unt.hbar) / 2 / np.pi
            * cr._memory_ops["ind_hamil"][(grad_el, B_idx)]
        )
    else:
        raise NotImplementedError


def partial_squared_omega_mn_phi(
    cr: Circuit,
    loop: Loop,
    grad_el: Union[Capacitor, Inductor, Junction],
    states: Tuple[int, int]
):
    """ Calculate the second derivative of the difference between the `m`th 
    and `n`th eigenfrequencies with respect to `loop` and `grad_el`.

    Parameters
    ----------
        cr:
            The `Circuit` object to differentiate the eigenfrequencies of
        loop:
            A loop in `cr` to differentiate with respect to
        grad_el:
            A circuit element to differentiate with respect to
        states:
            The numbers `(m, n)` of the eigenfrequencies to differentiate.
    """
    partial_H = cr._get_partial_H(loop)
    partial_H_squared = partial_squared_H_phi(cr, loop, grad_el)

    return partial_squared_omega(
        cr,
        grad_el,
        partial_H,
        partial_H_squared,
        states
    )


def partial_flux_dec(
    cr: Circuit,
    grad_el: Element,
    states: Tuple[int, int]
):
    """ Calculate the derivative of the flux decoherence between the states 
    given in `states` with respect to `grad_el`.

    Parameters
    ----------
        cr:
            The `Circuit` object.
        states:
            A tuple `(m, n)` of states to consider charge decoherence between.
        grad_el:
            The circuit element to differentiate with respect to
    """
    dec_rate_grad = 0
    for loop in cr.loops:
        partial_omega_mn = sqf.numpy(cr._get_partial_omega_mn(
            loop,
            states=states
        ))
        partial_squared_omega_mn = partial_squared_omega_mn_phi(
            cr,
            loop,
            grad_el,
            states
        )

        A = loop.A
        partial_A = 0

        dec_rate_grad += partial_dephasing_rate(
            A,
            partial_A,
            partial_omega_mn,
            partial_squared_omega_mn
        )
    return dec_rate_grad


###############################################################################
# Torch nodes
###############################################################################

def dec_rate_cc_torch(circuit: Circuit, states: Tuple[int, int]):
    return DecRateCC.apply(
        torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]),
        circuit,
        states
    )


def dec_rate_charge_torch(circuit: Circuit, states: Tuple[int, int]):
    return DecRateCharge.apply(
        torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]),
        circuit,
        states
    )


def dec_rate_flux_torch(circuit: Circuit, states: Tuple[int, int]):
    return DecRateFlux.apply(
        torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]),
        circuit,
        states
    )


class DecRateCC(Function):
    @staticmethod
    def forward(
        element_tensors: Tensor,
        circuit: 'Circuit',
        states: Tuple[int, int]
    ) -> Tensor:

        return torch.as_tensor(circuit._dec_rate_cc_np(states))

    @staticmethod
    def setup_context(ctx, inputs, output):
        circuit_parameters, circuit, states = inputs

        ctx.circuit = circuit.safecopy(save_eigs=True)
        ctx.states = states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor, None, None]:
        output_grad = torch.zeros(len(ctx.circuit._parameters))

        for idx, elem in enumerate(ctx.circuit._parameters.keys()):
            output_grad[idx] = grad_output * partial_cc_dec(
                ctx.circuit,
                elem,
                ctx.states
            )

        return output_grad, None, None


class DecRateCharge(Function):
    @staticmethod
    def forward(
        circuit_parameters: Tensor,
        circuit: 'Circuit',
        states: Tuple[int, int]
    ) -> Tensor:
        return torch.as_tensor(circuit._dec_rate_charge_np(states))

    @staticmethod
    def setup_context(ctx, inputs, output):
        circuit_parameters, circuit, states = inputs

        ctx.circuit = circuit.safecopy(save_eigs=True)
        ctx.states = states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor, None, None]:
        output_grad = torch.zeros(len(ctx.circuit._parameters))

        for idx, elem in enumerate(ctx.circuit._parameters.keys()):
            output_grad[idx] = grad_output * partial_charge_dec(
                ctx.circuit,
                elem,
                ctx.states
            )

        return output_grad, None, None


class DecRateFlux(Function):
    @staticmethod
    def forward(
        circuit_parameters: Tensor,
        circuit: 'Circuit',
        states: Tuple[int, int]
    ) -> Tensor:
        return torch.as_tensor(circuit._dec_rate_flux_np(states))

    @staticmethod
    def setup_context(ctx, inputs, output):
        circuit_parameters, circuit, states = inputs

        ctx.circuit = circuit.safecopy(save_eigs=True)
        ctx.states = states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor, None, None]:
        output_grad = torch.zeros(len(ctx.circuit._parameters))

        for idx, elem in enumerate(ctx.circuit._parameters.keys()):
            output_grad[idx] = grad_output * partial_flux_dec(
                ctx.circuit,
                elem,
                ctx.states
            )

        return output_grad, None, None
