"""Contains code for defining loss functions used in circuit optimization."""
from typing import Callable, Dict, List, Optional, Tuple

from SQcircuit import Circuit
from SQcircuit.settings import get_optim_mode
import torch

from .functions import (
    charge_sensitivity,
    flux_sensitivity,
    element_sensitivity,
    first_resonant_frequency,
    reset_charge_modes,
    zero,
    decoherence_time,
    total_dec_time,
    fastest_gate_speed,
    number_of_gates
)
from .utils import (
    detach_if_optim,
    dotdict,
    hinge_loss,
    SQValType
)

# Added on to a loss when it is close to zero, so that loss values are never
# exactly zero (to allow logarithms, etc.)
EPSILON = 1e-13

def t1_loss(
    circuit: Circuit,
    dec_type: str = 'total',
) -> Tuple[SQValType, SQValType]:
    """Penalizes short T1 time.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.
        dec_type:
            A decoherence channel to consider, or ``'total'`` to use all
            available.

    Returns
    ----------
        loss:
            ``-T1``.
        T1:
            The T1 time of ``circuit``.
    """
    t1 = decoherence_time(
        circuit=circuit,
        t_type='t1',
        dec_type=dec_type
    )

    return -t1 + EPSILON, t1


def t_phi_loss(
        circuit: Circuit,
        dec_type='total'
) -> Tuple[SQValType, SQValType]:
    """Penalizes short dephasing time.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.
        dec_type:
            A dephasing channel to consider, or ``'total'`` to use all
            available.

    Returns
    ----------
        loss:
            ``-T_phi``.
        T_phi:
            The dephasing time of ``circuit``.
    """
    t_phi = decoherence_time(
        circuit=circuit,
        t_type='t_phi',
        dec_type=dec_type
    )

    return -t_phi + EPSILON, t_phi


def t2_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    """Penalizes short T2 time, computed via all available decay channels.
    See ``functions.total_dec_time`` for a description of the decay channels
    considered.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.s

    Returns
    ----------
        loss:
            ``-T2``.
        T2:
            The T2 time of ``circuit``.
    """
    t2 = total_dec_time(circuit)

    return -t2 + EPSILON, t2


def element_sensitivity_loss(
    circuit: Circuit,
    n_samples=100,
    error=0.01,
) -> Tuple[SQValType, SQValType]:
    """Computes the sensitivity of the single-qubit gate number of the qubit
    with respect to variation in the element values. See
    ``functions.element_sensitivity`` for more details on how this is computed.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.
        n_samples:
            The number of randomly sampled circuits to calculate. More samples
            provide more reproducible results.
        fabrication_error:
            The percentage fabrication error to simulate.

    Returns
    ----------
        loss:
            The sensitivity ``sens``.
        sensitivity:
            The sensitivity of the gate number to element variation.
    """
    sens = element_sensitivity(circuit, number_of_gates, n_samples, error)

    return sens + EPSILON, sens


def gate_speed_loss(circuit: Circuit) -> Tuple[SQValType, SQValType]:
    """Penalizes a slow single-qubit gate speed for the circuit. See
    ``functions.fastest_gate_speed`` for more details on how this is computed.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.

    Returns
    ----------
        loss:
            1 / ``gate_speed``.
        gate_speed:
            The gate speed of ``circuit``.
    
    """
    gate_speed = fastest_gate_speed(circuit)

    return (1 / gate_speed) + EPSILON, gate_speed


def anharmonicity_loss(
    circuit: Circuit
) -> Tuple[SQValType, SQValType]:
    """Computes a loss to penalize low anharmonicitiy. The absolute anharmonicity is
        ``A = omega_{21} - omega_{10}``
    and the relative anharmonicity is
        ``Ar = (omega_{21} - omega_{10})/omega_{10}``.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.

    Returns
    ----------
        loss:
            1 / Ar, the reciprocal of the relative anharmonicity.
        anharmonicity:
            A, the absolute anharmonicity.
    """

    omega_10 = circuit.efreqs[1] - circuit.efreqs[0]
    omega_21 = circuit.efreqs[2] - circuit.efreqs[1]

    A = omega_21 - omega_10

    return (omega_10 / A) + EPSILON, A


def frequency_loss(
        circuit: Circuit,
        freq_threshold: float = 100.0,
) -> Tuple[SQValType, SQValType]:
    """Loss function which applies a quadratic hinge loss to qubits whose
    frequency exceeds ``freq_threshold``. When ``f_{10} < freq_threshold``, the
    loss is zero, and otherwise it is ``(f_{10} - freq_threshold)**2``.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.
        freq_thresholdD:
            The frequency threshold (in the frequency unit of SQcircuit)
            above which to apply the quadratic hinge loss. Default 100 GHz.

    Returns
    ----------
        loss:
            The hinge loss applied to ``f``.
        f:
            The qubit frequency of ``circuit``.
     """
    freq = first_resonant_frequency(circuit)
    if freq > freq_threshold:
        loss = (freq - freq_threshold)**2
    else:
        loss = zero()
    return loss + EPSILON, freq


def flux_sensitivity_loss(
    circuit: Circuit,
    a=0.1,
    b=1,
) -> Tuple[SQValType, SQValType]:
    """Applies a hinge loss to flux sensitivity of circuit. The flux sensitivity
    is the normalized variation in qubit frequency for small perturbations of
    the external flux about the operating point. See ``functions.flux_sensitivity``
    for more details.
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.
        a:
            The cutoff above which to apply the hinge loss.
        b:
            The slope for the hinge loss.

    Returns
    ----------
        loss:
            The hinge loss applied to ``S``. 
        S:
            The flux sensitivity of ``circuit``.
    """

    sens = flux_sensitivity(circuit)

    return hinge_loss(sens, a, b) + EPSILON, sens


def charge_sensitivity_loss(
    circuit: Circuit,
    a=0.02,
    b=1,
) -> Tuple[SQValType, SQValType]:
    """Applies a linear hinge loss to charge sensitivity of circuit. See
    ``functions.charge_sensitivity`` for details on how the charge sensitivity
    is calculated.
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.
        a:
            The cutoff above which to apply the hinge loss.
        b:
            The slope for the hinge loss.

    Returns
    ----------
        loss:
            The hinge loss applied to ``S``. 
        S:
            The charge sensitivity of ``circuit``.
    """

    sens = charge_sensitivity(circuit)

    reset_charge_modes(circuit)
    return hinge_loss(sens, a, b) + EPSILON, sens


def number_of_gates_loss(
    circuit: Circuit,
) -> Tuple[SQValType, SQValType]:
    """Applies a loss ``1/N`` to the number of single-qubit gates ``N`` for the
    circuit. See ``functions.number_of_gates`` for details on how the number
    of gates is calculated.
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object specifying the qubit.

    Returns
    ----------
        loss:
            The loss ``1/N``.
        N:
            The number of single-qubit gates.
    """

    N = number_of_gates(circuit)

    loss = 1 / N

    return loss, N


###############################################################################
# Incorporating all losses into one loss function
###############################################################################

ALL_METRICS = dotdict({
    ###########################################################################
    'frequency': frequency_loss,
    'flux_sensitivity': flux_sensitivity_loss,
    'charge_sensitivity': charge_sensitivity_loss,
    'number_of_gates': number_of_gates_loss,
    ###########################################################################
    'anharmonicity': anharmonicity_loss,
    'element_sensitivity': element_sensitivity_loss,
    'gate_speed': gate_speed_loss,
    't2': t2_loss,
    't1': t1_loss,
    't1_capacitive': lambda cr: t1_loss(cr, dec_type='capacitive'),
    't1_inductive': lambda cr: t1_loss(cr, dec_type='inductive'),
    't1_quasiparticle': lambda cr: t1_loss(cr, dec_type='quasiparticle'),
    't_phi': t_phi_loss,
    't_phi_charge': lambda cr: t_phi_loss(cr, dec_type='charge'),
    't_phi_cc': lambda cr: t_phi_loss(cr, dec_type='cc'),
    't_phi_flux': lambda cr: t_phi_loss(cr, dec_type='flux')
})


def get_all_metrics() -> List[str]:
    """
    Provides a list of all available metrics which can be calculated for a
    circuit or used to construct a loss function. Each function can be accessed
    via ``ALL_METRICS[key]``, where ``key`` is any string in the returned
    list.

    Returns
    ----------
        List of names of available metrics.
    """
    return list(ALL_METRICS.keys())


def describe_metric(name: str) -> None:
    """
    Provides a description of a given metric function (if available).

    Parameters
    ----------
        Name of metric.
    """
    descrip = ALL_METRICS[name].__doc__

    if descrip is None:
        print('No description available.')
    else:
        print(descrip)


def add_to_metrics(
    name: str,
    function: Callable[[Circuit], Tuple[SQValType, SQValType]]
) -> None:
    """Add a function to the available metrics to construct a loss function.

    Parameters
    ----------
        name:
            Name of new metric.
        function:
            Function computing the metric and loss.
    """
    ALL_METRICS[name] = function


LossFunctionRetType = Tuple[SQValType, Dict[str, SQValType], Dict[str, SQValType]]
LossFunctionType = Callable[
    [Circuit],
    LossFunctionRetType
]


def build_loss_function(
    use_losses: Dict[str, float],
    use_metrics: Optional[List[str]] = None,
    master_use_grad: bool = True
) -> LossFunctionType:
    """Build a loss function based on the metrics provided in ``use_losses``.
    The constructed loss function will also calculate metrics in
    ``use_metrics``.
    
    The loss function takes in a circuit and returns
        - Total loss: the sum of metrics in ``use_losses``.
        - Loss values: A dictionary of ``metric: value`` for metrics in
        ``use_losses``.
        - Metric values: A dictionary of ``metric: value`` for metrics in
        ``use_metrics``.

    Parameters
    ----------
        use_losses:
            A dictionary of ``metric: weight`` pairs, where ``metric`` is the 
            name of a metric returned by ``get_all_metrics()`` and ``weight`` 
            is the weight in the total loss function.
        use_metrics:
            A list of metrics to calculate, where the available metrics are 
            provided in the list returned by ``get_all_metrics()``.
        master_use_grad:
            Whether to enable gradient when calculating the total loss.

    Returns
    ----------
        A loss function which computes the total loss using ``use_losses`` as
        well as the metrics in ``use_metrics``.
    """
    if use_metrics is None:
        use_metrics = []

    if not isinstance(use_losses, dict):
        raise ValueError('You must pass in dictionary of metric_name: weight'
                         ' for `use_losses`.')
    for metric_name in use_losses.keys():
        if metric_name not in get_all_metrics():
            raise ValueError(f'The metric \'{metric_name}\' is not available.')
    if not isinstance(use_metrics, list):
        raise ValueError('You must pass in list of metrics for `use_metrics`.')
    for metric_name in use_metrics:
        if metric_name not in get_all_metrics():
            raise ValueError(f'The metric \'{metric_name}\' is not available.')

    return lambda circuit: calculate_loss_metrics(
        circuit,
        use_losses,
        use_metrics,
        master_use_grad=master_use_grad
    )


def calculate_loss_metrics(
    circuit: Circuit,
    use_losses: Dict[str, float],
    use_metrics: List[str],
    master_use_grad: bool = True,
) -> LossFunctionRetType:
    """Calculates an overall loss function for ``circuit``, based on metrics 
    provided in ``use_losses``. Additionally calculates other metrics provided
    in ``use_metrics``, but does not include them in the overall loss.

    Parameters
    ----------
        circuit:
            The circuit to calculate the loss/metrics for.
        use_losses:
            A dictionary of ``metric: weight`` pairs, where ``metric`` is the 
            name of a metric returned by ``get_all_metrics()`` and ``weight`` 
            is the weight in the overall loss function.
        use_metrics:
            A list of metrics to calculate, where the available metrics are 
            provided in the list returned by ``get_all_metrics()``.
        master_use_grad:
            Whether to enable gradient when calculating metrics to use in
            the overall loss.

    Returns
    ----------
        total_loss:
            The total loss function, which is ``sum(weight*metric)`` over 
            the metrics in ``use_losses``.
        loss_values
            A dictionary of ``metric: value`` for each metric in
            ``use_losses``.
        metric_values
            A dictionary of ``metric: value`` for each metric in
            ``use_metrics``.
    """

    if get_optim_mode():
        loss = torch.zeros((), requires_grad=master_use_grad)
    else:
        loss = 0.0

    loss_values: Dict[str, SQValType] = {}
    metrics: Dict[str, SQValType] = {}

    try:
        for key in use_losses:
            with torch.set_grad_enabled(master_use_grad):
                specific_loss, specific_metric = ALL_METRICS[key](circuit)
                loss = loss + use_losses[key] * specific_loss
            with torch.no_grad():
                loss_values[key + '_loss'] = detach_if_optim(specific_loss)
                metrics[key] = detach_if_optim(specific_metric)
        for key in use_metrics:
            if key not in use_losses:
                with torch.no_grad():
                    specific_loss, specific_metric = ALL_METRICS[key](circuit)

                    loss_values[key + '_loss'] = detach_if_optim(specific_loss)
                    metrics[key] = detach_if_optim(specific_metric)
    except KeyError as e:
        raise ValueError(f'The metric {e} is not available!') from None

    loss_values['total_loss'] = detach_if_optim(loss)

    return loss, loss_values, metrics
