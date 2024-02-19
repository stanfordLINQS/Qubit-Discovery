"""Contains helper functions for estimating optimal truncation numbers"""

from typing import Tuple, List, Optional

import numpy as np
import scipy, scipy.signal
from matplotlib.axes import Axes

import SQcircuit.functions as sqf
from SQcircuit import get_optim_mode, Circuit


def get_reshaped_eigvec(
    circuit: Circuit,
    eig_vec_idx: int,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Return the eigenvec, index1_eigenvec and index2_eigenvec part of
    the eigenvectors.
    """

    assert len(circuit._efreqs) != 0, "circuit should be diagonalized first."

    # Reshape eigenvector dimensions to correspond to individual modes
    if get_optim_mode():
        eigenvector = np.array(circuit._evecs[eig_vec_idx].detach().numpy())
    else:
        eigenvector = circuit._evecs[eig_vec_idx].full()
    eigenvector_reshaped = np.reshape(eigenvector, circuit.m)

    if len(circuit.m) == 1:
        eigvec_mag = np.abs(eigenvector) ** 2
        return eigvec_mag, (eigvec_mag, )
    elif len(circuit.m) == 2:
        # Extract maximum magnitudes of eigenvector entries along each mode axis
        mode_2_magnitudes = np.max(np.abs(eigenvector_reshaped) ** 2, axis=1)
        offset_idx = np.argmax(mode_2_magnitudes)
        mode_1_magnitudes = np.abs(eigenvector_reshaped[offset_idx, :]) ** 2
        eigvec_mag = np.abs(eigenvector) ** 2
        return eigvec_mag, (mode_1_magnitudes, mode_2_magnitudes)
    else:
        raise NotImplementedError

def fit_mode(
    mode_vector,
    num_points=15,
    peak_height_threshold=5e-3,
    axis: Optional[Axes]=None,
    both_parities=False
) -> List[Tuple[float, float, int]]:
    """
    For an input vector corresponding to the absolute eigenvector magnitudes
    for a specific harmonic mode, return the decay constant for an exponential
    decay fit starting with the last peak of the input.

    To adjust for patterns of variation between odd and even indices,
    the fit function matches to even/odd parity indices separately and
    returns the fit parameters corresponding to each parity as a 2-element list.

    If both_parities is false, returns the fit corresponding to those indices
    with the same parity as the maximum element in the list.
    """

    def monoExp(x, m, k):
        return m * np.exp(-k * x)

    fit_results = []
    if both_parities:
        parities = [0, 1]
        fit_peak_idxs = []
    else:
        parities = [np.argmax(mode_vector) % 2, ]
    for parity in parities:
        fit_points_parity = mode_vector[parity::2]

        peaks, _ = scipy.signal.find_peaks(
            # fit_points_parity[:-(num_points-1)],
            fit_points_parity,
            height=peak_height_threshold
        )
        peaks = np.insert(peaks, 0, 0)

        fit_peak_idx = sorted(peaks)[-1] * 2 + parity

        fit_points = mode_vector[fit_peak_idx:: 2]
        num_points = len(fit_points)
        fit_range = np.arange(0, 2 * num_points, 2)
        fit_param_guess = (fit_points[0], -np.log(fit_points[-1]) / num_points)

        try:
            params, _ = scipy.optimize.curve_fit(
                monoExp,
                fit_range,
                fit_points,
                fit_param_guess
            )
        except:
            params = fit_param_guess

        m, k = params
        fit_results.append((k, m, fit_peak_idx))

    # Plot slowest decaying curve out of the two parities
    if both_parities:
        #   k1, k2 = fit_results[0][0], fit_results[1][0]
        #   k = np.minimum(k1, k2)
        #   m = fit_results[0][1] if k1 < k2 else fit_results[1][1]
        k, m, peak_idx = get_slow_fit(fit_results)
    else:
        k, m, peak_idx = fit_results[0]
    if axis:
        axis.plot(peak_idx + fit_range, monoExp(fit_range, m, k), 'k')

    return fit_results

def get_slow_fit(
    fit_results, 
    ignore_threshold=1e-5
) -> Tuple[float, float, float]:

    if len(fit_results) == 1:
        return fit_results[0]

    else:
        k1, m1, peak1 = fit_results[0]
        k2, m2, peak2 = fit_results[1]

        if m1 < ignore_threshold:
            return k2, m2, peak2
        elif m2 < ignore_threshold:
            return k1, m1, peak1
        elif k1 < k2:
            return k1, m1, peak1
        else:
            return k2, m2, peak2

def trunc_num_heuristic(
    circuit: Circuit,
    eig_vec_idx: int = 0,
    K: int=1000,
    min_trunc: int=4,
    axes: Optional[Axes]=None
) -> List[int]:
    """
    For a diagonalized circuit with internal trunc numbers, suggests a set of
    trunc numbers for rediagonalization that will maximize likelihood of
    convergence, defined under the `convergence_test` function.
    """
    assert len(circuit._efreqs) != 0, "Circuit should be diagonalized first"

    trunc_nums = circuit.m

    # Ensure each mode has at least `min_trunc` by allocating truncation
    # numbers as proportion of `min_trunc`
    K = int(K / min_trunc ** 2)

    _, (mode_1_magnitudes, mode_2_magnitudes) = get_reshaped_eigvec(
        circuit,
        eig_vec_idx,
    )

    axis_0, axis_1 = None, None
    if axes is not None:
        assert len(axes) == 2, "Should have one axis for each mode"
        axis_0 = axes[0]
        axis_1 = axes[1]
    k1, _, _ = fit_mode(mode_1_magnitudes, axis=axis_0, both_parities=False)[0]

    fit_results = fit_mode(mode_2_magnitudes, axis=axis_1, both_parities=True)
    # k_even, k_odd = fit_results[0][0], fit_results[1][0]
    # k2 = np.minimum(k_even, k_odd)
    k2, _, _ = get_slow_fit(fit_results)

    # If fit outputs negative decay constant, set decay rates/trunc nums equal
    if k1 < 0 and k2 < 0:
        k1 = k2 = 1
    if k1 < 0:
        k1 = k2
    if k2 < 0:
        k2 = k1

    # Allocate relative trunc number ratio based on decay constant ratio
    ratio = np.abs(k2 / k1)
    # Reweight [m1, m2] such that m2/m1=r, m1*m2=K (where r is ratio)
    mode_1_result = int(np.sqrt(K / ratio))
    mode_2_result = int(np.sqrt(ratio * K))

    # Edge case: If a trunc number is greater than K, set it to K
    if mode_1_result > K:
        mode_1_result = K
    if mode_2_result > K:
        mode_2_result = K

    # Edge case: If one mode equals 
    # 0 (ex. after preceding code), rescale it to 1
    if mode_1_result == 0:
        mode_1_result = 1
    if mode_2_result == 0:
        mode_2_result = 1

    # When returning, go back from proportion of `sqrt(min_trunc)`
    return [mode_1_result * min_trunc, mode_2_result * min_trunc]

def assign_trunc_nums(circuit: Circuit, 
                      total_trunc_num: int) -> List[int]:
    """
    Heuristically re-assign truncation numbers for a circuit with one
    or two modes (not yet implemented for more).

    Parameters
    ----------
        circuit:
            Circuit to assign truncation numbers to
        total_trunc_num:
            Maximum allowed total truncation number
    Returns
    ----------
        trunc_nums:
            List of truncation numbers for each mode of circuit
    """
    if len(circuit.m) == 1:
        print("re-allocate truncation numbers (one mode)")
        circuit.set_trunc_nums([total_trunc_num, ])
        return [total_trunc_num, ]

    # circuit that has only charge modes 
    elif len(circuit.m) == len(circuit.omega==0):
        print(
            "keep equal truncaction numbers for "
            "all modes (circuit with only charge modes)"
        )
        return circuit.trunc_nums

    elif len(circuit.m) == 2:
        print("re-allocate truncation numbers (two modes)")
        trunc_nums = trunc_num_heuristic(
            circuit,
            K=total_trunc_num,
            eig_vec_idx=1,
            axes=None
        )
        circuit.set_trunc_nums(trunc_nums)
        return trunc_nums
    else:
        raise NotImplementedError

def test_convergence(
    circuit: Circuit,
    eig_vec_idx: int = 0,
    t: int = 10,
    threshold: float = 1e-5,
) -> Tuple[bool, Tuple[float, ...]]:
    """
    Test convergence of a circuit with one or two modes (test for more modes
    not yet implemented).

    Requires the last `t` (if available) elements corresponding to 
    each mode individually of the `eig_vec_idx`th eigenvector are each on 
    average less than `threshold`.

    Returns a boolean of whether the convergence test passed, and a tuple
    of the average values of the last `t` components for each mode.
    """
    assert len(circuit._efreqs) != 0, "Circuit should be diagonalized first"

    if len(circuit.m) == 1:
        eigvec_mag, (mode_1_magnitudes, ) = get_reshaped_eigvec(
            circuit,
            eig_vec_idx
        )

        epsilon = np.average(mode_1_magnitudes[-t:])
        
        return epsilon < threshold, (epsilon, )

    elif len(circuit.m) == 2:
        eigvec_mag, (mode_1_magnitudes, mode_2_magnitudes) = get_reshaped_eigvec(
            circuit,
            eig_vec_idx,
        )

        y1, y2 = mode_1_magnitudes[-t:], mode_2_magnitudes[-t:]

        assert_message = "Need at least 4 modes to check both parities"
        assert len(y1) >= 4 and len(y2) >= 4, assert_message
        if len(y1) <= t:
            epsilon_1 = (y1[-1] + y1[-2]) / 2
        else:
            epsilon_1 = np.average(y1)
        if len(y2) <= t:
            epsilon_2 = (y2[-1] + y2[-2]) /2
        else:
            epsilon_2 = np.average(y2)

        if epsilon_1 > threshold or epsilon_2 > threshold:
            return False, (epsilon_1, epsilon_2)

        return True, (epsilon_1, epsilon_2)
    else:
        raise NotImplementedError
