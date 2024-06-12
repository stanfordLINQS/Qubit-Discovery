"""Contains helper functions for estimating optimal truncation numbers"""

from typing import Tuple, List, Optional

import numpy as np
import scipy, scipy.signal
from matplotlib.axes import Axes

from SQcircuit import get_optim_mode, Circuit


def get_reshaped_eigvec(
    circuit: Circuit,
    eig_vec_idx: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns the eigenvec and maximum magnitudes per mode index of the eigenvectors.
    """

    assert len(circuit.efreqs) != 0, "circuit should be diagonalized first."

    # Reshape eigenvector dimensions to correspond to individual modes
    if get_optim_mode():
        eigenvector = np.array(circuit.evecs[eig_vec_idx].detach().numpy())
    else:
        eigenvector = circuit.evecs[eig_vec_idx].full()
    eigenvector_reshaped = np.reshape(eigenvector, circuit.m)

    eigvec_mag = np.abs(eigenvector) ** 2
    mode_magnitudes = []
    total_dim = np.shape(eigenvector)[0]
    for mode_idx, mode_size in enumerate(circuit.m):
      mode_eigvec = np.moveaxis(eigenvector_reshaped, mode_idx, 0)
      mode_eigvec = np.reshape(mode_eigvec, (mode_size, total_dim // mode_size))
      M = np.max(np.abs(mode_eigvec)**2, -1)
      mode_magnitudes.append(M)

    return eigvec_mag, mode_magnitudes


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
    min_trunc: int=1,
    charge_mode_cutoff: int=15,
    axes: Optional[Axes]=None
) -> List[int]:
    """
    For a diagonalized circuit with internal trunc numbers, suggests a set of
    trunc numbers for rediagonalization that will maximize likelihood of
    convergence, defined under the `convergence_test` function.
    """
    assert len(circuit.efreqs) != 0, "Circuit should be diagonalized first"

    trunc_nums = np.zeros_like(circuit.m)
    harmonic_modes = np.array(circuit.m)[circuit.omega != 0]
    num_charge_modes = np.sum(circuit.omega == 0)
    # charge_mode_cutoff = trunc_num_average = np.ceil(K ** (1 / len(circuit.omega)))
    # TODO: Assign charge mode more cleverly rather than hard-coding

    if axes is not None:
        assert len(axes) >= len(circuit.omega),\
            "Number of axes for fitting plots should match or exceed number of modes"

    # Ensure each mode has at least `min_trunc` by allocating truncation
    # numbers as proportion of `min_trunc`
    K = K / min_trunc ** len(harmonic_modes) / charge_mode_cutoff ** num_charge_modes

    _, mode_magnitudes = get_reshaped_eigvec(
        circuit,
        eig_vec_idx,
    )
    # harmonic_indices = np.nonzero(circuit.omega)[0]
    # harmonic_mode_magnitudes = [mode_magnitudes[int(harmonic_idx)] for harmonic_idx in harmonic_indices]

    k = np.zeros_like(harmonic_modes, dtype=np.float64)
    d = np.zeros_like(harmonic_modes, dtype=np.float64)

    for mode_idx, mode in enumerate(circuit.omega):
        axis = axes[mode_idx] if axes is not None else None
        if mode != 0:
            fit_results = fit_mode(mode_magnitudes[mode_idx],
                                   both_parities=True,
                                   axis=axis)
            ki, _, di = get_slow_fit(fit_results)
            k[mode_idx] = ki
            d[mode_idx] = di

    print(f"k: {k}")
    if all([ki < 0 for ki in k]):
        k = np.array([1 for _ in range(len(k))])

    k[k < 0] = np.mean(k[k > 0])

    # Allocate relative trunc number ratio based on decay constant ratio
    ratio = np.power(np.prod(k), 1 / len(k)) / k

    # Weight [mi, ] such that mi/mj=kj/ki, *mi=K
    mode_results = np.power(K * ratio, 1 / len(harmonic_modes))

    # Shift by relative peaks
    mode_results += d
    mode_results *= np.power(K / np.prod(mode_results), 1 / len(harmonic_modes))

    # Edge case: If mode number less than 1, rescale to 1 and rescale other mode numbers
    # to keep total product constant
    small_mode_results = mode_results[mode_results < 1]
    large_mode_results = mode_results[mode_results >= 1]
    rescale_factor = np.power(np.prod(small_mode_results), 1 / len(large_mode_results))
    mode_results[mode_results >= 1] *= rescale_factor
    mode_results[mode_results < 1] = 1

    # Edge case: If a trunc number is greater than K, set it to K
    mode_results = np.minimum(mode_results, K)

    # Round to nearest integer and assign harmonic modes
    harmonic_mode_values = np.floor(mode_results * min_trunc)
    trunc_nums[circuit.omega != 0] = harmonic_mode_values
    # Assign charge modes
    trunc_nums[circuit.omega == 0] = charge_mode_cutoff

    return list(trunc_nums)


def assign_trunc_nums(
    circuit: Circuit,
    total_trunc_num: int,
    axes=None,
    min_trunc=1
) -> List[int]:
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
        print("re-allocate truncation numbers (single mode)")
        circuit.set_trunc_nums([total_trunc_num, ])
        return [total_trunc_num, ]

    # circuit that has only charge modes 
    elif len(circuit.m) == sum(circuit.omega == 0):
        print(
            "keep equal truncation numbers for "
            "all modes (circuit with only charge modes)"
        )
        return circuit.trunc_nums

    else:
        print("re-allocate truncation numbers (2+ modes)")
        trunc_nums = trunc_num_heuristic(
            circuit,
            K=total_trunc_num,
            eig_vec_idx=1,
            min_trunc=min_trunc,
            axes=axes
        )
        circuit.set_trunc_nums(trunc_nums)
        return trunc_nums


def test_convergence(
    circuit: Circuit,
    eig_vec_idx: int = 0,
    t: int = 10,
    threshold: float = 1e-5,
) -> Tuple[bool, List[float]]:
    """
    Test convergence of a circuit with one or two modes (test for more modes
    not yet implemented).

    Requires the last `t` (if available) elements corresponding to
    each mode individually of the `eig_vec_idx`th eigenvector are each on
    average less than `threshold`.

    Returns a boolean of whether the convergence test passed, and a list
    of the average values of the last `t` components for each mode.
    """
    assert len(circuit.efreqs) != 0, "Circuit should be diagonalized first"

    eigvec_mag, mode_magnitudes = get_reshaped_eigvec(
        circuit,
        eig_vec_idx
    )

    y = [M[-t:] for M in mode_magnitudes]
    assert_message = "Need at least 4 modes to check both parities"
    assert all([len(yi) >= 4 for yi in y]), assert_message
    epsilon = []
    for yi in y:
        if len(yi) <= t:
            epsilon_i = (yi[-1] + yi[-2]) / 2
        else:
            epsilon_i = np.average(yi)
        epsilon.append(epsilon_i)
    for mode_idx, epsilon_i in enumerate(epsilon):
        # Exclude charge modes (for now) as they are fixed for a given K
        if circuit.omega[mode_idx] != 0 and epsilon_i > threshold:
            return False, epsilon
    # if any([epsilon_i > threshold for epsilon_i in epsilon]):
    #     return False, epsilon
    return True, epsilon
