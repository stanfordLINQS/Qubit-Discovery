"""Contains helper functions for estimating optimal truncation numbers"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import scipy, scipy.signal
from matplotlib.axes import Axes

from SQcircuit import Circuit, Junction
import SQcircuit.functions as sqf
import SQcircuit.units as unt
from SQcircuit.exceptions import CircuitStateError
from SQcircuit.settings import get_optim_mode


logger = logging.getLogger(__name__)


def get_reshaped_eigvec(
    circuit: Circuit,
    eig_vec_idx: int,
) -> List[np.ndarray]:
    """Returns the component of the ``eig_vec_idx`` eigenvector of ``circuit``
    in each of the modes.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object.
        eig_vec_idx:
            The index of the eigenvector to use.

    Returns
    ----------
        A list of the component of the eigenvector in each of the circuit's
        modes.
    """
    if len(circuit.evecs) == 0:
        raise CircuitStateError('The circuit must be diagonalized first.')

    # Reshape eigenvector dimensions to correspond to individual modes
    if get_optim_mode():
        eigenvector = np.array(circuit.evecs[eig_vec_idx].detach().numpy())
    else:
        eigenvector = circuit.evecs[eig_vec_idx].full()
    eigenvector_reshaped = np.reshape(eigenvector, circuit.m)

    # Get the component of the eigenvector's magnitude in each mode
    mode_magnitudes = []
    total_dim = np.shape(eigenvector)[0]
    for mode_idx, mode_size in enumerate(circuit.m):
        mode_eigvec = np.moveaxis(eigenvector_reshaped, mode_idx, 0)
        mode_eigvec = np.reshape(mode_eigvec, (mode_size, total_dim // mode_size))
        M = np.max(np.abs(mode_eigvec)**2, -1)
        mode_magnitudes.append(M)

    return mode_magnitudes


def fit_mode(
    vector,
    peak_height_threshold=5e-3,
    both_parities=False,
    axis: Optional[Axes]=None,
) -> List[Tuple[float, float, int]]:
    """Given an input vector corresponding to the absolute eigenvector
    magnitudes for a specific harmonic mode, return the decay constant for an
    exponential decay fit starting with the last peak of the input.

    Since eigenstates often have definite parity, when ``both_parities`` is
    ``True``, the fit is done to even/odd indices separately. Otherwise, the
    fit is performed to those indices with the same parity as the maximum
    element in the list.

    Parameters
    ----------
        vector:
            A vector to fit the decay to.
        peak_height_treshold:
            The minimum threshold of an entry to allow to be a peak.
        both_parities:
            Whether to fit to both even and odd indices of ``vector``,
            or only the parity with the maximum element.
        axis:
            An optional matplotlib axis to plot the fit on.

    Returns
    ----------
        A list of tuples ``(decay constant, peak size, peak index)`` for
        each of the parities. 
    """

    def monoExp(x, m, k):
        return m * np.exp(-k * x)

    # Either fit to even and odd indices separately, or just the parity
    # matching the index with the largest value.
    if both_parities:
        parities = [0, 1]
    else:
        parities = [np.argmax(vector) % 2]

    # Fit an exponential decay to each parity
    fit_results = []
    for parity in parities:
        # Get only entries with specified parity
        fit_points_parity = vector[parity::2]

        # Find the peaks in the data
        peaks, _ = scipy.signal.find_peaks(
            fit_points_parity,
            height=peak_height_threshold
        )
        # Edge case: No peaks found, ensure we can use the first entry
        peaks = np.insert(peaks, 0, 0)

        # And get the final one
        fit_peak_idx_parity = sorted(peaks)[-1]
        # Compute the actual index of the peak, to return later
        fit_peak_idx = fit_peak_idx_parity * 2 + parity

        # Fit an exponential decay starting with that peak
        fit_points = fit_points_parity[fit_peak_idx_parity:]

        num_points = len(fit_points)
        fit_range = np.arange(0, 2 * num_points, 2)
        # Choose a smart first guess
        fit_param_guess = (fit_points[0], -np.log(fit_points[-1]) / num_points)
        # And perform the curve fitting
        try:
            params, _ = scipy.optimize.curve_fit(
                monoExp,
                fit_range,
                fit_points,
                fit_param_guess
            )
        except RuntimeError:
            # If cannot fit, just use the guess
            params = fit_param_guess

        m, k = params
        fit_results.append((k, m, fit_peak_idx))

    # Plot slowest decaying curve out of the two parities if `axis` passed.
    if axis:
        if both_parities:
            k, m, peak_idx = get_slow_fit(fit_results)
        else:
            k, m, peak_idx = fit_results[0]
        axis.plot(peak_idx + fit_range, monoExp(fit_range, m, k), 'k')

    return fit_results


def get_slow_fit(
    fit_results: List[Tuple[float, float, int]],
    ignore_threshold=5e-3
) -> Tuple[float, float, float]:
    """Given a list of fit results from ``fit_mode()``, return the one with the
    slowest decay whose peak is at least ``ignore_threshold``.
    
    Parameters
    ----------
        fit_results
            A list of fit results as computed by ``fit_mode``.
        ignore_threshold:
            The minimum size of a peak.

    Returns
    ----------
        The slowest decaying curve whose peak is at least ``ignore_threshold``.
        If all results are below the threshold, it logs a warning and returns
        the first result.
    """
    res_above_threshold = [res for res in fit_results
                           if res[1] >= ignore_threshold]
    if res_above_threshold == []:
        logging.warning('All fit results were below threshold. Returning '
                        'the first result.')
        return fit_results[0]

    # Sort the results in ascending order by decay constant
    res_above_threshold.sort(key = lambda res: res[0])

    # Return result with the smallest constant (slowest-decaying)
    return res_above_threshold[0]


def charge_mode_heuristic(
    circuit: Circuit,
    n_stds: int = 3
) -> List[int]:
    """Calculate truncation numbers for the charge modes of a circuit by
    approximating each mode as an uncoupled harmonic oscillator and computing
    the width of the (Gaussian) ground state in the number basis.

    Currently only supports a single charge mode.
    
    Parameters
    ----------
        circuit:
            A ``Circuit`` object to provide truncation numbers for the charge
            modes.
        n_stds:
            The number of standard deviations to include in the truncated
            Hilbert space.

    Returns
    ----------
        List of truncation numbers for each charge mode of ``circuit``.
    """
    charge_modes_idxs = np.flatnonzero(circuit.omega == 0)

    if len(charge_modes_idxs) == 0:
        return []
    if len(charge_modes_idxs) == 1:
        charge_mode_idx = charge_modes_idxs[0]
        EC_eff = (
            (unt.e ** 2) / (2 * np.pi * unt.hbar) / 1e9
            * circuit.cInvTrans[charge_mode_idx, charge_mode_idx]
        )

        EJ_eff = 0
        for _, el, _, w_id in circuit.elem_keys[Junction]:
            EJ = np.squeeze(sqf.to_numpy(el.get_value('GHz'))) / 2 / np.pi
            EJ_eff += EJ * (circuit.wTrans[w_id, charge_mode_idx]) ** 2

        sigma = (EJ_eff / (8 * EC_eff)) ** (1 / 4)
        charge_truncation = int(n_stds * sigma)

        return [charge_truncation]
    else:
        raise NotImplementedError('The charge mode heuristic currently only '
                                  'supports circuits with one charge mode.')


def dimension_to_charge_trunc(max_dim: int) -> int:
    """Convert the maximum dimension of a Hilbert space to the appropriate
    truncation number for a charge mode.
    
    Given a truncation number ``m`` for a charge mode, the resulting Hilbert
    space dimension is ``2*m - 1``.

    Parameters
    ----------
        max_dim:
            The maximum dimension for the Hilbert space.

    Returns
    ----------
        The largest truncation number such that the resulting Hilbert space
        dimension is no greater than ``max_dim``.
    """

    return int(np.floor((max_dim + 1) / 2))


def trunc_num_heuristic(
    circuit: Circuit,
    eig_vec_idx: int = 0,
    K: int=1000,
    min_trunc: int = 1,
    charge_mode_cutoff: Optional[int] = None,
    use_charge_heuristic=False,
    axes: Optional[Axes] = None
) -> List[int]:
    """For a diagonalized circuits, suggests a set of truncation numbers for
    for rediagonalization that will maximize the likelihood of convergence,
    as defined in ``circuit.check_convergence()``.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to provide truncation numbers for.
        eig_vec_idx:
            The index of the eigenvector to use in the harmonic mode heuristic.
        K:
            The maximum Hilbert space dimension.
        min_trunc:
            The minimum truncation number for any mode to have.
        charge_mode_cutoff:
            An optional maximum truncation number for charge modes.
        use_charge_heuristic:
            Whether to use the charge mode heuristic, or just assign the
            charge mode truncation numbers evenly.
        axes:
            Optionally, a set of Matplotlib Axes to plot the exponential fits
            during the charge mode heuristic.

    Returns
    ----------
        List of truncation numbers for each mode of ``circuit``.
    """
    # Perform checks on parameters
    if len(circuit.evecs) == 0:
        raise CircuitStateError('The circuit must be diagonalized first.')
    if axes is not None:
        if len(axes) < len(circuit.omega):
            raise ValueError(
                'Number of axes for fitting plots should match or exceed '
                'number of modes'
            )
    if K < 1:
        raise ValueError('The maximum dimension of the Hilbert space must '
                         'be an integer >= 1.')
    if min_trunc < 1:
        raise ValueError('Each mode must have a minimum truncation number '
                         'of at least 1.')

    # Compute the average dimension, if modes had dimensions allocated equally.
    average_dim = int(np.floor(K ** (1 / circuit.n)))
    if average_dim < min_trunc:
        raise ValueError("The 'K' passed is not large enough to guarantee "
                         f"each mode a truncation number >={min_trunc}.")

    # Set up trunc nums and list of charge, harmonic modes
    trunc_nums = np.zeros_like(circuit.trunc_nums)
    harmonic_modes = circuit.omega != 0
    n_h = np.count_nonzero(harmonic_modes)
    charge_modes = circuit.omega == 0
    n_c = np.count_nonzero(charge_modes)

    # Assign charge mode truncation numbers
    if n_c > 0:
        if use_charge_heuristic:
            trunc_nums[charge_modes] = charge_mode_heuristic(circuit)
        else:
            trunc_nums[charge_modes] = dimension_to_charge_trunc(average_dim)

        # Cut off truncation numbers, if necessary
        if charge_mode_cutoff is not None:
            trunc_nums[charge_modes & (trunc_nums > charge_mode_cutoff)] = charge_mode_cutoff


        # Compute remaining Hilbert space dimension

        # We ensure each mode has at least `min_trunc` by allocating truncation
        # numbers as proportion of `min_trunc`.
        charge_space_dim = np.prod(trunc_nums[charge_modes])
        # To do this, compute an effective max Hilbert space dimension K
        K = (K
            / (min_trunc ** n_h)
            / charge_space_dim
        )

        if K < 1:
            raise ValueError(
                'The charge truncation numbers were too large to provide each '
                f'harmonic mode a truncation number >={min_trunc}. Please '
                'pass a smaller value for the `charge_mode_cutoff`.'
            )

    # Assign harmonic mode truncation numbers
    if n_h > 0:
        # Compute the component of the state in each of the mode
        mode_magnitudes = get_reshaped_eigvec(
            circuit,
            eig_vec_idx,
        )

        decay_constants = np.zeros_like(harmonic_modes, dtype=np.float64)
        peak_locations = np.zeros_like(harmonic_modes, dtype=np.float64)

        # Fit exponential decays to each mode
        for mode_idx in np.flatnonzero(harmonic_modes):
            axis = axes[mode_idx] if axes is not None else None
            fit_results = fit_mode(mode_magnitudes[mode_idx],
                                   both_parities=True,
                                   axis=axis)
            ki, _, di = get_slow_fit(fit_results)

            decay_constants[mode_idx] = ki
            peak_locations[mode_idx] = di

        # Allocate truncation numbers inverse to decay constants
        harmonic_trunc_nums = (
            np.power(K * np.prod(decay_constants), 1 / n_h)
            / decay_constants
        )

        # Shift by relative peaks
        harmonic_trunc_nums += peak_locations
        # and renormalize to be below K.
        harmonic_trunc_nums *= np.power(K / np.prod(harmonic_trunc_nums),
                                        1 / n_h)

        # Edge case: If mode number less than 1, rescale to 1 and rescale other
        # mode numbers to keep total product constant.
        while np.any(harmonic_trunc_nums < 1):
            small_trunc_nums = harmonic_trunc_nums[harmonic_trunc_nums <= 1]
            large_trunc_nums = harmonic_trunc_nums[harmonic_trunc_nums > 1]
            rescale_factor = np.power(np.prod(small_trunc_nums),
                                      1 / len(large_trunc_nums))
            harmonic_trunc_nums[harmonic_trunc_nums > 1] *= rescale_factor
            harmonic_trunc_nums[harmonic_trunc_nums <= 1] = 1

        # Edge case: If a trunc number is greater than K, set it to K
        harmonic_trunc_nums = np.minimum(harmonic_trunc_nums, K)

        # Round to nearest integer and assign harmonic modes
        # Multiply by `min_trunc` because we allocated as a proportion of that
        harmonic_trunc_nums = np.floor(np.real(harmonic_trunc_nums * min_trunc))

        # Because everything is integers, it is possible in the process of taking
        # the floor we cut off too much. Now, maximize each individual mode cutoff
        # while ensuring product less than K
        for idx in range(len(harmonic_trunc_nums)):
            harmonic_trunc_nums[idx] += np.floor(
                (K - np.prod(harmonic_trunc_nums)) 
                * harmonic_trunc_nums[idx]
                / np.prod(harmonic_trunc_nums),
            ).astype(int)

        trunc_nums[harmonic_modes] = harmonic_trunc_nums

    return list(trunc_nums)


def assign_trunc_nums(
    circuit: Circuit,
    total_trunc_num: int,
    axes=None,
    min_trunc=1,
    use_charge_heuristic=False,
) -> List[int]:
    """Heuristically re-assign truncation numbers for a circuit with multiple
    modes. (In the case of a single mode, it simply assigns the truncation
    number to ``total_trunc_num``.)

    Parameters
    ----------
        circuit:
            Circuit to assign truncation numbers to.
        total_trunc_num:
            Maximum allowed total truncation number.

    Returns
    ----------
        trunc_nums:
            List of truncation numbers for each mode of ``circuit``.
    """
    if len(circuit.m) == 1:
        logger.info('re-allocate truncation numbers (single mode)')
        # Charge mode
        if sum(circuit.omega) == 0:
            circuit.set_trunc_nums([dimension_to_charge_trunc(total_trunc_num),])
        # Harmonic mode
        else:
            circuit.set_trunc_nums([total_trunc_num, ])
    else:
        logger.info('re-allocate truncation numbers (2+ modes)')
        trunc_nums = trunc_num_heuristic(
            circuit,
            K=total_trunc_num,
            eig_vec_idx=1,
            min_trunc=min_trunc,
            use_charge_heuristic=use_charge_heuristic,
            axes=axes
        )
        circuit.set_trunc_nums(trunc_nums)

    return circuit.trunc_nums


def test_convergence(
    circuit: Circuit,
    eig_vec_idx: int = 0,
    t: int = 10,
    threshold: float = 1e-5,
) -> Tuple[bool, List[float]]:
    """Test convergence of a circuit.

    Requires the last ``t`` (if available) entries corresponding to
    each mode individually of the ``eig_vec_idx`` eigenvector are on
    average less than ``threshold``.

    Parameters
    ----------
        circuit:
            Circuit to test convergence of. Must be diagonalized.
        eig_vec_idx:
            The index of the eigenvector to use to check convergence.
        t:
            The number of entries of the eigenvector to use to check
            convergence. Must be at least 2.
        threshold:
            The maximum for the last ``t`` entries of the eigenvector.

    Returns
    ----------
        is_converged:
            A boolean of whether the circuit converged.
        epsilons:
            The average values of the last ``t`` components in each mode.
    """
    if len(circuit.efreqs) == 0:
        raise CircuitStateError('The circuit must be diagonalized first.')
    if np.any(np.array(circuit.m) < 4):
        raise ValueError('In order to check both parities, each mode must '
                         'have a dimension >= 4.')
    if t < 2:
        raise ValueError('The value for `t` must be at least 2.')

    mode_magnitudes = get_reshaped_eigvec(
        circuit,
        eig_vec_idx
    )

    # Get the last `t` entries of the component of the eigenvector in each mode.
    y = [M[-t:] for M in mode_magnitudes]
    # Average them.
    epsilons = [np.average(yi) for yi in y]

    for mode_idx, epsilon_i in enumerate(epsilons):
        # Exclude charge modes (for now) as they are fixed for a given K
        if circuit.omega[mode_idx] != 0 and epsilon_i > threshold:
            return False, epsilons

    return True, epsilons
