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


def charge_trunc_to_dimension(trunc: int) -> int:
    """Convert the truncation number for a charge mode to the dimension
    of the Hilbert space.
    
    Given a truncation number ``m`` for a charge mode, the resulting Hilbert
    space dimension is ``2*m - 1``.

    Parameters
    ----------
        trunc:
            The truncation number.

    Returns
    ----------
        The Hilbert space dimension of the corresponding charge mode.
    """

    return 2 * trunc - 1


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


def harmonic_mode_heuristic(
    circuit: Circuit,
    total_trunc_num: int,
    min_trunc_harmonic: int,
    eig_vec_idx: int,
    axes: Optional[Axes]
) -> List[int]:
    """For a diagonalized circuit, suggests a set of truncation numbers for
    harmonic modes that will maximize the likelihood of convergence,
    as defined by ``circuit.check_convergence()``.

    If the circuit has a single harmonic mode, it simply sets the truncation
    number to ``total_trunc_num``.

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to provide truncation numbers for.
        eig_vec_idx:
            The index of the eigenvector to use in the heuristic.
        total_trunc_num:
            The maximum Hilbert space dimension for the harmonic mode subspace.
        min_trunc_harmonic:
            The minimum truncation number for harmonic modes to have.
        axes:
            Optionally, a set of Matplotlib Axes to plot the exponential fits
            used in the he

    Returns
    ----------
        List of truncation numbers for the harmonic modes of ``circuit``.
    """
    harmonic_modes = circuit.omega != 0
    n_h = np.count_nonzero(harmonic_modes)

    if n_h == 0:
        return [0]
    elif n_h == 1:
        return [total_trunc_num]

    K = total_trunc_num

    K_eff = K / (min_trunc_harmonic ** n_h)

    # Compute the component of the state in each of the mode
    mode_magnitudes = get_reshaped_eigvec(
        circuit,
        eig_vec_idx,
    )

    decay_constants = np.zeros(n_h, dtype=np.float64)
    peak_locations = np.zeros(n_h, dtype=np.float64)

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
        np.power(K_eff * np.prod(decay_constants), 1 / n_h)
        / decay_constants
    )

    # Shift by relative peaks
    harmonic_trunc_nums += peak_locations
    # and renormalize to be below K.
    harmonic_trunc_nums *= np.power(K_eff / np.prod(harmonic_trunc_nums),
                                    1 / n_h)

    # Edge case: For mode numbers less than 1, rescale to 1 and rescale
    # other mode numbers to keep total product constant.
    while np.any(harmonic_trunc_nums < 1):
        small_trunc_nums = harmonic_trunc_nums[harmonic_trunc_nums <= 1]
        large_trunc_nums = harmonic_trunc_nums[harmonic_trunc_nums > 1]
        rescale_factor = np.power(np.prod(small_trunc_nums),
                                  1 / len(large_trunc_nums))
        harmonic_trunc_nums[harmonic_trunc_nums > 1] *= rescale_factor
        harmonic_trunc_nums[harmonic_trunc_nums <= 1] = 1

    # Edge case: If a trunc number is greater than K_eff, set it to K_eff
    harmonic_trunc_nums = np.minimum(harmonic_trunc_nums, K_eff)

    # Round to nearest integer and assign harmonic modes. Multiply by
    # `min_trunc_harmonic` because we allocated as a proportion of that.
    harmonic_trunc_nums = np.floor(
        harmonic_trunc_nums * min_trunc_harmonic
    ).astype(int)

    return harmonic_trunc_nums


def trunc_num_heuristic(
    circuit: Circuit,
    eig_vec_idx: int = 0,
    total_trunc_num: int = 1000,
    min_trunc_harmonic: int = 1,
    min_trunc_charge: int = 1,
    max_trunc_charge: Optional[int] = None,
    use_harmonic_heuristic=True,
    use_charge_heuristic=False,
    axes: Optional[Axes] = None
) -> List[int]:
    """For a diagonalized circuits, suggests a set of truncation numbers for
    rediagonalization that will maximize the likelihood of convergence,
    as defined in ``circuit.check_convergence()``.

    If ``use_charge_heuristic`` is ``False``, the truncation numbers for the
    charge modes are set to ``K**(1/circuit.n)``, subject to the values
    of ``min_trunc_charge`` and ``max_trunc_charge``.

    If the circuit has a single mode, it simply sets the truncation number
    to ``total_trunc_num`` (appropriately divided for charge modes).

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to provide truncation numbers for.
        eig_vec_idx:
            The index of the eigenvector to use in the harmonic mode heuristic.
        total_trunc_num:
            The maximum Hilbert space dimension.
        min_trunc_harmonic:
            The minimum truncation number for harmonic modes to have.
        min_trunc_charge:
            The minimum truncation number for charge modes to have. Note a
            truncation number ``m`` for a charge mode results in a Hilbert space
            of dimension ``2*m - 1``.
        max_trunc_charge:
            An optional maximum truncation number for charge modes.
        use_harmonic_heuristic:
            Whether to use the harmonic mode heuristic, or otherwise assign
            the harmonic truncation numbers evenly using the remaining
            Hilbert dimension after assigning the charge modes.  
        use_charge_heuristic:
            Whether to use the charge mode heuristic, or otherwise assign the
            charge mode truncation numbers evenly.
        axes:
            Optionally, a set of Matplotlib Axes to plot the exponential fits
            used in the harmonic mode heuristic.

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
    if total_trunc_num < 1:
        raise ValueError('The maximum dimension of the Hilbert space must '
                         'be an integer >= 1.')
    if min_trunc_harmonic < 1:
        raise ValueError('Harmonic modes must have a minimum truncation number '
                         'of at least 1.')
    if min_trunc_charge  < 1:
        raise ValueError('Charge modes must have a minimum truncation number '
                         'of at least 1.')

    # Deal with the easy case of a single mode (no heuristic necessary)
    if circuit.n == 1:
        logger.info('Single mode (no heuristic required).')
        # Charge mode
        if sum(circuit.omega) == 0:
            return [dimension_to_charge_trunc(total_trunc_num)]
        # Harmonic mode
        else:
            return [total_trunc_num]

    # Set up trunc nums and list of charge, harmonic modes
    trunc_nums = np.zeros_like(circuit.trunc_nums)
    harmonic_modes = circuit.omega != 0
    n_h = np.count_nonzero(harmonic_modes)
    charge_modes = circuit.omega == 0
    n_c = np.count_nonzero(charge_modes)

    # Check that `total_trunc_num` is large enough
    min_dim_req = (
        charge_trunc_to_dimension(min_trunc_charge) ** n_c
        * min_trunc_harmonic ** n_h
    )
    if min_dim_req > total_trunc_num:
        raise ValueError('The `total_trunc_num` passed is not large enough to '
                         'provide each mode the minimum truncation numbers '
                         'requested.')

    # Assign charge mode truncation numbers
    K = total_trunc_num
    if n_c > 0:
        if use_charge_heuristic:
            trunc_nums[charge_modes] = charge_mode_heuristic(circuit)
        else:
            average_dim = K ** (1/len(trunc_nums))
            trunc_nums[charge_modes] = dimension_to_charge_trunc(average_dim)

        # Cut off truncation numbers, if necessary
        if max_trunc_charge is not None:
            trunc_nums[charge_modes & (trunc_nums > max_trunc_charge)] = max_trunc_charge

        trunc_nums[charge_modes & (trunc_nums < min_trunc_charge)] = min_trunc_charge

        # Compute the Hilbert space dimension remaining for harmonic modes
        charge_space_dim = np.prod(charge_trunc_to_dimension(trunc_nums[charge_modes]))
        K //= charge_space_dim

        if K < (min_trunc_harmonic ** n_h):
            raise ValueError(
                'The charge truncation numbers were too large to provide each '
                f'harmonic mode a truncation number >={min_trunc_harmonic}. '
                'Please pass a smaller value for `max_trunc_charge`.'
            )

    # Assign harmonic mode truncation numbers
    if n_h > 0:
        if use_harmonic_heuristic:
            harmonic_trunc_nums = harmonic_mode_heuristic(
                circuit,
                K,
                min_trunc_harmonic,
                eig_vec_idx,
                axes
            )
        else:
            harmonic_trunc_nums = [np.floor(K**(1/n_h)).astype(int)] * n_h

        # Because everything is integers, it is possible in the process of
        # taking the floor we cut off too much. Thus, now maximize each
        # individual truncation number while ensuring product less than K.
        for idx in range(len(harmonic_trunc_nums)):
            curr_dim = np.prod(harmonic_trunc_nums)
            if curr_dim == K:
                break

            harmonic_trunc_nums[idx] += np.floor(
                (K - curr_dim) 
                * harmonic_trunc_nums[idx]
                / curr_dim,
            ).astype(int)

        trunc_nums[harmonic_modes] = harmonic_trunc_nums

    return list(trunc_nums)


def assign_trunc_nums(
    circuit: Circuit,
    total_trunc_num: int,
    min_trunc_harmonic: int = 1,
    min_trunc_charge: int = 1,
    max_trunc_charge: Optional[int] = None,
    use_charge_heuristic=False,
    axes: Optional[Axes] = None
) -> List[int]:
    """Heuristically re-assign truncation numbers for a circuit with multiple
    modes. (In the case of a single mode, it simply assigns the truncation
    number to ``total_trunc_num``.)

    Parameters
    ----------
        circuit:
            A ``Circuit`` object to provide truncation numbers for.
        total_trunc_num:
            The maximum Hilbert space dimension.
        min_trunc_harmonic:
            The minimum truncation number for harmonic modes.
        min_trunc_charge:
            The minimum truncation number for charge modes.
        max_trunc_charge:
            An optional maximum truncation number for charge modes.
        use_charge_heuristic:
            Whether to use the charge mode heuristic, or just assign the
            charge mode truncation numbers evenly.
        axes:
            Optionally, a set of Matplotlib Axes to plot the exponential fits
            during the charge mode heuristic.

    Returns
    ----------
        trunc_nums:
            List of truncation numbers for each mode of ``circuit``.
    """
    logger.info('Reallocating truncation numbers.')
    trunc_nums = trunc_num_heuristic(
            circuit,
            total_trunc_num=total_trunc_num,
            eig_vec_idx=1,
            min_trunc_harmonic=min_trunc_harmonic,
            min_trunc_charge=min_trunc_charge,
            max_trunc_charge=max_trunc_charge,
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
    test_charge_modes = False
) -> Tuple[bool, List[float]]:
    """Test convergence of a circuit.

    Requires the last ``t`` (if available) entries corresponding to
    each mode individually of the ``eig_vec_idx`` eigenvector are on
    average less than ``threshold``.

    By default this does **not** test charge modes, since this convergence test
    is written with harmonic modes (exponentially decaying magnitudes) in mind.

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
        test_charge_modes:
            Whether to test the charge modes (False by default).

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

    # Average them. If there are fewer than `t` entires, only average the last
    # two (to avoid averaging the whole component).
    epsilons = []
    for yi in y:
        if len(yi) <= t:
            epsilon_i = np.average(yi[-2:])
        else:
            epsilon_i = np.average(yi)
        epsilons.append(epsilon_i)
    # epsilons = [np.average(yi) for yi in y]

    for mode_idx, epsilon_i in enumerate(epsilons):
        if epsilon_i > threshold:
            if circuit.omega[mode_idx] == 0 and not test_charge_modes:
                continue
            return False, epsilons

    return True, epsilons
