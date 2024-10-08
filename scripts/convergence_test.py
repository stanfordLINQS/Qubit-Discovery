"""
Evaluate convergence test on randomly sampled (or fixed) circuit.

Usage:
  convergence_test.py <yaml_file> [--seed=<seed>] [--K=<K>]
                      [--circuit_code=<circuit_code>] [--init_circuit=<init_circuit>]
                      [--verbose]
  convergence_test.py -h | --help
  convergence_test.py --version

Arguments:
  <yaml_file>   YAML file containing details about the optimization.
  
Options:
  -h, --help    Show this screen.
  --version     Show version.

  -s, --seed=<seed>                         Seed for random generators.
  -K, --K=<K>                               Maximum truncation number.
  -c, --circuit_code=<circuit_code>         Circuit code.
  -i, --init_circuit=<init_circuit>         Set initial circuit params.
  -v, --verbose                             Turn on verbose output.
"""

import os
import random
import dill as pickle

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import torch

import SQcircuit as sq

import qubit_discovery as qd
from qubit_discovery import CircuitSampler
from qubit_discovery.optim.truncation import (
    assign_trunc_nums,
    get_reshaped_eigvec,
    test_convergence
)
from qubit_discovery.optim.utils import float_list
from utils import add_stdout_to_logger, load_final_circuit
from inout import add_command_line_keys, Directory, load_yaml_file

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
CONVERGENCE_REQUIRED_KEYS = [
    'seed',
    'K',
    'circuit_code',
]

CONVERGENCE_OPTIONAL_KEYS = [
    'init_circuit'
]

N_EIG_DIAG = 10
N_EIG_FLUX_SPECTRA = 10
PHI_VALUES = np.concatenate([
    np.linspace(0, 0.4, 30),
    np.linspace(0.4, 0.5, 31)[1:]
])
DEFAULT_FLUX_POINT = 0.5 - 1e-2


################################################################################
# Helper functions.
################################################################################


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_flux_spectrum(circuit):
    spectrum = np.zeros((N_EIG_FLUX_SPECTRA, len(PHI_VALUES)))

    loop = circuit.loops[0]
    for flux_idx, phi in enumerate(PHI_VALUES):
        loop.set_flux(phi)
        spectrum[:, flux_idx] = circuit.diag(n_eig=N_EIG_FLUX_SPECTRA)[0].detach().numpy()

    return spectrum


def plot_flux_spectrum(spectrum, axis):
    for eigen_idx in range(N_EIG_FLUX_SPECTRA):
        axis.plot(PHI_VALUES,
                  (spectrum[eigen_idx, :] - spectrum[0, :]),
                  marker='o',
                  markersize=1.5)
    axis.axvline(x=DEFAULT_FLUX_POINT, color='b', linestyle='--')
    axis.set_xlabel(r"$\Phi_{ext}/\Phi_0$")
    axis.set_ylabel(r" $\omega_n / 2\pi$  (GHz)")


def write_test_results(axis, text):
    # bbox features
    props = dict(boxstyle='round', facecolor='yellow', alpha=0.1)
    axis.text(
        1.03, 0.98, text.strip(),
        transform=axis.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=props
    )


def evaluate_trunc_number(circuit, trunc_nums, axis):
    if circuit.loops:
        circuit.loops[0].set_flux(DEFAULT_FLUX_POINT)
    circuit.set_trunc_nums(trunc_nums)
    circuit.diag(N_EIG_DIAG)
    passed_test, test_values = test_convergence(circuit, eig_vec_idx=1)
    spectrum = calculate_flux_spectrum(circuit)
    plot_flux_spectrum(spectrum, axis)
    output_text = f'Trunc numbers: {trunc_nums}\n'
    output_text += f'Test passed: {passed_test}\n'
    output_text += f'Test values (epsilon):\n'
    for test_value in test_values:
        output_text += f'{test_value:.3E}\n'
    write_test_results(axis, output_text)
    return passed_test


def plot_mode_magnitudes(mode_magnitudes, axis):
    mode_magnitudes = np.array(mode_magnitudes)
    axis.plot(mode_magnitudes,
              marker='o',
              linestyle='dotted',
              markersize=1.5)
    axis.set_xlabel(r"Eigenvector Entry Index $i$")
    axis.set_ylabel(r"Magnitude $|\psi_1^{[i]}|^2$")


################################################################################
# Main.
################################################################################


def main() -> None:
    ############################################################################
    # Load the Yaml file and command line parameters.
    ############################################################################

    arguments = docopt(__doc__, version='Optimize 0.8')

    parameters = load_yaml_file(arguments['<yaml_file>'])

    parameters = add_command_line_keys(
        parameters=parameters,
        arguments=arguments,
        keys=CONVERGENCE_REQUIRED_KEYS,
        optional_keys=CONVERGENCE_OPTIONAL_KEYS
    )

    if arguments['--verbose']:
        add_stdout_to_logger(sq.get_logger())
        add_stdout_to_logger(qd.get_logger())

    parameters["optim_type"] = "convergence"

    directory = Directory(parameters, arguments['<yaml_file>'])
    plot_output_dir = directory.get_plots_dir()
    records_dir = directory.get_records_dir()

    ############################################################################
    # Initiate the optimization settings.
    ############################################################################

    sq.set_engine('PyTorch')

    capacitor_range = float_list(parameters['capacitor_range'])
    junction_range = float_list(parameters['junction_range'])
    inductor_range = float_list(parameters['inductor_range'])

    if "flux_range" in parameters.keys():
        flux_range = float_list(parameters['flux_range'])
        elements_not_to_optimize = []
    else:
        flux_range = [0.5, 0.5]
        elements_not_to_optimize = [sq.Loop]

    set_seed(int(parameters['seed']))

    if parameters['init_circuit'] is None:
        sampler = CircuitSampler(
            capacitor_range=capacitor_range,
            inductor_range=inductor_range,
            junction_range=junction_range,
            flux_range=flux_range,
            elems_not_to_optimize=elements_not_to_optimize
        )
        circuit = sampler.sample_circuit_code(parameters['circuit_code'])
        if circuit.loops:
            circuit.loops[0].set_flux(DEFAULT_FLUX_POINT)
        print("Circuit sampled!")
    else:
        circuit = load_final_circuit(parameters['init_circuit'])
        circuit.update()
        print("Circuit loaded!")

    fig, axes = plt.subplots(3, 3, figsize=(27, 14))
    # fig.add_gridspec(nrows=3, height_ratios=[2, 1, 1])

    ########################################################################
    # Test even distribution of truncation numbers.
    ########################################################################

    even_trunc_nums = circuit.truncate_circuit(
        int(parameters['K'])
    )
    print(f"even_trunc_nums: {even_trunc_nums}")
    even_split_passed = evaluate_trunc_number(
        circuit, even_trunc_nums, axes[0, 2]
    )

    # Plot mode magnitudes used in heuristic test (up to three modes)
    mode_magnitudes = get_reshaped_eigvec(circuit, eig_vec_idx=1)
    for mode_magnitude_idx in range(min(3, len(mode_magnitudes))):
        plot_mode_magnitudes(
            mode_magnitudes[mode_magnitude_idx],
            axes[mode_magnitude_idx, 0]
        )

    ########################################################################
    # Test heuristic truncation numbers.
    ########################################################################

    heuristic_trunc_nums = assign_trunc_nums(
        circuit,
        int(parameters['K']),
        axes=axes[:, 0],
        min_trunc_harmonic=4,
        min_trunc_charge=12,
        use_charge_heuristic=False
    )
    heuristic_passed = evaluate_trunc_number(
        circuit, heuristic_trunc_nums, axes[2, 2]
    )

    # Plot mode magnitudes used in heuristic test (up to three modes)
    mode_magnitudes = get_reshaped_eigvec(circuit, eig_vec_idx=1)
    for mode_magnitude_idx in range(min(3, len(mode_magnitudes))):
        print(
            f"mode magnitudes: {mode_magnitudes[mode_magnitude_idx].shape}"
        )

        plot_mode_magnitudes(
            mode_magnitudes[mode_magnitude_idx],
            axes[mode_magnitude_idx, 1]
        )

        ####################################################################
        # Add metadata to plot.
        ####################################################################

        props = dict(boxstyle='round', facecolor='yellow', alpha=0.1)
        # bbox features
        plt.text(1.03, 0, f"Circuit code: {parameters['circuit_code']}",
                 fontsize=12,
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=axes[-1, -1].transAxes,
                 bbox=props)

    ########################################################################
    # Save circuit.
    ########################################################################

    save_suffix = (
        f"{parameters['circuit_code']}"
        f"_{parameters['name']}"
        f"_{int(parameters['seed'])}"
    )
    print(f"save_suffix:{save_suffix}")
    circuit_save_url = os.path.join(
        records_dir,
        f'circuit_record_{save_suffix}.pickle'
    )
    with open(circuit_save_url, 'wb') as f:
        pickle.dump(circuit.picklecopy(), f)

    ########################################################################
    # Save test summary.
    ########################################################################

    if even_split_passed and heuristic_passed:
        test_summary = "0"
    elif not even_split_passed and heuristic_passed:
        test_summary = "1"
    elif even_split_passed and not heuristic_passed:
        test_summary = "2"
    else:
        test_summary = "3"

    summary_save_url = os.path.join(
        records_dir,
        f'summary_{save_suffix}.txt'
    )

    with open(summary_save_url, 'w') as f:
        f.write(test_summary)
    f.close()

    ########################################################################
    # Save figure.
    ########################################################################

    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_output_dir, f'flux_spectra_{save_suffix}.png'),
        dpi=300,
        bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
