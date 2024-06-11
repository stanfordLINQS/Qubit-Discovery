"""
Evaluate convergence test on randomly sampled (or fixed) circuit.

Usage:
  test_convergence_test.py <yaml_file> [--seed=<seed> --circuit_code=<circuit_code>\
  --init_circuit=<init_circuit>]
  test_convergence_test.py -h | --help
  test_convergence_test.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

  -c, --circuit_code=<circuit_code>         Circuit code
  -s, --seed=<seed>                         Seed for random generators
  -i, --init_circuit=<init_circuit>         Set initial circuit params
"""

import os
import random
import dill as pickle

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import torch

import SQcircuit as sq

from qubit_discovery.utils.sampler import CircuitSampler
from qubit_discovery.optimization.truncation import (
    assign_trunc_nums,
    test_convergence,
    get_reshaped_eigvec
)
from plot_utils import load_final_circuit
from inout import load_yaml_file, add_command_line_keys, Directory

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
YAML_OR_COMMANDLINE_KEYS = [
    "seed",
    "circuit_code",
    "init_circuit",
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


def eval_list(ls: list) -> list:
    """Evaluates elements of a list and returns as a list.
    Warning: this can execute arbitrary code! Don't accept uninspected YAML
    files from strangers.
    """
    return [eval(i) for i in ls]


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
        keys=YAML_OR_COMMANDLINE_KEYS,
    )

    directory = Directory(parameters, arguments)
    plot_output_dir = directory.get_plots_dir()
    records_dir = directory.get_records_dir()

    ############################################################################
    # Initiate the optimization settings.
    ############################################################################

    sq.set_optim_mode(True)

    capacitor_range = eval_list(parameters['capacitor_range'])
    junction_range = eval_list(parameters['junction_range'])
    inductor_range = eval_list(parameters['inductor_range'])

    seed = parameters['seed']
    seeds = np.arange(10)
    for seed in seeds:
        print(f"seed: {seed}")
        set_seed(int(seed))
        circuit_code = parameters['circuit_code']
        name = parameters['name']

        if parameters['init_circuit'] == "":
            sampler = CircuitSampler(
                num_elements=len(parameters['circuit_code']),
                capacitor_range=capacitor_range,
                inductor_range=inductor_range,
                junction_range=junction_range
            )
            circuit = sampler.sample_circuit_code(parameters['circuit_code'])
            if circuit.loops:
                circuit.loops[0].set_flux(DEFAULT_FLUX_POINT)
            print("Circuit sampled!")
        else:
            circuit = load_final_circuit(parameters['init_circuit'])
            circuit.update()
            circuit._toggle_fullcopy = True
            print("Circuit loaded!")

        fig, axes = plt.subplots(3, 3, figsize=(27, 14))
        # fig.add_gridspec(nrows=3, height_ratios=[2, 1, 1])

        ########################################################################
        # Test baseline truncation numbers.
        ########################################################################

        baseline_trunc_nums = np.array(circuit.truncate_circuit(
            parameters['K'],
            heuristic=True)
        )
        baseline_trunc_nums[baseline_trunc_nums < 4] = 4
        baseline_trunc_nums = list(baseline_trunc_nums)
        print(f"baseline_trunc_nums: {baseline_trunc_nums}")
        evaluate_trunc_number(circuit, baseline_trunc_nums, axes[1, 2])

        ########################################################################
        # Test even distribution of truncation numbers.
        ########################################################################

        even_trunc_nums = circuit.truncate_circuit(
            parameters['K'],
            heuristic=False
        )
        print(f"even_trunc_nums: {even_trunc_nums}")
        even_split_passed = evaluate_trunc_number(
            circuit, even_trunc_nums, axes[0, 2]
        )

        # Plot mode magnitudes used in heuristic test (up to three modes)
        _, mode_magnitudes = get_reshaped_eigvec(circuit, eig_vec_idx=1)
        for mode_magnitude_idx in range(min(3, len(mode_magnitudes))):
            plot_mode_magnitudes(
                mode_magnitudes[mode_magnitude_idx],
                axes[mode_magnitude_idx, 0]
            )

        ########################################################################
        # Test heuristic truncation numbers.
        ########################################################################

        heuristic_trunc_nums = np.array(assign_trunc_nums(
            circuit,
            parameters['K'],
            axes=axes[:, 0],
            min_trunc=4)
        )
        heuristic_trunc_nums = list(heuristic_trunc_nums)
        print(f"heuristic_trunc_nums: {heuristic_trunc_nums}")
        heuristic_passed = evaluate_trunc_number(
            circuit, heuristic_trunc_nums, axes[2, 2]
        )

        # Plot mode magnitudes used in heuristic test (up to three modes)
        _, mode_magnitudes = get_reshaped_eigvec(circuit, eig_vec_idx=1)
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
            plt.text(1.03, 0, f"Circuit code: {circuit_code}",
                     fontsize=12,
                     horizontalalignment='left',
                     verticalalignment='bottom',
                     transform=axes[-1, -1].transAxes,
                     bbox=props)

        ########################################################################
        # Save circuit.
        ########################################################################

        save_suffix = f'{circuit_code}_{name}_{seed}'
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

        with open(
                summary_save_url, 'w'
        ) as f:
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
