"""
Evaluate convergence test on randomly sampled (or fixed) circuit.

Usage:
  convergence_test_simple.py <yaml_file>  [--seed=<seed>] [--circuit_code=<circuit_code>]
                             [--init_circuit=<init_circuit>] [--verbose]
  convergence_test_simple.py -h | --help
  convergence_test_simple.py --version

Arguments:
  <yaml_file>   YAML file containing details about the optimization.

Options:
  -h, --help    Show this screen.
  --version     Show version.

  -c, --circuit_code=<circuit_code>         Circuit code
  -s, --seed=<seed>                         Seed for random generators
  -i, --init_circuit=<init_circuit>         Set initial circuit params
  -v, --verbose                             Add verbose output
"""

import os
import random

from docopt import docopt
import numpy as np
import torch

import SQcircuit as sq
import qubit_discovery as qd
from qubit_discovery.optimization.sampler import CircuitSampler
from qubit_discovery.optimization.truncation import (
    assign_trunc_nums, test_convergence
)

from utils import load_final_circuit, add_stdout_to_logger
from inout import load_yaml_file, add_command_line_keys, Directory

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
CONVERGENCE_SIMPLE_REQUIRED_KEYS = [
    'seed',
    'circuit_code',
]

CONVERGENCE_SIMPLE_OPTIONAL_KEYS = [
    'init_circuit'
]

N_EIG_DIAG = 10
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


def evaluate_trunc_number(circuit, trunc_nums):
    if circuit.loops:
        circuit.loops[0].set_flux(DEFAULT_FLUX_POINT)
    circuit.set_trunc_nums(trunc_nums)
    circuit.diag(N_EIG_DIAG)
    passed_test, test_values = test_convergence(circuit, eig_vec_idx=1)
    output_text = f'Trunc numbers: {trunc_nums}\n'
    output_text += f'Test passed: {passed_test}\n'
    output_text += f'Test values (epsilon):\n'
    for test_value in test_values:
        output_text += f'{test_value:.3E}\n'
    return passed_test


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
        keys=CONVERGENCE_SIMPLE_REQUIRED_KEYS,
        optional_keys=CONVERGENCE_SIMPLE_OPTIONAL_KEYS
    )

    if arguments['--verbose']:
        add_stdout_to_logger(sq.get_logger())
        add_stdout_to_logger(qd.get_logger())

    directory = Directory(parameters, arguments['<yaml_file>'])
    records_dir = directory.get_records_dir()

    ############################################################################
    # Initiate the optimization settings.
    ############################################################################

    sq.set_engine('PyTorch')

    capacitor_range = eval_list(parameters['capacitor_range'])
    junction_range = eval_list(parameters['junction_range'])
    inductor_range = eval_list(parameters['inductor_range'])

    # seed = parameters['seed']
    # seeds = [90, ]
    seeds = np.arange(10)
    for seed in seeds:
        print(f"seed: {seed}")
        set_seed(int(seed))
        circuit_code = parameters['circuit_code']
        name = parameters['name']

        if parameters['init_circuit'] is None:
            sampler = CircuitSampler(
                capacitor_range=capacitor_range,
                inductor_range=inductor_range,
                junction_range=junction_range,
                flux_range=[0.5, 0.5],
                elems_not_to_optimize=[sq.Loop]
            )
            circuit = sampler.sample_circuit_code(parameters['circuit_code'])
            if circuit.loops:
                circuit.loops[0].set_flux(DEFAULT_FLUX_POINT)
            print("Circuit sampled!")
        else:
            circuit = load_final_circuit(parameters['init_circuit'])
            circuit.update()
            print("Circuit loaded!")

        ########################################################################
        # Test even distribution of truncation numbers.
        ########################################################################

        even_trunc_nums = circuit.truncate_circuit(
            parameters['K'],
            heuristic=False
        )
        print(f"even_trunc_nums: {even_trunc_nums}")
        even_split_passed = evaluate_trunc_number(circuit, even_trunc_nums)

        ########################################################################
        # Test heuristic truncation numbers.
        ########################################################################

        heuristic_trunc_nums = np.array(assign_trunc_nums(
            circuit,
            parameters['K'],
            min_trunc=4
        ))
        heuristic_trunc_nums = list(heuristic_trunc_nums)
        print(f"heuristic_trunc_nums: {heuristic_trunc_nums}")
        heuristic_passed = evaluate_trunc_number(circuit, heuristic_trunc_nums)

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

        save_suffix = f'{circuit_code}_{name}_{seed}'
        summary_save_url = os.path.join(
            records_dir,
            f'summary_{save_suffix}.txt'
        )

        print(f"summary_save_url: {summary_save_url}")
        with open(summary_save_url, 'w') as f:
            f.write(test_summary)
        f.close()


if __name__ == "__main__":
    main()
