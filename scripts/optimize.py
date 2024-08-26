"""
Optimize a single circuit.

Usage:
  optimize.py <yaml_file> [--seed=<seed>] [--circuit_code=<circuit_code>]
              [--optim_type=<optim_type>] [--init_circuit=<init_circuit>]
              [--save-intermediate] [--verbose]
  optimize.py -h | --help
  optimize.py --version

Arguments:
  <yaml_file>   YAML file containing details about the optimization.

Options:
  -h, --help    Show this screen.
  --version     Show version.

  -c, --circuit_code=<circuit_code>         Code for circuit topology.
  -s, --seed=<seed>                         Seed (integer) for random generators.
  -o, --optim_type=<optim_type>             Optimization method to use.
  -i, --init_circuit=<init_circuit>         Set intial circuit to <init_circuit>.
  --save-intermediate                       Save intermediate circuits during
                                            optimization to file.
  -v, --verbose                             Turn on verbose output.

Notes: Optional arguments to optimize.py must either be provided on the
command line or in <yaml_file>.
"""

import random

from docopt import docopt
import torch
import numpy as np

import qubit_discovery as qd
from qubit_discovery import (
    build_loss_function,
    CircuitSampler,
    run_BFGS,
    run_SGD
)
from qubit_discovery.optim.utils import float_list
import SQcircuit as sq

from utils import add_stdout_to_logger, load_final_circuit
from inout import add_command_line_keys, Directory, load_yaml_file

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
OPTIMIZE_REQUIRED_KEYS = [
    'seed',
    'circuit_code',
    'optim_type',
    'save-intermediate',
]

OPTIMIZE_OPTIONAL_KEYS = [
    'init_circuit'
]

################################################################################
# Helper functions.
################################################################################


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

################################################################################
# Main.
################################################################################


def main() -> None:

    ############################################################################
    # Loading the Yaml file and command line parameters.
    ############################################################################

    arguments = docopt(__doc__, version='Optimize 0.8')

    parameters = load_yaml_file(arguments['<yaml_file>'])

    parameters = add_command_line_keys(
        parameters=parameters,
        arguments=arguments,
        keys=OPTIMIZE_REQUIRED_KEYS,
        optional_keys=OPTIMIZE_OPTIONAL_KEYS
    )

    directory = Directory(parameters, arguments['<yaml_file>'])

    if arguments['--verbose']:
        add_stdout_to_logger(sq.get_logger())
        add_stdout_to_logger(qd.get_logger())

    ############################################################################
    # Initiating the optimization settings.
    ############################################################################

    sq.set_engine('PyTorch')
    sq.set_max_eigenvector_grad(2)

    capacitor_range = float_list(parameters['capacitor_range'])
    junction_range = float_list(parameters['junction_range'])
    inductor_range = float_list(parameters['inductor_range'])

    if 'flux_range' in parameters.keys():
        flux_range = float_list(parameters['flux_range'])
        elements_not_to_optimize = []
    else:
        flux_range = [0.5, 0.5]
        elements_not_to_optimize = [sq.Loop]

    sampler = CircuitSampler(
            capacitor_range=capacitor_range,
            inductor_range=inductor_range,
            junction_range=junction_range,
            flux_range=flux_range,
            elems_not_to_optimize=elements_not_to_optimize
    )

    set_seed(int(parameters['seed']))

    my_loss_function = build_loss_function(
        use_losses=parameters['use_losses'],
        use_metrics=parameters['use_metrics']
    )

    if parameters['init_circuit'] is None or parameters['init_circuit'] == '':
        circuit = sampler.sample_circuit_code(parameters['circuit_code'])
        print('Circuit sampled!')
    else:
        circuit = load_final_circuit(parameters['init_circuit'])
        circuit.update()
        print('Circuit loaded!')

    circuit.truncate_circuit(parameters['K'])

    ############################################################################
    # Run the optimizations.
    ############################################################################

    if parameters['optim_type'] == 'SGD':
        run_SGD(
            circuit=circuit,
            loss_metric_function=my_loss_function,
            max_iter=parameters['epochs'],
            total_trunc_num=parameters['K'],
            bounds=sampler.bounds,
            num_eigenvalues=parameters['num_eigenvalues'],
            identifier = f'{parameters["circuit_code"]}_{parameters["name"]}_SGD_{parameters["seed"]}',
            save_loc=directory.get_records_dir(),
            save_intermediate_circuits=parameters['save-intermediate']
        )
    elif parameters['optim_type'] == 'BFGS':
        run_BFGS(
            circuit=circuit,
            loss_metric_function=my_loss_function,
            max_iter=parameters['epochs'],
            total_trunc_num=parameters['K'],
            bounds=sampler.bounds,
            num_eigenvalues=parameters['num_eigenvalues'],
            identifier = f'{parameters["circuit_code"]}_{parameters["name"]}_BFGS_{parameters["seed"]}',
            save_loc=directory.get_records_dir(),
            save_intermediate_circuits=parameters['save-intermediate']
        )
    else:
        raise ValueError(f'Optimization with {parameters["optim_type"]} '
                         'is not supported.')


if __name__ == '__main__':
    main()
