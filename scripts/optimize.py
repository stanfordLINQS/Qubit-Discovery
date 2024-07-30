"""
Optimize a single circuit.

Usage:
  optimize.py <yaml_file> [--seed=<seed>] [--circuit_code=<circuit_code>]
              [--optim_type=<optim_type>] [--init_circuit=<init_circuit>]
              [--save-intermediate]
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

Notes: Optional arguments to optimize.py must either be provided on the
command line or in <yaml_file>.
"""

import random
from typing import List

from docopt import docopt
import numpy as np
import torch

from qubit_discovery.optimization import run_SGD, run_BFGS
from qubit_discovery.losses import build_loss_function
from qubit_discovery.optimization.sampler import CircuitSampler
from qubit_discovery.optimization.utils import float_list
import SQcircuit as sq
from SQcircuit import Circuit

from plot_utils import load_final_circuit
from inout import load_yaml_file, add_command_line_keys, Directory

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

    directory = Directory(parameters, arguments)

    ############################################################################
    # Initiating the optimization settings.
    ############################################################################

    sq.set_engine('PyTorch')
    sq.set_max_eigenvector_grad(2)

    capacitor_range = float_list(parameters['capacitor_range'])
    junction_range = float_list(parameters['junction_range'])
    inductor_range = float_list(parameters['inductor_range'])

    if "flux_range" in parameters.keys():
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
        use_losses=parameters["use_losses"],
        use_metrics=parameters["use_metrics"]
    )

    if parameters['init_circuit'] is None or parameters['init_circuit'] == "":
        circuit = sampler.sample_circuit_code(parameters['circuit_code'])
        print(circuit.loops[0].value() / 2 / np.pi)
        print("Circuit sampled!")
    else:
        circuit = load_final_circuit(parameters['init_circuit'])
        circuit.update()
        print("Circuit loaded!")

    circuit.truncate_circuit(parameters['K'])

    ############################################################################
    # Run the optimizations.
    ############################################################################

    if parameters['optim_type'] == "SGD":
        raise ValueError('SGD is currently deprecated.')
        # run_SGD(
        #     circuit=circuit,
        #     circuit_code=parameters['circuit_code'],
        #     loss_metric_function=my_loss_function,
        #     num_eigenvalues=parameters['num_eigenvalues'],
        #     baseline_trunc_nums=baseline_trunc_num,
        #     total_trunc_num=parameters['K'],
        #     num_epochs=parameters['epochs'],
        #     save_loc=directory.get_records_dir(),
        #     save_intermediate_circuits=parameters['save-intermediate']
        # )
    elif parameters['optim_type'] == "BFGS":
        run_BFGS(
            circuit=circuit,
            loss_metric_function=my_loss_function,
            max_iter=parameters['epochs'],
            total_trunc_num=parameters['K'],
            bounds=sampler.bounds,
            num_eigenvalues=parameters['num_eigenvalues'],
            identifier = f'{parameters["circuit_code"]}_{parameters["name"]}_{parameters["seed"]}',
            save_loc=directory.get_records_dir(),
            save_intermediate_circuits=parameters['save-intermediate'],
            verbose=True
        )


if __name__ == "__main__":
    main()
