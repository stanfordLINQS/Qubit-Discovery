"""
Optimize.

Usage:
  optimize yaml <yaml_file>  [--seed=<seed> --circuit_code=<circuit_code>\
  --optim_type=<optim_type> --init_circuit=<init_circuit> --save-intermediate]
  optimize -h | --help
  optimize --version

Options:
  -h --help     Show this screen.
  --version     Show version.

  -c, --circuit_code=<circuit_code>         Circuit code
  -s, --seed=<seed>                         Seed for random generators
  -o, --optim_type=<optim_type>             Optimization method
  -i, --init_circuit=<init_circuit>         Set initial circuit params
  --save-intermediate                       Save intermediate circuits
"""

import random

import numpy as np
import torch

from docopt import docopt

import SQcircuit as sq

from SQcircuit import Circuit

from qubit_discovery.optimization.utils import create_sampler
from qubit_discovery.optimization import run_SGD, run_BFGS
from qubit_discovery.losses.loss import calculate_loss_metrics
from plot_utils import load_final_circuit
from inout import load_yaml_file, add_command_line_keys, Directory

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
YAML_OR_COMMANDLINE_KEYS = [
    "seed",
    "circuit_code",
    "optim_type",
    "save-intermediate",
    "init_circuit",
]

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
        keys=YAML_OR_COMMANDLINE_KEYS,
    )

    directory = Directory(parameters, arguments)

    ############################################################################
    # Initiating the optimization settings.
    ############################################################################

    sq.set_optim_mode(True)

    capacitor_range = eval_list(parameters['capacitor_range'])
    junction_range = eval_list(parameters['junction_range'])
    inductor_range = eval_list(parameters['inductor_range'])

    bounds = {
        sq.Junction: torch.tensor(junction_range),
        sq.Inductor: torch.tensor(inductor_range),
        sq.Capacitor: torch.tensor(capacitor_range)
    }

    set_seed(int(parameters['seed']))

    def my_loss_function(cr: Circuit):
        return calculate_loss_metrics(
            cr,
            use_losses=parameters["use_losses"],
            use_metrics=parameters["use_metrics"],
        )

    if parameters['init_circuit'] == "":
        sampler = create_sampler(
            len(parameters['circuit_code']),
            capacitor_range,
            inductor_range,
            junction_range
        )
        circuit = sampler.sample_circuit_code(parameters['circuit_code'])
        print("Circuit sampled!")
    else:
        circuit = load_final_circuit(parameters['init_circuit'])
        circuit.update()
        circuit._toggle_fullcopy = True
        print("Circuit loaded!")

    baseline_trunc_num = circuit.truncate_circuit(parameters['K'])

    ############################################################################
    # Running the optimizations.
    ############################################################################

    if parameters['optim_type'] == "SGD":
        run_SGD(
            circuit=circuit,
            circuit_code=parameters['circuit_code'],
            loss_metric_function=my_loss_function,
            name=parameters['name'] + '_' + str(parameters['seed']),
            num_eigenvalues=parameters['num_eigenvalues'],
            baseline_trunc_nums=baseline_trunc_num,
            total_trunc_num=parameters['K'],
            num_epochs=parameters['epochs'],
            save_loc=directory.get_records_dir(),
            save_intermediate_circuits=parameters['save-intermediate']
        )
    elif parameters['optim_type'] == "BFGS":
        run_BFGS(
            circuit=circuit,
            circuit_code=parameters['circuit_code'],
            loss_metric_function=my_loss_function,
            name=parameters['name'] + '_' + str(parameters['seed']),
            num_eigenvalues=parameters['num_eigenvalues'],
            total_trunc_num=parameters['K'],
            bounds=bounds,
            save_loc=directory.get_records_dir(),
            max_iter=parameters['epochs'],
            tolerance=1e-10,
            verbose=True,
            save_intermediate_circuits=parameters['save-intermediate']
        )


if __name__ == "__main__":
    main()
