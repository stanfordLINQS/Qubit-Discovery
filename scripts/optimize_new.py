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
import os
import random
import yaml

import numpy as np
import torch

from docopt import docopt

import SQcircuit as sq

from SQcircuit import Circuit

from qubit_discovery.optimization.utils import create_sampler
from qubit_discovery.optimization import run_SGD, run_BFGS
from qubit_discovery.losses.loss import calculate_loss_metrics_new
from plot_utils import load_final_circuit


# Keys that must be included in Yaml file.
YAML_KEYS = [
    'name',
    'K',
    "epochs",
    "num_eigenvalues",
    "use_losses",
    "use_metrics",
    "capacitor_range",
    "inductor_range",
    "junction_range",
]

# Keys that should be in either command line or Yaml file.
YAML_OR_COMMANDLINE_KEYS = [
    "seed",
    "circuit_code",
    "optim_type",
    "save-intermediate",
    "init_circuit",
]


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


def main() -> None:

    ###########################################################################
    # Loading the Yaml file and command line parameters.
    ###########################################################################

    arguments = docopt(__doc__, version='Optimize 0.8')

    with open(arguments['<yaml_file>'], 'r') as f:
        parameters = yaml.safe_load(f.read())

    for key in YAML_KEYS:
        assert key in parameters.keys(), (
            f"Yaml file must include \"{key}\" key."
        )

    for key in YAML_OR_COMMANDLINE_KEYS:
        if arguments['--' + key] is not None:
            parameters[key] = arguments['--' + key]

        assert key in parameters.keys(), (
            f"\"{key}\" key must be either passed "
            f"in the command line or yaml file"
        )

    ###########################################################################
    # Initiating the optimization settings.
    ###########################################################################

    sq.set_optim_mode(True)

    capacitor_range = eval_list(parameters['capacitor_range'])
    junction_range = eval_list(parameters['junction_range'])
    inductor_range = eval_list(parameters['inductor_range'])

    bounds = {
        sq.Junction: torch.tensor(junction_range),
        sq.Inductor: torch.tensor(inductor_range),
        sq.Capacitor: torch.tensor(capacitor_range)
    }

    print(bounds)
    print(type(bounds))

    set_seed(int(parameters['seed']))

    # Setup output folders for data
    record_folder = os.path.join(
        os.path.dirname(os.path.abspath(arguments['<yaml_file>'])),
        f'{parameters["optim_type"]}_{parameters["name"]}', 'records'
    )
    os.makedirs(record_folder, exist_ok=True)

    def my_loss_function(cr: Circuit):
        return calculate_loss_metrics_new(
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

    ###########################################################################
    # Running the optimizations.
    ###########################################################################

    if parameters['optim_type'] == "SGD":
        run_SGD(
            circuit=circuit,
            circuit_code=parameters['circuit_code'],
            loss_metric_function=my_loss_function,
            name=parameters['name'] + '_' + str(parameters['seed']),
            num_eigenvalues=parameters['num_eigenvalues'],
            baseline_trunc_nums=circuit.truncate_circuit(parameters['K']),
            total_trunc_num=parameters['K'],
            num_epochs=parameters['epochs'],
            save_loc=record_folder,
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
            save_loc=record_folder,
            max_iter=parameters['epochs'],
            tolerance=1e-10,
            verbose=True,
            save_intermediate_circuits=parameters['save-intermediate']
        )


if __name__ == "__main__":
    main()
