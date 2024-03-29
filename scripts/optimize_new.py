"""
Optimize.

Usage:
  optimize yaml <yaml_file>  [--seed=<seed> --circuit_code=<circuit_code> --optim_type=<optim_type> --init_circuit=<init_circuit> --save-intermediate]
  optimize -h | --help
  optimize --version

Options:
  -h --help     Show this screen.
  --version     Show version.

  -c, --circuit_code=<circuit_code>         Circuit code
  -s, --seed=<seed>                         Seed for random generators
  -o, --optim_type=<optim_type>             Optimization method
  -d, --output_dir=<output_dir>             Set output directory
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

from settings import RESULTS_DIR

# Default optimization settings.
DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), 'defaults_new.yaml')
element_verbose = False

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
    global RESULTS_DIR
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

    capacitor_range = eval_list(parameters['capacitor_range'])
    junction_range = eval_list(parameters['junction_range'])
    inductor_range = eval_list(parameters['inductor_range'])

    set_seed(int(parameters['seed']))
    run_name = parameters['name'] + '_' + str(parameters['seed'])

    sq.set_optim_mode(True)

    # Setup output folders for data
    record_folder = os.path.join(
        RESULTS_DIR,
        f'{parameters["optim_type"]}_{parameters["name"]}',
        'records'
    )
    os.makedirs(record_folder, exist_ok=True)

    def my_loss_function(cr: Circuit):
        return calculate_loss_metrics_new(
            cr,
            use_losses=parameters["use_losses"],
            use_metrics=parameters["use_metrics"],
        )

    sampler = create_sampler(
        len(parameters['circuit_code']),
        capacitor_range,
        inductor_range,
        junction_range
    )

    bounds = {
        sq.Junction: torch.tensor(junction_range),
        sq.Inductor: torch.tensor(inductor_range),
        sq.Capacitor: torch.tensor(capacitor_range)
    }

    if parameters['init_circuit'] == "":
        circuit = sampler.sample_circuit_code(parameters['circuit_code'])
        print("Circuit sampled!")
    else:
        circuit = load_final_circuit(parameters['init_circuit'])
        circuit.update()
        circuit._toggle_fullcopy = True
        print("Circuit loaded!")

    # Begin by allocating truncation numbers equally amongst all modes
    baseline_trunc_nums = circuit.truncate_circuit(parameters['K'])

    if parameters['optim_type'] == "SGD":
        run_SGD(
            circuit,
            parameters['circuit_code'],
            my_loss_function,
            run_name,
            parameters['num_eigenvalues'],
            baseline_trunc_nums,
            parameters['K'],
            parameters['epochs'],
            bounds,
            record_folder,
            save_intermediate_circuits=parameters['save-intermediate']
        )
    elif parameters['optim_type'] == "BFGS":

        run_BFGS(
            circuit,
            parameters['circuit_code'],
            my_loss_function,
            run_name,
            parameters['num_eigenvalues'],
            parameters['K'],
            record_folder,
            bounds=bounds,
            max_iter=parameters['epochs'],
            tolerance=0,
            verbose=True,
            save_intermediate_circuits=parameters['save-intermediate']
        )


if __name__ == "__main__":
    main()
