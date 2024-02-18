"""
Optimize.

Usage:
  optimize <code> <seed> <optimization-type> --name=<name> [--save-intermediate] [--output_dir=<output_dir>]
  optimize yaml <yaml_file> [--code=<code> --seed=<seed> --optimization-type=<optim_type> --name=<name> --save-intermediate]
  optimize -h | --help
  optimize --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  
  -c, --code=<code>                         Circuit code
  -s, --seed=<seed>                         Seed for random generators
  -o, --optimization-type=<optim_type>      Optimization method
  -n, --name=<name>                         Name to label the run with
  -d, --output_dir=<output_dir>             Set output directory
  --save-intermediate                       Save intermediate circuits
"""
import os
import random
import sys

from qubit_discovery.optimization.utils import create_sampler
from qubit_discovery.optimization import run_SGD, run_BFGS, run_PSO
from qubit_discovery.losses import loss_functions

from docopt import docopt
import numpy as np
import SQcircuit as sq
import torch
import yaml

from settings import RESULTS_DIR

# Default optimization settings
DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), 'defaults.yaml')
element_verbose = False

def eval_list(ls: list) -> list:
    """
    Evaluates elements of a list and returns as a list.
    Warning: this can execute arbitary code! Don't accept uninspected YAML
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

    # Load default parameters
    with open(DEFAULTS_FILE, 'r') as f:
        parameters = yaml.safe_load(f.read())
    
    # Parse based on whether we use the `yaml` subcommand (and pass in a `yaml`
    # file) or the old option-only based interface.
    if arguments['yaml']:
        with open(arguments['<yaml_file>'], 'r') as f:
            data = yaml.safe_load(f.read())

        # Set parameters required to be put in YAML file
        try:
            parameters['K'] = data['K']
            parameters['K0']  = data['K0']
            parameters['epochs']  = data['epochs']
            parameters['losses']  = data['losses']
            parameters['name']  = data['name']
        except KeyError:
            sys.exit('Yaml file must include keys {K, K0, num_epoch, losses}')

        # Set parameters which may be overwritten on the commmand line
        if arguments['--name'] is not None:
            parameters['name'] = arguments['--name']

        # Set parameters which may be either passed in the YAML file
        # or on the command line; the command line overrides the YAML file
        if arguments['--seed'] is not None:
            parameters['seed'] = int(arguments['--seed'])
        else:
            try:
                parameters['seed'] = data['seed']
            except KeyError:
                sys.exit('A seed must be either passed in the command line or'
                         + ' yaml file')
        if arguments['--code'] is not None:
            parameters['circuit_code'] = arguments['--code']
        else:
            try:
                parameters['circuit_code'] = data['circuit_code']
            except KeyError:
                sys.exit('An circuit code must be either passed in the command'
                          + ' line or yaml file')
        if arguments['--optimization-type'] is not None:
            parameters['optim_type'] = arguments['--optimization-type'] 
        else:
            try:
                parameters['optim_type'] = data['optim_type']
            except KeyError:
                sys.exit('An optimization type must be either passed in the'
                         + ' command line or yaml file')
                
        # Load optional parameters which are otherwise set by default
        for key in ['loss_function', 'capacitor_range', 'inductor_range',
                    'junction_range']:
            try:
                parameters[key] = data[key]
            except KeyError:
                pass
    else:
        parameters['seed'] = int(arguments['<seed>'])
        parameters['circuit_code'] = arguments['<code>']
        parameters['optim_type'] = arguments['<optimization-type>']
        parameters['name'] = arguments['--name']
        if arguments['--output_dir'] is not None:
            RESULTS_DIR = arguments['--output_dir']

    # Compute any derived parameters and set up environment
    save_intermediate_circuits = arguments['--save-intermediate']
    loss_metric_function = loss_functions[parameters['loss_function']]

    capacitor_range = eval_list(parameters['capacitor_range'])
    junction_range = eval_list(parameters['junction_range'])
    inductor_range = eval_list(parameters['inductor_range'])

    parameters['N'] = len(parameters['circuit_code'])
    set_seed(parameters['seed'])
    run_name = parameters['name'] + '_' + str(parameters['seed'])

    sq.set_optim_mode(True)

    # Setup output folders for data
    record_folder = os.path.join(RESULTS_DIR, f'{parameters["optim_type"]}_{parameters["name"]}', 'records')
    os.makedirs(record_folder, exist_ok=True)

    # Run the correct optimization 
    if parameters['optim_type'] == 'PSO':
        run_PSO(
            parameters['circuit_code'],
            capacitor_range,
            inductor_range,
            junction_range,
            loss_metric_function,
            parameters['seed'],
            parameters['num_eigenvalues'],
            parameters['K'],
            parameters['epochs'],
            record_folder,
            f'PSO_{parameters["circuit_code"]}_{parameters["seed"]}'
        )
        return

    sampler = create_sampler(parameters['N'], capacitor_range, inductor_range, junction_range)
    circuit = sampler.sample_circuit_code(parameters['circuit_code'])
    print("Circuit sampled!")

    # Begin by allocating truncation numbers equally amongst all modes
    baseline_trunc_nums = circuit.truncate_circuit(parameters['K0'])

    if parameters['optim_type'] == "SGD":
        bounds = {
            sq.Capacitor: (capacitor_range[0], capacitor_range[1]),
            sq.Inductor: (inductor_range[0], inductor_range[1]),
            sq.Junction: (junction_range[0], junction_range[1])
        }
        run_SGD(
            circuit,
            parameters['circuit_code'],
            lambda cr: loss_metric_function(
                cr,
                use_frequency_loss=parameters['losses']['frequency_loss'], 
                use_anharmonicity_loss=parameters['losses']['anharmonicity_loss'],
                use_flux_sensitivity_loss=parameters['losses']['flux_sensitivity_loss'], 
                use_charge_sensitivity_loss=parameters['losses']['charge_sensitivity_loss'],
                use_T1_loss=parameters['losses']['T1_loss']
            ),
            run_name,
            parameters['num_eigenvalues'],
            baseline_trunc_nums,
            parameters['K'],
            parameters['epochs'],
            bounds,
            record_folder,
            save_intermediate_circuits=save_intermediate_circuits
        )
    elif parameters['optim_type'] == "BFGS":
        bounds = {
            sq.Junction: torch.tensor(junction_range),
            sq.Inductor: torch.tensor(inductor_range),
            sq.Capacitor: torch.tensor(capacitor_range)
        }

        run_BFGS(
            circuit,
            parameters['circuit_code'],
            lambda cr, master_use_grad=True: loss_metric_function(
                cr,
                use_frequency_loss=parameters['losses']['frequency_loss'], 
                use_anharmonicity_loss=parameters['losses']['anharmonicity_loss'],
                use_flux_sensitivity_loss=parameters['losses']['flux_sensitivity_loss'], 
                use_charge_sensitivity_loss=parameters['losses']['charge_sensitivity_loss'],
                use_T1_loss=parameters['losses']['T1_loss'],
                master_use_grad=master_use_grad
            ),
            run_name,
            parameters['num_eigenvalues'],
            parameters['K'],
            record_folder,
            bounds=bounds,
            max_iter=parameters['epochs'],
            tolerance=0,
            verbose=True
        )


if __name__ == "__main__":
    main()
