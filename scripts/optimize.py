"""
Optimize.

Usage:
  optimize <code> <id> <optimization-type> [--no-save-circuit]
  optimize yaml <yaml_file> [--code=<code> --id=<id> --optimization-type=<optim_type> --no-save-circuit]
  optimize -h | --help
  optimize --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  
  -c, --code=<code>                         Circuit code
  -i, --id=<id>                             Seed for random generators
  -o, --optimization-type=<optim_type>      Optimization method
  -n, --no-save-circuit                     Don't save intermediate circuits
"""
import random
import sys

from qubit_discovery.optimization.utils import create_sampler
from qubit_discovery.optimization import run_SGD, run_BFGS, run_PSO
from qubit_discovery.losses import calculate_loss_metrics

from docopt import docopt
import numpy as np
import SQcircuit as sq
import torch
import yaml

from settings import RESULTS_DIR

# Default optimization settings
num_eigenvalues = 10
num_epochs = 100  # number of training iterations
total_trunc_num = 140
baseline_trunc_num = 100 # ≤ total_trunc_nums; initial guess at necessary size
losses = {
    'frequency_loss': True,
    'anharmonicity_loss': False,
    'flux_sensitivity_loss': False,
    'charge_sensitivity_loss': False,
    'T1_loss': False
}

# Target parameter range
capacitor_range = (1e-15, 12e-12)  # F
inductor_range = (2e-8, 5e-6)  # H
junction_range = (1e9 * 2 * np.pi, 100e9 * 2 * np.pi)  # Hz
# capacitor_range = (8e-15, 12e-14) # F
# inductor_range = (2e-7, 5e-6) # H
# junction_range = (1e9 * 2 * np.pi, 12e9 * 2 * np.pi) # Hz

element_verbose = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    global num_eigenvalues, num_epochs, total_trunc_num, baseline_trunc_num, losses

    arguments = docopt(__doc__, version='Optimize 0.7')
    
    seed, circuit_code, optim_type = None, None, None
    name = ''
    if arguments['yaml']:
        with open(arguments['<yaml_file>'], 'r') as f:
            data = yaml.safe_load(f.read())
        try:
            total_trunc_num = data['K']
            baseline_trunc_num = data['K0']
            num_epochs = data['epochs']
            losses = {k: v for d in data['losses'] for k, v in d.items()}
            name = data['name'] + '_'
        except KeyError:
            sys.exit('Yaml file must include keys {K, K0, num_epoch, losses}')

        if arguments['--id'] is not None:
            seed = int(arguments['--id'])
        else:
            try:
                seed = data['id']
            except KeyError:
                sys.exit('An id must be either passed in the command line or'
                         + ' yaml file')
        if arguments['--code'] is not None:
            circuit_code = arguments['--code']
        else:
            try:
                circuit_code = data['code']
            except KeyError:
                sys.exit('An circuit code must be either passed in the command'
                          + ' line or yaml file')
        if arguments['--optimization-type'] is not None:
            optim_type = arguments['--optimization-type'] 
        else:
            try:
                optim_type = data['optimization_type']
            except KeyError:
                sys.exit('An optimization type must be either passed in the'
                         + ' command line or yaml file')
    else:
        seed = int(arguments['<id>'])
        circuit_code = arguments['<code>']
        optim_type = arguments['<optimization-type>']
    save_circuit = not arguments['--no-save-circuit']

    print(losses)
        
    N = len(circuit_code)
    set_seed(seed)
    name += str(seed)
    sq.set_optim_mode(True)

    if optim_type == 'PSO':
        run_PSO(
            circuit_code,
            capacitor_range,
            inductor_range,
            junction_range,
            calculate_loss_metrics,
            name,
            num_eigenvalues,
            total_trunc_num,
            num_epochs,
            RESULTS_DIR,
            f'SGD_{circuit_code}_{seed}'
        )
        return

    sampler = create_sampler(N, capacitor_range, inductor_range, junction_range)
    circuit = sampler.sample_circuit_code(circuit_code)
    print("Circuit sampled!")

    # Begin by allocating truncation numbers equally amongst all modes
    circuit.truncate_circuit(baseline_trunc_num)

    if optim_type == "SGD":
        run_SGD(circuit,
                circuit_code,
                lambda cr: calculate_loss_metrics(cr,
                                                  use_frequency_loss=losses['frequency_loss'], 
                                                  use_anharmonicity_loss=losses['anharmonicity_loss'],
                                                  use_flux_sensitivity_loss=losses['flux_sensitivity_loss'], 
                                                  use_charge_sensitivity_loss=losses['charge_sensitivity_loss'],
                                                  use_T1_loss=losses['T1_loss']),
                name,
                num_eigenvalues,
                total_trunc_num,
                num_epochs,
                RESULTS_DIR,
                save_circuit=save_circuit
                )
    elif optim_type == "BFGS":
        bounds = {
            sq.Junction: torch.tensor([junction_range[0], junction_range[1]]),
            sq.Inductor: torch.tensor([inductor_range[0], inductor_range[1]]),
            sq.Capacitor: torch.tensor([capacitor_range[0], capacitor_range[1]])
        }

        run_BFGS(circuit,
                 circuit_code,
                 lambda cr, master_use_grad=True: calculate_loss_metrics(cr,
                                                                         use_frequency_loss=losses['frequency_loss'], 
                                                                         use_anharmonicity_loss=losses['anharmonicity_loss'],
                                                                         use_flux_sensitivity_loss=losses['flux_sensitivity_loss'], 
                                                                         use_charge_sensitivity_loss=losses['charge_sensitivity_loss'],
                                                                         use_T1_loss=losses['T1_loss'],
                                                                         master_use_grad=master_use_grad),
                 name,
                 num_eigenvalues,
                 total_trunc_num,
                 RESULTS_DIR,
                 bounds=bounds,
                 max_iter=num_epochs,
                 tolerance=0,
                 verbose=True)


if __name__ == "__main__":
    main()
