import argparse
import random

from qubit_discovery.optimization.utils import create_sampler
from qubit_discovery.optimization import run_SGD, run_BFGS, run_PSO
from qubit_discovery.losses import calculate_loss_metrics

import numpy as np
import SQcircuit as sq
import torch

from settings import RESULTS_DIR

# Optimization settings

num_epochs = 20  # number of training iterations
num_eigenvalues = 10
total_trunc_num = 140
baseline_trunc_num = 100 # ≤ total_trunc_nums; initial guess at necessary size

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
    # Assign keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("code")
    parser.add_argument("id")
    parser.add_argument("optimization_type")
    parser.add_argument('-n', '--no-save-circuit', action='store_false')
    args = parser.parse_args() # ['JL', '0', 'SGD']

    seed = int(args.id)
    set_seed(seed)

    sq.set_optim_mode(True)

    circuit_code = args.code
    run_id = int(args.id)
    N = len(circuit_code)

    if args.optimization_type == 'PSO':
        run_PSO(
            circuit_code,
            capacitor_range,
            inductor_range,
            junction_range,
            calculate_loss_metrics,
            seed,
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

    if args.optimization_type == "SGD":
        run_SGD(circuit,
                circuit_code,
                lambda cr: calculate_loss_metrics(cr,
                                                  use_frequency_loss=True, 
                                                  use_anharmonicity_loss=True,
                                                  use_flux_sensitivity_loss=True, 
                                                  use_charge_sensitivity_loss=True,
                                                  use_T1_loss=False),
                run_id,
                num_eigenvalues,
                total_trunc_num,
                num_epochs,
                RESULTS_DIR,
                save_circuit=args.no_save_circuit
                )
    elif args.optimization_type == "BFGS":
        bounds = {
            sq.Junction: torch.tensor([junction_range[0], junction_range[1]]),
            sq.Inductor: torch.tensor([inductor_range[0], inductor_range[1]]),
            sq.Capacitor: torch.tensor([capacitor_range[0], capacitor_range[1]])
        }

        run_BFGS(circuit,
                 circuit_code,
                 lambda cr, master_use_grad=True: calculate_loss_metrics(cr,
                                                                         use_frequency_loss=True, 
                                                                         use_anharmonicity_loss=True,
                                                                         use_flux_sensitivity_loss=True, 
                                                                         use_charge_sensitivity_loss=False,
                                                                         use_T1_loss=False,
                                                                         master_use_grad=master_use_grad),
                 run_id,
                 num_eigenvalues,
                 total_trunc_num,
                 RESULTS_DIR,
                 bounds=bounds,
                 max_iter=num_epochs,
                 tolerance=0,
                 verbose=True)


if __name__ == "__main__":
    main()
