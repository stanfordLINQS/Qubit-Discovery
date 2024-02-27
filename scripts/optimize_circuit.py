"""Contains code for optimization of single circuit instance."""

import argparse
import random

from qubit_discovery.optimization.utils import create_sampler
from qubit_discovery.optimization import run_SGD, run_BFGS
from qubit_discovery.losses import loss_functions
from analysis import (
    build_circuit,
)
from build_literature_circuits import build_flux_qubit, build_quantronium

import numpy as np
import SQcircuit as sq
import torch

from settings import RESULTS_DIR

# Optimization settings
num_epochs = 100  # number of training iterations
num_eigenvalues = 10
total_trunc_num = 800
baseline_trunc_num = 400
# total_trunc_num = 140
# baseline_trunc_num = 100

# Target parameter range
# 1e-15 F
# 2e-8 H
capacitor_range = [1e-15, 12e-12]  # F
inductor_range = [1e-15, 5e-6]  # H
junction_range = [1e9 * 2 * np.pi, 100e9 * 2 * np.pi]  # Hz
# capacitor_range = [8e-15, 12e-14] # F
# inductor_range = [2e-7, 5e-6] # H
# junction_range = [1e9 * 2 * np.pi, 12e9 * 2 * np.pi] # Hz

element_verbose = False

k_B = 1.38e-23  # J/K
h = 6.626e-34  # J/Hz
hbar = h / (2 * np.pi)
q_e = 1.602e-19


def set_charge_offsets(circuit, charge_values):
    for charge_island_idx in circuit.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        circuit.set_charge_offset(charge_mode, charge_values[charge_island_idx])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Assign keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit_type")
    parser.add_argument("id")
    parser.add_argument("optimization_type")
    args = parser.parse_args()

    seed = int(args.id)
    set_seed(seed)

    sq.set_optim_mode(True)

    circuit_code = args.circuit_type
    run_id = int(args.id)
    N = len(circuit_code)

    if args.circuit_type == "quantronium":
        circuit = build_quantronium()
    elif args.circuit_type == "flux_qubit":
        circuit = build_flux_qubit()
    else:
        print("Error: Circuit name not recognized. Sampling random JL circuit...")
        sampler = create_sampler(N, capacitor_range, inductor_range, junction_range)
        circuit = sampler.sample_circuit_code('JL')
        print("Circuit sampled!")

    baseline_trunc_nums = circuit.truncate_circuit(baseline_trunc_num)
    circuit.diag(num_eigenvalues)
    loss_metric_function = loss_functions['constant_norm']

    # circuit, circuit_code, seed, num_eigenvalues, total_trunc_num, num_epochs
    if args.optimization_type == "SGD":
        bounds = {
            sq.Capacitor: (capacitor_range[0], capacitor_range[1]),
            sq.Inductor: (inductor_range[0], inductor_range[1]),
            sq.Junction: (junction_range[0], junction_range[1])
        }
        run_SGD(circuit,
                circuit_code,
                lambda cr: loss_metric_function(cr,
                                                use_frequency_loss=False, 
                                                use_anharmonicity_loss=True,
                                                use_flux_sensitivity_loss=True, 
                                                use_charge_sensitivity_loss=True,
                                                use_T1_loss=False),
                args.circuit_type,
                num_eigenvalues,
                baseline_trunc_nums,
                total_trunc_num,
                num_epochs,
                bounds,
                RESULTS_DIR,
                save_intermediate_circuits=False
                )
    elif args.optimization_type == "BFGS":
        bounds = {
            sq.Junction: torch.tensor([junction_range[0], junction_range[1]]),
            sq.Inductor: torch.tensor([inductor_range[0], inductor_range[1]]),
            sq.Capacitor: torch.tensor([capacitor_range[0], capacitor_range[1]])
        }

        run_BFGS(circuit,
                 circuit_code,
                 run_id,
                 num_eigenvalues,
                 baseline_trunc_num,
                 total_trunc_num,
                 bounds=bounds,
                 max_iter=num_epochs,
                 tolerance=0)


if __name__ == "__main__":
    main()
