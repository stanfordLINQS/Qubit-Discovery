"""Contains code for optimization of single circuit instance."""

import random

from BFGS import run_BFGS
from SGD import run_SGD

from functions import (
    create_sampler,
)

import argparse
import numpy as np
import SQcircuit as sq
import torch


# Assign keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument("code")
parser.add_argument("id")
parser.add_argument("optimization_type")
args = parser.parse_args()

# Optimization settings

num_epochs = 100  # number of training iterations
num_eigenvalues = 10
total_trunc_num = 8000
baseline_trunc_num = 2000
# total_trunc_num = 140
# baseline_trunc_num = 100

# Target parameter range
# 1e-15 F
# 2e-8 H
capacitor_range = [1e-15, 12e-12]  # F
inductor_range = [1e-15, 5e-6]  # H
junction_range = [1e9 * 2 * np.pi, 100e9 * 2 * np.pi] # Hz
# capacitor_range = [8e-15, 12e-14] # F
# inductor_range = [2e-7, 5e-6] # H
# junction_range = [1e9 * 2 * np.pi, 12e9 * 2 * np.pi] # Hz

element_verbose = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seed = int(args.id)
    set_seed(seed)

    sq.set_optim_mode(True)

    circuit_code = args.code
    run_id = int(args.id)
    N = len(circuit_code)

    sampler = create_sampler(N, capacitor_range, inductor_range, junction_range)
    circuit = sampler.sample_circuit_code(circuit_code)
    print("Circuit sampled!")

    baseline_trunc_nums = circuit.truncate_circuit(baseline_trunc_num)
    # trunc_nums = [100, 100]
    circuit.set_trunc_nums(baseline_trunc_nums)
    circuit.diag(num_eigenvalues)

    # circuit, circuit_code, seed, num_eigenvalues, total_trunc_num, num_epochs
    if args.optimization_type == "SGD":
        run_SGD(circuit,
                circuit_code,
                run_id,
                num_eigenvalues,
                baseline_trunc_num,
                total_trunc_num,
                num_epochs
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
