"""Contains code for optimization of single circuit instance."""

import random

from BFGS import run_BFGS
from SGD import run_SGD

from functions import (
    create_sampler, set_params,
)

from loss import (
    calculate_loss,
    frequency_loss,
    anharmonicity_loss
)

import argparse
import numpy as np
import SQcircuit as sq
import torch

import psutil

# Assign keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument("code")
parser.add_argument("id")
parser.add_argument("optimization_type")
args = parser.parse_args()

# Optimization settings

num_epochs = 100  # number of training iterations
num_eigenvalues = 10
total_trunc_num = 140

# Target parameter range
capacitor_range = [1e-15, 12e-12]  # F
inductor_range = [2e-8, 5e-6]  # H
junction_range = [1e9 * 2 * np.pi, 100e9 * 2 * np.pi]  # Hz
# capacitor_range = [8e-15, 12e-14] # F
# inductor_range = [2e-7, 5e-6] # H
# junction_range = [1e9 * 2 * np.pi, 12e9 * 2 * np.pi] # Hz

element_verbose = False

def check_memory():
    print(f"Total RAM usage (in MB): {psutil.Process().memory_info().rss / (1024 * 1024)}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def objective_func(circuit, x, num_eigenvalues):
    set_params(circuit, x)
    circuit.diag(num_eigenvalues)
    loss_frequency = frequency_loss(circuit)
    loss_anharmonicity = anharmonicity_loss(circuit)

    return loss_frequency + loss_anharmonicity

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

    # TEMP
    '''trunc_nums = circuit.truncate_circuit(total_trunc_num)'''
    trunc_nums = [20, 20]
    circuit.set_trunc_nums(trunc_nums)
    print("Circuit truncated...")

    circuit.diag(num_eigenvalues)
    print("Circuit diagonalized")

    for iteration in range(num_epochs):
        check_memory()
        params = torch.stack(circuit.parameters).clone()
        params = params + torch.rand(1) * params * 1e-5
        loss = objective_func(circuit, params, num_eigenvalues)
        loss.backward()

if __name__ == "__main__":
    main()