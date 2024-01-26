"""Contains code for optimization of single circuit instance."""

import random

from BFGS import run_BFGS
from SGD import run_SGD

from functions import (
    create_sampler,
    build_circuit,
)

import argparse
import numpy as np
import SQcircuit as sq
import torch

# Assign keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument("circuit_type")
parser.add_argument("id")
parser.add_argument("optimization_type")
args = parser.parse_args()

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


def init_quantronium(param_std=0.01):
    E_J = 0.865 * k_B / h  # Hz
    E_C = E_J / 1.27  # Hz
    J_ratio = 20
    C_ratio = 2
    E_J0 = J_ratio * E_J
    C_j = (2 * q_e) ** 2 / (2 * h * E_C)

    E_J1 = E_J / 2 * np.random.normal(loc=1.0, scale=param_std)
    cap_1 = C_j / C_ratio * np.random.normal(loc=1.0, scale=param_std)
    E_J2 = E_J / 2 * np.random.normal(loc=1.0, scale=param_std)
    cap_2 = C_j / C_ratio * np.random.normal(loc=1.0, scale=param_std)
    E_J3 = E_J0 * np.random.normal(loc=1.0, scale=param_std)
    cap_3 = C_j * np.random.normal(loc=1.0, scale=param_std)

    element_dictionary = {(0, 1): [('J', E_J1, 'Hz'), ('C', cap_1, 'F')],
                          (1, 2): [('J', E_J2, 'Hz'), ('C', cap_2, 'F')],
                          (0, 2): [('J', E_J3, 'Hz'), ('C', cap_3, 'F')]}
    quantronium = build_circuit(element_dictionary)

    charge_values = [0.5, 0, 0]
    set_charge_offsets(quantronium, charge_values)
    flux_value = 0.0
    quantronium.loops[0].set_flux(flux_value)

    return quantronium


def init_flux_qubit(param_std=0.01):
    alpha = 0.43
    E_Ja = 36.2  # GHz
    E_ca = 0.35 * 50  # GHz
    E_J = E_Ja / alpha
    # E_c = E_ca * alpha
    C_alpha = 2 * (q_e ** 2 / (E_ca * 1e9 * h))
    C = C_alpha / alpha
    C_sh = 51.0  # fF

    E_J1 = E_J * np.random.normal(loc=1.0, scale=param_std)
    cap_1 = C * np.random.normal(loc=1.0, scale=param_std)
    E_J2 = E_J * np.random.normal(loc=1.0, scale=param_std)
    cap_2 = C * np.random.normal(loc=1.0, scale=param_std)
    E_J3 = E_Ja * np.random.normal(loc=1.0, scale=param_std)
    cap_3 = C_alpha * np.random.normal(loc=1.0, scale=param_std)
    cap_shunt = C_sh * np.random.normal(loc=1.0, scale=param_std)
    flux_qubit_dictionary = {(0, 1): [('J', E_J1, 'GHz'), ('C', cap_1, 'F')],
                             (0, 2): [('J', E_J2, 'GHz'), ('C', cap_2, 'F')],
                             (1, 2): [('J', E_J3, 'GHz'), ('C', cap_3, 'F'),
                                      ('C', cap_shunt, 'fF')]}
    flux_qubit = build_circuit(flux_qubit_dictionary)
    charge_values = [0, 0]
    set_charge_offsets(flux_qubit, charge_values)
    flux_value = 0.4513  # 0.4606
    flux_qubit.loops[0].set_flux(flux_value)
    return flux_qubit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seed = int(args.id)
    set_seed(seed)

    sq.set_optim_mode(True)

    circuit_code = args.circuit_type
    run_id = int(args.id)
    N = len(circuit_code)

    if args.circuit_type == "quantronium":
        circuit = init_quantronium()
    elif args.circuit_type == "flux_qubit":
        circuit = init_flux_qubit()
    else:
        print("Error: Circuit name not recognized. Sampling random JL circuit...")
        sampler = create_sampler(N, capacitor_range, inductor_range, junction_range)
        circuit = sampler.sample_circuit_code('JL')
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
