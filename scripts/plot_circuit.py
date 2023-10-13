import argparse
from collections import defaultdict
import os

import numpy as np
import dill as pickle
from matplotlib import pyplot as plt
from SQcircuit import Circuit

from settings import RESULTS_DIR
from plot_utils import load_record

HIGH_RES_PHI = np.concatenate([np.linspace(0, 0.4, 15),
                               np.linspace(0.4, 0.6, 31)[1:],
                               np.linspace(0.6, 1, 16)[1:]])
LOW_RES_PHI = np.concatenate([np.linspace(0, 0.4, 5),
                              np.linspace(0.4, 0.6, 11)[1:],
                              np.linspace(0.6, 1, 6)[1:]])
METRIC_TITLES = ['All Loss', 'Frequency', 'Flux Sensitivity',
                     'Charge Sensitivity', 'Anharmonicity', r'$T_1$']
METRIC_KEYS = ['all_loss', 'omega', 'flux_sensitivity',
               'charge_sensitivity', 'A', 'T1']
LOSS_TITLES = ['Frequency Loss', 'Anharmonicity Loss', '$T_1$ Loss',
                 'Flux Sensitivity Loss', 'Charge Sensitivity Loss', 'Total Loss']
LOSS_KEYS = ['frequency_loss', 'anharmonicity_loss', 'T1_loss',
             'flux_sensitivity_loss', 'charge_sensitivity_loss', 'total_loss']

def load_final_circuit(circuit_record: str) -> Circuit:
    with open(circuit_record, 'rb') as f:
        try:
            while True:
                last_circ = pickle.load(f)
        except EOFError:
            pass
    return last_circ

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--code', type=str, required=True)
    parser.add_argument('-o', '--optimization_type', type=str, required=True)
    parser.add_argument('-n', '--name')
    parser.add_argument('-i', '--ids')
    args = parser.parse_args()

    name = args.name
    circuit_codes = args.code
    optim_type = args.optimization_type

    ids = args.ids.split(',')

    for id_num in ids:
        identifier = f'{name}_{id_num}' if name is not None else f'{id_num}'

        loss_record = load_record(os.path.join(
                RESULTS_DIR, f'{optim_type}_loss_record_{circuit_codes}_{identifier}.pickle'))
        


            
if __name__ == "__main__":
    main()