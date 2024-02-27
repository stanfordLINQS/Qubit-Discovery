import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import SQcircuit as sq

import analysis as an
from plot_utils import add_file_args, load_final_circuit
from qubit_discovery.losses import loss_functions
from settings import RESULTS_DIR

N_EIG = 10

METRICS = {'omega': 'Frequency',
           'flux_sensitivity': 'Flux Sensitivity',
           'charge_sensitivity': 'Charge Sensitivity',
           'A': 'Anharmonicity',
           'T1': 'T_1 Time (s)',
           'T2': 'T_2 Time (s)'}
LOSSES = {'frequency_loss': 'Frequency Loss',
          'anharmonicity_loss': 'Anharmonicity Loss',
          'T1_loss': 'T_1 Loss',
          'flux_sensitivity_loss': 'Flux Sensitivity Loss',
          'charge_sensitivity_loss': 'Charge Sensitivity Loss',
          'total_loss': 'Total Loss'
          }

def main():       
    parser = argparse.ArgumentParser()
    add_file_args(parser)
    parser.add_argument('-i', '--ids', required=True,
                        help='Id numbers of circuit to plot, in comma-delimited list')
    args = parser.parse_args()

    name = args.name
    circuit_code = args.codes
    optim_type = args.optimization_type
    ids = args.ids.split(',')

    experiment_folder = f"{args.optimization_type}_{args.name}"
    records_folder = os.path.join(experiment_folder, 'records/')
    plot_output_folder = os.path.join(RESULTS_DIR, experiment_folder, "plots")
    os.makedirs(plot_output_folder, exist_ok=True)

    sq.set_optim_mode(True)
    for id_num in ids:
        circuit_path = os.path.join(
            RESULTS_DIR, records_folder, f'{optim_type}_circuit_record_{circuit_code}_{name}_{id_num}.pickle')
        cr = load_final_circuit(circuit_path)
        cr._toggle_fullcopy = True

        cr.update() # rebuild op memory
        cr.diag(N_EIG)

        # Summarize circuit loss and metrics
        out_txt = ""
        out_txt += f"Description:\n{cr.description(_test=True)}\n"
        summary_path = os.path.join(
            RESULTS_DIR, plot_output_folder, f'{optim_type}_circuit_record_{circuit_code}_{name}_{id_num}.txt')
        loss_metric_function = loss_functions['constant_norm']
        total_loss, loss_values, metric_values = loss_metric_function(cr)

        out_txt += "\nMetrics:\n"
        for metric_key, metric_title in METRICS.items():
            out_txt += f"{metric_title}: {metric_values[metric_key]}\n"
        out_txt += "\nLosses:\n"
        for loss_key, loss_title in LOSSES.items():
            out_txt += f"{loss_title}: {loss_values[loss_key]}\n"

        with open(summary_path, 'w') as f:
            f.write(out_txt)


if __name__ == '__main__':
    main()