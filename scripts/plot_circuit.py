import argparse
from collections import defaultdict
import os

import numpy as np
import dill as pickle
from matplotlib import pyplot as plt
import SQcircuit as sq

from settings import RESULTS_DIR

N_EIG = 10
HIGH_RES_PHI = np.concatenate([np.linspace(0, 0.4, 15),
                               np.linspace(0.4, 0.6, 31)[1:],
                               np.linspace(0.6, 1, 15)[1:]])
LOW_RES_PHI = np.concatenate([np.linspace(0, 0.4, 5),
                              np.linspace(0.4, 0.6, 11)[1:],
                              np.linspace(0.6, 1, 5)[1:]])
METRIC_TITLES = ['All Loss', 'Frequency', 'Flux Sensitivity',
                     'Charge Sensitivity', 'Anharmonicity', r'$T_1$']
METRIC_KEYS = ['all_loss', 'omega', 'flux_sensitivity',
               'charge_sensitivity', 'A', 'T1']
LOSS_TITLES = ['Frequency Loss', 'Anharmonicity Loss', '$T_1$ Loss',
                 'Flux Sensitivity Loss', 'Charge Sensitivity Loss', 'Total Loss']
LOSS_KEYS = ['frequency_loss', 'anharmonicity_loss', 'T1_loss',
             'flux_sensitivity_loss', 'charge_sensitivity_loss', 'total_loss']

def load_final_circuit(circuit_record: str) -> sq.Circuit:
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
    parser.add_argument('-l', '--low_res', action='store_true')
    args = parser.parse_args()

    name = args.name
    circuit_code = args.code
    optim_type = args.optimization_type

    ids = args.ids.split(',')

    sq.set_optim_mode(True)
    out_txt = ''
    for id_num in ids:
        identifier = f'{name}_{id_num}' if name is not None else f'{id_num}'

        circuit_path= os.path.join(RESULTS_DIR, 
                                   f'{optim_type}_circuit_record_{circuit_code}_{identifier}.pickle')
        loss_path= os.path.join(RESULTS_DIR, 
                                   f'{optim_type}_loss_record_{circuit_code}_{identifier}.pickle')
        metrics_path= os.path.join(RESULTS_DIR, 
                                   f'{optim_type}_metrics_record_{circuit_code}_{identifier}.pickle')
        old_cr = load_final_circuit(circuit_path)
        with open(loss_path, 'rb') as f:
            loss_record = pickle.load(f)
        with open(metrics_path, 'rb') as f:
            metric_record = pickle.load(f)

        cr = sq.Circuit(old_cr.elements)
        cr.set_trunc_nums(old_cr.m)

        out_txt += identifier + '\n'
        elem_values = {}
        for node in cr.elements.keys():
            elem_values[node] = []
            for elem in cr.elements[node]:
                elem_values[node].append((type(elem), elem._value.item()))
        out_txt += cr.description(tp='txt', _test=True) + '\n'
        out_txt += str(elem_values) + '\n'

        if args.low_res:
            phi_ext = LOW_RES_PHI
        else:
            phi_ext = HIGH_RES_PHI

        spec = np.zeros((N_EIG, len(phi_ext)))

        loop1 = cr.loops[0]
        for i, phi in enumerate(phi_ext):
            loop1.set_flux(phi)
            spec[:, i] = cr.diag(n_eig=N_EIG)[0].detach().numpy()

        n_eig = 10

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        loss_text = ''
        for (key, title) in zip(LOSS_KEYS, LOSS_TITLES):
            loss_text += f'{title}: {loss_record[key][-1]:.3e}\n'
            
        metric_text = ''
        for (key, title) in zip(METRIC_KEYS, METRIC_TITLES):
            metric_text += f'{title}: {metric_record[key][-1]:.3e}\n'

        for i in range(n_eig):
            ax.plot(phi_ext, (spec[i, :] - spec[0, :]), marker='o', markersize=3.5)

        ax.set_xlabel(r"$\Phi_{ext}/\Phi_0$")
        ax.set_ylabel(r" $\omega_n / 2\pi$  (GHz)")

        props = dict(boxstyle='round', facecolor='yellow', alpha=0.1)  # bbox features
        ax.text(1.03, 0.98, loss_text.strip(), transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        props2 = dict(boxstyle='round', facecolor='blue', alpha=0.1)  # bbox features
        ax.text(1.03, 0.45, metric_text.strip(), transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props2)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'circuit_graph_{circuit_code}_{identifier}.png'), dpi=300)

        out_txt += '\n' + '-' * 20 + '\n'
    text_out_identifier = f'{circuit_code}_{name}' if name is not None else circuit_code
    with open(os.path.join(RESULTS_DIR, 'circuit_data_' + text_out_identifier + '.txt'), 'w') as f:
        f.write(out_txt)


            
if __name__ == "__main__":
    main()