import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import SQcircuit as sq

import analysis as an
from plot_utils import add_file_args, load_final_circuit
from settings import RESULTS_DIR

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
        cr.update() # rebuild op memory

        save_prefix = f'{optim_type}_plot_{circuit_code}_{name}_{id_num}'
        # Plot flux
        if (sum(cr.omega) > 0):
            fig, ax = plt.subplots()
            flux_vals, flux_spectra = an.calculate_flux_spectrum(cr)
            an.plot_flux_spectrum(flux_vals, flux_spectra, ax)
            plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.flux.png'), dpi=300)

        # Plot charge
        if (np.count_nonzero(cr.omega == 0) > 0):
            # Plot diagonal
            fig, ax = plt.subplots()
            ng_vals, charge_specta = an.sweep_charge_spectrum(cr, [True] * len(cr.m))
            an.plot_1D_charge_spectrum(ng_vals, charge_specta, ax)
            plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.charge_diag.png'), dpi=300)

            # And along each of axes
            for charge_mode_idx in cr.charge_islands.keys():
                modes_to_sweep = [False] * len(cr.m)
                modes_to_sweep[charge_mode_idx] = True
                fig, ax = plt.subplots()
                ng_vals, charge_specta = an.sweep_charge_spectrum(cr, modes_to_sweep)
                an.plot_1D_charge_spectrum(ng_vals, charge_specta, ax)
                plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.charge_{charge_mode_idx}.png'), dpi=300)
        if (np.count_nonzero(cr.omega == 0) == 2):
            # 2d plot
            modes_to_sweep = [True if w == 0 else False for w in cr.omega]
            fig, ax = plt.subplots()
            ng_vals, charge_specta = an.grid_charge_spectrum(cr, modes_to_sweep)
            an.plot_2D_charge_spectrum(ng_vals[0], ng_vals[1], charge_specta, ax)
            plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.charge_2D.png'), dpi=300)

        fig, axs = plt.subplots(1, 2)
        phi_range, state0 = an.calc_state(cr, 0, len(cr.m))
        an.plot_state_phase(phi_range, state0, axs[0])
        phi_range, state1 = an.calc_state(cr, 1, len(cr.m))
        an.plot_state_phase(phi_range, state1, axs[1])
        fig.tight_layout()
        plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.states.png'), dpi=300)


if __name__ == '__main__':
    main()