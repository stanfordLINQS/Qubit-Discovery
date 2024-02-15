import argparse
import os

import matplotlib.pyplot as plt
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

    for id_num in ids:
        circuit_path = os.path.join(
            RESULTS_DIR, records_folder, f'{optim_type}_circuit_record_{circuit_code}_{name}_{id_num}.pickle')
        cr = load_final_circuit(circuit_path)

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs[1,1]

        sq.set_optim_mode(True)
        flux_vals, flux_spectra = an.calculate_flux_spectrum(cr)
        an.plot_flux_spectrum(flux_vals, flux_spectra, axs[0, 0])
        try:
            ng_vals, charge_specta = an.calculate_charge_spectrum(cr, [True] * len(cr.m))
            an.plot_charge_spectrum(ng_vals, charge_specta, axs[0, 1])
        except TypeError:
            pass
        # phi_range, state0 = an.calc_state(cr, 0, len(cr.m))
        # an.plot_state_phase(phi_range, state0, axs[1, 0])
        # phi_range, state1 = an.calc_state(cr, 1, len(cr.m))
        # an.plot_state_phase(phi_range, state1, axs[1, 1])

        plt.savefig(os.path.join(plot_output_folder, '{optim_type}_circuit_record_{circuit_code}_{identifier}_analysis.png'), dpi=300)


    


if __name__ == '__main__':
    main()