"""
Plot flux/charge spectra and computational subspace phase plots for a given set of circuits.

Usage:
  plot_analysis.py <yaml_file>  [--ids=<ids> \
    --circuit_code=<circuit_code> --optim_type=<optim_type>]
  plot_analysis.py -h | --help
  plot_analysis.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

  -i, --ids=<ids>                           ids of the circuits to be summarized
  -c, --circuit_code=<circuit_code>         Circuit code
  -o, --optim_type=<optim_type>             Optimization method
"""

import os

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import SQcircuit as sq

import analysis as an
from plot_utils import load_final_circuit, set_plotting_defaults
from inout import (
    load_yaml_file,
    add_command_line_keys,
    Directory,
)

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
YAML_OR_COMMANDLINE_KEYS = [
    "ids",
    "circuit_code",
    "optim_type",
]


def main() -> None:
    ############################################################################
    # Load the Yaml file and command line parameters.
    ############################################################################

    arguments = docopt(__doc__, version='Optimize 0.8')

    parameters = load_yaml_file(arguments['<yaml_file>'])

    parameters = add_command_line_keys(
        parameters=parameters,
        arguments=arguments,
        keys=YAML_OR_COMMANDLINE_KEYS,
    )

    name = parameters['name']
    circuit_code = parameters['circuit_code']
    optim_type = parameters['optim_type']
    ids = parameters['ids'].split(',')

    directory = Directory(parameters, arguments)
    plot_output_folder = directory.get_plots_dir()

    set_plotting_defaults()

    sq.set_optim_mode(True)
    for id_num in ids:
        circuit_path = directory.get_record_file_dir(
            record_type="circuit",
            circuit_code=circuit_code,
            idx=id_num,
        )
        cr = load_final_circuit(circuit_path)
        cr.update()  # rebuild op memory

        save_prefix = f'{optim_type}_plot_{circuit_code}_{name}_{id_num}'
        # Plot flux
        fig, ax = plt.subplots()
        flux_vals, flux_spectra = an.calculate_flux_spectrum(cr)
        an.plot_flux_spectrum(flux_vals, flux_spectra, ax)
        plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.flux.png'),
                    bbox_inches="tight",
                    dpi=300)
        ax.clear()

        # Plot charge
        if np.count_nonzero(cr.omega == 0) > 0:
            # Plot diagonal
            ng_vals, charge_spectra = an.sweep_charge_spectrum(cr, [True] * len(cr.m))
            an.plot_1D_charge_spectrum(ng_vals, charge_spectra, ax)
            plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.charge_diag.png'),
                        dpi=300,
                        bbox_inches="tight")
            ax.clear()

            # And along each of axes
            for charge_mode_idx in cr.charge_islands.keys():
                modes_to_sweep = [False] * len(cr.m)
                modes_to_sweep[charge_mode_idx] = True
                fig, ax = plt.subplots()
                ng_vals, charge_spectra = an.sweep_charge_spectrum(cr, modes_to_sweep)
                an.plot_1D_charge_spectrum(ng_vals, charge_spectra, ax)
                plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.charge_{charge_mode_idx}.png'),
                            bbox_inches="tight",
                            dpi=300)
                ax.clear()

        if np.count_nonzero(cr.omega == 0) == 2:
            # 2d plot
            modes_to_sweep = [True if w == 0 else False for w in cr.omega]
            fig, ax = plt.subplots()
            ng_vals, charge_spectra = an.grid_charge_spectrum(cr, modes_to_sweep)
            an.plot_2D_charge_spectrum(ng_vals[0], ng_vals[1], charge_spectra, ax)
            plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.charge_2D.png'),
                        bbox_inches="tight",
                        dpi=300)
            ax.clear()

        # fig, axs = plt.subplots(1, 2)
        # phi_range, state0 = an.calc_state(cr, 0, len(cr.m))
        # an.plot_state_phase(phi_range, state0, axs[0])
        # phi_range, state1 = an.calc_state(cr, 1, len(cr.m))
        # an.plot_state_phase(phi_range, state1, axs[1])
        # fig.tight_layout()
        # plt.savefig(os.path.join(plot_output_folder, f'{save_prefix}.states.png'),
        #             bbox_inches="tight",
        #             dpi=300)


if __name__ == '__main__':
    main()
