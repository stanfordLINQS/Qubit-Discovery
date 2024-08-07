"""
Get a summary of the circuit metrics and loss valuations from saved circuit
files after optimizations, and output as a text file.

Usage:
  circuit_summary.py <yaml_file> [--ids=<ids>] [--circuit_code=<circuit_code>]
                     [--optim_type=<optim_type>]
  circuit_summary.py  -h | --help
  circuit_summary.py  --version

Arguments:
  <yaml_file>   YAML file containing details about the optimization.

Options:
  -h, --help    Show this screen.
  --version     Show version.

  -i, --ids=<ids>                           Comma-delimited list of circuits to
                                            summarize.
  -c, --circuit_code=<circuit_code>         Circuit code.
  -o, --optim_type=<optim_type>             Optimization method used.
"""

from docopt import docopt
import qubit_discovery as qd
from qubit_discovery.losses.loss import calculate_loss_metrics
import SQcircuit as sq

import analysis as an
from inout import (
    load_yaml_file,
    add_command_line_keys,
    Directory,
    get_metrics_dict,
    get_units,
)
from utils import load_final_circuit

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
SUMMARY_REQUIRED_KEYS = [
    'ids',
    'circuit_code',
    'optim_type',
]

################################################################################
# Main.
################################################################################


def main():

    ############################################################################
    # Loading the Yaml file and command line parameters.
    ############################################################################

    arguments = docopt(__doc__, version='Optimize 0.8')

    parameters = load_yaml_file(arguments['<yaml_file>'])

    parameters = add_command_line_keys(
        parameters=parameters,
        arguments=arguments,
        keys=SUMMARY_REQUIRED_KEYS,
    )

    directory = Directory(parameters, arguments['<yaml_file>'])

    ############################################################################
    # Summarize the Circuit.
    ############################################################################

    sq.set_engine('PyTorch')
    qd.log_to_stdout()

    for id_num in parameters['ids'].split(','):
        cr = load_final_circuit(directory.get_record_file_dir(
                record_type='circuit',
                circuit_code=parameters['circuit_code'],
                idx=id_num,
        ))
        cr.update()  # rebuild op memory
        cr.diag(parameters['num_eigenvalues'])

        metrics_in_optim, metrics_not_in_optim = get_metrics_dict(parameters)
        # add in element sensitivity for summary file
        # metrics_not_in_optim.append('element_sensitivity')

        # Prepare summary text for the circuit.
        summary_text = f'Description:\n{cr.description(_test=True)}\n'
        summary_text += an.build_circuit_topology_string(cr)

        summary_text += f'\ntrunc_nums: {cr.trunc_nums}\n'

        total_loss, loss_details, metrics = calculate_loss_metrics(
            circuit=cr,
            use_losses=parameters['use_losses'],
            use_metrics=metrics_not_in_optim
        )
        ########################################################################
        # In optimization losses
        ########################################################################
        loss_in_optim_summary = 'Loss in Optimization:\n'
        for key in metrics_in_optim:
            if loss_details[key + '_loss'] == 0.0:
                continue
            loss_in_optim_summary += (
                f"{key + '_loss'}: {loss_details[key + '_loss']}\n"
            )
        loss_in_optim_summary += (
            f'{"total_loss"}: {loss_details["total_loss"]}\n'
        )
        ########################################################################
        # Not in optimization losses
        ########################################################################
        loss_not_in_optim_summary = 'Other Losses:\n'
        for key in metrics_not_in_optim:
            if loss_details[key + '_loss'] == 0.0:
                continue
            loss_not_in_optim_summary += (
                f'{key + "_loss"}: {loss_details[key+ "_loss"]}\n'
            )
        ########################################################################
        # All the metrics
        ########################################################################
        metrics_summary = '\nMetrics:\n'
        for key in metrics_in_optim + metrics_not_in_optim:
            metrics_summary += f"{key} {get_units()[key]}: {metrics[key]}\n"

        summary_text += (
            metrics_summary +
            '\n' +
            loss_in_optim_summary +
            '\n' +
            loss_not_in_optim_summary
        )

        with open(
            directory.get_summary_file_dir(parameters['circuit_code'], id_num),
            'w'
        ) as f:
            f.write(summary_text)


if __name__ == '__main__':
    main()
