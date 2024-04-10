"""
Get the summary of the circuit metrics and loss valuations.

Usage:
  summary yaml <yaml_file>  [--ids=<ids> --circuit_code=<circuit_code> \
--optim_type=<optim_type>]
  summary -h | --help
  summary --version

Options:
  -h --help     Show this screen.
  --version     Show version.

  -i, --ids=<ids>                           ids of the circuits to be summarized
  -c, --circuit_code=<circuit_code>         Circuit code
  -o, --optim_type=<optim_type>             Optimization method
"""

from docopt import docopt

import SQcircuit as sq

import analysis as an
<<<<<<< HEAD
from plot_utils import add_file_args, load_final_circuit
from qubit_discovery.losses import loss_functions
from settings import RESULTS_DIR

N_EIG = 10

METRICS = {'omega': 'Frequency',
           'flux_sensitivity': 'Flux Sensitivity',
           'charge_sensitivity': 'Charge Sensitivity',
           'A': 'Anharmonicity',
           'T1': 'T_1 Time (s)',
           'T2': 'T_2 Time (s)',
           'gate': 'Gate Estimate'}
LOSSES = {'frequency_loss': 'Frequency Loss',
          'anharmonicity_loss': 'Anharmonicity Loss',
          'T1_loss': 'T_1 Loss',
          'flux_sensitivity_loss': 'Flux Sensitivity Loss',
          'charge_sensitivity_loss': 'Charge Sensitivity Loss',
          'experimental_sensitivity_loss': 'Experimental Sensitivity Loss',
          'gate_loss': 'Gate Loss',
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
=======
from plot_utils import load_final_circuit
from qubit_discovery.losses.loss import calculate_loss_metrics
from inout import (
    load_yaml_file,
    add_command_line_keys,
    Directory,
    get_metrics_dist,
    get_units,
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
        keys=YAML_OR_COMMANDLINE_KEYS,
    )

    directory = Directory(parameters, arguments)

    ############################################################################
    # Summarize the Circuit.
    ############################################################################
>>>>>>> origin/dev-tr

    sq.set_optim_mode(True)
    for id_num in parameters['ids'].split(','):

        cr = load_final_circuit(directory.get_record_file_dir(
                record_type="circuit",
                circuit_code=parameters['circuit_code'],
                idx=id_num,
        ))
        cr._toggle_fullcopy = True
        cr.update()  # rebuild op memory
        cr.diag(parameters["num_eigenvalues"])

        metrics_in_optim, metrics_not_in_optim = get_metrics_dist(parameters)

<<<<<<< HEAD
        # Summarize circuit loss and metrics
        out_txt = ""
        out_txt += f"Description:\n{cr.description(_test=True)}\n"
        summary_path = os.path.join(
            RESULTS_DIR, plot_output_folder, f'{optim_type}_circuit_record_{circuit_code}_{name}_{id_num}.txt')
        loss_metric_function = loss_functions['default']
        total_loss, loss_values, metric_values = loss_metric_function(cr)
=======
        # Prepare summary text for the circuit.
        summary_text = f"Description:\n{cr.description(_test=True)}\n"
        summary_text += an.build_circuit_topology_string(cr)
>>>>>>> origin/dev-tr

        total_loss, loss_details, metrics = calculate_loss_metrics(
            circuit=cr,
            use_losses=parameters['use_losses'],
            use_metrics=metrics_not_in_optim
        )
        ########################################################################
        # In optimization losses
        ########################################################################
        loss_in_optim_summary = "Loss in Optimization:\n"
        for key in metrics_in_optim:
            if loss_details[key + '_loss'] == 0.0:
                continue
            loss_in_optim_summary += (
                f"{key + '_loss'}: {loss_details[key + '_loss']}\n"
            )
        loss_in_optim_summary += (
            f"{'total_loss'}: {loss_details['total_loss']}\n"
        )
        ########################################################################
        # Not in optimization losses
        ########################################################################
        loss_not_in_optim_summary = "Other Losses:\n"
        for key in metrics_not_in_optim:
            if loss_details[key + '_loss'] == 0.0:
                continue
            loss_not_in_optim_summary += (
                f"{key + '_loss'}: {loss_details[key+ '_loss']}\n"
            )
        ########################################################################
        # All the metrics
        ########################################################################
        metrics_summary = "Metrics:\n"
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
