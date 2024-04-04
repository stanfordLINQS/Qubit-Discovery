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
from collections import defaultdict
from docopt import docopt

import SQcircuit as sq

import analysis as an
from plot_utils import load_final_circuit
from qubit_discovery.losses.loss import calculate_loss_metrics_new
from inout import load_yaml_file, add_command_line_keys, Directory

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
YAML_OR_COMMANDLINE_KEYS = [
    "ids",
    "circuit_code",
    "optim_type",
]

USE_LOSSES = {
    'frequency': 1.0,
    'anharmonicity': 1.0,
    'flux_sensitivity': 1.0,
    'charge_sensitivity': 1.0,
    'T1': 1.0,
}

USE_METRICS = ["T2"]

# unit keys for metrics.
UNITS = {
    'frequency': '[GHz]',
    'T1': '[s]',
    "T2": '[s]'
}
UNITS = defaultdict(lambda: "", UNITS)
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

        # Prepare summary text for the circuit.
        summary_text = f"Description:\n{cr.description(_test=True)}\n"
        total_loss, loss_details, metrics = calculate_loss_metrics_new(
            circuit=cr,
            use_losses=USE_LOSSES,
            use_metrics=USE_METRICS
        )
        summary_text += an.build_circuit_topology_string(cr)
        summary_text += (
            "Metrics:\n" +
            "\n".join(
                f"{key}{UNITS[key]}: {metrics[key]}" for key in metrics.keys()
            ) +
            "\n\nLosses:\n" +
            "\n".join(
                f"{key}: {loss_details[key]}" for key in loss_details.keys()
            )
        )

        with open(
            directory.get_summary_file_dir(parameters['circuit_code'], id_num),
            'w'
        ) as f:
            f.write(summary_text)


if __name__ == '__main__':
    main()
