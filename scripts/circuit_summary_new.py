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
from plot_utils import load_final_circuit
from qubit_discovery.losses import loss_functions
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

METRICS = {
    'omega': 'Frequency',
    'flux_sensitivity': 'Flux Sensitivity',
    'charge_sensitivity': 'Charge Sensitivity',
    'A': 'Anharmonicity',
    'T1': 'T_1 Time (s)',
    'T2': 'T_2 Time (s)'
}
LOSSES = {
    'frequency_loss': 'Frequency Loss',
    'anharmonicity_loss': 'Anharmonicity Loss',
    'T1_loss': 'T_1 Loss',
    'flux_sensitivity_loss': 'Flux Sensitivity Loss',
    'charge_sensitivity_loss': 'Charge Sensitivity Loss',
    'total_loss': 'Total Loss'
}

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
        loss_function = loss_functions['default']
        total_loss, loss_details, metrics = loss_function(cr)
        summary_text += an.build_circuit_topology_string(cr)
        summary_text += (
            "Metrics:\n" +
            "\n".join(f"{METRICS[key]}: {metrics[key]}" for key in METRICS) +
            "\n\nLosses:\n" +
            "\n".join(f"{LOSSES[key]}: {loss_details[key]}" for key in LOSSES)
        )

        with open(
            directory.get_summary_file_dir(parameters['circuit_code'], id_num),
            'w'
        ) as f:
            f.write(summary_text)


if __name__ == '__main__':
    main()
