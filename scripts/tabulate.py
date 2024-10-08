"""
Tabulate results from convergence test.

Usage:
  tabulate.py <yaml_file> [--num_runs=<num_runs>] [--circuit_code=<circuit_code>]
              [--num_best=<num_best>]
  tabulate.py -h | --help
  tabulate.py --version

Arguments:
  <yaml_file>   YAML file containing details about the optimization.

Options:
  -h, --help        Show this screen.
  -v, --version     Show version.

  -n, --num_runs=<num_runs>                 Number of runs to consider.
  -c, --circuit_code=<circuit_code>         Circuit code.
  -b, --num_best=<num_best>                 <num_best> number of best.
"""

from collections import defaultdict
import os

from docopt import docopt

from inout import (
    add_command_line_keys,
    Directory,
    load_yaml_file
)

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
YAML_OR_COMMANDLINE_KEYS = [
    "num_runs",
    "circuit_code"
]

################################################################################
# Main.
################################################################################


def main() -> None:

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

    directory = Directory(parameters, arguments['<yaml_file>'])
    records_dir = directory.get_records_dir()

    ############################################################################
    # Read files
    ############################################################################

    circuit_codes = parameters['circuit_code'].split(',')
    results = defaultdict(lambda:0)

    for circuit_code in circuit_codes:
        for seed in range(int(parameters['num_runs'])):
            save_suffix = (
                f"{circuit_code}"
                f"_{parameters['name']}"
                f"_{seed}"
            )

            summary_save_url = os.path.join(
                records_dir,
                f'summary_{save_suffix}.txt'
            )

            with open(summary_save_url, 'r') as f:
                result = f.readline()
                results[result] += 1
            f.close()

    print("Results:")
    for code in results.keys():
        print(f"Code {code} value: {results[code]}")


if __name__ == "__main__":
    main()
