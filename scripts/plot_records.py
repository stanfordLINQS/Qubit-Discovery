"""
Plot optimization results.

Usage:
  plot_records.py <yaml_file>  [--num_runs=<num_runs> \
--circuit_code=<circuit_code> --optim_type=<optim_type> --num_best=<num_best>]
  plot_records.py -h | --help
  plot_records.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.

  -n, --num_runs=<num_runs>                 Number of runs to consider
  -c, --circuit_code=<circuit_code>         Circuit code
  -o, --optim_type=<optim_type>             Optimization method
  -b, --num_best=<num_best>                 <num_best> number of best [Optional]
"""

from collections import defaultdict
from typing import Dict, List

from docopt import docopt
from matplotlib import pyplot as plt

from plot_utils import load_record
from inout import (
    load_yaml_file,
    add_command_line_keys,
    Directory,
    get_units,
)

################################################################################
# General Settings.
################################################################################

# Keys that should be in either command line or Yaml file.
YAML_OR_COMMANDLINE_KEYS = [
    "num_runs",
    "circuit_code",
    "optim_type",
]
# optional keys.
OPTIONAL_KEYS = ["num_best"]

PLOT_SCHEME = {
    'JJ': 'b', 'JL': 'darkorange',
    'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'
}

# metric keys for plotting.
METRIC_KEYS = [
    'flux_sensitivity',             # (0, 0) plot position
    'charge_sensitivity',           # (0, 1) plot position
    'anharmonicity',                # (0, 2) plot position
    't2_flux',                      # (1, 0) plot position
    't2_charge',                    # (1, 1) plot position
    't2_cc',                        # (1, 2) plot position
    't1',                           # (2, 0) plot position
    't2',                           # (2, 1) plot position
    'frequency',                    # (2, 2) plot position
    't',                            # (3, 0) plot position
    'gate_speed',                   # (3, 1) plot position
    'number_of_gates',              # (3, 2) plot position
]

# loss keys for plotting .
LOSS_KEYS = [
    'flux_sensitivity_loss',        # (0, 0) plot position
    'charge_sensitivity_loss',      # (0, 1) plot position
    'frequency_loss',               # (0, 2) plot position
    'number_of_gates_loss',         # (1, 0) plot position
    'total_loss'                    # (1, 1) plot position
]

################################################################################
# Helper functions.
################################################################################


def capitalize_metric(input_str):
    """Capitalize metric string for title purposes."""

    # Split the input string by underscores
    parts = input_str.split('_')
    # Capitalize the first letter of each part and join them back with a space
    formatted_str = ' '.join(part.capitalize() for part in parts)
    return formatted_str


def compute_best_ids(
    aggregate_loss_records,
    n: int,
    codes: List[str]
) -> Dict[str, List[int]]:
    out = {}
    for circuit_code in codes:
        code_loss_record = aggregate_loss_records[circuit_code].items()

        sorted_runs = sorted(
            code_loss_record,
            key=lambda x: x[1]['total_loss'][-1]
        )

        out[circuit_code] = [id_num for id_num, run in sorted_runs][:n]

        print(
            f'Top {n} runs in order are '
            f'{out[circuit_code]} for code {circuit_code}.'
        )
    return out


def plot_circuit_metrics(
    run,
    axs,
    plot_type: str,
    circuit_code: str,
    best: bool,
) -> None:

    record_keys = METRIC_KEYS if plot_type == 'metrics' else LOSS_KEYS
    # l = 3 if plot_type == 'metrics' else 2

    for plot_idx in range(len(record_keys)):
        key = record_keys[plot_idx]
        i, j = plot_idx // 3, plot_idx % 3
        try:
            axs[i, j].plot(
                run[key],
                PLOT_SCHEME[circuit_code],
                label=circuit_code if best else None,
                alpha=None if best else 0.3,
                linestyle=None if best else '--',
            )
            axs[i, j].set_title(
                capitalize_metric(key) + f" {get_units()[key]}"
            )
            # if key not in ['number_of_gates_loss', 'total_loss']:
            axs[i, j].set_yscale('log')
            axs[i, j].legend(loc="upper left")
        except KeyError:
            pass


def plot_results(
    record,
    plot_folder_directory,
    best_ids: Dict[str, List[int]],
    plot_type: str,
    save_prefix: str = '',
) -> None:

    if plot_type == 'metrics':
        fig, axs = plt.subplots(4, 3, figsize=(28, 16))
    elif plot_type == 'loss':
        fig, axs = plt.subplots(2, 3, figsize=(20, 8))
    else:
        raise ValueError(
            f"Unknown plot type: {plot_type}. Must be "
            f"'metrics' or 'loss'"
        )

    for circuit_code, ids_list in best_ids.items():
        # Plot best run for that specific code
        runs_list = [record[circuit_code][id_num] for id_num in ids_list]

        for i, run in enumerate(runs_list):
            # i=0 is always the best circuit.
            plot_circuit_metrics(
                run=run,
                axs=axs,
                plot_type=plot_type,
                circuit_code=circuit_code,
                best=True if i == 0 else False,
            )

    plt.savefig(
        f'{plot_folder_directory}/{save_prefix}_{plot_type}_record.png',
        dpi=500
    )


def build_save_prefix(parameters) -> str:
    save_prefix = ""
    for code in parameters['circuit_code']:
        save_prefix += code
    if parameters['num_runs'] is not None:
        save_prefix += f"_n_{parameters['num_runs']}"
    if parameters['num_best'] is not None:
        save_prefix += f"_b_{parameters['num_best']}"
    if parameters['optim_type'] is not None:
        save_prefix += f"_{parameters['optim_type']}"
    if parameters['name'] is not None:
        save_prefix += f"_{parameters['name']}"

    return save_prefix

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
        optional_keys=OPTIONAL_KEYS,
    )

    if parameters['num_best'] is not None:
        best_n = int(parameters['num_best'])
    else:
        best_n = int(parameters['num_runs'])

    directory = Directory(parameters, arguments)

    ############################################################################
    # Plotting
    ############################################################################

    circuit_codes = parameters['circuit_code'].split(',')
    aggregate_loss_record = defaultdict(dict)
    aggregate_metrics_record = defaultdict(dict)

    success_count = 0
    for circuit_code in circuit_codes:
        for id_num in range(int(parameters['num_runs'])):

            loss_record = load_record(directory.get_record_file_dir(
                record_type="loss",
                circuit_code=circuit_code,
                idx=id_num,
            ))

            metrics_record = load_record(directory.get_record_file_dir(
                record_type="metrics",
                circuit_code=circuit_code,
                idx=id_num,
            ))

            if loss_record is not None and metrics_record is not None:
                success_count += 1
                aggregate_loss_record[circuit_code][id_num] = loss_record
                aggregate_metrics_record[circuit_code][id_num] = metrics_record

    save_prefix = build_save_prefix(parameters)
    best_ids = compute_best_ids(aggregate_loss_record, best_n, circuit_codes)

    plot_results(
        record=aggregate_loss_record,
        plot_folder_directory=directory.get_plots_dir(),
        best_ids=best_ids,
        plot_type='loss',
        save_prefix=save_prefix
    )
    plot_results(
        record=aggregate_metrics_record,
        plot_folder_directory=directory.get_plots_dir(),
        best_ids=best_ids,
        plot_type='metrics',
        save_prefix=save_prefix
    )
    print(
        f"Loaded {success_count} of "
        f"{len(circuit_codes) * int(parameters['num_runs'])} "
        f"successful runs."
    )


if __name__ == "__main__":
    main()
