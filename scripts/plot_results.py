import argparse
from collections import defaultdict
import os
from typing import Dict, List

from matplotlib import pyplot as plt

from plot_utils import add_file_args, code_to_codename, load_record
from qubit_discovery.losses.loss import OMEGA_TARGET
from settings import RESULTS_DIR


def compute_best_ids(
    aggregate_loss_records,
    n: int,
    codes: List[str]
) -> Dict[str, List[int]]:
    out = {}
    for codename in codes:
        code_loss_record = aggregate_loss_records[codename].items()
        sorted_runs = sorted(code_loss_record, key=lambda x: x[1]['total_loss'][-1])
        out[codename] = [id_num for id_num, run in sorted_runs][:n]
        print(f'Top {n} runs in order are {out[codename]} for code {codename}.')
    return out


def plot_results(
    record,
    plot_folder,
    best_ids: Dict[str, List[int]],
    plot_type: str,
    title: str = '',
    save_prefix: str = '',
) -> None:

    PLOT_SCHEME = {'Transmon': 'b', 'Fluxonium': 'darkorange',
                   'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'}

    METRIC_TITLES = [r'$T_2$ Time (s)', 'Frequency (GHz)', 'Flux Sensitivity',
                     'Charge Sensitivity', 'Anharmonicity', r'$T_1$ Time (s)']

    METRIC_KEYS = ['T2', 'omega', 'flux_sensitivity',
                   'charge_sensitivity', 'A', 'T1']

    LOSS_TITLES = ['Frequency Loss', 'Anharmonicity Loss',
                   'T1 Loss', 'Flux Sensitivity Loss',
                   'Charge Sensitivity Loss', 'Total Loss']

    LOSS_KEYS = ['frequency_loss', 'anharmonicity_loss',
                 'T1_loss', 'flux_sensitivity_loss',
                 'charge_sensitivity_loss', 'total_loss']
    
    fig, axs = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle(title, fontsize=32)
    plot_titles = METRIC_TITLES if plot_type == 'metrics' else LOSS_TITLES
    record_keys = METRIC_KEYS if plot_type == 'metrics' else LOSS_KEYS

    def plot_circuit_metrics(
        run,
        code: str,
        best: bool,
        show_label: bool = False,
    ) -> None:
        codename = code_to_codename(code)
        label = codename if show_label else None
        for plot_idx in range(6):
            axs[plot_idx % 2, plot_idx // 2].set_title(plot_titles[plot_idx])
            axs[plot_idx % 2, plot_idx // 2].set_yscale('log')
        
        if best:
            alpha = None
            linestyle = None
        else:
            alpha = 0.3
            linestyle = '--'

        for plot_idx in range(6):
            try:
                axs[plot_idx % 2, plot_idx // 2].plot(
                    run[record_keys[plot_idx]],
                    PLOT_SCHEME[codename],
                    label=label,
                    alpha=alpha,
                    linestyle=linestyle
                )
            except KeyError:
                print(plot_type)
                print(record)
        axs[0, 0].legend(loc='upper right')
        axs[1, 0].legend(loc='lower right')
        axs[0, 1].legend(loc='upper right')
        axs[1, 1].legend(loc='center right')
        axs[0, 2].legend(loc='upper right')
        axs[1, 2].legend(loc='lower left')

    if plot_type == 'metrics':
        axs[1, 0].axhline(y=OMEGA_TARGET, color='m', linestyle=':')
        axs[0, 2].axhline(y=22, color='m', linestyle=':')

    for codename, ids_list in best_ids.items():
        # Plot best run for that specific code
        runs_list = [record[codename][id_num] for id_num in ids_list]
        plot_circuit_metrics(runs_list[0], codename, True, True)
        # Plot the remaining ones
        for run in runs_list[1:]:
            plot_circuit_metrics(run, codename, False, False)

    plt.savefig(
        f'{plot_folder}/{save_prefix}_{plot_type}_record.png',
        dpi=300
    )


def build_save_prefix(args) -> str:
    save_prefix = ""
    for code in args.codes:
        save_prefix += code
    if args.num_runs is not None:
        save_prefix += f"_n_{args.num_runs}"
    if args.best_n is not None:
        save_prefix += f"_b_{args.best_n}"
    if args.optimization_type is not None:
        save_prefix += f"_{args.optimization_type}"
    if args.name is not None:
        save_prefix += f"_{args.name}"

    return save_prefix


def main() -> None:
    global RESULTS_DIR

    # Assign keyword arguments
    parser = argparse.ArgumentParser()
    add_file_args(parser)
    parser.add_argument(
        "num_runs", type=int,
        help="Number of runs to plot, starting at id = 0"
    )
    parser.add_argument(
        '-b', '--best_n', type=int,
        help="If used, plot only the <best_n> of each circuit type."
    )
    parser.add_argument(
        '-d', '--directory', type=int,
        help="Directory from results, if different than default."
    )
    args = parser.parse_args()

    num_runs = int(args.num_runs)
    if args.best_n is not None:
        best_n = args.best_n
    else:
        best_n = num_runs
    
    if args.directory is not None:
        RESULTS_DIR = args.directory

    circuit_codes = args.codes.split(',')
    aggregate_loss_record = defaultdict(dict)
    aggregate_metrics_record = defaultdict(dict)
    experiment_folder = f"{args.optimization_type}_{args.name}"
    records_folder = os.path.join(experiment_folder, "records")

    success_count = 0
    for codename in circuit_codes:
        for id_num in range(num_runs):
            path = os.path.join(
                RESULTS_DIR, records_folder, f'{args.optimization_type}_loss_record_{codename}_{args.name}_{id_num}.pickle')
            print(f"path: {path}")
            loss_record = load_record(os.path.join(
                RESULTS_DIR, records_folder, f'{args.optimization_type}_loss_record_{codename}_{args.name}_{id_num}.pickle'))
            metrics_record = load_record(os.path.join(
                RESULTS_DIR, records_folder, f'{args.optimization_type}_metrics_record_{codename}_{args.name}_{id_num}.pickle'))
            
            if loss_record is not None and metrics_record is not None:
                success_count += 1
                aggregate_loss_record[codename][id_num] = loss_record
                aggregate_metrics_record[codename][id_num] = metrics_record

    save_prefix = build_save_prefix(args)
    title = f"Optimization with {args.optimization_type}"
    if args.name is not None:
        title += f': {args.name}.'

    best_ids = compute_best_ids(aggregate_loss_record, best_n, circuit_codes)
    plot_output_folder = os.path.join(RESULTS_DIR, experiment_folder, "plots")
    os.makedirs(plot_output_folder, exist_ok=True)
    print(f"experiment_folder: {experiment_folder}")
    plot_results(
        aggregate_loss_record,
        plot_output_folder,
        best_ids,
        plot_type='loss',
        title=title,
        save_prefix=save_prefix
    )
    plot_results(
        aggregate_metrics_record,
        plot_output_folder,
        best_ids,
        plot_type='metrics',
        title=title,
        save_prefix=save_prefix
    )
    print(
        f"Loaded {success_count} of {len(circuit_codes) * num_runs} "
        f"successful runs."
    )


if __name__ == "__main__":
    main()
