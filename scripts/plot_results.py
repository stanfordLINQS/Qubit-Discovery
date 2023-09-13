# Optionally plot circuit spectrum if desired
# spectrum_optimal_circuit(loss_record, codename="Fluxonium")
import argparse
from collections import OrderedDict
from typing import Any, List

from matplotlib import pyplot as plt
import dill as pickle
from SQcircuit import Circuit

from plot_utils import code_to_codename
from qubit_discovery.losses.loss import OMEGA_TARGET
from settings import RESULTS_DIR

def load_record(url: str) -> Any:
    try:
        with open(url, 'rb') as f:
            record = pickle.load(f)
        return record
    except FileNotFoundError:
        return None

def get_optimal_n_runs(loss_record, n: int, code=None):
    if code is not None:
        loss_record = [(i, run) for i, run in loss_record if 
                       run['circuit_code'] == code]
    sort_key = 'total_loss' if 'total_loss' in loss_record[0][1] else 'all_loss'
    sorted_runs = sorted(loss_record, key=lambda x: x[1][sort_key][-1])
    print(f'Top {n} runs in order are {[i for i, run in sorted_runs[:n]]} for code {code}')
    return [run for i, run in sorted_runs[:n]]


def plot_results(loss_record, 
                 circuit_codes: List[str], 
                 best_n: int,
                 type='metrics', 
                 title='',
                 save_prefix='') -> None:
    plot_scheme = {'Transmon': 'b', 'Fluxonium': 'darkorange',
                   'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'}

    fig, axs = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle(title, fontsize = 32)
    metric_titles = ['All Loss', 'Frequency', 'Flux Sensitivity',
                     'Charge Sensitivity', 'Anharmonicity', r'$T_1$']
    metric_keys = ['all_loss', 'omega', 'flux_sensitivity',
                   'charge_sensitivity', 'A', 'T1']
    loss_titles = ['Frequency Loss', 'Anharmonicity Loss', 'T1 Loss',
                     'Flux Sensitivity Loss', 'Charge Sensitivity Loss', 'Total Loss']
    loss_keys = ['frequency_loss', 'anharmonicity_loss', 'T1_loss',
                   'flux_sensitivity_loss', 'charge_sensitivity_loss', 'total_loss']
    plot_titles = metric_titles if type == 'metrics' else loss_titles
    record_keys = metric_keys if type == 'metrics' else loss_keys

    def plot_circuit_metrics(run, 
                             code: str, 
                             best: bool,
                             show_label=False) -> None:
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
            axs[plot_idx % 2, plot_idx // 2].plot(run[record_keys[plot_idx]],
                           plot_scheme[codename], label=label, alpha=alpha,
                           linestyle=linestyle)
        axs[0, 0].legend(loc='upper right')
        axs[1, 0].legend(loc='lower right')
        axs[0, 1].legend(loc='upper right')
        axs[1, 1].legend(loc='center right')
        axs[0, 2].legend(loc='upper right')
        axs[1, 2].legend(loc='lower left')

    if type == 'metrics':
        axs[1, 0].axhline(y=OMEGA_TARGET, color='m', linestyle=':')
        axs[0, 2].axhline(y=22, color='m', linestyle=':')

    optimal_runs = [(get_optimal_n_runs(loss_record, n=best_n, code=code), code)
                    for code in circuit_codes]

    for runs_list, code in optimal_runs:
        # Plot best run for that specific cocde
        plot_circuit_metrics(runs_list[0], code, True, True)
        # Plot the remaining ones
        for run in runs_list[1:]:
            plot_circuit_metrics(run, code, False, False)

    plt.savefig(f'{RESULTS_DIR}/{save_prefix}_{type}_record.png', dpi=300)

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

    return save_prefix

def main() -> None:
    # Assign keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("num_runs", type=int)
    parser.add_argument('-c', '--codes', type=str, required=True)
    parser.add_argument('-b', '--best_n', type=int)
    parser.add_argument('-o', '--optimization_type', type=str)
    parser.add_argument('-s', '--save_circuits', action='store_true') #TODO: implement
    parser.add_argument('-n', '--name')
    args = parser.parse_args() # ['5', '-c', 'JL', '-o', 'SGD'])

    num_runs = int(args.num_runs)
    if args.best_n is not None:
        best_n = args.best_n
    else:
        best_n = num_runs

    if args.name is None:
        name = ''
    else:
        name = args.name + '_'

    circuit_codes = [code for code in args.codes.split(',')]
    aggregate_loss_record = []
    aggregate_metrics_record = []
    prefix = args.optimization_type
    if prefix != "":
        prefix += '_'

    for codename in circuit_codes:
        for id in range(num_runs):
            loss_record = load_record(
                f'{RESULTS_DIR}/{prefix}loss_record_{codename}_{name}{id}.pickle')
            metrics_record = load_record(
                f'{RESULTS_DIR}/{prefix}metrics_record_{codename}_{name}{id}.pickle')
            if loss_record is not None and metrics_record is not None:
                aggregate_loss_record.append((id, loss_record))
                aggregate_metrics_record.append((id, metrics_record))

    save_prefix = build_save_prefix(args)
    title = f"Optimization with {args.optimization_type}"
    plot_results(aggregate_loss_record, circuit_codes, best_n, type='loss', title=title,
                 save_prefix=save_prefix)
    plot_results(aggregate_metrics_record, circuit_codes, best_n, type='metrics', title=title,
                 save_prefix=save_prefix)


if __name__ == "__main__":
    main()
