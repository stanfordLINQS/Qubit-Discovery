import argparse
import os
from typing import Any, Dict, List

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
    
def compute_best_ids(aggregate_loss_records, n: int, codes=List[str]
                     ) -> Dict[str, List[int]]:
    out = {}
    for code in codes:
        code_loss_record = [(id_num, run) for id_num, run in aggregate_loss_records 
                            if run['circuit_code'] == code]
        sorted_runs = sorted(code_loss_record, key=lambda x: x[1]['total_loss'][-1])
        out[code] = [id_num for id_num, run in sorted_runs]
        print(f'Top {n} runs in order are {out[code]} for code {code}.')
    return out


def plot_results(loss_record, 
                 best_ids: Dict[str, List[int]],
                 plot_type: str, 
                 title: str = '',
                 save_prefix: str = '') -> None:
    PLOT_SCHEME = {'Transmon': 'b', 'Fluxonium': 'darkorange',
                   'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'}
    METRIC_TITLES = ['All Loss', 'Frequency', 'Flux Sensitivity',
                     'Charge Sensitivity', 'Anharmonicity', r'$T_1$']
    METRIC_KEYS = ['all_loss', 'omega', 'flux_sensitivity',
                   'charge_sensitivity', 'A', 'T1']
    LOSS_TITLES = ['Frequency Loss', 'Anharmonicity Loss', 'T1 Loss',
                     'Flux Sensitivity Loss', 'Charge Sensitivity Loss', 'Total Loss']
    LOSS_KEYS = ['frequency_loss', 'anharmonicity_loss', 'T1_loss',
                   'flux_sensitivity_loss', 'charge_sensitivity_loss', 'total_loss']
    
    fig, axs = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle(title, fontsize = 32)
    plot_titles = METRIC_TITLES if plot_type == 'metrics' else LOSS_TITLES
    record_keys = METRIC_KEYS if plot_type == 'metrics' else LOSS_KEYS

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
                           PLOT_SCHEME[codename], label=label, alpha=alpha,
                           linestyle=linestyle)
        axs[0, 0].legend(loc='upper right')
        axs[1, 0].legend(loc='lower right')
        axs[0, 1].legend(loc='upper right')
        axs[1, 1].legend(loc='center right')
        axs[0, 2].legend(loc='upper right')
        axs[1, 2].legend(loc='lower left')

    if plot_type == 'metrics':
        axs[1, 0].axhline(y=OMEGA_TARGET, color='m', linestyle=':')
        axs[0, 2].axhline(y=22, color='m', linestyle=':')

    for code, runs_list in best_ids:
        # Plot best run for that specific cocde
        plot_circuit_metrics(runs_list[0], code, True, True)
        # Plot the remaining ones
        for run in runs_list[1:]:
            plot_circuit_metrics(run, code, False, False)

    plt.savefig(f'{RESULTS_DIR}/{save_prefix}_{plot_type}_record.png', dpi=300)

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
    # Assign keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("num_runs", type=int)
    parser.add_argument('-c', '--codes', type=str, required=True)
    parser.add_argument('-o', '--optimization_type', type=str, required=True)
    parser.add_argument('-b', '--best_n', type=int)
    parser.add_argument('-s', '--save_circuits', action='store_true') #TODO: implement
    parser.add_argument('-n', '--name')
    args = parser.parse_args()

    num_runs = int(args.num_runs)
    if args.best_n is not None:
        best_n = args.best_n
    else:
        best_n = num_runs

    name = args.name
    circuit_codes = args.codes.split(',')
    aggregate_loss_record = []
    aggregate_metrics_record = []
    optim_type = args.optimization_type

    for codename in circuit_codes:
        for id_num in range(num_runs):
            identifier = f'{name}_{id_num}' if name is not None else f'{id_num}'

            loss_record = load_record(os.path.join(
                RESULTS_DIR, f'{optim_type}_loss_record_{codename}_{identifier}.pickle'))
            metrics_record = load_record(os.path.join(
                RESULTS_DIR, f'{optim_type}_metrics_record_{codename}_{identifier}.pickle'))
            
            if loss_record is not None and metrics_record is not None:
                aggregate_loss_record.append((id_num, loss_record))
                aggregate_metrics_record.append((id_num, metrics_record))

    save_prefix = build_save_prefix(args)
    title = f"Optimization with {args.optimization_type}"
    if name is not None:
        title += f': {name}.'


    best_ids = compute_best_ids(aggregate_loss_record, best_n)
    plot_results(aggregate_loss_record, best_ids, plot_type='loss', title=title,
                 save_prefix=save_prefix)
    plot_results(aggregate_metrics_record, best_ids, plot_type='metrics', title=title,
                 save_prefix=save_prefix)


if __name__ == "__main__":
    main()
