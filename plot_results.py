# Optionally plot circuit spectrum if desired
# spectrum_optimal_circuit(loss_record, codename="Fluxonium")

from collections import OrderedDict
from typing import Any, List

from functions import(
    code_to_codename,
    flatten,
    get_element_counts,
    get_optimal_key
)
from loss import OMEGA_TARGET
from settings import RESULTS_DIR

import argparse
from matplotlib import pyplot as plt
import dill as pickle

# Assign keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--codes', type=str, required=True)
parser.add_argument("num_runs", type=int)
parser.add_argument('-b', '--best_n', type=int)
parser.add_argument('-o', '--optimization_type', type=str)
args = parser.parse_args()

num_runs = int(args.num_runs)
if args.best_n is not None:
    best_n = args.best_n
else:
    best_n = num_runs

def load_record(url: str) -> Any:
    file = open(url, 'rb')
    record = pickle.load(file)
    file.close()
    return record

def get_optimal_n_keys(loss_record, n: int, code=None):
    optimal_keys_losses = OrderedDict({})

    for circuit, circuit_code, l in loss_record.keys():
        key = (circuit, circuit_code, 'total_loss')
        loss = loss_record[key][-1]
        if (code in key or code is None) and key not in optimal_keys_losses:

            if len(optimal_keys_losses) < n:
                optimal_keys_losses[key] = loss
            else:
                last_key = next(reversed(optimal_keys_losses))
                if loss < optimal_keys_losses[last_key]:
                    del optimal_keys_losses[last_key]
                    optimal_keys_losses[key] = loss
            optimal_keys_losses = OrderedDict(sorted(optimal_keys_losses.items(),
                                                     key = lambda t: t[1]))

    print(f"optimal_keys_losses.keys(): {list(optimal_keys_losses.keys())}")
    return list(optimal_keys_losses.keys())


def plot_results(loss_record, 
                 circuit_codes: List[str], 
                 type='metrics', 
                 title='',
                 save_prefix='') -> None:
    plot_scheme = {'Transmon': 'b', 'Fluxonium': 'darkorange',
                   'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'}

    fig, axs = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle(title, fontsize = 32)
    metric_titles = ['Total Loss', 'Frequency', 'Flux Sensitivity',
                     'Charge Sensitivity', 'Anharmonicity', r'$T_1$']
    metric_keys = ['total_loss', 'omega', 'flux_sensitivity',
                   'charge_sensitivity', 'A', 'T1']
    loss_titles = ['Frequency Loss', 'Anharmonicity Loss', 'T1 Loss',
                     'Flux Sensitivity Loss', 'Charge Sensitivity Loss', 'Total Loss']
    loss_keys = ['frequency_loss', 'anharmonicity_loss', 'T1_loss',
                   'flux_sensitivity_loss', 'charge_sensitivity_loss', 'total_loss']
    plot_titles = metric_titles if type == 'metrics' else loss_titles
    record_keys = metric_keys if type == 'metrics' else loss_keys

    def plot_circuit_metrics(circuit: Circuit, 
                             loss_record, 
                             code: str, 
                             optimal_keys,
                             show_label=False) -> None:
        codename = code_to_codename(code)
        label = codename if show_label else None
        for plot_idx in range(6):
            axs[plot_idx % 2, plot_idx // 2].set_title(plot_titles[plot_idx])
            axs[plot_idx % 2, plot_idx // 2].set_yscale('log')

        key = (circuit, code, 'total_loss')
        if key in optimal_keys:
            alpha = None
            linestyle = None
        else:
            alpha = 0.3
            linestyle = '--'

        for plot_idx in range(6):
            axs[plot_idx % 2, plot_idx // 2].plot(loss_record[
                           (circuit, code, record_keys[plot_idx])],
                           plot_scheme[codename], label=label, alpha=alpha,
                           linestyle=linestyle)
            # if record_keys[plot_idx] == 'flux_sensitivity_loss':
            #     print(loss_record[
            #                (circuit, code, record_keys[plot_idx])])
        axs[0, 0].legend(loc='upper right')
        axs[1, 0].legend(loc='lower right')
        axs[0, 1].legend(loc='upper right')
        axs[1, 1].legend(loc='center right')
        axs[0, 2].legend(loc='upper right')
        axs[1, 2].legend(loc='lower left')

    if type == 'metrics':
        axs[1, 0].axhline(y=OMEGA_TARGET, color='m', linestyle=':')
        axs[0, 2].axhline(y=22, color='m', linestyle=':')

    optimal_keys = [get_optimal_key(loss_record, code=code) for code
                    in circuit_codes]
    optimal_n_keys = flatten([get_optimal_n_keys(loss_record, n=best_n, code=code) 
                              for code in circuit_codes])

    plotted_codenames = set()
    for key in optimal_n_keys:
        circuit, code, _ = key
        show_label = False
        junction_count, inductor_count, _ = get_element_counts(circuit)
        # codename = lookup_codename(junction_count, inductor_count)
        if code not in plotted_codenames:
            show_label = True
            plotted_codenames.add(code)
        plot_circuit_metrics(circuit, loss_record, code, optimal_keys,
                             show_label=show_label)

    # plt.savefig('/home/mckeehan/sqcircuit/Qubit-Discovery/results/output.png')
    plt.savefig(f'{RESULTS_DIR}/{save_prefix}_{type}_record.png')

def build_save_prefix() -> str:
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
    circuit_codes = [code for code in args.codes.split(',')]
    aggregate_loss_record = {}
    aggregate_metrics_record = {}
    prefix = args.optimization_type
    if prefix != "":
        prefix += '_'
    for codename in circuit_codes:
        for id in range(num_runs):
            loss_record = load_record(f'{RESULTS_DIR}/{prefix}loss_record_{codename}_{id}.pickle')
            metrics_record = load_record(
                f'{RESULTS_DIR}/{prefix}metrics_record_{codename}_{id}.pickle')
            aggregate_loss_record.update(loss_record)
            aggregate_metrics_record.update(metrics_record)
    save_prefix = build_save_prefix()
    title = f"Optimization with {args.optimization_type}"
    plot_results(aggregate_loss_record, circuit_codes, type='loss', title=title,
                 save_prefix=save_prefix)
    plot_results(aggregate_metrics_record, circuit_codes, type='metrics', title=title,
                 save_prefix=save_prefix)


if __name__ == "__main__":
    main()
