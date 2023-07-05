# Optionally plot circuit spectrum if desired
# spectrum_optimal_circuit(loss_record, codename="Fluxonium")

from functions import lookup_codename, get_element_counts

from matplotlib import pyplot as plt
import dill as pickle

# TODO: Consolidate following definition with that in loss.py
omega_target = 1 # GHz

codenames = ["JL", ]
num_runs = 1

def load_record(url):
    file = open(url, 'rb')
    record = pickle.load(file)
    file.close()
    return record

def get_optimal_key(loss_record, codename=None):
  optimal_loss = 1e100
  optimal_key = None

  for circuit, codename_key, _ in loss_record.keys():
    key = (circuit, codename_key, 'total_loss')
    print(f"keys: {loss_record.keys()}")
    print(f"key: {key}")
    print(f"values: {loss_record.values()}")
    print(f"lr: {loss_record[key]}")
    loss = loss_record[key][-1]
    if loss < optimal_loss and (codename in key or codename is None):
        optimal_loss = loss
        optimal_key = key

  return optimal_key

def plot_results(loss_record):
    plot_scheme = {'Transmon': 'b', 'Fluxonium': 'darkorange',
                   'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'}

    fig, axs = plt.subplots(2, 3, figsize=(22, 11))

    # fig.suptitle('Scaling from N=2 to N=3 Inductive Elements')

    def plot_circuit_metrics(circuit, loss_record, codename, optimal_keys,
                             show_label=False):
        label = codename if show_label else None
        axs[0, 0].set_title(f'Total Loss')
        axs[1, 0].set_title(f'Frequency')
        axs[0, 1].set_title(f'Flux Sensitivity')
        axs[1, 1].set_title(f'Charge Sensitivity')
        axs[0, 2].set_title(f'Anharmonicity')
        axs[1, 2].set_title(r'$T_1$')

        axs[0, 0].set_yscale('log')
        axs[0, 1].set_yscale('log')
        axs[0, 2].set_yscale('log')
        axs[1, 0].set_yscale('log')
        axs[1, 1].set_yscale('log')
        axs[1, 2].set_yscale('log')
        key = (circuit, codename, 'total_loss')
        if key in optimal_keys:
            alpha = None
            linestyle = None
        else:
            alpha = 0.3
            linestyle = '--'

        axs[0, 0].plot(loss_record[(circuit, codename, 'total_loss')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[0, 0].legend(loc='upper right')
        axs[1, 0].plot(loss_record[(circuit, codename, 'omega')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[1, 0].legend(loc='lower right')
        axs[0, 1].plot(loss_record[(circuit, codename, 'flux_sensitivity')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[0, 1].legend(loc='upper right')
        axs[1, 1].plot(loss_record[(circuit, codename, 'charge_sensitivity')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[1, 1].legend(loc='center right')
        axs[0, 2].plot(loss_record[(circuit, codename, 'A')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[0, 2].legend(loc='upper right')
        axs[1, 2].plot(loss_record[(circuit, codename, 'T1')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[1, 2].legend(loc='lower left')

    axs[1, 0].axhline(y=omega_target, color='m', linestyle=':')
    axs[0, 2].axhline(y=22, color='m', linestyle=':')

    optimal_keys = [get_optimal_key(loss_record, codename=codename) for codename
                    in codenames]

    plotted_codenames = set()
    for key in loss_record.keys():
        circuit, codename, index = key
        show_label = False
        junction_count, inductor_count, _ = get_element_counts(circuit)
        codename = lookup_codename(junction_count, inductor_count)
        if codename not in plotted_codenames:
            show_label = True
            plotted_codenames.add(codename)
        plot_circuit_metrics(circuit, loss_record, codename, optimal_keys,
                             show_label=show_label)

    plt.savefig('/home/mckeehan/sqcircuit/Qubit-Discovery/results/output.png')

def main():
    loss_record = load_record('/home/mckeehan/sqcircuit/Qubit-Discovery/results/loss_record.pickle')
    metric_record = load_record('/home/mckeehan/sqcircuit/Qubit-Discovery/results/metric_record.pickle')
    plot_results(metric_record)


if __name__ == "__main__":
    main()
