# `Qubit-Discovery` `scripts/` tutorial

This document provides a tutorial on using the programs in the [`scripts/`](../scripts/) directory to run replicable parallelized optimization of circuits using the [`qubit_discovery`](../qubit_discovery) package.

The following pipeline has been used in [TODO]() to run optimization on Stanford's [Sherlock](https://www.sherlock.stanford.edu/) HPC cluster.

## 1. Creating a YAML file

To ensure replicability, all metadata about the optimization is recorded in a YAML file. It contains, at minimum, the following information:
- `name`: A unique identifier for the optimization run.
- `K`: The total truncation number used for circuit quantization.
- `epochs`: The maximum number of optimization iterations to perform.
- `num_eigenvalues`: The number of eigenvalues to calculate during optimization.
- `capacitor_range`: Range of capacitor values in Farads, specified as `[min, max]`.
- `inductor_range`: List of floats. Range of inductor values in Henries, specified as `[min, max]`.
- `junction_range`: List of floats. Range of junction frequencies in radians per second, specified as `[min, max]`.
- `use_losses`: Specifies how to construct the loss function. A list of metrics is specified as `metric: weight` pairs, and the total loss function is constructed by $$\mathcal{L} = \sum \text{weight} \times \text{metric}(\text{circuit}).$$ The available metrics can be found by calling `qubit_discovery.losses.loss.get_all_metrics()`. 
- `use_metrics`: List of metrics to calculate during optimization, but not to include in loss function. Useful if you want to track how something evolves during optimization, but don't want to optimize _for_ it. The available metrics can found by calling `qubit_discovery.losses.loss.get_all_metrics()`.

An example YAML file is provided in [`optim_data-min.yaml`](optim_data-min.yaml). 

Other metadata about the optimization can be provided either directly to the Python scripts or as keys in the YAML file, when needed. These are:
- `circuit_code`: A string describing the topology of the circuit (see [TODO]() for details about the conventions).
- `optim_type`: The optimization algorithm to use. Currently gradient descent (`'SGD'`) and BFGS (`'BFGS'`) are supported.
- `seed`: An integer to set the seed of random number generators. Crucially, this determines the random initialization used to begin optimization.
- `init_circuit`: The path to a pickled `SQcircuit.Circuit` object to use as the initial circuit for optimization, rather than a random seed.
- `save_intermediate`: Whether to save circuits during intermediate steps of the optimization (rather than only the final optimized circuit).

An example YAML file with all possible keys is provided in [`optim_data-max.yaml`](optim_data-max.yaml).

When running the following scripts, the following directory structure will automatically be created in the folder containing the YAML file. The different files generated are discussed below.
```
 main_folder/                       # main directory
 ├── yaml_file.yaml
 │
 └── {optim_type}_{name}/           # experiment_directory
     │
     ├── records/                   # records_directory
     │   │
     │   ├── {optim_type}_loss_record_{circuit_code}_{name}_{seed}.pickle
     │   ├── {optim_type}_metrics_record_{circuit_code}_{name}_{seed}.pickle
     │   └── {optim_type}_circuits_record_{circuit_code}_{name}_{seed}.pickle
     │
     ├── plots/                     # plots_directory
     │   │
     │   ├── {circuit_code}_n_{num_runs}_{optim_type}_{name}_loss.png
     │   └── {circuit_code}_n_{num_runs}_{optim_type}_{name}_metrics.png
     │
     └── summaries/                 # summaries_directory
         │
         └── {optim_type}_circuit_summary_{circuit_code}_{name}_{id_num}.txt

```

## 2. Running the optimization

To run the optimization after constructing a YAML file, we use [`optimize.py`](../scripts/optimize.py). An overview of its usage can be found by running
```
python optimize.py --help
```

It constructs a loss function based on the provided YAML file, and runs optimization of the type specified by `optim_type` starting with a randomly sampled circuit of `circuit_code` within the element ranges provided (or starting with `init_circuit`, if given). Since optimization can become stuck in local minima, it is usually necessary to run multiple times with different random intializiations, for example,
```
for i in {1..10}
do
    python scripts/optimize.py tutorials/optim_data-min.yaml --circuit_code=JL --seed=$i --optim_type=BFGS
done
```

_Note: If you pass in metadata both in the YAML file and via a command-line argument, the command-line argument will be used._

Running `optimize.py` generates three files in the `{optim_type}_{name}/{records}` directory:
- `{optim_type}_circuits_record_{circuit_code}_{name}_{seed}.pickle`: contains the final optimized circuit (and the intermediate circuits, if `--save-intermediate` is passed to `optimize.py`).
- `{optim_type}_loss_record_{circuit_code}_{name}_{seed}.pickle`: contains the total loss calculated at each epoch, in addition to the values for each of the individual component metrics.
- `{optim_type}_metrics_record_{circuit_code}_{name}_{seed}.pickle`: contains the value for all the metrics passed in the `use_metrics` list for the YAML file, at each epoch.

## 3. Summarizing results

After running many different optimization runs, the next step is to compare how well they perform. This is done with [`plot_records.py`](../scripts/plot_records.py). An overview of its usage can be found by running
```
python plot_records.py --help
```

It takes in many optimization runs, plots learning curves of the ones with the best final total losses, and prints the seeds of the best performing runs. For instance, after example the `optimize.py` command above, to find the best 5 performing runs, execute 
```
python scripts/plot_records.py tutorials/optim_data-min.yaml --circuit_code=JL --num_runs=10 --optim_type=BFGS --num_best=5
```
The learning curves of the `num_best`-performing circuits are outputted as `{circuit_code}_n_{num_runs}_{optim_type}_{name}_loss.png` and `{circuit_code}_n_{num_runs}_{optim_type}_{name}_metrics.png` in the `{optim_type}_{name}/plots` directory.

## 4. Exploring details of best-performing circuits

After having identified the best circuits as a result of the above optimization procedure, we then examine the details. This can be done by hand by loading the appropriate `{optim_type}_circuits_record_{circuit_code}_{name}_{seed}.pickle` file into a notebook. We also provide two useful scripts which automatate the main process.

The [`circuit_summary.py`](../scripts/circuit_summary.py) program prints to file the values of all the elements in a list of circuits, along with the value for all calculated metrics.
```
python scripts/circuit_summary.py tutorials/optim_data-min.yaml --circuit_code=JL --optim_type=BFGS --ids=1,6,9
```
The outputs are saved in `{optim_type}_circuit_summary_{circuit_code}_{name}_{id_num}.txt` files in the `{optim_type}_{name}/summaries`.


The [`plot_analysis.py`](../scripts/plot_analysis.py) program plots the flux and charge spectrum of a list of circuits.
```
python scripts/plot_analysis.py tutorials/optim_data-min.yaml --circuit_code=JL --optim_type=BFGS --ids=1,6,9
```
The outputs are saved as `{optim_type}_plot_{circuit_code}_{name}_{id_num}.flux.png` and `{optim_type}_plot_{circuit_code}_{name}_{id_num}.charge_diag.png` in the `{optim_type}_{name}/plots` directory.
