<br />
<p align="center">
<img src = pics/README_logo.png width= 500px" />
</p>

# Qubit-Discovery
[**Introduction**](#Introduction)
|[**Tutorial**](#Tutorial)
|[**Installation**](#Installation)
|[**Examples**](#Examples)
|[**Contribution**](#Contribution)

## Introduction

Qubit-Discovery is a set of software tools to perform gradient-based optimization of superconducting qubits using the [`SQcircuit`](https://github.com/stanfordLINQS/SQcircuit) Python package. It provides a Python package `qubit_discovery` implementing loss functions and optimization algorithms, and a set of scripts using it in a reproducible workflow for multiple optimization runs.

With these tools, superconducting qubits can be optimized over the values of their elements for metrics such as decoherence time, anharmonicity, qubit frequency, or other user-defined targets.

A description of the theory and example usage is provided in the following paper:
> TODO

## Usage

### `qubit_discovery`

The `qubit_discovery` module provides the basic ingredients for running optimization with `SQcircuit`. The `losses` submodule implements functions to evaluate qubit metrics (anharmonicity, decoherence time, …) and utilities to build loss functions out of them. The `optimization` submodule implements optimization algorithms (SGD and BFGS), utilities to set truncation numbers, and a qubit sampler.

See the notebooks in the [`tutorials`](tutorials/) directory for instructions on how to use these features:
- [`QD_overview.ipynb`](tutorials/QD_overview.ipynb) gives a basic overview of the core functionality.
- [`QD_advanced-features.ipynb`](tutorials/QD_advanced-features.ipynb) shows the advanced features and customization possible.
- [`QD_transmon-optim.ipynb`](tutorials/QD_transmon-optim.ipynb) shows how the package can be used to optimize the $T_2$ time of a transmon.

### Scripts

The `scripts` directory provides a set of programs which can be used to automate multiple optimization runs via the following workflow:
1. Run [`optimize.py`](scripts/optimize.py) in parallel with different seeds, which correspond to distinct intialization points.
2. Collate the results of the different runs using [`plot_records.py`](scripts/plot_records.py).
3. Output the details of the best-performing circuits with [`circuit_summary.py`](scripts/circuit_summary.py) and [`plot_analysis.py`](scripts/plot_analysis.py).

For a detailed description of the workflow, see [`parallel_optimization.md`](tutorials/parallel_optimization.md).

## Installation

To install the `qubit_discovery` package, download from  PyPI:
```
pip install qubit_discovery
```

To use the provided scripts, after installing the `qubit_discovery` package clone the `Qubit-Discovery` repository and run with Python >=3.9. 

## Contribution

## License
