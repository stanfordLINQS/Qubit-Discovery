<div align="center"> <picture>
  <source media="(prefers-color-scheme: dark)" srcset="pics/dark_logo_qd.png">
  <source media="(prefers-color-scheme: light)" srcset="pics/light_logo_qd.png">
  <img alt="Logo image" src="pics/dark_logo_qd.png" width="150" height="auto">
</picture></div>

# Qubit-Discovery
[**Introduction**](##Introduction)
| [**Usage**](##Usage)
| [**Installation**](##Installation)
| [**Contribution**](##Contribution)

## Introduction

Qubit-Discovery is a set of software tools to perform gradient-based optimization of superconducting circuits using the [`SQcircuit`](https://github.com/stanfordLINQS/SQcircuit) Python package, with a special focus on qubit design. It provides a Python package, `qubit-discovery`, implementing loss functions and optimization algorithms, and a set of scripts using it in a reproducible workflow for parallelized optimization runs.

With these tools, superconducting circuits can be optimized over the values of their elements for metrics such as decoherence time, anharmonicity, qubit frequency, or other user-defined targets.

A description of the theory and example application (employing the provided scripts) is provided in the following paper:
> TODO

## Usage

### `qubit-discovery`

The `qubit-discovery` package provides the basic ingredients for running optimization with `SQcircuit`. The `losses` subpackage implements functions to evaluate qubit metrics (anharmonicity, decoherence time, …) and utilities to build loss functions out of them. The `optimization` subpackage implements optimization algorithms (SGD, BFGS, and other PyTorch optimizers), utilities to set truncation numbers, and a qubit sampler.

See the notebooks in the [`tutorials`](tutorials/) directory for instructions on how to use these features.
- [`QD_transmon-optim.ipynb`](tutorials/QD_transmon-optim.ipynb) provides a quick demo on how to use `qubit-discovery` to optimize the $T_2$ time, qubit frequency, and anharmonicity of a transmon, in only a few lines of code.
- [`QD_overview.ipynb`](tutorials/QD_overview.ipynb) gives an introduction to the core functionality of the package.
- [`QD_advanced-features.ipynb`](tutorials/QD_advanced-features.ipynb) shows the advanced features and customization possible.

### Scripts

The `scripts` directory provides a set of programs which can be used to automate multiple optimization runs via the following workflow:
1. Execute [`optimize.py`](scripts/optimize.py) in parallel with different seeds, which correspond to distinct intialization points.
2. Collate the results of the different runs using [`plot_records.py`](scripts/plot_records.py).
3. Output the details of the best-performing circuits with [`circuit_summary.py`](scripts/circuit_summary.py) and [`plot_analysis.py`](scripts/plot_analysis.py).

For a detailed description of this workflow, see [`parallel_optimization.md`](tutorials/parallel_optimization.md).

## Installation

To install the `qubit-discovery` package, download from  PyPI:
```
pip install qubit-discovery
```

To use the provided scripts, first install `qubit-discovery` with scripting support:
```
pip install qubit-discovery[scripts]
```
Then clone the `Qubit-Discovery` repository and run the scripts with Python >= 3.9.

## Contribution

We welcome contributions to the develoment of Qubit-Discovery! Please feel free to fork the repository and send pull requests, or file bug reports on the [issues](https://github.com/stanfordLINQS/Qubit-Discovery/issues) page. All code contributions are acknowledged in the contributors' section below.
