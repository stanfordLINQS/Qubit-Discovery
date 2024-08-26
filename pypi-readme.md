<div align="center"> <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stanfordLINQS/Qubit-Discovery/main/pics/dark_logo_qd.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/stanfordLINQS/Qubit-Discovery/main/pics/light_logo_qd.png">
  <img alt="Logo image" src="https://raw.githubusercontent.com/stanfordLINQS/Qubit-Discovery/main/pics/dark_logo_qd.png" width="150" height="auto">
</picture></div>

# qubit-discovery

qubit-discovery is an open-source Python library for optimizing superconducting circuits, built on top of [SQcircuit](https://sqcircuit.org/) and [PyTorch](https://pytorch.org/). It provides:
* Composable loss functions with a special focus on qubit design, and straightforward methods to add new custom ones.
* Fine-tuned BFGS and SGD algorithms to optimize circuits, along with an interface to use other PyTorch optimizers. 
* Utility features including random circuit sampling and functions to automatically choose circuit truncation numbers.

With these capabilities, you can easily optimize any superconducting circuit for decoherence time, anharmonicity, charge sensitivity, or other desired targets. 

A description of the theory involved and example application is provided in the following paper:
> Taha Rajabzadeh, Alex Boulton-McKeehan, Sam Bonkowsky, David I. Schuster, Amir H. Safavi-Naeini, _A General Framework for Gradient-Based Optimization of Superconducting Quantum Circuits using Qubit Discovery as a Case Study_, arXiv:2408.12704 (2024), https://arxiv.org/abs/2408.12704.

If qubit-discovery is useful to you, we welcome contributions to its development and maintenance! Use of the package in publications may be acknowledged by citing the above paper.