# VQ‑CNNI: hybrid variational quantum–classical neural network interferometer for Phase Estimation

This repository implements a hybrid quantum–classical model for estimating an unknown phase \(\phi\).  
A **parametrized quantum circuit (PQC)** is used to generate a probability distribution over the computational basis states.  
These probabilities are then aggregated according to the imbalance quantum number \(m = (\#0 - \#1)\) and fed into a classical **multi‑layer perceptron (MLP)**. The MLP outputs a two‑dimensional vector whose arctangent directly yields the estimate \(\hat{\phi}\).

> In short: **PQC → probability distribution → aggregation by \(m\) → MLP → \(\hat{\phi}\)**.

The code is written with **PennyLane** and supports multiple activation functions (`softsign`, `sigmoid`, `softplus`, `ELU`, `tanh`, `ReLU`). It also includes utilities for computing the Quantum Fisher Information (QFI) and squared wrapped phase error (SWPE) of the trained estimator.

The repository contains six variants of the VQ‑CNNI (each with a different activation function), separate scripts for **VQI extreme regime training/testing**, and a Jupyter notebook `data_analysis.ipynb` that produces all figures used in the paper.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Running the Code](#running-the-code)
- [Key Functions](#key-functions)
- [Extending the Code](#extending-the-code)
- [Citation](#citation)

---

## Overview

The model consists of the following stages:

1. **Encoding layers** – apply \(R_Z\) twists, \(R_X\) twists, and collective \(R_X\) gates (controlled by trainable parameters \(\theta\)).
2. **Phase encoding** – encode the unknown \(\phi\) via \(R_Z(\phi)\) on every qubit.
3. **Decoding layers** – symmetric structure to the encoding but with independent trainable parameters \(\boldsymbol{\chi}\).
4. **Measurement** – the circuit outputs a probability distribution \(p(s)\) over the \(2^{n}\) computational basis states \(|s\rangle\).
5. **Aggregation by \(m\)** – for each basis state, compute \(m = \#0 - \#1\). Sum probabilities of all states with the same \(m\) to obtain a vector \(\mathbf{p}_m\) of length \(2n+1\).
6. **Classical neural network** – an MLP maps \(\mathbf{p}_m\) to a 2‑D vector \((y_0, y_1)\). The estimated phase is \(\hat{\phi} = \arctan2(y_0, y_1)\).
7. **Loss function** – squared wrapped phase error (SWPE): \(2\bigl(1-\cos(\phi - \hat{\phi})\bigr)\), averaged over a uniform training grid of \(\phi\) values in \([-\pi, \pi]\).

Training uses the Adam optimizer (PennyLane’s built‑in) with analytic gradients (statevector simulation).

---

## Repository Structure

```text
├── vqc_mlp_sigmoid.py              # Activation = sigmoid
├── vqc_mlp_softsign.py             # Activation = softsign
├── vqc_mlp_softsign-shift.py             # Activation = softsign-shift
├── vqc_mlp_elu.py                  # Activation = ELU
├── vqc_mlp_tanh.py                 # Activation = tanh
├── vqc_mlp_arctan.py                 # Activation = arctan
├── vqc_mlp_softsign-fixedParam.py        # Activation = softsign，fixed quantum params
├── VQI.py                         # VQI baseline
├── data_analysis.ipynb             # Jupyter notebook for generating all paper figures
└── README.md


---

## Dependencies

- Python ≥ 3.8
- [PennyLane](https://pennylane.ai/) ≥ 0.42
- NumPy
- Matplotlib
- Jupyter (for the analysis notebook)

Install all required packages with:

```bash
pip install pennylane numpy matplotlib jupyter
```

---

## Running the Code

To train a model:

```bash
python vqc_mlp_xxx.py
```

During training, the script prints the loss and evaluates the test Loss and QFI every 10 iterations.  
Early stopping is implemented: training stops if the loss does not improve for 100 consecutive iterations (after a minimum of 500 iterations).

All results (including the final and best models) are saved in the directory:

```text
VQ-CNNI/<n_qubits>/vqc_<n_enc>_<n_dec>/<activation>/
```

For example:  
`VQ-CNNI/8/vqc_1_1/softsign/`

---

## Key Functions

- `circuit_state(phi, q_params)` – returns the full statevector for given \(\phi\) and quantum parameters.
- `circuit_probs(phi, q_params)` – returns the probability distribution over computational basis states.
- `probs_to_p_m(probs)` – aggregates the probabilities into the \(m\) subspaces, producing the vector \(\mathbf{p}_m\).
- `MLP` – classical neural network that maps \(\mathbf{p}_m\) to a 2‑D output.
- `objective(x)` – the loss function evaluated over the whole training set.
- `compute_QFI(phi, q_params)` – computes the quantum Fisher information of the pure output state using a central‑difference derivative.
- `evaluate_model(x)` – returns the test Loss (on the test grid) and QFI (at \(\phi=0\)).
- `adjust_phase_wrap(phi_true, phi_pred)` – corrects the \(\pm 2\pi\) ambiguity of the arctan output.

---

## Citation

If you use this code in your research, please cite the associated paper (to be added upon publication).  
For questions or contributions, please open an issue in the repository.

**Acknowledgments**  
This work uses PennyLane for quantum circuit simulation and automatic differentiation. The ansatz design is inspired by [[arXiv:2107.01860](https://www.nature.com/articles/s41586-022-04435-4)].
