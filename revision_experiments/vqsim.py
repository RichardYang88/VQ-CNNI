"""Pure-NumPy replica of the original notebooks' (VQI.ipynb /
vqc_mlp_*.ipynb) circuit, MLP readout and PCA, used to re-render vector
figure panels and regenerate finite-shot samples without PennyLane.

The gate sequence mirrors ``apply_circuit`` in ``vqcnni_lib.py``
(enc = dec = 1, noise-free):
  RY(pi/2)^N -> RZZ(t1/2) pairs -> H^N -> RZZ(t2/2) pairs -> H^N
  -> RX(t3)^N -> RZ(phi)^N            [phase imprinting]
  -> RX(v3)^N -> H^N -> RZZ(v2/2) pairs -> H^N -> RZZ(v1/2) pairs
  -> RX(pi/2)^N
with RZZ(chi,i,j) = CNOT(i,j)-RZ(chi,j)-CNOT(i,j).  Wire 0 is the most
significant bit of the computational-basis index (PennyLane convention;
all observables used here are qubit-permutation invariant anyway).
"""

import itertools
import pickle

import numpy as np


# ----------------------------------------------------------------------
# m-value structure (population imbalance m = #0 - #1)
# ----------------------------------------------------------------------
def m_structure(N):
    """Return (index_to_m, unique_m, masks) for N qubits."""
    index_to_m = np.array(
        [format(k, f"0{N}b").count("0") - format(k, f"0{N}b").count("1")
         for k in range(2 ** N)], dtype=int)
    unique_m = np.unique(index_to_m)
    masks = [np.where(index_to_m == mm)[0] for mm in unique_m]
    return index_to_m, unique_m, masks


# ----------------------------------------------------------------------
# statevector primitives
# ----------------------------------------------------------------------
def _bit(idx, q, N):
    return (idx >> (N - 1 - q)) & 1


def _apply1(st, U, q, N):
    st = st.reshape([2] * N)
    st = np.tensordot(U, st, axes=([1], [q]))
    st = np.moveaxis(st, 0, q)
    return st.reshape(-1)


def _cnot(st, c, t, N):
    idx = np.arange(2 ** N)
    ctrl = _bit(idx, c, N) == 1
    flip = 1 << (N - 1 - t)
    new = st.copy()
    on = np.where(ctrl)[0]
    new[on] = st[on ^ flip]
    return new


def _rz(theta, N):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _rzz(st, chi, i, j, N):
    st = _cnot(st, i, j, N)
    st = _apply1(st, _rz(chi, N), j, N)
    return _cnot(st, i, j, N)


def circuit_probs(phi, theta, curly, N=8):
    """Exact outcome probabilities (2^N,) for one phase."""
    t1, t2, t3 = theta
    v1, v2, v3 = curly
    st = np.zeros(2 ** N, dtype=complex)
    st[0] = 1.0
    pairs = list(itertools.combinations(range(N), 2))
    for q in range(N):
        st = _apply1(st, _ry(np.pi / 2), q, N)
    for i, j in pairs:
        st = _rzz(st, t1 / 2.0, i, j, N)
    for q in range(N):
        st = _apply1(st, _H, q, N)
    for i, j in pairs:
        st = _rzz(st, t2 / 2.0, i, j, N)
    for q in range(N):
        st = _apply1(st, _H, q, N)
    for q in range(N):
        st = _apply1(st, _rx(t3), q, N)
    for q in range(N):
        st = _apply1(st, _rz(phi, N), q, N)
    for q in range(N):
        st = _apply1(st, _rx(v3), q, N)
    for q in range(N):
        st = _apply1(st, _H, q, N)
    for i, j in pairs:
        st = _rzz(st, v2 / 2.0, i, j, N)
    for q in range(N):
        st = _apply1(st, _H, q, N)
    for i, j in pairs:
        st = _rzz(st, v1 / 2.0, i, j, N)
    for q in range(N):
        st = _apply1(st, _rx(np.pi / 2), q, N)
    return np.abs(st) ** 2


def probs_to_p_m(probs, masks):
    if probs.ndim == 1:
        return np.array([probs[idx].sum() for idx in masks])
    return np.stack([probs[:, idx].sum(axis=1) for idx in masks], axis=1)


# ----------------------------------------------------------------------
# Softsign MLP readout  (N+1) -> 128 -> 64 -> 2, L2-normalized output
# ----------------------------------------------------------------------
def softsign(x):
    return x / (1 + np.abs(x))


class _PLTensor(np.ndarray):
    """Unpickle stand-in for pennylane.numpy.tensor (state = nd + grad)."""

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) > 1:
            super().__setstate__(state[:-1])
        else:
            super().__setstate__(state)


class _RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "pennylane" in module:
            return _PLTensor
        return super().find_class(module, name)


_MLP_SHAPES = [(128, 9), (128,), (64, 128), (64,), (2, 64), (2,)]


def load_mlp(path):
    """Return [W1, b1, W2, b2, W3, b3] as plain ndarrays."""
    with open(path, "rb") as f:
        obj = _RemapUnpickler(f).load()
    if isinstance(obj, np.ndarray) and obj.ndim == 1:  # flat concat
        parts, off = [], 0
        for shp in _MLP_SHAPES:
            n = int(np.prod(shp))
            parts.append(np.asarray(obj[off:off + n]).reshape(shp))
            off += n
        return parts
    return [np.asarray(v) for v in obj]


def mlp_latent(params, p_m):
    """Second hidden-layer representation h2 (64-d)."""
    W1, b1, W2, b2, _W3, _b3 = params
    h1 = softsign(W1 @ p_m + b1)
    return softsign(W2 @ h1 + b2)


def mlp_predict(params, p_m):
    W1, b1, W2, b2, W3, b3 = params
    h2 = mlp_latent(params, p_m)
    out = W3 @ h2 + b3
    out = out / (np.linalg.norm(out) + 1e-6)
    return np.arctan2(out[0], out[1])


# ----------------------------------------------------------------------
# PCA with the sklearn svd_flip sign convention
# ----------------------------------------------------------------------
def pca2(X):
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    max_abs = np.argmax(np.abs(Vt[:2]), axis=1)
    signs = np.sign(Vt[np.arange(2), max_abs])
    Vt[:2] = Vt[:2] * signs[:, None]
    return Xc @ Vt[:2].T
