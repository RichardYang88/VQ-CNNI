"""
Shared library for revision experiments (VQ-CNNI, PRR WT10346).

Replicates the quantum circuit and training conventions of the original
notebooks (VQI.ipynb / vqc_mlp_softsign.ipynb) with:
  * PennyLane broadcasting (batched training phases) for speed,
  * a manual Adam implementation (the old opt.step(x, grad) interface is
    deprecated in recent PennyLane versions),
  * identical circuit topology: RY(pi/2) input, two RZZ twists built from
    CNOT-RZ-CNOT with Hadamard basis changes, collective RX, phase layer
    RZ(phi) on every qubit, decoding layers, final RX(pi/2).
"""

import itertools
import numpy as np
import autograd
import pennylane as qml
from pennylane import numpy as pnp


# ----------------------------------------------------------------------
# m-value structure (population imbalance m = #0 - #1)
# ----------------------------------------------------------------------
def m_structure(N):
    """Return (index_to_m, unique_m, masks) for N qubits."""
    index_to_m = np.array(
        [format(k, f"0{N}b").count("0") - format(k, f"0{N}b").count("1")
         for k in range(2 ** N)],
        dtype=int,
    )
    unique_m = np.unique(index_to_m)
    masks = [np.where(index_to_m == mm)[0] for mm in unique_m]
    return index_to_m, unique_m, masks


def probs_to_p_m(probs, masks):
    """Aggregate computational-basis probs onto m sectors.

    probs: shape (2^N,) or (B, 2^N); returns (N+1,) or (B, N+1).
    """
    if probs.ndim == 1:
        return pnp.stack([pnp.sum(probs[idx]) for idx in masks])
    return pnp.stack(
        [pnp.sum(probs[:, idx], axis=1) for idx in masks], axis=1)


# ----------------------------------------------------------------------
# Circuit operations (shared by probs/state qnodes)
# ----------------------------------------------------------------------
def _rzz(chi, i, j):
    qml.CNOT(wires=[i, j])
    qml.RZ(chi, wires=j)
    qml.CNOT(wires=[i, j])


def apply_circuit(N, phi, theta_flat, curly_flat, noise_p=0.0):
    """Apply VQI/VQ-CNNI circuit ops (enc=dec=1); phi may be batched.
    noise_p>0 adds DepolarizingChannel(noise_p) after every gate."""

    def _noise(q):
        if noise_p > 0:
            qml.DepolarizingChannel(noise_p, wires=q)

    for q in range(N):
        qml.RY(np.pi / 2, wires=q)
        _noise(q)
    t1, t2, t3 = theta_flat[0], theta_flat[1], theta_flat[2]
    for i, j in itertools.combinations(range(N), 2):
        _rzz(t1 / 2.0, i, j)
        _noise(i)
        _noise(j)
    for q in range(N):
        qml.Hadamard(wires=q)
        _noise(q)
    for i, j in itertools.combinations(range(N), 2):
        _rzz(t2 / 2.0, i, j)
        _noise(i)
        _noise(j)
    for q in range(N):
        qml.Hadamard(wires=q)
        _noise(q)
    for q in range(N):
        qml.RX(t3, wires=q)
        _noise(q)
    for q in range(N):
        qml.RZ(phi, wires=q)
        _noise(q)
    v1, v2, v3 = curly_flat[0], curly_flat[1], curly_flat[2]
    for q in range(N):
        qml.RX(v3, wires=q)
        _noise(q)
    for q in range(N):
        qml.Hadamard(wires=q)
        _noise(q)
    for i, j in itertools.combinations(range(N), 2):
        _rzz(v2 / 2.0, i, j)
        _noise(i)
        _noise(j)
    for q in range(N):
        qml.Hadamard(wires=q)
        _noise(q)
    for i, j in itertools.combinations(range(N), 2):
        _rzz(v1 / 2.0, i, j)
        _noise(i)
        _noise(j)
    for q in range(N):
        qml.RX(np.pi / 2, wires=q)
        _noise(q)


def build_probs_qnode(N, noise_p=0.0, diff_method="best"):
    """qnode -> probs over 2^N outcomes; mixed-state device if noise_p>0."""
    if noise_p > 0:
        dev = qml.device("default.mixed", wires=N)
    else:
        dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="autograd", diff_method=diff_method)
    def circuit_probs(phi, theta_flat, curly_flat):
        apply_circuit(N, phi, theta_flat, curly_flat, noise_p=noise_p)
        return qml.probs(wires=list(range(N)))

    return circuit_probs


def build_state_qnode(N, mixed=False):
    """qnode -> statevector (pure) or density matrix (mixed)."""
    dev = qml.device("default.mixed" if mixed else "default.qubit", wires=N)

    @qml.qnode(dev, interface="autograd")
    def circuit_state(phi, theta_flat, curly_flat, noise_p=0.0):
        apply_circuit(N, phi, theta_flat, curly_flat, noise_p=noise_p)
        return qml.state()

    return circuit_state


# ----------------------------------------------------------------------
# QFI
# ----------------------------------------------------------------------
def qfi_pure(state_fn, theta_flat, curly_flat, phi0=0.0, dphi=1e-5):
    """Pure-state QFI at phi0 by central finite differences."""
    psi_p = np.asarray(state_fn(phi0 + dphi, theta_flat, curly_flat))
    psi_m = np.asarray(state_fn(phi0 - dphi, theta_flat, curly_flat))
    dpsi = (psi_p - psi_m) / (2 * dphi)
    psi0 = (psi_p + psi_m) / 2
    return float(4 * (np.vdot(dpsi, dpsi).real
                      - np.abs(np.vdot(psi0, dpsi)) ** 2))


def qfi_mixed(state_fn, theta_flat, curly_flat, phi0=0.0, dphi=1e-5,
              noise_p=0.0):
    """SLD-QFI of a (possibly mixed) state rho(phi)."""
    rho_p = np.asarray(state_fn(phi0 + dphi, theta_flat, curly_flat,
                                noise_p=noise_p))
    rho_m = np.asarray(state_fn(phi0 - dphi, theta_flat, curly_flat,
                                noise_p=noise_p))
    rho0 = (rho_p + rho_m) / 2
    drho = (rho_p - rho_m) / (2 * dphi)
    vals, vecs = np.linalg.eigh(rho0)
    vals = np.clip(vals, 0.0, None)
    d = vecs.conj().T @ drho @ vecs
    F = 0.0
    for i in range(len(vals)):
        for j in range(len(vals)):
            if vals[i] + vals[j] > 1e-12:
                F += 2 * np.abs(d[i, j]) ** 2 / (vals[i] + vals[j])
    return float(F)


# ----------------------------------------------------------------------
# Readout (bit-flip) noise kernel
# ----------------------------------------------------------------------
def readout_kernel(N, p_flip):
    """K[s', s] = P(observe s' | true s) for independent per-qubit flips."""
    K1 = np.array([[1 - p_flip, p_flip], [p_flip, 1 - p_flip]])
    K = K1
    for _ in range(N - 1):
        K = np.kron(K, K1)
    return K


def depol_equiv_flip(N, p):
    """Equivalent readout bit-flip probability for gate-level depolarizing
    noise of strength ``p`` applied after every gate of the VQI/VQ-CNNI
    circuit (enc=dec=1).

    Depolarizing channels are unitarily covariant, so the channel after every
    gate can be slid to the end of the circuit and composed per qubit:
    each qubit carries ``Lq = 4 (N - 1) + 9`` channels, composing to an
    effective depolarizing probability ``(3/4) [1 - (1 - 4 p / 3)^Lq]``,
    whose computational-basis measurement statistics equal independent
    bit-flips with probability ``f = (1/2) [1 - (1 - 4 p / 3)^Lq]``.
    The equivalence is exact for IsingZZ-type entangling gates and accurate
    to a few 1e-3 in outcome probabilities for the CNOT-based RZZ
    decomposition used here (verified numerically against default.mixed).
    """
    Lq = 4 * (N - 1) + 9
    return 0.5 * (1.0 - (1.0 - 4.0 * p / 3.0) ** Lq)


# ----------------------------------------------------------------------
# Softsign MLP (same architecture/initialization as the notebook)
# ----------------------------------------------------------------------
def softsign(x):
    return x / (1 + pnp.abs(x))


class MLP:
    """(N+1) -> 128 -> 64 -> 2 MLP, softsign hidden activations, linear
    output, L2-normalized; phase = arctan2(v0, v1)."""

    def __init__(self, dim_in, dim_hidden=128, seed=0):
        r = np.random.RandomState(seed)
        h = dim_hidden
        self.W1 = pnp.array(r.randn(h, dim_in) * 0.1, requires_grad=True)
        self.b1 = pnp.zeros(h, requires_grad=True)
        self.W2 = pnp.array(r.randn(h // 2, h) * 0.1, requires_grad=True)
        self.b2 = pnp.zeros(h // 2, requires_grad=True)
        self.W3 = pnp.array(r.randn(2, h // 2) * 0.1, requires_grad=True)
        self.b3 = pnp.zeros(2, requires_grad=True)

    def __call__(self, p_m):
        h1 = softsign(self.W1 @ p_m + self.b1)
        h2 = softsign(self.W2 @ h1 + self.b2)
        out = self.W3 @ h2 + self.b3
        return out / (pnp.linalg.norm(out) + 1e-6)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_parameters(self, params):
        (self.W1, self.b1, self.W2, self.b2,
         self.W3, self.b3) = params

    def n_params(self):
        return sum(p.size for p in self.parameters())


# ----------------------------------------------------------------------
# Manual Adam — bit-equivalent to qml.AdamOptimizer(stepsize, beta1=0.9,
# beta2=0.99, eps=1e-8) as implemented in PennyLane >= 0.30:
# moments are accumulated WITHOUT bias correction and the correction is
# folded into the step size instead,
#     eta_t = eta * sqrt(1 - beta2^t) / (1 - beta1^t),
#     x <- x - eta_t * fm / (sqrt(sm) + eps).
# NOTE: PennyLane's default is beta2 = 0.99 (NOT 0.999); using any other
# value changes the training trajectory and breaks reproducibility of the
# original notebooks, which used qml.AdamOptimizer(stepsize=0.02).
# ----------------------------------------------------------------------
class Adam:
    def __init__(self, lr=0.02, beta1=0.9, beta2=0.99, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.fm = None
        self.sm = None
        self.t = 0

    def step(self, x, g):
        if self.fm is None:
            self.fm = np.zeros_like(np.asarray(x), dtype=float)
            self.sm = np.zeros_like(np.asarray(x), dtype=float)
        g = np.asarray(g, dtype=float)
        self.fm = self.b1 * self.fm + (1 - self.b1) * g
        self.sm = self.b2 * self.sm + (1 - self.b2) * g ** 2
        self.t += 1
        # bias correction folded into the step size (PennyLane convention)
        eta = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        return pnp.array(x - eta * self.fm / (np.sqrt(self.sm) + self.eps),
                         requires_grad=True)


# ----------------------------------------------------------------------
# SWPE / phase utilities
# ----------------------------------------------------------------------
def swpe_db(phi_pred, phi_true):
    diff = np.angle(np.exp(1j * (np.asarray(phi_pred)
                                 - np.asarray(phi_true))))
    return 10 * np.log10(diff ** 2 + 1e-12)


def test_phases(n_phi=100, n_test=50):
    phi_train = np.linspace(-np.pi, np.pi, n_phi)
    phi_trues = np.linspace(-np.pi + np.pi / n_phi,
                            np.pi + np.pi / n_phi, n_test)
    return phi_train, phi_trues


def hermgauss_phis(sigma_phi, n):
    from numpy.polynomial.hermite import hermgauss
    xs, ws = hermgauss(n)
    phis = np.sqrt(2) * sigma_phi * xs
    weights = ws / np.sqrt(np.pi)
    return weights, phis
