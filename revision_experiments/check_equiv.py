import os
import numpy as np
import pennylane as qml
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqcnni_lib import apply_circuit

N, p = 2, 0.005
theta = np.array([0.3, -0.7, 1.1])
curly = np.array([-0.4, 0.9, 0.2])
phi = 0.7
Lq = 4 * (N - 1) + 9
c_eff = 0.75 * (1.0 - (1.0 - 4.0 * p / 3.0) ** Lq)

dev = qml.device("default.mixed", wires=N)

@qml.qnode(dev)
def noisy():
    apply_circuit(N, phi, theta, curly, noise_p=p)
    return qml.probs(wires=list(range(N)))

@qml.qnode(dev)
def end_raw():
    apply_circuit(N, phi, theta, curly, noise_p=0.0)
    for _ in range(Lq):
        for q in range(N):
            qml.DepolarizingChannel(p, wires=q)
    return qml.probs(wires=list(range(N)))

@qml.qnode(dev)
def end_composed():
    apply_circuit(N, phi, theta, curly, noise_p=0.0)
    for q in range(N):
        qml.DepolarizingChannel(c_eff, wires=q)
    return qml.probs(wires=list(range(N)))

a = np.asarray(noisy())
b = np.asarray(end_raw())
c = np.asarray(end_composed())
print('Lq =', Lq, flush=True)
print('max|noisy - end_raw|       =', float(np.max(np.abs(a - b))), flush=True)
print('max|end_raw - end_composed|=', float(np.max(np.abs(b - c))), flush=True)
print('max|noisy - end_composed|  =', float(np.max(np.abs(a - c))), flush=True)





