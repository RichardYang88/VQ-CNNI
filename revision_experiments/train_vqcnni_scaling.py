"""
VQ-CNNI (Softsign) training for the system-size scaling study
(PRR revision, Reviewer 2 Comment 5).

Identical architecture/hyperparameters to the original softsign notebook:
  * N qubits, enc=dec=1 circuit (6 quantum parameters)
  * MLP: (N+1) -> 128 -> 64 -> 2, softsign hidden activations, linear
    L2-normalized output, phi_pred = arctan2(v0, v1)
  * circular loss over 100 uniform training phases in [-pi, pi)
  * Adam, lr=0.02, T=3000, patience=100, M_min=500, eval every K=10

Usage:
  python train_vqcnni_scaling.py --N 4 --seed 0 --out results/vqcnni_N4_s0.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import autograd
from pennylane import numpy as pnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqcnni_lib import (m_structure, probs_to_p_m, build_probs_qnode,
                        build_state_qnode, qfi_pure, swpe_db, test_phases,
                        Adam, MLP, softsign)


def get_act(name):
    """Hidden-layer activation functions for the activation comparison."""
    if name == "softsign":
        return softsign
    if name == "tanh":
        return pnp.tanh
    if name == "arctan":
        return pnp.arctan
    if name == "sigmoid":
        return lambda x: 1.0 / (1.0 + pnp.exp(-x))
    if name == "elu":
        return lambda x: pnp.where(x > 0, x, pnp.exp(x) - 1.0)
    if name == "softsign_shift":
        return lambda x: x / (1 + pnp.abs(x)) + 1.0
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="softsign")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--min_iters", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    N = args.N
    act = get_act(args.act)
    np.random.seed(args.seed)
    index_to_m, unique_m, masks = m_structure(N)
    n_m = len(unique_m)

    circuit_probs = build_probs_qnode(N)
    circuit_state = build_state_qnode(N)

    phi_train, phi_trues = test_phases(100, 50)
    phi_train_pnp = pnp.array(phi_train)

    theta = pnp.array(np.random.uniform(-0.1, 0.1, 3), requires_grad=True)
    curly = pnp.array(np.random.uniform(-0.1, 0.1, 3), requires_grad=True)
    net = MLP(dim_in=n_m, dim_hidden=args.hidden, seed=args.seed)
    mlp_flat = np.concatenate([np.ravel(p) for p in net.parameters()])

    mlp_shapes = [p.shape for p in net.parameters()]
    mlp_sizes = [p.size for p in net.parameters()]

    def unpack_mlp(flat):
        params, off = [], 0
        for shp, sz in zip(mlp_shapes, mlp_sizes):
            params.append(flat[off:off + sz].reshape(shp))
            off += sz
        return params

    def net_batch(p_m, params):
        W1, b1, W2, b2, W3, b3 = params
        h1 = act(p_m @ W1.T + b1)
        h2 = act(h1 @ W2.T + b2)
        out = h2 @ W3.T + b3
        norm = pnp.sqrt(pnp.sum(out ** 2, axis=1, keepdims=True) + 1e-12)
        return out / norm

    def loss_fn(x):
        theta, curly = x[:3], x[3:6]
        params = unpack_mlp(x[6:])
        probs = circuit_probs(phi_train_pnp, theta, curly)      # (B,2^N)
        p_m = probs_to_p_m(probs, masks)                         # (B,n_m)
        v = net_batch(p_m, params)
        phi_pred = pnp.arctan2(v[:, 0], v[:, 1])
        return pnp.mean(1 - pnp.cos(phi_train_pnp - phi_pred))

    grad_fn = autograd.grad(loss_fn)
    opt = Adam(lr=args.lr)
    x = pnp.array(np.concatenate([np.asarray(theta), np.asarray(curly),
                                  mlp_flat]), requires_grad=True)

    n_total = int(x.size)
    n_quantum = 6
    n_mlp = n_total - 6

    def evaluate(x):
        theta, curly = x[:3], x[3:6]
        params = unpack_mlp(x[6:])
        probs = np.asarray(circuit_probs(pnp.array(phi_trues), theta, curly))
        p_m = np.asarray(probs_to_p_m(pnp.array(probs), masks))
        v = np.asarray(net_batch(pnp.array(p_m), params))
        phi_preds = np.arctan2(v[:, 0], v[:, 1])
        swpe = swpe_db(phi_preds, phi_trues)
        qfi = qfi_pure(circuit_state, theta, curly)
        return phi_preds, swpe, qfi

    loss_hist, med_hist, qfi_hist, epoch_list = [], [], [], []
    best_loss, best_x, no_improve = np.inf, np.array(x), 0
    t0 = time.time()
    for it in range(1, args.maxiter + 1):
        g = grad_fn(x)
        x = opt.step(x, g)
        loss = float(loss_fn(x))
        loss_hist.append(loss)
        if loss < best_loss:
            best_loss, best_x, no_improve = loss, np.array(x), 0
        else:
            no_improve += 1
        if it % args.eval_every == 0:
            _, swpe_e, qfi_e = evaluate(x)
            med_hist.append(float(np.median(swpe_e)))
            qfi_hist.append(qfi_e)
            epoch_list.append(it)
            print(f"[{args.act} N={N} s={args.seed}] it={it} loss={loss:.4e} "
                  f"medSWPE={med_hist[-1]:.2f} dB QFI={qfi_e:.2f}",
                  flush=True)
        if it > args.min_iters and no_improve >= args.patience:
            print(f"Early stop at {it}", flush=True)
            break

    x_best = pnp.array(best_x, requires_grad=False)
    phi_preds, swpe, qfi = evaluate(x_best)

    # finite-shot evaluation: 20 realizations x 1e6 shots
    theta_b, curly_b = x_best[:3], x_best[3:6]
    params_b = unpack_mlp(x_best[6:])
    probs_test = np.asarray(
        circuit_probs(pnp.array(phi_trues), theta_b, curly_b))
    n_shots = int(1e6)
    rng = np.random.RandomState(args.seed + 1000)
    shot_preds = np.zeros((20, len(phi_trues)))
    for r in range(20):
        counts = np.array([rng.multinomial(n_shots, p) for p in probs_test])
        p_hat = np.array(probs_to_p_m(pnp.array(counts / n_shots), masks))
        v = np.asarray(net_batch(pnp.array(p_hat), params_b))
        shot_preds[r] = np.arctan2(v[:, 0], v[:, 1])
    swpe_shots = swpe_db(shot_preds, phi_trues[None, :])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             x_best=np.asarray(x_best),
             theta_best=np.asarray(x_best[:3]),
             curly_best=np.asarray(x_best[3:6]),
             mlp_flat_best=np.asarray(x_best[6:]),
             phi_trues=phi_trues, phi_preds=phi_preds, swpe_db=swpe,
             shot_preds=shot_preds, swpe_shots_db=swpe_shots,
             loss_hist=np.array(loss_hist),
             med_swpe_hist=np.array(med_hist),
             qfi_hist=np.array(qfi_hist), epochs=np.array(epoch_list),
             qfi=np.array(qfi),
             meta=json.dumps({"model": "VQ-CNNI", "activation": args.act,
                              "N": N, "seed": args.seed, "lr": args.lr,
                              "maxiter": args.maxiter,
                              "patience": args.patience,
                              "min_iters": args.min_iters,
                              "eval_every": args.eval_every,
                              "hidden": args.hidden,
                              "n_train": 100, "n_test": 50,
                              "n_shot_trials": 20, "shots": n_shots,
                              "n_params_total": n_total,
                              "n_quantum": n_quantum,
                              "n_classical": n_mlp,
                              "time_s": time.time() - t0}))
    print(f"Saved {args.out}; median SWPE={np.median(swpe):.2f} dB, "
          f"QFI={qfi:.3f}, total params={n_total}")


if __name__ == "__main__":
    main()
