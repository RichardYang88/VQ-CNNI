"""
VQ-CNNI-fixed protocol (PRR revision; matches the original
VQ-CNNI_fixedParam experiment): freeze the circuit parameters of a
trained VQI-local model and train only the Softsign MLP decoder under
the circular loss. Since the circuit is fixed, the measurement
probabilities for all training phases are precomputed once, making the
training purely classical and fast.

Usage:
  python train_fixed.py --vqi results/vqi_local_N8_s0.npz --N 8 \
      --out results/vqcnni_fixed_N8.npz
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
                        Adam, softsign)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqi", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--min_iters", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    N = args.N
    vqi = np.load(args.vqi)
    theta = np.asarray(vqi["theta_best"])
    curly = np.asarray(vqi["curly_best"])
    index_to_m, unique_m, masks = m_structure(N)
    n_m = len(unique_m)

    circuit_probs = build_probs_qnode(N)
    phi_train, phi_trues = test_phases(100, 50)

    # precompute training/test probabilities with frozen circuit
    pm_train = np.asarray(probs_to_p_m(
        pnp.array(np.asarray(circuit_probs(pnp.array(phi_train),
                                           pnp.array(theta),
                                           pnp.array(curly)))), masks))
    pm_test = np.asarray(probs_to_p_m(
        pnp.array(np.asarray(circuit_probs(pnp.array(phi_trues),
                                           pnp.array(theta),
                                           pnp.array(curly)))), masks))

    r = np.random.RandomState(args.seed)
    h = args.hidden
    W1 = pnp.array(r.randn(h, n_m) * 0.1, requires_grad=True)
    b1 = pnp.zeros(h, requires_grad=True)
    W2 = pnp.array(r.randn(h // 2, h) * 0.1, requires_grad=True)
    b2 = pnp.zeros(h // 2, requires_grad=True)
    W3 = pnp.array(r.randn(2, h // 2) * 0.1, requires_grad=True)
    b3 = pnp.zeros(2, requires_grad=True)
    shapes = [(h, n_m), (h,), (h // 2, h), (h // 2,), (2, h // 2), (2,)]

    def unpack(flat):
        params, off = [], 0
        for shp in shapes:
            sz = int(np.prod(shp))
            params.append(flat[off:off + sz].reshape(shp))
            off += sz
        return params

    def net(pm, params):
        w1, bb1, w2, bb2, w3, bb3 = params
        h1 = softsign(pm @ w1.T + bb1)
        h2 = softsign(h1 @ w2.T + bb2)
        out = h2 @ w3.T + bb3
        return out / pnp.sqrt(pnp.sum(out ** 2, axis=1, keepdims=True)
                              + 1e-12)

    pm_train_p = pnp.array(pm_train)

    def loss_fn(flat):
        v = net(pm_train_p, unpack(flat))
        phi_pred = pnp.arctan2(v[:, 0], v[:, 1])
        return pnp.mean(1 - pnp.cos(pnp.array(phi_train) - phi_pred))

    grad_fn = autograd.grad(loss_fn)
    x = pnp.array(np.concatenate([np.ravel(p) for p in
                                  [W1, b1, W2, b2, W3, b3]]),
                  requires_grad=True)
    opt = Adam(lr=args.lr)

    def eval_swpe(flat):
        v = np.asarray(net(pnp.array(pm_test), unpack(flat)))
        preds = np.arctan2(v[:, 0], v[:, 1])
        return preds, swpe_db(preds, phi_trues)

    best_loss, best_x, no_improve = np.inf, np.array(x), 0
    t0 = time.time()
    for it in range(1, args.maxiter + 1):
        g = grad_fn(x)
        x = opt.step(x, g)
        loss = float(loss_fn(x))
        if loss < best_loss:
            best_loss, best_x, no_improve = loss, np.array(x), 0
        else:
            no_improve += 1
        if it % 200 == 0:
            print(f"it={it} loss={loss:.4e}", flush=True)
        if it > args.min_iters and no_improve >= args.patience:
            print(f"Early stop at {it}", flush=True)
            break

    preds, swpe = eval_swpe(best_x)
    x_full = np.concatenate([theta, curly, np.asarray(best_x)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, x_best=x_full, mlp_flat_best=np.asarray(best_x),
             theta_best=theta, curly_best=curly,
             phi_trues=phi_trues, phi_preds=preds, swpe_db=swpe,
             meta=json.dumps({"model": "VQ-CNNI-fixed", "N": N,
                              "seed": args.seed, "lr": args.lr,
                              "vqi_model": args.vqi,
                              "time_s": time.time() - t0}))
    print(f"Saved {args.out}; median SWPE={np.median(swpe):.2f} dB")


if __name__ == "__main__":
    main()
