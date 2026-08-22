"""
VQI baselines for the PRR revision (Reviewer 2, Comment 1).

Variants (all use the SAME quantum circuit, enc=dec=1, 6 quantum params):
  local         : original VQI; weighted BMSE under Gaussian prior
                  (Gauss-Hermite quadrature, sigma_phi=0.7), linear
                  estimator phi_hat(m) = a*m with a re-fitted analytically
                  every iteration.  [reproduces the manuscript baseline]
  global_linear : circular loss 1 - E[cos(phi - phi_hat)] over uniform
                  phi in [-pi, pi), estimator phi_hat(m) = a*m (shared a).
  global_lookup : same circular loss, estimator phi_hat(m_k) = c_k free
                  for each of the N+1 m-sectors.

Usage:
  python train_vqi_global.py --variant global_lookup --N 8 --seed 0 \
      --out results/vqi_global_lookup_N8_s0.npz
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
                        hermgauss_phis, Adam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["local", "global_linear", "global_lookup"])
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--min_iters", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    N = args.N
    rng = np.random.RandomState(args.seed)
    index_to_m, unique_m, masks = m_structure(N)
    n_m = len(unique_m)

    circuit_probs = build_probs_qnode(N)
    circuit_state = build_state_qnode(N)

    phi_train_uniform, phi_trues = test_phases(100, 50)
    weights_gh, phis_gh = hermgauss_phis(0.7, 100)

    theta = pnp.array(rng.uniform(-0.1, 0.1, 3), requires_grad=True)
    curly = pnp.array(rng.uniform(-0.1, 0.1, 3), requires_grad=True)
    a = pnp.array(0.3, requires_grad=True)
    c = pnp.array(np.linspace(-np.pi, np.pi, n_m) * 0.1, requires_grad=True)

    def local_loss(theta, curly):
        probs = circuit_probs(pnp.array(phis_gh), theta, curly)  # (B,2^N)
        p_m = probs_to_p_m(probs, masks)                          # (B,n_m)
        m_mean = pnp.sum(p_m * unique_m[None, :], axis=1)
        m2_mean = pnp.sum(p_m * (unique_m ** 2)[None, :], axis=1)
        a_opt = (pnp.sum(weights_gh * phis_gh * m_mean)
                 / pnp.sum(weights_gh * m2_mean))
        bmse = pnp.sum(p_m * (phis_gh[:, None]
                              - a_opt * unique_m[None, :]) ** 2, axis=1)
        return pnp.sum(bmse * weights_gh), a_opt

    def global_linear_loss(theta, curly, a):
        probs = circuit_probs(pnp.array(phi_train_uniform), theta, curly)
        p_m = probs_to_p_m(probs, masks)
        cos_t = pnp.sum(p_m * pnp.cos(phi_train_uniform[:, None]
                                      - a * unique_m[None, :]), axis=1)
        return pnp.mean(1 - cos_t)

    def global_lookup_loss(theta, curly, c):
        probs = circuit_probs(pnp.array(phi_train_uniform), theta, curly)
        p_m = probs_to_p_m(probs, masks)
        cos_t = pnp.sum(p_m * pnp.cos(phi_train_uniform[:, None]
                                      - c[None, :]), axis=1)
        return pnp.mean(1 - cos_t)

    def unpack(x):
        theta, curly = x[:3], x[3:6]
        if args.variant == "global_linear":
            return theta, curly, x[6], None
        if args.variant == "global_lookup":
            return theta, curly, None, x[6:6 + n_m]
        return theta, curly, None, None

    def loss_of(x):
        theta, curly, a_, c_ = unpack(x)
        if args.variant == "local":
            return local_loss(theta, curly)[0]
        if args.variant == "global_linear":
            return global_linear_loss(theta, curly, a_)
        return global_lookup_loss(theta, curly, c_)

    grad_fn = autograd.grad(loss_of)

    if args.variant == "global_linear":
        extra = [float(a)]
    elif args.variant == "global_lookup":
        extra = np.asarray(c)
    else:
        extra = np.zeros(0)
    x = pnp.array(np.concatenate([np.asarray(theta), np.asarray(curly),
                                  extra]), requires_grad=True)
    opt = Adam(lr=args.lr)

    loss_hist = []
    best_loss, best_x, no_improve = np.inf, np.array(x), 0
    t0 = time.time()
    for it in range(1, args.maxiter + 1):
        g = grad_fn(x)
        x = opt.step(x, g)
        loss = float(loss_of(x))
        loss_hist.append(loss)
        if loss < best_loss:
            best_loss, best_x, no_improve = loss, np.array(x), 0
        else:
            no_improve += 1
        if it % 100 == 0:
            print(f"[{args.variant} N={N} s={args.seed}] it={it} "
                  f"loss={loss:.4e} best={best_loss:.4e}", flush=True)
        if it > args.min_iters and no_improve >= args.patience:
            print(f"Early stop at {it}", flush=True)
            break

    theta_b, curly_b, a_b, c_b = unpack(
        pnp.array(best_x, requires_grad=False))
    a_b = None if a_b is None else float(a_b)

    if args.variant == "local":
        _, a_opt = local_loss(theta_b, curly_b)
        phi_hat_m = float(a_opt) * unique_m
    elif args.variant == "global_linear":
        phi_hat_m = a_b * unique_m
    else:
        phi_hat_m = np.asarray(c_b)

    probs_test = np.asarray(
        circuit_probs(pnp.array(phi_trues), theta_b, curly_b))
    p_m_test = np.asarray(probs_to_p_m(pnp.array(probs_test), masks))
    phi_preds = p_m_test @ phi_hat_m
    swpe = swpe_db(phi_preds, phi_trues)

    n_shots = int(1e6)
    shot_preds = np.zeros((20, len(phi_trues)))
    for r in range(20):
        counts = np.array([rng.multinomial(n_shots, p) for p in probs_test])
        p_hat = np.asarray(probs_to_p_m(pnp.array(counts / n_shots), masks))
        shot_preds[r] = p_hat @ phi_hat_m
    swpe_shots = swpe_db(shot_preds, phi_trues[None, :])

    qfi = qfi_pure(circuit_state, theta_b, curly_b)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out,
             theta_best=np.asarray(theta_b), curly_best=np.asarray(curly_b),
             a_best=np.array(a_b if a_b is not None else np.nan),
             c_best=(np.asarray(c_b) if c_b is not None
                     else np.zeros(n_m) * np.nan),
             phi_hat_m=phi_hat_m, unique_m=unique_m,
             phi_trues=phi_trues, phi_preds=phi_preds, swpe_db=swpe,
             shot_preds=shot_preds, swpe_shots_db=swpe_shots,
             loss_hist=np.array(loss_hist), qfi=np.array(qfi),
             meta=json.dumps({"variant": args.variant, "N": N,
                              "seed": args.seed, "lr": args.lr,
                              "maxiter": args.maxiter,
                              "patience": args.patience,
                              "min_iters": args.min_iters,
                              "n_train": 100, "n_test": 50,
                              "n_shot_trials": 20, "shots": n_shots,
                              "time_s": time.time() - t0}))
    print(f"Saved {args.out}; median SWPE={np.median(swpe):.2f} dB, "
          f"QFI={qfi:.3f}")


if __name__ == "__main__":
    main()
