"""
Noise-robustness study for the PRR revision (Reviewer 2, Comment 4).

Stages:
  eval      : evaluate a trained VQ-CNNI model under gate depolarizing
              noise or readout bit-flip noise (no retraining).
  finetune  : noise-aware fine-tuning starting from the clean-trained
              model, with the noise channel inside the training loop.

Noise models:
  depol  : DepolarizingChannel(p) applied after every gate
           (default.mixed device, exact expectation values).
  readout: independent per-qubit bit flip with probability q applied to
           the measurement statistics (linear kernel, differentiable).

Usage:
  python noise_study.py --stage eval --noise depol --levels 0 0.001 0.005 \
      --N 8 --model results/vqcnni_N8_s0.npz --out results/noise_eval_N8.npz
  python noise_study.py --stage finetune --noise readout --level 0.01 \
      --N 8 --model results/vqcnni_N8_s0.npz --maxiter 1500 \
      --out results/noise_ft_readout_N8.npz
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
                        build_state_qnode, qfi_mixed, swpe_db, test_phases,
                        Adam, softsign, readout_kernel, depol_equiv_flip)


def load_model(path):
    d = np.load(path)
    x = d["x_best"]
    meta = json.loads(str(d["meta"]))
    return pnp.array(np.asarray(x), requires_grad=True), meta


def make_net_batch(N, hidden=128):
    """Shapes for MLP (N+1)->hidden->hidden//2->2."""
    n_m = N + 1
    shapes = [(hidden, n_m), (hidden,),
              (hidden // 2, hidden), (hidden // 2,),
              (2, hidden // 2), (2,)]
    sizes = [int(np.prod(s)) for s in shapes]

    def unpack(flat):
        params, off = [], 0
        for shp, sz in zip(shapes, sizes):
            params.append(flat[off:off + sz].reshape(shp))
            off += sz
        return params

    def net_batch(p_m, params):
        W1, b1, W2, b2, W3, b3 = params
        h1 = softsign(p_m @ W1.T + b1)
        h2 = softsign(h1 @ W2.T + b2)
        out = h2 @ W3.T + b3
        norm = pnp.sqrt(pnp.sum(out ** 2, axis=1, keepdims=True) + 1e-12)
        return out / norm

    return unpack, net_batch


def predict_phases(circuit_probs, x, masks, net_unpack, net_batch,
                   phi_grid, K=None):
    theta, curly = x[:3], x[3:6]
    params = net_unpack(x[6:])
    probs = circuit_probs(pnp.array(phi_grid), theta, curly)
    if K is not None:
        probs = pnp.array(probs) @ K.T
    p_m = probs_to_p_m(probs, masks)
    v = net_batch(p_m, params)
    return pnp.arctan2(v[:, 0], v[:, 1])


def classical_fi(circuit_probs, x, masks, phi0=0.0, dphi=1e-5, K=None):
    """Classical FI of the (optionally readout-noisy) m statistics."""
    theta, curly = x[:3], x[3:6]
    probs_p = np.asarray(circuit_probs(pnp.array(phi0 + dphi), theta, curly))
    probs_m = np.asarray(circuit_probs(pnp.array(phi0 - dphi), theta, curly))
    if K is not None:
        probs_p = probs_p @ K.T
        probs_m = probs_m @ K.T
    pm_p = np.asarray(probs_to_p_m(pnp.array(probs_p), masks))
    pm_m = np.asarray(probs_to_p_m(pnp.array(probs_m), masks))
    pm0 = (pm_p + pm_m) / 2
    dpm = (pm_p - pm_m) / (2 * dphi)
    mask = pm0 > 1e-12
    return float(np.sum(np.where(mask, dpm ** 2 / np.maximum(pm0, 1e-300),
                                 0.0)))



def run_eval(args):
    N = args.N
    x, meta = load_model(args.model)
    index_to_m, unique_m, masks = m_structure(N)
    _, phi_trues = test_phases(100, 50)
    unpack, net_batch = make_net_batch(N, hidden=meta.get("hidden", 128))

    levels = [float(v) for v in args.levels.split(",")]
    med_swpe_list, fi_list = [], []

    if args.noise == "depol":
        for p in levels:
            cp = build_probs_qnode(N, noise_p=p)
            cs = build_state_qnode(N, mixed=True)
            preds = np.asarray(predict_phases(cp, x, masks, unpack,
                                              net_batch, phi_trues))
            sw = swpe_db(preds, phi_trues)
            fi = qfi_mixed(cs, x[:3], x[3:6], noise_p=p)
            med_swpe_list.append(float(np.median(sw)))
            fi_list.append(fi)
            print(f"depol p={p}: median SWPE={med_swpe_list[-1]:.2f} dB, "
                  f"QFI={fi:.3f}", flush=True)
    else:
        cp = build_probs_qnode(N)
        for q in levels:
            K = readout_kernel(N, q) if q > 0 else None
            preds = np.asarray(predict_phases(cp, x, masks, unpack,
                                              net_batch, phi_trues, K=K))
            sw = swpe_db(preds, phi_trues)
            fi = classical_fi(cp, x, masks, K=K)
            med_swpe_list.append(float(np.median(sw)))
            fi_list.append(fi)
            print(f"readout q={q}: median SWPE={med_swpe_list[-1]:.2f} dB, "
                  f"FI(phi=0)={fi:.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, levels=np.array(levels),
             med_swpe_db=np.array(med_swpe_list), fi=np.array(fi_list),
             meta=json.dumps({"stage": "eval", "noise": args.noise,
                              "N": N, "model": args.model}))
    print(f"Saved {args.out}")


def run_finetune(args):
    N = args.N
    x0, meta = load_model(args.model)
    level = float(args.level)
    index_to_m, unique_m, masks = m_structure(N)
    phi_train, phi_trues = test_phases(args.n_train, 50)
    phi_train_pnp = pnp.array(phi_train, requires_grad=False)
    unpack, net_batch = make_net_batch(N, hidden=meta.get("hidden", 128))

    # Both noise types are trained with a differentiable readout bit-flip
    # kernel applied to exact (noiseless) probabilities: for readout noise
    # this is the exact model; for gate-level depolarizing noise it is the
    # equivalent end-of-circuit channel (see vqcnni_lib.depol_equiv_flip),
    # because backpropagation through ~37 Kraus channels per qubit on
    # default.mixed is prohibitively memory-hungry. The reported noisy SWPE
    # values are re-evaluated below on the exact gate-noise model.
    if args.noise == "readout":
        K = readout_kernel(N, level) if level > 0 else None
    else:
        K = readout_kernel(N, depol_equiv_flip(N, level))
    circuit_probs = build_probs_qnode(N)

    def loss_fn(x):
        theta, curly = x[:3], x[3:6]
        params = unpack(x[6:])
        probs = circuit_probs(phi_train_pnp, theta, curly)
        if K is not None:
            probs = probs @ K.T
        p_m = probs_to_p_m(probs, masks)
        v = net_batch(p_m, params)
        phi_pred = pnp.arctan2(v[:, 0], v[:, 1])
        return pnp.mean(1 - pnp.cos(phi_train_pnp - phi_pred))

    grad_fn = autograd.grad(loss_fn)
    opt = Adam(lr=args.lr)
    x = pnp.array(np.asarray(x0), requires_grad=True)

    def val_swpe(x):
        preds = np.asarray(predict_phases(circuit_probs, x, masks, unpack,
                                          net_batch, phi_trues, K=K))
        return float(np.median(swpe_db(preds, phi_trues)))

    swpe_noisy_before = val_swpe(x)
    cp_clean = build_probs_qnode(N)
    preds_clean = np.asarray(predict_phases(cp_clean, x, masks, unpack,
                                            net_batch, phi_trues))
    swpe_clean_before = float(np.median(swpe_db(preds_clean, phi_trues)))

    hist, best_val, best_x = [], np.inf, np.array(x)
    no_improve, t0 = 0, time.time()
    for it in range(1, args.maxiter + 1):
        g = grad_fn(x)
        x = opt.step(x, g)
        if it % 10 == 0:
            v = val_swpe(x)
            hist.append(v)
            if v < best_val:
                best_val, best_x, no_improve = v, np.array(x), 0
            else:
                no_improve += 10
            print(f"[ft {args.noise} {level}] it={it} valSWPE={v:.2f} dB "
                  f"best={best_val:.2f}", flush=True)
            if it > args.min_iters and no_improve >= args.patience:
                print(f"Early stop at {it}", flush=True)
                break

    x_ft = pnp.array(best_x, requires_grad=False)
    swpe_noisy_after = val_swpe(x_ft)
    preds_clean_after = np.asarray(predict_phases(cp_clean, x_ft, masks,
                                                  unpack, net_batch,
                                                  phi_trues))
    swpe_clean_after = float(np.median(swpe_db(preds_clean_after,
                                               phi_trues)))

    meta_out = {"stage": "finetune", "noise": args.noise,
                "level": level, "N": N, "model": args.model,
                "lr": args.lr, "maxiter": args.maxiter,
                "patience": args.patience,
                "min_iters": args.min_iters,
                "time_s": time.time() - t0}
    if args.noise == "depol":
        # noisy SWPE saved here is under the training surrogate
        # (equivalent readout bit-flip channel); run --stage patch_exact
        # afterwards to overwrite it with the exact gate-level noise value.
        meta_out["flip_equiv"] = depol_equiv_flip(N, level)
        meta_out["exact_eval"] = False
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, x_before=np.asarray(x0), x_after=np.asarray(x_ft),
             val_hist=np.array(hist),
             swpe_noisy_before=np.array(swpe_noisy_before),
             swpe_noisy_after=np.array(swpe_noisy_after),
             swpe_clean_before=np.array(swpe_clean_before),
             swpe_clean_after=np.array(swpe_clean_after),
             meta=json.dumps(meta_out))
    print(f"Saved {args.out}\n"
          f"  noisy  SWPE: {swpe_noisy_before:.2f} -> "
          f"{swpe_noisy_after:.2f} dB\n"
          f"  clean  SWPE: {swpe_clean_before:.2f} -> "
          f"{swpe_clean_after:.2f} dB")


def run_patch_exact(args):
    """Re-evaluate a fine-tuned model under the exact gate-level noise
    model (default.mixed, forward only) and overwrite the noisy SWPE
    values in the saved npz. Kept separate from training because the
    mixed-state evaluation can crash the process on this machine."""
    d = np.load(args.model)
    meta = json.loads(str(d["meta"]))
    N, level = int(meta["N"]), float(meta["level"])
    index_to_m, unique_m, masks = m_structure(N)
    unpack, net_batch = make_net_batch(N, hidden=meta.get("hidden", 128))
    _, phi_trues = test_phases(100, 50)
    cp_exact = build_probs_qnode(N, noise_p=level)
    vals = {}
    for key in ("x_before", "x_after"):
        x = pnp.array(np.asarray(d[key]), requires_grad=False)
        preds = np.asarray(predict_phases(cp_exact, x, masks, unpack,
                                          net_batch, phi_trues))
        vals[key] = float(np.median(swpe_db(preds, phi_trues)))
        print(f"{key}: exact noisy median SWPE = {vals[key]:.2f} dB",
              flush=True)
    data = {k: d[k] for k in d.files
            if k not in ("meta", "swpe_noisy_before", "swpe_noisy_after")}
    meta["exact_eval"] = True
    np.savez(args.model, **data,
             swpe_noisy_before=np.array(vals["x_before"]),
             swpe_noisy_after=np.array(vals["x_after"]),
             meta=json.dumps(meta))
    print(f"Patched {args.model} with exact-noise values.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["eval", "finetune", "patch_exact"])
    ap.add_argument("--noise", default="readout",
                    choices=["depol", "readout"])
    ap.add_argument("--levels", default="0")
    ap.add_argument("--level", default="0.01")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--model", required=True)
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--min_iters", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.stage == "eval":
        run_eval(args)
    elif args.stage == "patch_exact":
        run_patch_exact(args)
    else:
        if args.out is None:
            ap.error("--out is required for stage finetune")
        run_finetune(args)


if __name__ == "__main__":
    main()

