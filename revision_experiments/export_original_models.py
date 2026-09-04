"""Export the original paper's trained models as revision checkpoints.

The revised figures must all describe the SAME experiments as the
original manuscript.  The original activation models trained with the
*.ipynb notebooks are saved under VQ-CNNI/8/vqc_1_1/<act>/; this script
converts them into the revision npz format used by make_figures.py:

  * results/vqcnni_N8_s0.npz      <- VQ-CNNI/8/vqc_1_1/softsign
  * results/act_<act>_N8.npz      <- VQ-CNNI/8/vqc_1_1/<act>

Exact-probability predictions are taken from the saved phi_preds.npy
(and cross-checked against a NumPy re-evaluation of the saved
parameters); finite-shot predictions use the saved 20-trial arrays when
available (softsign: phi_preds_shots20.npy) and otherwise re-sample 20
trials of 10^6 shots from the exact outcome probabilities of the saved
model with a documented RNG seed (RandomState(1000)).

This guarantees that Fig.2, Fig.3g, Fig.5 and Fig.6b all quote the very
same trained models that appear in the original paper (instead of
retrainings, which in this environment are sensitive to low-level
numerics and can land in neighboring local optima).  The retraining
pipeline (train_vqcnni_scaling.py) remains strictly aligned with the
notebook protocol and is used for the N=4/6 scaling runs, the seed-1/2
repeats, the VQI variants and the fixed-decoder experiment.

Usage:
  python export_original_models.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vqsim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BASE = os.path.join(ROOT, "VQ-CNNI", "8", "vqc_1_1")
ACTS = ["softsign", "tanh", "arctan", "sigmoid", "elu", "softsign_shift"]
N = 8
N_SHOTS = int(1e6)
N_TRIALS = 20
SHOT_RNG_SEED = 1000  # = train_vqcnni_scaling.py convention (seed 0 + 1000)


def swpe_db(pred, true):
    d = np.angle(np.exp(1j * (np.asarray(pred) - np.asarray(true))))
    return 10 * np.log10(d ** 2 + 1e-12)


def _act_fn(act):
    """Hidden-layer activation of the original notebooks."""
    if act == "softsign":
        return lambda x: x / (1 + np.abs(x))
    if act == "tanh":
        return np.tanh
    if act == "arctan":
        return np.arctan
    if act == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if act == "elu":
        return lambda x: np.where(x > 0, x, np.exp(x) - 1.0)
    if act == "softsign_shift":
        return lambda x: x / (1 + np.abs(x)) + 1.0
    raise ValueError(act)


def mlp_predict_act(params, p_m, act):
    """MLP forward with the model's own activation; returns phi_pred."""
    W1, b1, W2, b2, W3, b3 = params
    f = _act_fn(act)
    h2 = f(W2 @ f(W1 @ p_m + b1) + b2)
    out = W3 @ h2 + b3
    out = out / (np.linalg.norm(out) + 1e-6)
    return np.arctan2(out[0], out[1])


def export_act(act, out_name):
    d = os.path.join(BASE, act)
    qp = np.load(os.path.join(d, "quantum_params.npz"))
    theta, curly = [np.asarray(a) for a in qp["params"]]
    mlp = vqsim.load_mlp(os.path.join(d, "mlp_params.pkl"))
    phi_trues = np.load(os.path.join(d, "phi_trues.npy"))
    preds_saved = np.load(os.path.join(d, "phi_preds.npy"))

    # cross-check: recompute exact predictions from the saved parameters
    _, _, masks = vqsim.m_structure(N)
    preds_re = np.array([
        mlp_predict_act(
            mlp, vqsim.probs_to_p_m(vqsim.circuit_probs(p, theta, curly, N),
                                    masks), act)
        for p in phi_trues])
    wrap_diff = np.angle(np.exp(1j * (preds_re - preds_saved)))
    assert np.max(np.abs(wrap_diff)) < 1e-8, \
        "%s: saved predictions not reproduced (%.2e)" % (
            act, np.max(np.abs(wrap_diff)))

    # finite shots
    f20 = os.path.join(d, "phi_preds_shots20.npy")
    if os.path.exists(f20):
        shot_preds = np.load(f20)
        assert shot_preds.shape == (N_TRIALS, len(phi_trues))
        shot_source = "saved phi_preds_shots20.npy (original notebook)"
    else:
        rng = np.random.RandomState(SHOT_RNG_SEED)
        params = mlp
        shot_preds = np.zeros((N_TRIALS, len(phi_trues)))
        probs_exact = np.stack([
            vqsim.circuit_probs(p, theta, curly, N) for p in phi_trues])
        for r in range(N_TRIALS):
            for i, p in enumerate(probs_exact):
                counts = rng.multinomial(N_SHOTS, p)
                p_m = vqsim.probs_to_p_m(counts / N_SHOTS, masks)
                shot_preds[r, i] = mlp_predict_act(params, p_m, act)
        shot_source = ("re-sampled 20 trials x 1e6 shots from the exact "
                       "probabilities, RandomState(%d)" % SHOT_RNG_SEED)

    mlp_flat = np.concatenate([np.ravel(p) for p in mlp])
    x_best = np.concatenate([theta.ravel(), curly.ravel(), mlp_flat])
    qfi = float(np.load(os.path.join(d, "QFI.npy")).ravel()[0])
    meta = {
        "model": "VQ-CNNI", "activation": act, "N": N, "seed": 0,
        "init_seed": 42, "mlp_seed": 0,
        "provenance": "original trained model " + d,
        "shot_source": shot_source,
        "n_train": 100, "n_test": len(phi_trues),
        "n_shot_trials": N_TRIALS, "shots": N_SHOTS,
        "shot_rng_seed": SHOT_RNG_SEED,
        "n_params_total": int(x_best.size), "n_quantum": 6,
        "n_classical": int(mlp_flat.size),
        "notebook": "vqc_mlp_%s.ipynb" % act,
        "qfi": qfi,
    }
    np.savez(os.path.join(RES, out_name),
             x_best=x_best, theta_best=theta, curly_best=curly,
             mlp_flat_best=mlp_flat,
             phi_trues=phi_trues, phi_preds=preds_saved,
             swpe_db=swpe_db(preds_saved, phi_trues),
             shot_preds=shot_preds,
             swpe_shots_db=swpe_db(shot_preds, phi_trues[None, :]),
             qfi=np.array(qfi), meta=json.dumps(meta))
    med_e = float(np.median(swpe_db(preds_saved, phi_trues)))
    med_s = float(np.median(swpe_db(shot_preds, phi_trues[None, :])))
    print("exported %-22s -> %-22s exact median %7.2f dB | "
          "shots median %7.2f dB | QFI %.3f | %s"
          % (act, out_name, med_e, med_s, qfi, shot_source))


def main():
    os.makedirs(RES, exist_ok=True)
    export_act("softsign", "vqcnni_N8_s0.npz")
    for act in ACTS:
        export_act(act, "act_%s_N8.npz" % act)
    print("done.")


if __name__ == "__main__":
    main()
