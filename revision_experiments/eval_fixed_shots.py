"""Finite-shot evaluation (20 trials x 1e6 shots) for the retrained
VQ-CNNI-fixed checkpoint (results/vqcnni_fixed_N8.npz), mirroring the
protocol of train_vqcnni_scaling.py exactly (same decoding path, same
rng seeding seed+1000). The resulting shot_preds / swpe_shots_db are
added to the checkpoint so that Fig.3 panel g can be computed from the
same revision-protocol data as Fig.2b.

Usage:
  python eval_fixed_shots.py
"""

import json
import os
import sys

import numpy as np
from pennylane import numpy as pnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqcnni_lib import (m_structure, probs_to_p_m, build_probs_qnode,
                        swpe_db, test_phases, softsign)

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "results", "vqcnni_fixed_N8.npz")
N_TRIALS = 20


def main():
    d = np.load(PATH)
    meta = json.loads(str(d["meta"]))
    N = meta["N"]
    seed = meta["seed"]
    h = 128
    theta = np.asarray(d["theta_best"])
    curly = np.asarray(d["curly_best"])
    flat = np.asarray(d["mlp_flat_best"])

    index_to_m, unique_m, masks = m_structure(N)
    n_m = len(unique_m)
    shapes = [(h, n_m), (h,), (h // 2, h), (h // 2,), (2, h // 2), (2,)]

    def unpack(v):
        params, off = [], 0
        for shp in shapes:
            sz = int(np.prod(shp))
            params.append(v[off:off + sz].reshape(shp))
            off += sz
        assert off == v.size, "mlp_flat_best does not match decoder shapes"
        return params

    def net(pm, params):
        w1, b1, w2, b2, w3, b3 = params
        h1 = softsign(pm @ w1.T + b1)
        h2 = softsign(h1 @ w2.T + b2)
        out = h2 @ w3.T + b3
        # notebook normalization: out / (||out|| + 1e-6)
        return out / (np.sqrt((out ** 2).sum(axis=1)) + 1e-6)[:, None]

    # use the test grid stored in the checkpoint (the fixedParam protocol
    # evaluates on the shifted grid linspace(-pi, pi - 2*pi/100, 50))
    phi_trues = np.asarray(d["phi_trues"])
    circuit_probs = build_probs_qnode(N)
    probs_test = np.asarray(circuit_probs(pnp.array(phi_trues),
                                          pnp.array(theta),
                                          pnp.array(curly)))
    pm_test = np.asarray(probs_to_p_m(pnp.array(probs_test), masks))

    params = unpack(flat)
    v = np.asarray(net(pm_test, params))
    preds = np.arctan2(v[:, 0], v[:, 1])
    swpe = swpe_db(preds, phi_trues)
    # sanity: the reconstruction must reproduce the stored exact evaluation
    assert np.allclose(preds, np.asarray(d["phi_preds"]), atol=1e-8), \
        "exact predictions not reproduced"
    assert np.allclose(swpe, np.asarray(d["swpe_db"]), atol=1e-8), \
        "exact SWPE not reproduced"

    n_shots = int(1e6)
    rng = np.random.RandomState(seed + 1000)
    shot_preds = np.zeros((N_TRIALS, len(phi_trues)))
    for r in range(N_TRIALS):
        counts = np.array([rng.multinomial(n_shots, p) for p in probs_test])
        p_hat = np.array(probs_to_p_m(pnp.array(counts / n_shots), masks))
        vv = np.asarray(net(p_hat, params))
        shot_preds[r] = np.arctan2(vv[:, 0], vv[:, 1])
    swpe_shots = swpe_db(shot_preds, phi_trues[None, :])

    keep = phi_trues <= np.pi
    print("exact median SWPE (kept phases) :",
          round(float(np.median(swpe[keep])), 2), "dB")
    print("shots median SWPE (kept phases) :",
          round(float(np.median(swpe_shots[:, keep])), 2), "dB")

    data = {k: d[k] for k in d.files}
    data["shot_preds"] = shot_preds
    data["swpe_shots_db"] = swpe_shots
    meta["n_shot_trials"] = N_TRIALS
    meta["shots"] = n_shots
    meta["shot_rng_seed"] = seed + 1000
    data["meta"] = json.dumps(meta)
    np.savez(PATH, **data)
    print("updated", PATH)


if __name__ == "__main__":
    main()
