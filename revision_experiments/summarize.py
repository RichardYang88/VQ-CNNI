"""Summarize all revision results for the manuscript/response letter."""
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MODELS = [("vqi_local", "VQI-local"),
          ("vqi_global_linear", "VQI-LinearEst"),
          ("vqi_global_lookup", "VQI-NonParamEst"),
          ("vqcnni", "VQ-CNNI")]


def main():
    print("=" * 72)
    print("Median SWPE (dB) over test phases | exact vs finite-shot(20x1e6)")
    print("=" * 72)
    for N in (4, 6, 8):
        print(f"--- N={N} ---")
        for pref, name in MODELS:
            med_e, med_s, qfis = [], [], []
            for s in (0, 1, 2):
                p = os.path.join(RES, f"{pref}_N{N}_s{s}.npz")
                if not os.path.exists(p):
                    continue
                d = np.load(p)
                med_e.append(float(np.median(d["swpe_db"])))
                if "swpe_shots_db" in d:
                    med_s.append(float(np.median(d["swpe_shots_db"])))
                qfis.append(float(d["qfi"]))
            if med_e:
                se = f"{np.median(med_e):.2f} [{np.min(med_e):.2f}," \
                     f"{np.max(med_e):.2f}]"
                ss = (f"{np.median(med_s):.2f} [{np.min(med_s):.2f},"
                      f"{np.max(med_s):.2f}]" if med_s else "n/a")
                qf = f"{np.median(qfis):.2f} [{np.min(qfis):.2f}," \
                     f"{np.max(qfis):.2f}]"
                print(f"  {name:22s} exact {se:24s} shots {ss:24s} "
                      f"QFI {qf}")
    print()
    fx = os.path.join(RES, "vqcnni_fixed_N8.npz")
    if os.path.exists(fx):
        d = np.load(fx)
        msg = (f"fixed-decoder N=8: exact median SWPE="
               f"{np.median(d['swpe_db']):.2f} dB")
        if "swpe_shots_db" in d:
            msg += f", shots median={np.median(d['swpe_shots_db']):.2f} dB"
        print(msg)
    print()
    for key in ("depol", "readout"):
        ev = os.path.join(RES, f"noise_eval_{key}_N8.npz")
        if os.path.exists(ev):
            d = np.load(ev)
            print(f"noise eval {key}:")
            for lv, m in zip(d["levels"], d["med_swpe_db"]):
                print(f"   {key}={lv:.4f}: median SWPE={m:.2f} dB")
        ft = os.path.join(RES, f"noise_ft_{key}_N8.npz")
        if os.path.exists(ft):
            d = np.load(ft)
            meta = json.loads(str(d["meta"]))
            print(f"noise finetune {key} level={meta['level']}: "
                  f"noisy {d['swpe_noisy_before']:.2f} -> "
                  f"{d['swpe_noisy_after']:.2f} dB; "
                  f"clean {d['swpe_clean_before']:.2f} -> "
                  f"{d['swpe_clean_after']:.2f} dB")
    print()
    for a in ("softsign", "tanh", "arctan", "sigmoid", "elu",
              "softsign_shift"):
        p = os.path.join(RES, f"act_{a}_N8.npz")
        if os.path.exists(p):
            d = np.load(p)
            print(f"act {a:15s} exact median={np.median(d['swpe_db']):.2f} "
                  f"dB  shots median={np.median(d['swpe_shots_db']):.2f} dB")


if __name__ == "__main__":
    main()
