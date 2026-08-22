# VQ-CNNI — variational quantum–classical neural network interferometer

Code and data accompanying the paper **"Global Quantum Sensing with a
Variational Quantum-Classical Neural Network Interferometer"** (revised
manuscript: [`paper/manuscript.pdf`](paper/manuscript.pdf)).

The model estimates an unknown phase φ with a hybrid pipeline:
**parametrized quantum circuit → basis-state probabilities → aggregation by
population imbalance m → classical MLP → φ̂ = arctan2(y₀, y₁)**, trained with
the squared wrapped phase error (SWPE) loss.

---

## Repository structure

```text
├── vqc_mlp_softsign.ipynb          # Original VQ-CNNI training notebook (Softsign, main model)
├── vqc_mlp_tanh.ipynb              # Activation variants:
├── vqc_mlp_arctan.ipynb            #   tanh / arctan / sigmoid /
├── vqc_mlp_sigmoid.ipynb           #   ELU / softsign-shift
├── vqc_mlp_elu.ipynb
├── vqc_mlp_softsign-shift.ipynb
├── vqc_mlp_softsign-fixedParam.ipynb  # fixed-circuit-parameter variant
├── VQI.ipynb                       # Original VQI (baseline) training notebook
├── Data_analysis.ipynb             # Original analysis notebook (figures of the first submission)
├── VQ-CNNI/                        # Trained VQ-CNNI models, N=8, enc=dec=1 (per activation)
├── VQ-CNNI_fixedParam/             # Fixed-decoder ablation model
├── VQI/                            # Trained VQI-local baseline model
├── Epoch_heatmap_manifold.pdf      # Vector snapshot panels embedded in Fig. 4
├── revision_experiments/           # Reproducible pipeline used for the revised paper
│   ├── vqcnni_lib.py               #   PennyLane model library (circuit, MLP, Adam, QFI, SWPE)
│   ├── vqsim.py                    #   NumPy-only simulator used to re-render figure panels
│   ├── train_vqcnni_scaling.py     #   VQ-CNNI training (N=4/6/8, seeds) + activation comparison
│   ├── train_vqi_global.py         #   VQI baselines: local / linear estimator / lookup estimator
│   ├── train_fixed.py              #   fixed-decoder ablation
│   ├── noise_study.py              #   noise evaluation and noise-aware fine-tuning
│   ├── make_figures.py             #   generates every paper figure → paper/figures_rev/
│   ├── summarize.py                #   prints every number quoted in the manuscript
│   ├── check_equiv.py              #   sanity check of the noise-channel equivalence
│   ├── run_all.sh                  #   main batch runner (all revision experiments)
│   ├── chain_activations.sh        #   activation-comparison runner
│   ├── rerun_missing_vqi.sh        #   backfill runner (missing seeds, fixed decoder, depol FT)
│   ├── run_depol_ft.sh             #   retried depolarizing fine-tuning runner
│   ├── results/                    #   saved NumPy archives (.npz) of all archived runs
│   └── logs/                       #   training logs of the archived runs
├── paper/                          # Manuscript sources and final figures
│   ├── manuscript.tex, references.bib, manuscript.pdf
│   ├── response_to_reviewers.tex   # Point-by-point reply to the referee reports
│   ├── 1.pdf                       # Fig. 1 (schematic)
│   ├── figures_rev/                # Figs. 2–6 (vector PDFs from make_figures.py)
│   └── CHANGES.md                  # Detailed list of revision changes
├── CHANGES.md                      # Summary of revision changes with verified numbers
├── requirements.txt
└── README.md
```

---

## Environment

Requires Python ≥ 3.10 (tested with Python 3.10, PennyLane 0.45):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add `pip install jupyter` if you want to open the original notebooks.

---

## Reproducing the paper (fast path)

All trained arrays are included in the repository, so the figures and the
manuscript numbers can be regenerated without retraining:

```bash
python revision_experiments/make_figures.py   # writes paper/figures_rev/*.pdf
python revision_experiments/summarize.py      # prints all quoted SWPE/QFI numbers
```

Figure → source mapping:

| Paper figure | Generator (`make_figures.py`) | Data |
|---|---|---|
| Fig. 2 (`fig2_baselines.pdf`) | `fig2_baselines` | `results/{vqi_local,vqi_global_linear,vqi_global_lookup,vqcnni}_N8_s{0,1,2}.npz` |
| Fig. 3 (`fig3_geometry.pdf`) | `fig3_geometry` | saved models in `VQ-CNNI/…/softsign`, `VQ-CNNI_fixedParam/…/softsign`, `VQI/…` |
| Fig. 4 (`fig4_inset.pdf`) | `fig4_inset` | `VQ-CNNI/…/softsign` training curves + `Epoch_heatmap_manifold.pdf` |
| Fig. 5 (`fig5_active.pdf`) | `fig5_active` | `results/act_*_N8.npz` + saved models of all six activations |
| Fig. 6 (`fig6_combined.pdf`) | `fig6_combined` | scaling `results/*_N{4,6,8}_s*.npz` + `results/noise_eval_*_N8.npz` |
| ranking (`fig_ranking_shots.pdf`) | `fig_ranking_shots` | `phi_preds*.npy` of the saved models (used in the response letter) |

## Rerunning the experiments

To rerun everything from scratch (takes several hours on a workstation):

```bash
cd revision_experiments
bash run_all.sh              # VQ-CNNI scaling + noise study (chain A) and VQI baselines (chain B)
bash chain_activations.sh    # six-activation comparison at N=8
bash rerun_missing_vqi.sh    # backfills missing seeds, fixed-decoder run, depol fine-tuning
```

Individual runs, e.g.:

```bash
python revision_experiments/train_vqcnni_scaling.py --N 8 --seed 0 \
    --out revision_experiments/results/vqcnni_N8_s0.npz
python revision_experiments/noise_study.py --stage eval --noise depol \
    --levels 0,0.001,0.002,0.005,0.01,0.02 --N 8 \
    --model revision_experiments/results/vqcnni_N8_s0.npz \
    --out revision_experiments/results/noise_eval_depol_N8.npz
```

The original notebooks (`vqc_mlp_*.ipynb`, `VQI.ipynb`) train the models saved
under `VQ-CNNI/`, `VQ-CNNI_fixedParam/` and `VQI/`; `Data_analysis.ipynb`
reproduces the analysis figures of the first submission from those saved
models. All hyperparameters are listed in Table I of the manuscript.

---

## Model summary

1. **Encoding/decoding layers** — R_Z twists, R_X twists and collective R_X
   gates with trainable parameters (enc = dec = 1 for all results here).
2. **Phase encoding** — R_Z(φ) on every qubit.
3. **Measurement** — probability distribution p(s) over the 2ᴺ basis states.
4. **Aggregation by m** — probabilities summed over states with equal
   imbalance m = #0 − #1, giving a vector of length N+1.
5. **Classical MLP** — (N+1) → 128 → 64 → 2 with L2-normalized output;
   φ̂ = arctan2(y₀, y₁).
6. **Loss** — SWPE: 2(1 − cos(φ − φ̂)) over a uniform grid in [−π, π).

---

## Citation

If you use this code, please cite the paper (citation details upon
publication). For questions, open an issue in this repository.

**Acknowledgments** — This work uses [PennyLane](https://pennylane.ai/) for
quantum circuit simulation and automatic differentiation. The ansatz design is
inspired by [Arrazola et al., Nature 611, 679 (2022)](https://www.nature.com/articles/s41586-022-04435-4).

