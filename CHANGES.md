# Changes in the Revised Manuscript (PRR wt10346)

This file summarizes all changes made in the revision of "Global quantum
phase estimation with variational quantum circuit neural networks" in
response to the editor and referee reports. Manuscript changes are marked
by blue (\textcolor{blue}) text in the revised LaTeX source; the
point-by-point replies are in `paper/response_to_reviewers.tex`.

## Verified final numbers (taken from `revision_experiments/results/*.npz`)

### Matched-condition baselines (N=8, 3 seeds, finite-shot median SWPE)
- The three VQI variants are locally optimal estimators, reported at their optimal operating point (test phase closest to φ=0):
  - VQI-local: −55.3 dB
  - VQI-LinearEst (linear estimator, formerly "VQI-global (linear)"): −48.3 dB
  - VQI-NonParamEst (nonparametric single-outcome readout, formerly "VQI-global (lookup)"): −59.8 dB
  - Full-range medians degrade to −7.1 / −8.7 / −12.7 dB (VQI-local / LinearEst / NonParamEst): the local optimum cannot be sustained globally
- VQ-CNNI (global estimator, full-range median): −57.1 dB (IQR −61.8 to −56.3)
- VQ-CNNI matches the best local baseline value within ≈2.7 dB while remaining uniformly accurate over [−π, π)

### Fixed-circuit decoder ablation (N=8, 2 seeds, exact evaluation)
- VQ-CNNI-fixed: −5.7 dB (worst seed −5.7, best −5.8); VQ-CNNI: −62.5 dB
- Isolated decoder contribution: ≈ −56 dB

### Scaling (N=4,6,8; 3 seeds; finite-shot SWPE)
- VQ-CNNI (full-range median): −62.4 (N=4), −59.1 (N=6), −57.1 (N=8) dB
- VQI baselines reported at their optimal operating point φ≈0 (locally optimal estimators): VQI-local −44.9/−50.4/−55.3; VQI-LinearEst −32.2/−34.2/−48.3; VQI-NonParamEst −26.8/−19.0/−59.8 dB
- Full-range medians of the baselines remain between ≈+0.6 dB and ≈−12.7 dB with no systematic improvement in N (VQI-local −3.4/−5.6/−7.1; VQI-LinearEst −0.2/+0.6/−8.7; VQI-NonParamEst −9.4/−9.9/−12.7)
- Gap to the best operating-point baseline: ≈18 dB (N=4), ≈9 dB (N=6), ≈−2.7 dB (N=8, VQI-NonParamEst locally best at φ≈0 while its full-range median is −12.7 dB)

### Noise robustness (trained N=8 model, Softsign activation; median SWPE over 50 test phases, exact noisy outcome probabilities)
- Depolarizing: −75.4 dB (0) → −23.5 dB (p=1e-3) → −8.35 dB (p=5e-3) → +0.9 dB (p=0.02)
- Readout: −75.4 dB (0) → −30.19 dB (q=0.01) → −23.8 dB (q=0.02)
- Noise-aware fine-tuning:
  - Readout q=0.01: −30.19 → −74.57 dB (clean-condition cost: −75.44 → −31.49 dB)
  - Depolarizing p=0.005: −8.35 → −29.44 dB under full gate-level evaluation (fine-tuning loop uses the equivalent readout bit-flip channel; clean-condition cost: −75.44 → −10.41 dB)

### Activation functions (retrained, N=8)
- Exact-evaluation group structure: odd-symmetric (Softsign, Arctan, Tanh) statistically indistinguishable; asymmetric group (ELU, Softsign-shift, Sigmoid) clearly separated.
- Finite-shot median SWPE (10^6 shots, 20 trials, 3 models per activation): Softsign −61.8, Arctan −61.8, Tanh −61.7, ELU −52.4, Softsign-shift −52.2, Sigmoid −41.6 dB; odd-symmetric group saturates a common ≈−61.8 dB shot-noise floor (spread <0.1 dB).

## Changes by section

### Sec. I
- Explicit problem statement with target generator and platforms; positioning against prior unbiased/global estimation work (Sanders & Milburn, Hayashi, Lu et al., Gorecki et al.).
- Contribution narrowed to what joint optimization does to the measurement-representation geometry; Kawaguchi et al. (arXiv:2505.04958) and MacLellan et al. (arXiv:2409.01223) cited and discussed.

### Sec. II.A
- Decoder fully specified; activation placement stated (hidden layers only; output layer linear and l2-normalized).
- New paragraph "Calibration–inference protocol" resolving the loss-function and known-generator concerns (R1.4).

### Sec. II.B (new)
- VQI baseline and matched-objective variants: VQI-LinearEst (linear estimator with learned slope) and VQI-NonParamEst (nonparametric single-outcome readout, formerly "VQI-global (lookup)"), the most general single-outcome estimator (R2.1).

### Sec. II.C
- Terminology change (third round): "decoding Jacobian" replaced by "response gain" G(φ)=dφ̃/dφ throughout (manuscript, response letter, figures); the mean/variance statistics are now written as Ḡ and σ_G² (metrology-standard language; avoids the Jacobian-matrix misnomer for a scalar rate).
- The response gain is defined unambiguously as the local slope of the calibration curve (R1.3).
- Boundary handling clarified: all evaluation grids are the half-open interval [−π,π), so each physical phase (including ±π) is tested exactly once (R1.3).

### Sec. II.D
- Evaluation protocol separated into exact and finite-shot modes; PCA and QFI computation documented; noise channels documented; hyperparameters collected in Table I (R1.6).
- Table I reformatted: fixed-width wrapped columns and shortened entries so the table fits the page width.

### Sec. III.A
- Matched-condition comparison under finite shots (20 trials of 10^6 shots), three independent training runs per model (R2.1, R1.5, R1.3).
- Fig. 2 replaced by the matched-condition figure; caption states the [−π,π) convention and the curves are plotted strictly on that domain (the out-of-domain duplicate endpoint is removed).
- Baseline reporting made consistent with their local nature: the three VQI variants are locally optimal estimators and are quoted at their best operating point (test phase closest to φ=0): −55.3/−48.3/−59.8 dB (VQI-local / VQI-LinearEst / VQI-NonParamEst); the text states that these values are strictly local (a linear or single-outcome readout cannot sustain them globally; full-range medians degrade to −7.1/−8.7/−12.7 dB), while VQ-CNNI keeps its full-range median (−57.1 dB, IQR −61.8 to −56.3), within ≈2.7 dB of the best local value but uniform over [−π,π). Fig. 2(b) caption updated accordingly.

### Sec. III.B
- Fixed-circuit decoder ablation (VQ-CNNI-fixed) moved to the main text (R1.5).

### Sec. III.C
- Fig. 4 restored to the original layout with the SWPE axis corrected to a decibel scale, the VQI-local optimal-point reference line, and epoch-snapshot panels of the feature heatmap and latent manifold (R2.2).
- Second round: curves truncated at the minimal-SWPE epoch (710, the final reported model) so no post-minimum rise is shown; logarithmic inset removed; snapshot panels b-i moved to the empty upper region so they do not obscure the SWPE/QFI curves.
- Third round: snapshot panels removed from the curve axes; the combined four-in-one heatmap panel (feature heatmap + latent PCA snapshot at epochs 0/60/80/717), generated as `Epoch_heatmap_manifold.pdf`, is inserted as panels b–e; SWPE/QFI curves extended to epoch 717 (final saved model); epochs 60/80/710 annotated directly on the SWPE curve.

### Sec. III.D
- Activation discussion rewritten: expressivity equivalence acknowledged with proof sketch (Softsign-shift ≡ Softsign; Tanh ≡ Sigmoid via tanh(x)=2σ(2x)−1; ELU the only distinct function class); differences attributed to optimization dynamics under a fixed training budget, supported by LeCun et al. (1998), Glorot & Bengio (2010), Klambauer et al. (2017), Jacot et al. (2018), Ramachandran et al. (2017); the ELU observation shows that symmetry, not expressivity, controls the grouping (R2.3).
- Reference lines in Fig. 5 documented as the optimal-point (φ≈0) SWPE of the VQI baselines in the corresponding evaluation mode (R2.2).
- Finite-shot activation ranking added to the text (Sec. III.D) and shown as Fig. 5b,c; the same ranking figure is embedded in the response letter after R2.2 (R2.2).
- Third round: the two matched-objective baseline curves (VQI-LinearEst, VQI-NonParamEst) removed from Fig. 5a (they are already shown in Fig. 2); panels b,c now show the original trained models under exact expectation values and 10^6-shot finite sampling, respectively, each with a single VQI-local dashed reference at its optimal operating point φ≈0 evaluated in the same mode as the data (exact: −55.0 dB; finite-shot trial mean over 5000 regenerated 10^6-shot trials: −54.9 dB, values stated in the legends); Sec. III.D text and Fig. 5 caption updated accordingly.
- Fourth round (figure corrections): Fig. 5c VQI reference switched from the 20-trial median (which fluctuated below the exact value by ≈1 SE of the small-sample median) to the linear-space mean over 5000 regenerated independent 10^6-shot trials (−54.9 dB ≥ exact −55.0 dB, as required by the bias–variance decomposition); Fig. 5d latent manifolds re-rendered as true vector graphics from the original trained models (pure-NumPy replica of the original circuit/MLP, validated to ≤2e-10 against the stored predictions) and enlarged; Fig. 3 re-rendered fully vectorially from the original model parameters (heatmaps, PCA embeddings, latent manifolds); Fig. 4a QFI-max marker changed from arrow annotation to a vertical dashed line with the epoch labeled on the x axis, and panel letters b–i restyled to match (a).
- Seventh round (Fig. 5 layout): the six-panel latent-manifold block was squeezed into the narrow bottom-left column and rendered too small; it is moved to the wide top-right slot next to (a). Panels renumbered in reading order: (a) SWPE curves, (b) latent manifolds (enlarged), (c) exact ranking, (d) finite-shot ranking, (e) response-gain statistics (horizontally compressed); bottom row is now (c)+(d)+(e). Old→new mapping: b→c, c→d, d→b. Manuscript text, Fig. 5 caption and response-letter cross-references updated accordingly (reviewer quotes kept verbatim).

### Sec. III.E (new): scaling with system size and robustness to noise
- Scaling with N=4,6,8 (3 seeds per model, identical hyperparameters; Fig. 6a, finite-shot SWPE; the three locally optimal VQI baselines reported at their optimal operating point φ≈0, VQ-CNNI as the full-range median).
- Gate/readout noise impact study (Fig. 6b; Softsign activation stated) (R2.4, R2.5).
- Second round: noise-aware fine-tuning removed (noise mitigation is not required; mentioned as future work only); Fig. 7 removed and its noise-impact panel merged with the scaling panel into a two-panel Fig. 6.

### Abstract and conclusion
- Updated to reflect matched baselines, fixed-circuit ablation, scaling, and noise results.

### Code and data availability
- Statement corrected to reference the actual repository contents (notebooks, `revision_experiments/vqcnni_lib.py`, training scripts, checkpoints, `make_figures.py`).

## New/changed figures
- Fig. 2: matched-condition baselines, finite shots, 3 runs (new).
- Fig. 3: fixed-circuit ablation (new, main text); second round: rebuilt from the original trained models in the original composite layout (original heatmap/manifold PNGs at native aspect ratios, fixing the aspect distortion; panel g computed from the original per-phase predictions and now includes VQI-local, VQ-CNNI and VQ-CNNI-fixed).
- Fig. 4: training dynamics in the original layout — corrected dB scale, epoch snapshots of feature heatmap and latent manifold (revised); second round: truncated at epoch 710 (minimal SWPE), inset removed, snapshots repositioned clear of the curves; third round: combined heatmap panel `Epoch_heatmap_manifold.pdf` inserted as panels b–e, curves extended to epoch 717 with epoch markers; fifth round: panel labels unified to the parenthesized style (a)–(i) (previously (a) vs. bare b–i), and the embedding of the (b)–(i) snapshot images in `fig4_inset.pdf` re-verified by independent rasterization (poppler + ghostscript) at 100–300 dpi; sixth round: panels (b)–(i) re-embedded as true vector graphics — the original `Epoch_heatmap_manifold.pdf` page is merged as a uniformly scaled Form XObject with pypdf (all text, axes, curves and colorbars remain vector; only the heatmap fields stay as the author's native 600 ppi bitmaps), with a transparent matplotlib overlay carrying the white masks and the parenthesized (b)–(i) labels; verified by text extraction (pdftotext now recovers the embedded vector text) and high-resolution rendering.
- Fig. 5: activation comparison regenerated from the original data; second round: restructured to five panels; third round: simplified — (a) SWPE curves (VQI-local + six activations, exact evaluation); (b,c) ranking of the original trained models under exact and finite-shot evaluation, each with a single VQI-local reference at φ≈0 in the corresponding mode; (d) latent manifolds; (e) response-gain statistics (renamed from "Jacobian statistics"); seventh round: layout rearranged — top row (a)+(b) with the latent manifolds enlarged in the wide right slot, bottom row (c) exact ranking + (d) finite-shot ranking + (e) response-gain (horizontally compressed); panels renumbered in reading order (old b→c, c→d, d→b) and all manuscript/response-letter cross-references updated.
- Fig. 6: two panels — (a) SWPE scaling with N, all four models; (b) noise impact (SWPE vs noise level). Old Fig. 7 removed.
- Response letter: finite-shot activation ranking figure embedded after R2.2; the same panels also appear as Fig. 5b,c in the manuscript.

## Repository
- `revision_experiments/`: training scripts (`train_vqcnni_scaling.py`, `train_vqi_global.py`, `train_fixed.py`), shared library `vqcnni_lib.py`, batch runner `run_all.sh`, noise study `noise_study.py`, summarizer `summarize.py`, figure generator `make_figures.py`, trained checkpoints (`checkpoints/`), numerical results (`results/*.npz`), logs (`logs/`).

## Final revision round (2026-09-04)
- Canonical data basis: the revised figures (Figs. 2, 3g, 5, 6b) now quote the
  original trained models of the original manuscript, exported as revision
  checkpoints by revision_experiments/export_original_models.py (retraining is
  sensitive to low-level numerics in this environment). Softsign VQ-CNNI N=8
  numbers therefore coincide with the original submission again (exact median
  -69.2 dB; odd-symmetry breaking -69.2 -> -43.0 dB; response-gain variance
  5.55e-6, lowest of all six activations).
- Added a reviewer-facing clarification that the different appearances of the
  VQ-CNNI/Softsign SWPE distribution in Figs. 2b, 3g and 5c,d reflect only the
  different statistical dimensions displayed (phase-resolved trial band, box
  over the 50 test phases, and box pooled over 20 trials x 50 phases), not
  different data; Fig. 5c,d provides the most comprehensive pooled statistics.
- Statistics convention: one canonical trained model per VQ-CNNI/activation
  configuration; three seeds retained for the VQI baselines (Fig. 6a).
- All affected numbers updated in manuscript.tex and
  response_to_reviewers.tex; figures regenerated (paper/2,3,5,6.pdf);
  manuscript_changes.tex regenerated; details in paper/CHANGES.md.

