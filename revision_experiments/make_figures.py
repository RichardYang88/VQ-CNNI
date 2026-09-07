"""Generate revision figures for the PRR resubmission.

Outputs go to paper/figures_rev/. Requires results/*.npz from the
training scripts and the original VQ-CNNI/VQI saved models.
"""
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "revision_experiments", "results")
OUT = os.path.join(ROOT, "paper", "figures_rev")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.0,
                     "xtick.direction": "in", "ytick.direction": "in"})

MODELS = {
    "VQI-local": ("vqi_local", "#1f77b4", "o"),
    "VQI-LinearEst": ("vqi_global_linear", "#2ca02c", "s"),
    "VQI-NonParamEst": ("vqi_global_lookup", "#ff7f0e", "^"),
    "VQ-CNNI": ("vqcnni", "#d62728", "D"),
}


def load_runs(prefix, N, seeds=(0, 1, 2)):
    runs = []
    for s in seeds:
        p = os.path.join(RES, f"{prefix}_N{N}_s{s}.npz")
        if os.path.exists(p):
            runs.append(np.load(p))
    return runs


def fig2_baselines(N=8):
    """New Fig.2: fair baseline comparison under finite shots.

    The median/mean lines are computed from the seed-0 checkpoint
    (results/<prefix>_N8_s0.npz); for VQ-CNNI this is the
    notebook-exact retraining of the original softsign model used in
    Fig.5.  The IQR shading (25th–75th percentile) pools all available
    seeds (typically 0,1,2 for VQI variants; seed 0 only for VQ-CNNI),
    capturing inter-seed variation so that the shading is visible for
    all models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ax1, ax2 = axes
    for name, (pref, color, mk) in MODELS.items():
        runs = load_runs(pref, N, seeds=(0,))
        if not runs:
            continue
        trues_all = runs[0]["phi_trues"]
        # evaluation domain is [-pi, pi): drop the out-of-domain endpoint
        # (physically a duplicate of the first grid point) whose wrapped
        # prediction would otherwise draw a spurious jump at +pi
        keep = trues_all <= np.pi
        trues = trues_all[keep]
        all_shot = np.concatenate([d["shot_preds"][:, keep] for d in runs],
                                  axis=0)
        mean_p = all_shot.mean(axis=0)
        lo = np.percentile(all_shot, 25, axis=0)
        hi = np.percentile(all_shot, 75, axis=0)
        ax1.plot(trues, mean_p, color=color, marker=mk, ms=4, lw=1.3,
                 label=name)
        ax1.fill_between(trues, lo, hi, color=color, alpha=0.15)
        # SWPE line: seed-0 median (consistency convention)
        sw_seed0 = np.concatenate([d["swpe_shots_db"][:, keep] for d in runs],
                                  axis=0)
        med = np.median(sw_seed0, axis=0)
        ax2.plot(trues, med, color=color, lw=1.4, label=name)
        # IQR shading: pool all available seeds for visible inter-seed spread
        all_runs = load_runs(pref, N, seeds=(0, 1, 2))
        if len(all_runs) >= 1:
            sw_all = np.concatenate([d["swpe_shots_db"][:, keep]
                                     for d in all_runs], axis=0)
            q1 = np.percentile(sw_all, 25, axis=0)
            q3 = np.percentile(sw_all, 75, axis=0)
            ax2.fill_between(trues, q1, q3, color=color, alpha=0.25)
    ax1.plot(trues, trues, "k--", lw=1, alpha=0.6)
    ax1.set_xlabel(r"True phase $\phi$")
    ax1.set_ylabel(r"Predicted phase $\tilde{\phi}$")
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.grid(True, ls="--", alpha=0.3)
    ax2.set_xlabel(r"True phase $\phi$")
    ax2.set_ylabel("SWPE (dB)")
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax2.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.grid(True, ls="--", alpha=0.3)
    ax1.text(-0.12, 1.04, "(a)", transform=ax1.transAxes,
             fontsize=13, fontweight="bold")
    ax2.text(-0.12, 1.04, "(b)", transform=ax2.transAxes,
             fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_baselines.pdf"))
    plt.close(fig)
    print("saved fig2_baselines.pdf")


def fig6_combined():
    """New Fig.6: (a) SWPE scaling with qubit number N; (b) noise
    robustness (median SWPE versus noise level). The noise-mitigation
    (fine-tuning) panel of the previous revision has been removed."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax1 = axes[0]
    Ns = [4, 6, 8]
    styles = {"vqcnni": ("#d62728", "D", "VQ-CNNI"),
              "vqi_local": ("#1f77b4", "o", "VQI-local"),
              "vqi_global_lookup": ("#ff7f0e", "^", "VQI-NonParamEst"),
              "vqi_global_linear": ("#2ca02c", "s", "VQI-LinearEst")}
    for pref, (color, mk, label) in styles.items():
        # The three VQI variants are locally optimal estimators trained to
        # concentrate sensitivity around phi=0, so they are reported at
        # their optimal operating point (evaluation phase closest to
        # phi=0); VQ-CNNI is a global estimator and keeps the full-range
        # median.
        at_phi0 = pref != "vqcnni"
        if at_phi0:
            label += r" ($\phi\!=\!0$)"
        meds = []
        for N in Ns:
            runs = load_runs(pref, N)
            if not runs:
                meds.append(np.nan)
                continue
            key = "swpe_shots_db" if "swpe_shots_db" in runs[0] else "swpe_db"
            # Unified whisker convention: for every method, pool all
            # individual finite-shot evaluations (across seeds and trials)
            # and report the 5th–95th percentile spread around the pooled
            # median.  For VQ-CNNI this pools the 20×50 full-range
            # evaluations, consistent with the box whiskers in Fig. 5(d).
            # For the VQI baselines this pools the evaluations at the
            # single operating phase φ≈0 across all seeds and trials.
            if at_phi0:
                # VQI: pool all trial evaluations at φ≈0 across all seeds
                idx = int(np.argmin(np.abs(runs[0]["phi_trues"])))
                all_vals = np.concatenate(
                    [d[key][:, idx].ravel() for d in runs])
            else:
                # VQ-CNNI: pool all 20×50 full-range evaluations
                all_vals = np.concatenate(
                    [d[key].ravel() for d in runs])
            point = np.median(all_vals)
            meds.append(point)
            ax1.errorbar([N], [point],
                         yerr=[[point - np.percentile(all_vals, 5)],
                               [np.percentile(all_vals, 95) - point]],
                         fmt=mk, color=color, capsize=3, ms=6)
        ax1.plot(Ns, meds, color=color, lw=1.2, label=label)
    ax1.set_xlabel("Number of particles $N$")
    ax1.set_ylabel("SWPE, finite shots (dB)")
    ax1.set_xticks(Ns)
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(True, ls="--", alpha=0.3)

    ax2 = axes[1]
    styles2 = {"depol": ("o-", "#d62728", "Depolarizing $p$"),
               "readout": ("s-", "#1f77b4", "Readout $q$")}
    for key, (sty, col, lab) in styles2.items():
        ev = os.path.join(RES, f"noise_eval_{key}_N8.npz")
        if not os.path.exists(ev):
            continue
        d = np.load(ev)
        lv, med = np.asarray(d["levels"]), np.asarray(d["med_swpe_db"])
        ax2.semilogx(lv[lv > 0], med[lv > 0], sty, color=col, label=lab)
        if np.any(lv == 0):
            ax2.scatter([lv[lv > 0].min() / 2], [med[lv == 0][0]], color=col,
                        zorder=5)
            ax2.annotate("0", (lv[lv > 0].min() / 2, med[lv == 0][0]),
                         textcoords="offset points", xytext=(-10, -2),
                         fontsize=8)
    ax2.set_xlabel("Noise level ($p$ or $q$)")
    ax2.set_ylabel("Median SWPE (dB)")
    noise_levels = lv[lv > 0]
    ax2.set_xticks(noise_levels)
    ax2.set_xticklabels([f"{v:.3f}".rstrip("0").rstrip(".") for v in noise_levels])
    ax2.tick_params(axis="x", which="minor", bottom=False)
    ax2.grid(True, ls="--", alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)

    for i, t in enumerate("(a) (b)".split()):
        axes[i].text(-0.14, 1.04, t, transform=axes[i].transAxes,
                     fontsize=13, fontweight="bold")
    fig.suptitle("VQ-CNNI with Softsign activation ($N=8$)", fontsize=10,
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_combined.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig6_combined.pdf")



def _transparent(png_path):
    """Load a PNG and make pure-white pixels transparent so that saved
    snapshot panels can be overlaid on the main axes (original style)."""
    from PIL import Image
    img = Image.open(png_path).convert("RGBA")
    data = np.array(img)
    data[(data[:, :, :3] == 255).all(axis=2), 3] = 0
    return data


def fig4_inset():
    """Fig.4: (a) dual-axis SWPE/QFI curve with the SWPE axis on a true
    decibel scale, shown only up to the epoch of minimal SWPE; (b-i) the
    epoch snapshots of the feature heatmap and latent manifold embedded
    as true vector graphics from the author-provided
    Epoch_heatmap_manifold.pdf (its internal panel letters a-h are
    replaced by b-i)."""
    from matplotlib.patches import Rectangle
    base = os.path.join(ROOT, "VQ-CNNI", "8", "vqc_1_1", "softsign")
    epochs = np.load(os.path.join(base, "epoch_list.npy"))
    mse_test = np.load(os.path.join(base, "test_mse_vs_epoch.npy"))
    qfi = np.load(os.path.join(base, "qfi_vs_epoch.npy"))
    swpe_db = 10 * np.log10(np.maximum(mse_test, 1e-16))
    # display training only up to the epoch of minimal SWPE
    im = int(np.argmin(swpe_db))
    epochs, swpe_db, qfi = (a[:im + 1] for a in (epochs, swpe_db, qfi))

    # (b-i): embed the author-provided snapshot composite as vector
    # graphics by merging the original PDF page as a Form XObject below
    # this figure's overlay layer (white masks + relabeled letters)
    src_pdf = os.path.join(ROOT, "Epoch_heatmap_manifold.pdf")
    PAGE_W_PT, PAGE_H_PT = 1012.32, 429.664
    # panel letters a-h of the source pdf (pdftotext -bbox, pt, y from top)
    letters = [(32.25, 19.59, 44.40, 56.01), (276.69, 19.59, 289.58, 56.01),
               (521.13, 19.59, 531.80, 56.01), (765.57, 19.59, 778.46, 56.01),
               (32.25, 213.58, 44.45, 250.00), (276.69, 213.58, 284.52, 250.00),
               (521.13, 213.44, 534.02, 249.86), (765.57, 213.58, 778.38, 250.00)]

    fig_w = 14.0
    img_h = fig_w * 0.98 * PAGE_H_PT / PAGE_W_PT  # native aspect ratio
    fig_h = img_h + 3.75
    fig = plt.figure(figsize=(fig_w, fig_h))
    y_img = 0.10 / fig_h
    h_img = img_h / fig_h
    ax1 = fig.add_axes([0.06, y_img + h_img + 0.60 / fig_h, 0.88,
                        2.6 / fig_h])
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("SWPE (dB)", color="tab:blue")
    ax1.plot(epochs, swpe_db, marker="o", ms=3, color="tab:blue",
             label="SWPE (dB)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(-70, 10)
    # horizontal reference: VQI-local at its optimal operating point
    # (same exact evaluation mode as the main text)
    vqi_swpe = np.load(os.path.join(RES, "vqi_local_N8_s0.npz"))["swpe_db"]
    ax1.axhline(float(np.min(vqi_swpe)), color="gray", ls=":", lw=1)
    ax2 = ax1.twinx()
    ax2.set_ylabel("QFI", color="tab:red")
    ax2.plot(epochs, qfi, marker="s", ms=3, ls="--", color="tab:red",
             label="QFI")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 34)
    max_idx = int(np.argmax(qfi))
    ax2.scatter([epochs[max_idx]], [qfi[max_idx]], color="k", zorder=5)
    # vertical dashed marker at the QFI-max epoch, value labeled on x axis
    ax1.axvline(epochs[max_idx], color="k", ls="--", lw=1, alpha=0.7)
    top = int(epochs[-1])
    ax1.set_xticks(sorted(set([0, int(epochs[max_idx])] +
                              list(range(200, top + 1, 200)))))
    ax1.grid(True, ls="--", alpha=0.3)
    ax1.text(-0.045, 1.06, "(a)", transform=ax1.transAxes, fontsize=13,
             fontweight="bold")
    ax1.text(0.58, 0.21, "SWPE (dB)", transform=ax1.transAxes,
             color="tab:blue", fontsize=10)
    ax2.text(0.58, 0.41, "QFI", transform=ax2.transAxes, color="tab:red",
             fontsize=10)

    # overlay layer: white masks over the source panel letters and the
    # new parenthesized labels, saved with a transparent background and
    # merged on top of the vector-embedded source page below
    axi = fig.add_axes([0.01, y_img, 0.98, h_img])
    axi.set_xlim(0, PAGE_W_PT)
    axi.set_ylim(PAGE_H_PT, 0)
    axi.set_axis_off()
    for new_lab, (x0, y0, x1, y1) in zip("bcdefghi", letters):
        axi.add_patch(Rectangle((x0 - 4, y0 - 2), (x1 - x0) + 8,
                                (y1 - y0) + 4, facecolor="white",
                                edgecolor="none", zorder=2))
        # parenthesized labels, consistent with "(a)" above and with the
        # other revised figures
        axi.text(x0 - 4, (y0 + y1) / 2, "(%s)" % new_lab, fontsize=13,
                 fontweight="bold", ha="left", va="center", zorder=3)
    overlay_pdf = os.path.join(OUT, "fig4_overlay.pdf")
    fig.savefig(overlay_pdf, transparent=True)
    plt.close(fig)

    # compose the final PDF: blank page <- source page (uniformly scaled,
    # fully vector) <- matplotlib overlay (panel a, masks, new letters)
    from pypdf import PdfReader, PdfWriter, Transformation
    W_PT, H_PT = fig_w * 72.0, fig_h * 72.0
    s = 0.98 * W_PT / PAGE_W_PT
    ctm = Transformation((s, 0.0, 0.0, s, 0.01 * W_PT, y_img * H_PT))
    writer = PdfWriter()
    page = writer.add_blank_page(width=W_PT, height=H_PT)
    page.merge_transformed_page(PdfReader(src_pdf).pages[0], ctm)
    page.merge_transformed_page(PdfReader(overlay_pdf).pages[0],
                                Transformation())
    with open(os.path.join(OUT, "fig4_inset.pdf"), "wb") as fh:
        writer.write(fh)
    print("saved fig4_inset.pdf (vector panels b-i)")


RANK_ACTS = ["softsign", "arctan", "tanh", "sigmoid", "elu", "softsign_shift"]
RANK_LABELS = {"softsign": "Softsign", "arctan": "Arctan", "tanh": "Tanh",
               "sigmoid": "Sigmoid", "elu": "ELU",
               "softsign_shift": "Softsign-shift"}
VQI_LINE_COLORS = {"VQI-local": "black"}


def _load_ranking_data():
    """Exact and finite-shot SWPE distributions of the retrained
    activation models, loaded from the revision checkpoints
    results/act_<act>_N8.npz (20 independent 10^6-shot trials each).

    Consistency note: with the notebook-aligned training protocol
    (loss 2(1-cos), Adam beta2=0.99, init_seed=42, mlp_seed=0) these
    checkpoints reproduce the original vqc_mlp_<act>.ipynb models, so
    this panel and Fig.5 use the same experiments.
    """
    exact, shots = {}, {}
    for a in RANK_ACTS:
        d = np.load(os.path.join(RES, f"act_{a}_N8.npz"))
        exact[a] = np.asarray(d["swpe_db"])
        shots[a] = np.asarray(d["swpe_shots_db"]).ravel()
    order = sorted(RANK_ACTS, key=lambda a: np.median(exact[a]))
    return exact, shots, order



def _vqi_ref_levels(mode):
    """SWPE of VQI-local at its optimal point phi~0 in the given mode.

    The VQI estimator is locally optimal, so the meaningful reference
    level is the value at the operating point phi~0 rather than the
    global mean (which is dominated by the wrapping penalty far from
    phi=0). mode='exact': exact-probability prediction at the test phase
    closest to 0. mode='shots': mean squared wrapped error over 5000
    independent 10^6-shot sampling trials at that phase (linear-space
    mean, i.e. the bias+variance value, which by construction lies above
    the exact reference; a small-trial median can fluctuate below it).
    """
    p = os.path.join(ROOT, "VQI", "8", "vqc_1_1_0.7")
    phi = np.linspace(-np.pi + np.pi / 100, np.pi + np.pi / 100, 50)

    def sq_err(pred):
        d = np.angle(np.exp(1j * (pred - phi)))
        return d ** 2

    i0 = int(np.argmin(np.abs(phi)))
    if mode == "exact":
        v = float(10 * np.log10(sq_err(
            np.load(os.path.join(p, "phi_preds.npy")))[i0] + 1e-12))
    else:
        s = sq_err(np.load(os.path.join(p, "phi_preds_shots5000.npy")))
        v = float(10 * np.log10(np.mean(s[:, i0]) + 1e-12))
    return {"VQI-local": v}


def _ranking_panel(ax, data, order, refs, title=None, qfi=None, colors=None):
    """SWPE box plots across activations with the VQI-local reference.

    ``qfi`` (optional dict act -> value) appends the pure-state QFI of the
    pre-measurement probe at phi=0 in parentheses below each activation
    name, as in the four-panel Fig.~2 draft; ``colors`` (optional dict
    act -> color) tints each quartile box with the activation's curve
    color instead of the uniform default fill.
    """
    labels = [RANK_LABELS[a] for a in order]
    if qfi is not None:
        labels = [f"{RANK_LABELS[a]}\n({qfi[a]:.1f})" for a in order]
    bp = ax.boxplot([data[a] for a in order], tick_labels=labels,
                    showfliers=False, patch_artist=True, widths=0.6,
                    whis=(5, 95))
    for i, patch in enumerate(bp["boxes"]):
        c = colors[order[i]] if colors is not None else "#9ecae1"
        patch.set_facecolor(c)
        patch.set_alpha(0.45 if colors is not None else 0.6)
        if colors is not None:
            patch.set_edgecolor(c)
    for lab, v in refs.items():
        ax.axhline(v, ls="--", lw=1.5, color=VQI_LINE_COLORS[lab],
                   label=f"{lab}, $\\phi\\approx0$: {v:.1f} dB")
    ax.set_ylabel("SWPE (dB)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    if qfi is not None:
        ax.tick_params(axis="x", rotation=0, labelsize=8.5)
    else:
        ax.tick_params(axis="x", rotation=25, labelsize=9)
    lo = np.percentile(np.concatenate([data[a] for a in order]), 1) - 3
    hi = np.percentile(np.concatenate([data[a] for a in order]), 99) + 9
    ax.set_ylim(lo, hi)
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              handlelength=2.2, labelspacing=0.35)


def fig_ranking_shots():
    """Exact vs finite-shot SWPE ranking across activations (original
    trained models), with VQI-local indicated at its optimal point."""
    exact, shots, order = _load_ranking_data()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _ranking_panel(axes[0], exact, order, _vqi_ref_levels("exact"),
                   title="Exact expectation values")
    _ranking_panel(axes[1], shots, order, _vqi_ref_levels("shots"),
                   title=r"Finite shots ($10^6$)")
    axes[0].text(-0.14, 1.02, "(a)", transform=axes[0].transAxes,
                 fontsize=13, fontweight="bold")
    axes[1].text(-0.14, 1.02, "(b)", transform=axes[1].transAxes,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ranking_shots.pdf"))
    plt.close(fig)
    rank_e = [RANK_LABELS[a] for a in order]
    order_s = sorted(RANK_ACTS, key=lambda a: np.median(shots[a]))
    rank_s = [RANK_LABELS[a] for a in order_s]
    refs_e = _vqi_ref_levels("exact")
    refs_s = _vqi_ref_levels("shots")
    print("saved fig_ranking_shots.pdf")
    print("exact ranking   :", rank_e)
    print("shots ranking   :", rank_s)
    print("VQI ref (exact) :", {k: round(v, 1) for k, v in refs_e.items()})
    print("VQI ref (shots) :", {k: round(v, 1) for k, v in refs_s.items()})




def pca2(X):
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def _pm_grid(x_q, N, phis):
    """p(m|phi) grid (len(phis), N+1) for given quantum params."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "revision_experiments"))
    from vqcnni_lib import build_probs_qnode, m_structure, probs_to_p_m
    from pennylane import numpy as pnp
    _, _, masks = m_structure(N)
    cp = build_probs_qnode(N)
    theta = pnp.array(np.asarray(x_q[:3]), requires_grad=False)
    curly = pnp.array(np.asarray(x_q[3:6]), requires_grad=False)
    probs = np.asarray(cp(pnp.array(phis), theta, curly))
    return np.asarray(probs_to_p_m(pnp.array(probs), masks))


def _pm_raw_grid(x_q, N, phis):
    """Outcome probabilities p(m|phi) via a pennylane-free numpy
    statevector simulation of the enc=dec=1 circuit; returns
    (len(phis), N+1) array ordered by m and the sorted m values."""
    import itertools
    n = N
    dim = 2 ** n
    t1, t2, t3 = np.asarray(x_q[:3], float)
    v1, v2, v3 = np.asarray(x_q[3:6], float)
    idx = np.array([format(k, f"0{n}b").count("0") -
                    format(k, f"0{n}b").count("1") for k in range(dim)])
    ums = np.unique(idx)
    bits = np.array([((np.arange(dim) >> (n - 1 - q)) & 1)
                     for q in range(n)])

    def rx(a):
        c, s = np.cos(a / 2), np.sin(a / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])

    def ry(a):
        c, s = np.cos(a / 2), np.sin(a / 2)
        return np.array([[c, -s], [s, c]])

    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)

    def single(st, U, q):
        s = np.tensordot(U, st.reshape((2,) * n), axes=(1, q))
        return np.moveaxis(s, 0, q).reshape(-1)

    def rz(st, a, q):
        phase = np.where(bits[q] == 1, np.exp(1j * a / 2),
                         np.exp(-1j * a / 2))
        return st * phase

    def cnot(st, i, j):
        kj = 1 << (n - 1 - j)
        ks = np.where((bits[i] == 1) & (bits[j] == 0))[0]
        st = st.copy()
        st[ks], st[ks ^ kj] = st[ks ^ kj], st[ks]
        return st

    def rzz(st, chi, i, j):
        return cnot(rz(cnot(st, i, j), chi, j), i, j)

    pairs = list(itertools.combinations(range(n), 2))
    out = np.zeros((len(phis), len(ums)))
    for r, phi in enumerate(phis):
        st = np.zeros(dim, complex)
        st[0] = 1.0
        for q in range(n):
            st = single(st, ry(np.pi / 2), q)
        for i, j in pairs:
            st = rzz(st, t1 / 2, i, j)
        for q in range(n):
            st = single(st, H, q)
        for i, j in pairs:
            st = rzz(st, t2 / 2, i, j)
        for q in range(n):
            st = single(st, H, q)
        for q in range(n):
            st = single(st, rx(t3), q)
        for q in range(n):
            st = rz(st, phi, q)
        for q in range(n):
            st = single(st, rx(v3), q)
        for q in range(n):
            st = single(st, H, q)
        for i, j in pairs:
            st = rzz(st, v2 / 2, i, j)
        for q in range(n):
            st = single(st, H, q)
        for i, j in pairs:
            st = rzz(st, v1 / 2, i, j)
        for q in range(n):
            st = single(st, rx(np.pi / 2), q)
        probs = np.abs(st) ** 2
        out[r] = [probs[idx == mm].sum() for mm in ums]
    return out, ums


def fig3_geometry():
    """Fig.3 rebuilt from the original trained models in the original
    composite style (column titles, panels a-g), fully vectorial: panels
    a-f are re-rendered from the original circuit/MLP parameters with
    matplotlib (heatmaps as pcolormesh, PCA embeddings as scatter/line,
    correcting the aspect distortion of the original figure), and panel
    g is computed from the original per-phase predictions, now including
    VQI-local, VQ-CNNI and VQ-CNNI-fixed."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    import vqsim
    cnn_d = os.path.join(ROOT, "VQ-CNNI", "8", "vqc_1_1", "softsign")
    fix_d = os.path.join(ROOT, "VQ-CNNI_fixedParam", "8", "vqc_1_1",
                         "softsign")
    vqi_d = os.path.join(ROOT, "VQI", "8", "vqc_1_1_0.7")
    phi50 = np.linspace(-np.pi + np.pi / 100, np.pi + np.pi / 100, 50)
    N = 8
    idx_m, uniq_m, masks = vqsim.m_structure(N)

    def swpe_of(pred):
        d = np.angle(np.exp(1j * (pred - phi50)))
        return 10 * np.log10(d ** 2 + 1e-12)

    def pmatrix(theta, curly):
        return np.stack([vqsim.probs_to_p_m(
            vqsim.circuit_probs(p, theta, curly, N), masks)
            for p in phi50])

    qp_vqi = np.asarray(np.load(os.path.join(
        fix_d, "quantum_params.npz"))["params"]).ravel()
    qp_cnn = np.asarray(np.load(os.path.join(
        cnn_d, "quantum_params.npz"))["params"]).reshape(2, 3)
    pm_vqi = pmatrix(qp_vqi[:3], qp_vqi[3:])
    pm_cnn = pmatrix(qp_cnn[0], qp_cnn[1])
    mlp_fix = vqsim.load_mlp(os.path.join(fix_d, "mlp_params.pkl"))
    mlp_cnn = vqsim.load_mlp(os.path.join(cnn_d, "mlp_params.pkl"))

    fig = plt.figure(figsize=(16.5, 8.4))
    outer = fig.add_gridspec(1, 4, left=0.02, right=0.99, top=0.88,
                             bottom=0.06, wspace=0.25,
                             width_ratios=[1.25, 1.15, 1.15, 0.85])
    cols = [GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[j],
                                    hspace=0.25) for j in range(3)]

    def letter(ax, txt):
        ax.text(-0.06, 1.04, txt, transform=ax.transAxes, fontsize=13,
                fontweight="bold")

    dphi = phi50[1] - phi50[0]

    def heatmap(ax, pm, ttl, lab):
        im = ax.imshow(pm.T, aspect="auto", cmap="plasma",
                       interpolation="nearest", origin="lower",
                       extent=[phi50[0] - dphi / 2, phi50[-1] + dphi / 2,
                               uniq_m[0] - 1, uniq_m[-1] + 1])
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r"p(m|$\phi$)", fontsize=9)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel(r"$\phi$", fontsize=9)
        ax.set_ylabel("m", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_title(ttl, fontsize=11)
        letter(ax, lab)

    def embed(ax, X2d, ttl, lab):
        sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=phi50, cmap="hsv", s=14)
        ax.plot(X2d[:, 0], X2d[:, 1], alpha=0.5, lw=0.7)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$\phi$", fontsize=9)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel("PCA dim 1", fontsize=9)
        ax.set_ylabel("PCA dim 2", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, lw=0.4)
        ax.set_title(ttl, fontsize=11)
        letter(ax, lab)

    # column 1: quantum feature heatmaps (a,b)
    heatmap(fig.add_subplot(cols[0][0]), pm_vqi, "VQI-local", "a")
    heatmap(fig.add_subplot(cols[0][1]), pm_cnn, "VQ-CNNI", "b")
    # column 2: quantum feature manifolds (c,d)
    embed(fig.add_subplot(cols[1][0]), vqsim.pca2(pm_vqi), "VQI-local", "c")
    embed(fig.add_subplot(cols[1][1]), vqsim.pca2(pm_cnn), "VQ-CNNI", "d")
    # column 3: latent manifolds (e,f)
    embed(fig.add_subplot(cols[2][0]),
          vqsim.pca2(np.stack([vqsim.mlp_latent(mlp_fix, x) for x in pm_vqi])),
          "VQ-CNNI-fixed", "e")
    embed(fig.add_subplot(cols[2][1]),
          vqsim.pca2(np.stack([vqsim.mlp_latent(mlp_cnn, x) for x in pm_cnn])),
          "VQ-CNNI", "f")

    # column 4: exact SWPE boxplot (g), from the seed-0 revision
    # checkpoints (same data as Fig.2b exact mode; the VQ-CNNI
    # checkpoint is the notebook-exact retraining of the softsign model
    # whose manifold is shown in panels (b) and (f))
    ax = fig.add_subplot(outer[3])
    names = ["VQI-local", "VQ-CNNI", "VQ-CNNI-fixed"]
    boxes = [np.asarray(np.load(os.path.join(
                 RES, "vqi_local_N8_s0.npz"))["swpe_db"]),
             np.asarray(np.load(os.path.join(
                 RES, "vqcnni_N8_s0.npz"))["swpe_db"]),
             np.asarray(np.load(os.path.join(
                 RES, "vqcnni_fixed_N8.npz"))["swpe_db"])]
    bp = ax.boxplot(boxes, tick_labels=names, showfliers=False,
                    patch_artist=True, whis=(5, 95))

    for patch, c in zip(bp["boxes"], ["#1f77b4", "#d62728", "#2ca02c"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
        patch.set_edgecolor("black")
    ax.set_ylabel("SWPE (dB)")
    ax.grid(True, ls="--", alpha=0.4, axis="y")
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    letter(ax, "g")

    # column titles (original style)
    for spec, txt, colr in [(outer[0], "Quantum feature heatmap", "navy"),
                            (outer[1], "Quantum feature manifold",
                             "darkgreen"),
                            (outer[2], "Latent manifold", "purple"),
                            (outer[3], "Estimation error", "navy")]:
        pos = spec.get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, 0.95, txt, ha="center",
                 fontsize=13, fontweight="bold", color=colr)
    fig.savefig(os.path.join(OUT, "fig3_geometry.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_geometry.pdf")


def fig5_active():
    """Fig.5 regenerated from the original trained models: (a) exact
    SWPE curves; (b) latent manifolds; (c,d) exact / finite-shot SWPE
    ranking panels with the VQI-local baseline indicated at its optimal
    operating point phi~0 in the corresponding evaluation mode; (e)
    response-gain statistics. Layout: top row (a)+(b), bottom row
    (c)+(d)+(e, horizontally compressed)."""
    import vqsim
    phi_trues = np.linspace(-np.pi + np.pi / 100, np.pi + np.pi / 100, 50)
    vqi_path = os.path.join(ROOT, "VQI", "8", "vqc_1_1_0.7")
    vb = os.path.join(ROOT, "VQ-CNNI", "8", "vqc_1_1")
    models = {
        "VQI": dict(path=vqi_path, label="VQI-local", color="black", marker="h",
                    linestyle="--", qfi=17.4),
        "tanh": dict(path=os.path.join(vb, "tanh"), label="Tanh",
                     color="#2ca02c", marker="X", linestyle="-", qfi=8.9),
        "sigmoid": dict(path=os.path.join(vb, "sigmoid"), label="Sigmoid",
                        color="#9467bd", marker="D", linestyle=":",
                        qfi=24.6),
        "softsign": dict(path=os.path.join(vb, "softsign"),
                         label="Softsign", color="#d62728", marker="o",
                         linestyle=(0, (3, 2)), qfi=12.8),
        "softsign_shift": dict(path=os.path.join(vb, "softsign_shift"),
                               label="softsign-shift", color="#8c564b",
                               marker="<",
                               linestyle=(0, (5, 2, 1, 2, 1, 2)), qfi=9.4),
        "elu": dict(path=os.path.join(vb, "elu"), label="ELU",
                    color="#e377c2", marker="*", linestyle=(0, (1, 1)),
                    qfi=20.7),
        "arctan": dict(path=os.path.join(vb, "arctan"), label="Arctan",
                       color="#17becf", marker="h", linestyle=(0, (5, 2)),
                       qfi=8.6),
    }
    jacobian_data = {
        "tanh": (1.0000294539239016, 1.8566152934941547e-05),
        "sigmoid": (1.000722381941264, 0.0015767558672795307),
        "elu": (0.9999968942878045, 0.00010310047034810306),
        "arctan": (1.00002755806634, 1.4348247494348468e-05),
        "softsign": (0.9999684924136879, 5.547654738055314e-06),
        "softsign_shift": (0.9979963921744658, 0.003680973121346913),
    }

    def compute_swpe_db(p, t):
        diff = np.angle(np.exp(1j * (p - t)))
        return 10 * np.log10(diff ** 2 + 1e-12)

    swpe_data = {}
    for name, cfg in models.items():
        f = os.path.join(cfg["path"], "phi_preds.npy")
        if os.path.exists(f):
            swpe_data[name] = compute_swpe_db(np.load(f), phi_trues)
    order = sorted(swpe_data, key=lambda m: np.mean(swpe_data[m]))

    fig = plt.figure(figsize=(18, 11))
    # top row: (a) SWPE curves + (b) latent-manifold block (wide, so the
    # six manifold panels are no longer squeezed into a narrow column);
    # bottom row: (c) exact ranking, (d) finite-shot ranking, (e)
    # response-gain statistics (horizontally compressed).
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.38)
    gs_top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.14)
    gs_bot = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.8],
                                  wspace=0.38)
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax2 = fig.add_subplot(gs_bot[0, 0])
    ax3 = fig.add_subplot(gs_bot[0, 1])
    ax5 = fig.add_subplot(gs_bot[0, 2])

    # (a) SWPE curves
    for name in order:
        cfg = models[name]
        y = swpe_data[name]
        ax1.scatter(phi_trues, y, s=12, alpha=0.08, color=cfg["color"],
                    zorder=1)
        ax1.plot(phi_trues, y, label=cfg["label"], marker=cfg["marker"],
                 linestyle=cfg["linestyle"], color=cfg["color"],
                 markersize=5.5, linewidth=1.8)
    ax1.set_xlabel(r"Ground-truth phase $\phi$", fontsize=13)
    ax1.set_ylabel("SWPE (dB)", fontsize=13)
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$",
                         r"$\pi$"])
    ax1.grid(True, ls="--", alpha=0.2)
    ax1.legend(frameon=True, fancybox=False, edgecolor="none",
               facecolor="white", framealpha=0.85, ncol=3,
               loc="upper center", handlelength=2.8, fontsize=9.5)
    ax1.text(-0.12, 1.03, "(a)", transform=ax1.transAxes, fontsize=14,
             fontweight="bold")

    # (c,d) SWPE ranking across activations under exact expectation
    # values and finite shots, with VQI-local indicated at its optimal
    # operating point phi~0 in the corresponding evaluation mode
    exact_rank, shots_rank, rank_order = _load_ranking_data()
    qfi_vals = {a: models[a]["qfi"] for a in RANK_ACTS}
    act_colors = {a: models[a]["color"] for a in RANK_ACTS}
    _ranking_panel(ax2, exact_rank, rank_order, _vqi_ref_levels("exact"),
                   qfi=qfi_vals, colors=act_colors)
    _ranking_panel(ax3, shots_rank, rank_order, _vqi_ref_levels("shots"),
                   qfi=qfi_vals, colors=act_colors)
    ax2.text(-0.16, 1.03, "(c)", transform=ax2.transAxes, fontsize=14,
             fontweight="bold")
    ax3.text(-0.16, 1.03, "(d)", transform=ax3.transAxes, fontsize=14,
             fontweight="bold")
    # (b) latent manifolds of the six activations: vector re-rendering
    # from the original trained models (PCA of the decoder's second
    # hidden layer, same layout as latent_manifold_all_models.pdf)
    N = 8
    idx_m, uniq_m, masks = vqsim.m_structure(N)
    gs_d = gs_top[0, 1].subgridspec(2, 3, wspace=0.55, hspace=0.8)
    lm_acts = [("softsign", "Softsign"), ("tanh", "Tanh"),
               ("arctan", "Arctan"), ("softsign_shift", "Softsign-shift"),
               ("sigmoid", "Sigmoid"), ("elu", "ELU")]
    for k, (key, ttl) in enumerate(lm_acts):
        b = os.path.join(vb, key)
        q = np.asarray(np.load(os.path.join(
            b, "quantum_params.npz"))["params"]).reshape(2, 3)
        mlp = vqsim.load_mlp(os.path.join(b, "mlp_params.pkl"))
        pm = np.stack([vqsim.probs_to_p_m(
            vqsim.circuit_probs(p, q[0], q[1], N), masks)
            for p in phi_trues])
        l2 = vqsim.pca2(np.stack([vqsim.mlp_latent(mlp, x) for x in pm]))
        axd = fig.add_subplot(gs_d[k // 3, k % 3])
        sc = axd.scatter(l2[:, 0], l2[:, 1], c=phi_trues, cmap="hsv", s=9)
        axd.plot(l2[:, 0], l2[:, 1], alpha=0.5, lw=0.6)
        cb = fig.colorbar(sc, ax=axd, pad=0.02)
        cb.set_label(r"$\phi$", fontsize=8)
        cb.ax.tick_params(labelsize=6)
        axd.set_title(ttl, fontsize=10)
        axd.set_xlabel("PCA dim 1", fontsize=8)
        axd.set_ylabel("PCA dim 2", fontsize=8)
        axd.tick_params(labelsize=6)
        axd.grid(True, alpha=0.3, lw=0.4)
        if k == 0:
            axd.text(-0.45, 1.12, "(b)", transform=axd.transAxes,
                     fontsize=14, fontweight="bold")

    # (e) response-gain statistics
    acts = list(jacobian_data)
    jm = [jacobian_data[a][0] for a in acts]
    jv = [jacobian_data[a][1] for a in acts]
    sc = ax5.scatter(jm, jv, s=70, c=np.log10(jv), cmap="viridis",
                     edgecolor="black", linewidth=0.6, alpha=0.9)
    for i, label in enumerate(["Tanh", "Sigmoid", "ELU", "Arctan",
                               "Softsign", "Softsign-shift"]):
        if label == "Sigmoid":
            ax5.annotate(label, (jm[i], jv[i]), xytext=(-8, 2),
                         textcoords="offset points", fontsize=12, ha="right")
        elif label == "Softsign":
            ax5.annotate(label, (jm[i], jv[i]), xytext=(5, -8),
                         textcoords="offset points", fontsize=12)
        else:
            ax5.annotate(label, (jm[i], jv[i]), xytext=(5, 2),
                         textcoords="offset points", fontsize=12)
    ax5.axvline(x=1.0, color="red", ls="--", lw=1.2, alpha=0.7,
                label=r"$\bar{G}=1$")
    ax5.set_xlabel(r"Response gain mean $\bar{G}$", fontsize=12)
    ax5.set_ylabel(r"Response gain variance $\sigma_G^2$", fontsize=12)
    ax5.set_yscale("log")
    cbar = fig.colorbar(sc, ax=ax5, pad=0.05)
    cbar.set_label(r"$\log_{10}(\sigma_G^2)$", fontsize=12)
    cbar.ax.tick_params(labelsize=8)
    ax5.grid(True, alpha=0.3, ls="--", lw=0.6)
    ax5.legend(loc="lower left", frameon=True, fancybox=False,
               edgecolor="black", fontsize=11)
    x_min, x_max = ax5.get_xlim()
    ax5.set_xlim(min(x_min, 0.996), max(x_max, 1.002))
    y_min, y_max = ax5.get_ylim()
    ax5.set_ylim(y_min, y_max * 0.8)
    ax5.text(-0.15, 1.05, "(e)", transform=ax5.transAxes, fontsize=14,
             fontweight="bold")

    fig.savefig(os.path.join(OUT, "fig5_active.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig5_active.pdf")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["fig2", "fig6", "fig4", "ranking", "fig3",
                            "fig5"]
    if "fig2" in todo:
        fig2_baselines()
    if "fig6" in todo:
        fig6_combined()
    if "fig4" in todo:
        fig4_inset()
    if "ranking" in todo:
        fig_ranking_shots()
    if "fig3" in todo:
        fig3_geometry()
    if "fig5" in todo:
        fig5_active()

