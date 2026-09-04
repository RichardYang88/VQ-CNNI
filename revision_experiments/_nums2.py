import numpy as np, json, os, sys
sys.path.insert(0, 'revision_experiments')

phi = np.linspace(-np.pi + np.pi/100, np.pi + np.pi/100, 50)
keep = phi <= np.pi

print("=== retrained act_*_N8.npz ===")
for a in ["softsign", "arctan", "tanh", "sigmoid", "elu", "softsign_shift"]:
    d = np.load(f'revision_experiments/results/act_{a}_N8.npz')
    sw = d['swpe_db'][keep]
    sh = d['swpe_shots_db'][:, keep]
    print(f"{a:<16} exact_med={np.median(sw):7.2f}  shots_med={np.median(sh):7.2f}"
          f"  qfi={float(d['qfi']):6.2f}")

print("\n=== original models VQ-CNNI/8/vqc_1_1 ===")
for a in ["softsign", "arctan", "tanh", "sigmoid", "elu", "softsign_shift"]:
    pred = np.load(f'VQ-CNNI/8/vqc_1_1/{a}/phi_preds.npy')
    dd = np.angle(np.exp(1j*(pred - phi)))
    swx = 10*np.log10(dd**2 + 1e-12)
    ps = np.load(f'VQ-CNNI/8/vqc_1_1/{a}/phi_preds_shots.npy')
    ds = np.angle(np.exp(1j*(ps - phi)))
    sws = 10*np.log10(ds**2 + 1e-12)
    print(f"{a:<16} exact_med={np.median(swx[keep]):7.2f}  shots_med={np.median(sws[keep]):7.2f}")

print("\n=== QFI field check (softsign) ===")
d1 = np.load('revision_experiments/results/vqcnni_N8_s0.npz')
d2 = np.load('revision_experiments/results/act_softsign_N8.npz')
print("s0 qfi:", np.asarray(d1['qfi']).shape, np.asarray(d1['qfi']))
print("act qfi:", np.asarray(d2['qfi']).shape, np.asarray(d2['qfi']))
from vqcnni_lib import build_state_qnode, qfi_pure
cs = build_state_qnode(8)
th = np.asarray(d1['theta_best']); cu = np.asarray(d1['curly_best'])
print("recomputed QFI from x_best:", qfi_pure(cs, th, cu))
print("qfi_hist tail s0:", np.asarray(d1['qfi_hist'])[-3:])

print("\n=== retrained vqi_local N8 s0 reference levels ===")
d = np.load('revision_experiments/results/vqi_local_N8_s0.npz')
i0 = int(np.argmin(np.abs(d['phi_trues'])))
sw = d['swpe_db']
print("phase grid pt closest to 0:", d['phi_trues'][i0])
print("exact SWPE at i0:", sw[i0])
sh = d['swpe_shots_db']
print("shots at i0 per trial:", np.round(sh[:, i0], 2))
print("shots median at i0:", np.median(sh[:, i0]), " mean:", np.mean(sh[:, i0]))
print("full-range exact median:", np.median(sw[keep]))

po = 'VQI/8/vqc_1_1_0.7'
predo = np.load(os.path.join(po, 'phi_preds.npy'))
ddo = np.angle(np.exp(1j*(predo - phi)))
swo = 10*np.log10(ddo**2 + 1e-12)
i0o = int(np.argmin(np.abs(phi)))
print("original VQI-local exact at i0:", swo[i0o])
if os.path.exists(os.path.join(po, 'phi_preds_shots5000.npy')):
    s5 = np.load(os.path.join(po, 'phi_preds_shots5000.npy'))
    d5 = np.angle(np.exp(1j*(s5 - phi)))**2
    print("original VQI-local shots5000 mean at i0 (dB):",
          10*np.log10(np.mean(d5[:, i0o]) + 1e-12))

print("\n=== Fig2 pooled VQ-CNNI sanity ===")
all_sw = []
for s in (0, 1, 2):
    d = np.load(f'revision_experiments/results/vqcnni_N8_s{s}.npz')
    all_sw.append(d['swpe_shots_db'][:, d['phi_trues'] <= np.pi])
print("pooled median:", np.median(np.concatenate(all_sw)))
