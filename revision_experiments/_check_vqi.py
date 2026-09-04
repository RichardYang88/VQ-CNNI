import numpy as np, json, glob, os

print(f"{'run':<30}{'niters':>7}{'best_it':>8}{'exact_med':>10}")
files = sorted(glob.glob('revision_experiments/results/vqi_*_N*_s*.npz'))
for f in files:
    d = np.load(f)
    meta = json.loads(str(d['meta']))
    lh = d['loss_hist']
    keep = d['phi_trues'] <= np.pi
    print(f"{os.path.basename(f):<30}{len(lh):>7}{int(np.argmin(lh))+1:>8}"
          f"{np.median(d['swpe_db'][keep]):>10.2f}")

print()
print("original model dirs:")
base = 'VQ-CNNI/8/vqc_1_1/softsign'
print(sorted(os.listdir(base)))
for f in os.listdir(base):
    if f.endswith('.npz'):
        d = np.load(os.path.join(base, f), allow_pickle=True)
        print(f, d.files)
