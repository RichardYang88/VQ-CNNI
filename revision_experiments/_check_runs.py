import numpy as np, json, glob

d1 = np.load('revision_experiments/results/vqcnni_N8_s0.npz')
d2 = np.load('revision_experiments/results/act_softsign_N8.npz')
print('x_best identical:', np.allclose(d1['x_best'], d2['x_best']))
print('swpe_db identical:', np.allclose(d1['swpe_db'], d2['swpe_db']))
print('shot_preds identical:', np.allclose(d1['shot_preds'], d2['shot_preds']))
print()
print(f"{'run':<30}{'stop_it':>8}{'maxit':>7}{'exact_med':>10}  eval tail")
files = sorted(glob.glob('revision_experiments/results/vqi_global_*_N*_s*.npz')) \
      + sorted(glob.glob('revision_experiments/results/vqi_local_N*_s*.npz'))
for f in files:
    try:
        d = np.load(f)
        meta = json.loads(str(d['meta']))
        ep = d['epochs']; hist = d['med_swpe_hist']
        keep = d['phi_trues'] <= np.pi
        print(f"{f.split('/')[-1]:<30}{int(ep[-1]):>8}{meta.get('maxiter','?'):>7}"
              f"{np.median(d['swpe_db'][keep]):>10.2f}  {np.round(hist[-2:],1)}")
    except Exception as e:
        print(f, 'ERR', e)
