#!/bin/bash
# Depol fine-tune with retry: earlier attempts died silently (segfault)
# mid-run; retry training until the npz appears, then patch with the
# exact gate-level noise evaluation (also retried).
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for attempt in 1 2 3 4 5 6 7 8; do
  if [ -f results/noise_ft_depol_N8.npz ]; then
    echo "[depol-ft] training result present"
    break
  fi
  echo "[depol-ft] train attempt $attempt $(date)"
  $PY noise_study.py --stage finetune --noise depol --level 0.005 \
      --N 8 --model results/vqcnni_N8_s0.npz --maxiter 1500 \
      --out results/noise_ft_depol_N8.npz >> logs/noise_ft_depol3.log 2>&1
  sleep 2
done

for attempt in 1 2 3 4 5 6; do
  if "$PY" -c "
import numpy as np, json
m = json.loads(str(np.load('results/noise_ft_depol_N8.npz')['meta']))
exit(0 if m.get('exact_eval') else 1)" 2>/dev/null; then
    echo "[depol-ft] exact eval present"
    break
  fi
  echo "[depol-ft] patch attempt $attempt $(date)"
  $PY noise_study.py --stage patch_exact \
      --model results/noise_ft_depol_N8.npz >> logs/noise_ft_depol3.log 2>&1
  sleep 2
done
echo "[depol-ft] FINISHED $(date)"
