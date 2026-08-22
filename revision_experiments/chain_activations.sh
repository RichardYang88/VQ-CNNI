#!/bin/bash
# Activation-function comparison (N=8): retrain all six activations with the
# shared pipeline; results include exact-probability and finite-shot SWPE.
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -c 0
PY="${PYTHON:-python3}"
cd "$(dirname "$0")"
mkdir -p results logs

echo "[activations] starting $(date)"

for act in softsign tanh arctan sigmoid elu softsign_shift; do
  f=results/act_${act}_N8.npz
  if [ ! -f "$f" ]; then
    for attempt in 1 2 3; do
      $PY train_vqcnni_scaling.py --act $act --N 8 --seed 0 --out $f \
        > logs/act_${act}.log 2>&1 && break
      echo "[activations] attempt $attempt failed for $act $(date)"
      sleep 2
    done
  fi
done
echo "[activations] DONE $(date)"
