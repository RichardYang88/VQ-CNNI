#!/bin/bash
# Wait for the in-flight canonical N=8 seed-0 retraining to finish, then
# run the full aligned revision pipeline (chains A+B, activations,
# fixed-decoder experiment, figures, numeric summary).
set -u
cd "$(dirname "$0")"
while pgrep -f "train_vqcnni_scaling.py --N 8 --act softsign --seed 0" \
        > /dev/null; do
    sleep 15
done
echo "[launcher] canonical run done $(date)"
bash rerun_aligned.sh > logs/launcher_master.log 2>&1
echo "[launcher] pipeline finished $(date)"
