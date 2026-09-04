#!/bin/bash
# Full re-run of all revision experiments with the notebook-aligned
# protocol (PRR WT10346). Chain A (VQ-CNNI scaling + noise) and chain B
# (VQI variants) run in parallel; then the activation comparison, the
# fixed-decoder experiment and its finite-shot evaluation, then figures
# and the numeric summary.
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -c 0
export PYTHON="${PYTHON:-/home/yqc/venv-qc/bin/python}"
cd "$(dirname "$0")"
mkdir -p results logs

echo "[master] start $(date)"
bash run_all.sh A > logs/chainA_master.log 2>&1 &
bash run_all.sh B > logs/chainB_master.log 2>&1 &
wait
echo "[master] chains A+B done $(date)"

bash chain_activations.sh > logs/activations_master.log 2>&1
echo "[master] activations done $(date)"

$PYTHON train_fixed.py --vqi results/vqi_local_N8_s0.npz --N 8 \
    --out results/vqcnni_fixed_N8.npz > logs/vqcnni_fixed.log 2>&1
$PYTHON eval_fixed_shots.py > logs/eval_fixed_shots.log 2>&1
echo "[master] fixed + shots done $(date)"

$PYTHON make_figures.py all > logs/make_figures.log 2>&1
$PYTHON summarize.py > logs/summarize.log 2>&1
echo "[master] ALL DONE $(date)"
