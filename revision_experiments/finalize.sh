#!/bin/bash
# Finalization after the aligned pipeline: re-run the noise stages on the
# canonical (original-model) N=8 checkpoint, then regenerate figures and
# the numeric summary.
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHON="${PYTHON:-/home/yqc/venv-qc/bin/python}"
cd "$(dirname "$0")"

# wait for the main pipeline to finish
while pgrep -f "rerun_aligned.sh" > /dev/null 2>&1; do
    sleep 20
done
echo "[finalize] pipeline done, re-running noise stages on canonical model $(date)"

$PY noise_study.py --stage eval --noise depol \
    --levels 0,0.001,0.002,0.005,0.01,0.02 --N 8 \
    --model results/vqcnni_N8_s0.npz \
    --out results/noise_eval_depol_N8.npz \
    > logs/noise_eval_depol.log 2>&1
$PY noise_study.py --stage eval --noise readout \
    --levels 0,0.001,0.002,0.005,0.01,0.02 --N 8 \
    --model results/vqcnni_N8_s0.npz \
    --out results/noise_eval_readout_N8.npz \
    > logs/noise_eval_readout.log 2>&1
$PY noise_study.py --stage finetune --noise readout --level 0.01 --N 8 \
    --model results/vqcnni_N8_s0.npz --maxiter 1500 \
    --out results/noise_ft_readout_N8.npz \
    > logs/noise_ft_readout.log 2>&1
$PY noise_study.py --stage finetune --noise depol --level 0.005 --N 8 \
    --model results/vqcnni_N8_s0.npz --maxiter 1500 \
    --out results/noise_ft_depol_N8.npz \
    > logs/noise_ft_depol.log 2>&1
echo "[finalize] noise done $(date)"

$PY make_figures.py all > logs/make_figures_final.log 2>&1
$PY summarize.py > logs/summarize_final.log 2>&1
echo "[finalize] ALL DONE $(date)"
