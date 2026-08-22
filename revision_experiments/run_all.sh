#!/bin/bash
# Run all revision experiments (PRR WT10346). Logs in logs/, results in results/.
set -u
PY="${PYTHON:-python3}"
cd "$(dirname "$0")"
mkdir -p results logs

chain_vqcnni() {
  for N in 8 6 4; do
    for s in 0 1 2; do
      echo "[chain A] VQ-CNNI N=$N seed=$s $(date)"
      $PY train_vqcnni_scaling.py --N $N --seed $s \
        --out results/vqcnni_N${N}_s${s}.npz \
        > logs/vqcnni_N${N}_s${s}.log 2>&1
    done
  done
  # noise study uses N=8 seed0 model
  echo "[chain A] noise eval depol $(date)"
  $PY noise_study.py --stage eval --noise depol \
    --levels 0,0.001,0.002,0.005,0.01,0.02 --N 8 \
    --model results/vqcnni_N8_s0.npz \
    --out results/noise_eval_depol_N8.npz \
    > logs/noise_eval_depol.log 2>&1
  echo "[chain A] noise eval readout $(date)"
  $PY noise_study.py --stage eval --noise readout \
    --levels 0,0.001,0.002,0.005,0.01,0.02 --N 8 \
    --model results/vqcnni_N8_s0.npz \
    --out results/noise_eval_readout_N8.npz \
    > logs/noise_eval_readout.log 2>&1
  echo "[chain A] finetune readout q=0.01 $(date)"
  $PY noise_study.py --stage finetune --noise readout --level 0.01 --N 8 \
    --model results/vqcnni_N8_s0.npz --maxiter 1500 \
    --out results/noise_ft_readout_N8.npz \
    > logs/noise_ft_readout.log 2>&1
  echo "[chain A] finetune depol p=0.005 $(date)"
  $PY noise_study.py --stage finetune --noise depol --level 0.005 --N 8 \
    --model results/vqcnni_N8_s0.npz --maxiter 1500 \
    --out results/noise_ft_depol_N8.npz \
    > logs/noise_ft_depol.log 2>&1
}

chain_vqi() {
  for N in 4 6 8; do
    for v in local global_linear global_lookup; do
      for s in 0 1 2; do
        echo "[chain B] VQI $v N=$N seed=$s $(date)"
        $PY train_vqi_global.py --variant $v --N $N --seed $s \
          --out results/vqi_${v}_N${N}_s${s}.npz \
          > logs/vqi_${v}_N${N}_s${s}.log 2>&1
      done
    done
  done
}

if [ "${1:-}" = "A" ]; then chain_vqcnni
elif [ "${1:-}" = "B" ]; then chain_vqi
else
  chain_vqcnni > logs/chainA_master.log 2>&1 &
  chain_vqi > logs/chainB_master.log 2>&1 &
  wait
  echo "ALL DONE $(date)"
fi
