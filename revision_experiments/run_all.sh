#!/bin/bash
# Run all revision experiments (PRR WT10346). Logs in logs/, results in results/.
set -u
PY="${PYTHON:-python3}"
cd "$(dirname "$0")"
mkdir -p results logs

chain_vqcnni() {
  for N in 8 6 4; do
    for s in 0 1 2; do
      out=results/vqcnni_N${N}_s${s}.npz
      if [ -f "$out" ]; then echo "[chain A] skip $out"; continue; fi
      # seed 0 keeps the notebook-default init_seed=42 (so vqcnni_N8_s0
      # is the exact retraining of vqc_mlp_softsign.ipynb); seeds 1,2
      # are protocol repeats with their own initialization seeds.
      INIT=""
      [ "$s" -ne 0 ] && INIT="--init_seed $s"
      echo "[chain A] VQ-CNNI N=$N seed=$s $(date)"
      $PY train_vqcnni_scaling.py --N $N --seed $s $INIT \
        --out $out \
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
        out=results/vqi_${v}_N${N}_s${s}.npz
        if [ -f "$out" ]; then echo "[chain B] skip $out"; continue; fi
        echo "[chain B] VQI $v N=$N seed=$s $(date)"
        $PY train_vqi_global.py --variant $v --N $N --seed $s \
          --out $out \
          > logs/vqi_${v}_N${N}_s${s}.log 2>&1
      done
    done
  done
}

# ---------------------- Chain 0: canonical checkpoints ----------------------
# Export the original paper's trained models (VQ-CNNI/8/vqc_1_1/<act>) as
# the canonical N=8 seed-0 checkpoints used by the revised figures, so
# that Fig.2, Fig.3g, Fig.5 and Fig.6b quote exactly the models of the
# original manuscript.
chain_export() {
  echo "[chain 0] export original models $(date)"
  $PY export_original_models.py > logs/export_original.log 2>&1
}

if [ "${1:-}" = "A" ]; then chain_vqcnni
elif [ "${1:-}" = "B" ]; then chain_vqi
elif [ "${1:-}" = "0" ]; then chain_export
else
  chain_export
  chain_vqcnni > logs/chainA_master.log 2>&1 &
  chain_vqi > logs/chainB_master.log 2>&1 &
  wait
  echo "ALL DONE $(date)"
fi
