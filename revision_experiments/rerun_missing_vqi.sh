#!/bin/bash
# Rerun any missing results sequentially with retries (segfault/OOM recovery).
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -c 0
PY="${PYTHON:-python3}"
cd "$(dirname "$0")"
mkdir -p results logs

run_retry() {  # $1=logfile  $2..=command
  local log=$1; shift
  for attempt in 1 2 3 4; do
    "$@" > "$log" 2>&1 && return 0
    echo "  attempt $attempt failed ($*) $(date)" >> logs/rerun_master.log
    sleep 2
  done
  echo "GAVE-UP: $*" >> logs/rerun_master.log
  return 1
}

for N in 4 6 8; do
  for v in local global_linear global_lookup; do
    for s in 0 1 2; do
      f=results/vqi_${v}_N${N}_s${s}.npz
      if [ ! -f "$f" ]; then
        echo "rerun $v N=$N s=$s $(date)" >> logs/rerun_master.log
        run_retry logs/vqi_${v}_N${N}_s${s}.log \
          $PY train_vqi_global.py --variant $v --N $N --seed $s --out $f
      fi
    done
  done
done

# fixed decoder experiment (needs vqi_local_N8_s0)
if [ ! -f results/vqcnni_fixed_N8.npz ] && [ -f results/vqi_local_N8_s0.npz ]; then
  echo "rerun fixed decoder $(date)" >> logs/rerun_master.log
  run_retry logs/train_fixed.log \
    $PY train_fixed.py --vqi results/vqi_local_N8_s0.npz --N 8 \
      --out results/vqcnni_fixed_N8.npz
fi

# depolarizing fine-tuning (was OOM-killed)
if [ ! -f results/noise_ft_depol_N8.npz ]; then
  echo "rerun finetune depol $(date)" >> logs/rerun_master.log
  run_retry logs/noise_ft_depol.log \
    $PY noise_study.py --stage finetune --noise depol --level 0.005 --N 8 \
      --model results/vqcnni_N8_s0.npz --maxiter 1500 \
      --out results/noise_ft_depol_N8.npz
fi

echo "RERUN DONE $(date)" >> logs/rerun_master.log
echo "RERUN DONE $(date)"

