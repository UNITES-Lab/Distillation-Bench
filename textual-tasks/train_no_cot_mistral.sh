#!/usr/bin/env bash
# run_eval.sh – evaluate several models one‑after‑another on the same task.
# chmod +x 
set -euo pipefail      # stop on the first error; catch unset vars; fail piped cmds

DATASETS=(
  "MATH"
  "GSM8K"
  "ANLI"
  "Date"
)

for DATASET in "${DATASETS[@]}"; do
  echo "===> Evaluating ${DATASET}"
CUDA_VISIBLE_DEVICES=1 python train_no_cot.py --task ${DATASET} --n 100 --model mistral-7b 
  echo "<=== Finished ${DATASET}"
done

echo "✅  All training done."
