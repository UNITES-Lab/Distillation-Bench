#!/usr/bin/env bash
# run_eval.sh – evaluate several models one‑after‑another on the same task.

set -euo pipefail      # stop on the first error; catch unset vars; fail piped cmds

DATASETS=(
  "SQA"
  "CSQA"
  "ARC"
  "MATH"
  "GSM8K"
  "ANLI"
  "Date"
)

for DATASET in "${DATASETS[@]}"; do
  echo "===> Evaluating ${DATASET}"
  CUDA_VISIBLE_DEVICES=3 python evaluate_zs.py \
      --model "llama-r1" \
      --task ${DATASET}
  echo "<=== Finished ${DATASET}"
done

echo "✅  All evaluations done."
