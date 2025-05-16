#!/usr/bin/env bash
# run_eval.sh – evaluate several models one‑after‑another on the same task.

set -euo pipefail      # stop on the first error; catch unset vars; fail piped cmds

DATASETS=(
  "GSM8K1"
  "GSM8K2"
  "MATH"
  "MATH2"
  "Date1"
  "Date2"
)

MODELS=(
    "llama-8b"
    "mistral-7b"
    "gemma-7b"
    "llama-r1"
)


for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}";  do 
        echo "===> Evaluating ${DATASET}"
        CUDA_VISIBLE_DEVICES=3 python evaluate_ood.py \
            --model ${MODEL} \
            --n 100 \
            --task ${DATASET}
        echo "<=== Finished ${DATASET}"
    done 
done

echo "✅  All evaluations done."
