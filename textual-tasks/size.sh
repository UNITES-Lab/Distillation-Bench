#!/usr/bin/env bash
# run_eval.sh – evaluate several models one‑after‑another on the same task.

set -euo pipefail      # stop on the first error; catch unset vars; fail piped cmds

DATASETS=(
  "SQA"
  "ARC"
  "GSM8K"
  "Date"
)

MODELS=(
    "qwen-smallest"
    "qwen-small"
    "qwen-medium"
)


for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do 
        echo "===> Evaluating ${DATASET} with model ${MODEL}"
          CUDA_VISIBLE_DEVICES=6 python evaluate.py \
            --model ${MODEL} \
            --n 100 \
            --task ${DATASET}
        echo "<=== Finished ${DATASET}"
    done
done

echo "✅  All evaluations done."
