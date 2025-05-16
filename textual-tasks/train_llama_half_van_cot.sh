#!/usr/bin/env bash
# run_eval.sh – evaluate several models one‑after‑another on the same task.
# chmod +x 
set -euo pipefail      # stop on the first error; catch unset vars; fail piped cmds


DATASETS=(
  "SQA"
  "ARC"
  "GSM8K"
  "Date"
)

RATIOS=(
    25
    50
    75
)

for DATASET in "${DATASETS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do 
        echo "===> Evaluating ${DATASET} with ratio ${RATIO}"
        CUDA_VISIBLE_DEVICES=6 python train_van_cot.py --task ${DATASET} --n ${RATIO} --model llama-8b
        echo "<=== Finished ${DATASET}"
    done
done

for DATASET in "${DATASETS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do 
        echo "===> Evaluating ${DATASET} with ratio ${RATIO}"
          CUDA_VISIBLE_DEVICES=6 python evaluate.py \
            --model llama-8b \
            --n ${RATIO} \
            --type "_vanilla_cot" \
            --task ${DATASET}
        echo "<=== Finished ${DATASET}"
    done
done


echo "✅  All training done."
