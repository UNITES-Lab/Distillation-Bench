#!/usr/bin/env bash
# run_eval.sh – evaluate several models one‑after‑another on the same task.

set -euo pipefail      # stop on the first error; catch unset vars; fail piped cmds

MODELS=(
  "llama-r1"
  "gemma-7b"
)

for model in "${MODELS[@]}"; do
  echo "===> Evaluating ${model}"
  CUDA_VISIBLE_DEVICES=6 python evaluate.py \
      --model "${model}" \
      --n 100 \
      --task MATH \
      --zero-shot
  echo "<=== Finished ${model}"
done

echo "✅  All evaluations done."
