#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob               



declare -a JSONS=()

for arg in "$@"; do
  if [[ -d $arg ]]; then
    JSONS+=("$arg"/*.json)
  else
    JSONS+=("$arg")
  fi
done

if [[ ${#JSONS[@]} -eq 0 ]]; then
  echo "No *.json files found among the given arguments." >&2
  exit 1
fi


for f in "${JSONS[@]}"; do
  out="${f%.json}_rephrased.json"
  if [[ -s $out ]]; then
    echo "✅  Skipping $f (already processed: $out)"
    continue
  fi

  echo "▶️  Rephrasing: $f"
  python rephrase_questions.py "$f"\
    --model "pro" 
done

echo "🎉  All done."
