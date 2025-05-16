# coding=utf-8
"""Evaluate a *base* model with **Zero‑shot‑CoT** prompting.

This script mirrors `evaluate.py` in the RevThink repo, but implements the
 two‑stage prompting procedure from *Large Language Models are Zero‑Shot
 Reasoners* (Kojima et al., 2023).

Key differences w.r.t. the original `evaluate.py`
-------------------------------------------------
1. **No LoRA adapter assumed by default.**  The method is purely inference‑time;
   you can still pass `--adapter_path` manually if you want to test a fine‑tuned
   adapter under Zero‑shot‑CoT.
2. **Two forward passes per example.**
   * First pass extracts a chain‑of‑thought (CoT) with the fixed trigger
     *"Let’s think step by step."*.
   * Second pass appends an answer‑extraction trigger that depends on the task
     (e.g. *"Therefore, the answer (Yes or No) is"* for BoolQ) and forces the
     model to commit to a short, final answer.
3. **Answer parsing & accuracy** reuse the utility functions already shipped in
   `utils.py` and `math_utils.py`.

Example
-------
CUDA_VISIBLE_DEVICES=0 python evaluate_zero_shot_cot.py \
  --task GSM8K --model mistral-7b --model_dir /path/to/hf-cache

"""

import argparse
import json
import os
from typing import Callable, Dict

import transformers
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest  # Optional – only if you load LoRA

from utils import (
    evaluate,
    get_alphabet_choice,
    get_yes_no,
)
from math_utils import parse_number, parse_math_boxed

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

COT_TRIGGER = "Let's think step by step."

ANSWER_TRIGGER_MAP: Dict[str, str] = {
    # yes / no
    "SQA": "\nTherefore, the answer (Yes or No) is",
    "BoolQ": "\nTherefore, the answer (Yes or No) is",
    # multiple‑choice – 5 options (A‑E)
    "CSQA": "\nTherefore, among (A) through (E), the answer is",
    "ARC": "\nTherefore, among (A) through (E), the answer is",
    "ANLI": "\nTherefore, among (A) through (E), the answer is",
    "OBQA": "\nTherefore, among (A) through (E), the answer is",
    "ESNLI": "\nTherefore, among (A) through (E), the answer is",
    # multiple‑choice – 6 options (A‑F)
    "Date": "\nTherefore, among (A) through (F), the answer is",
    # numeric
    "GSM8K": "\nTherefore, the answer (arabic numerals) is",
    "GSM8K-Rev": "\nTherefore, the answer (arabic numerals) is",
    # boxed maths / LaTeX
    "MATH": "\nTherefore, the answer (boxed) is",
    "TabMWP": "\nTherefore, the answer (boxed) is",
}

# ---------------------------------------------------------------------------
# Answer extraction dispatch
# ---------------------------------------------------------------------------

def get_answer_extractor(task: str) -> Callable[[str], str]:
    if task in {"SQA", "BoolQ"}:
        return get_yes_no
    if task in {"ANLI", "ARC", "Date", "CSQA", "OBQA", "ESNLI"}:
        return get_alphabet_choice
    if task in {"GSM8K", "GSM8K-Rev"}:
        return parse_number
    if task in {"MATH", "TabMWP"}:
        return parse_math_boxed
    raise ValueError(f"Unsupported task: {task}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="SQA", type=str)
    parser.add_argument("--model", default="mistral-7b", type=str)
    parser.add_argument("--model_dir", default="", type=str,
                        help="HF cache / download directory")
    parser.add_argument("--max_tokens_cot", default=128, type=int)
    parser.add_argument("--max_tokens_ans", default=32, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--adapter_path", default="", type=str,
                        help="Optional LoRA adapter to load on top of base model")
    args = parser.parse_args()

    # --------------------------- model mapping ---------------------------
    if args.model == "mistral-7b":
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model == "gemma-2b":
        base_model = "google/gemma-2b-it"
    elif args.model == "gemma-7b":
        base_model = "google/gemma-7b-it"
    elif args.model == "qwen-32b":
        base_model = "Qwen/QwQ-32B"
    elif args.model == "llama-8b":
        base_model = "meta-llama/Llama-3.1-8B-Instruct"
    elif args.model == "llama-r1":
        base_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    elif args.model == "qwen":
        base_model = "Qwen/Qwen2.5-7B"
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # --------------------------- tokenizer & templates -------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="right")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    if any(k in args.model for k in ("mistral", "llama")):
        template = ("<s>[INST] Answer the following question:\n### Question: {question} "
                    "[/INST] ### Answer: {answer}")
    elif "gemma" in args.model:
        template = ("<bos><start_of_turn>user\nAnswer the following question:\n"
                    "### Question: {question}<end_of_turn>\n<start_of_turn>model\n"
                    "### Answer: {answer}")
    else:
        raise ValueError("No prompt template for the selected model.")

    # --------------------------- data -----------------------------------
    data_path = f"./data/test_data/{args.task}_test.json"
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    with open(data_path) as f:
        test_samples = json.load(f)

    # --------------------------- LLM setup ------------------------------
    llm = LLM(model=base_model,
              enable_lora=bool(args.adapter_path),
              max_lora_rank=32,
              download_dir=args.model_dir,
              tensor_parallel_size=1)

    lora_req = (LoRARequest("zs_cot_adapter", 1, args.adapter_path)
                if args.adapter_path else None)

    # Sampler cfg – deterministic greedy decoding by default.
    cot_params = SamplingParams(n=1,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens_cot,
                                stop_token_ids=[tokenizer.eos_token_id])

    ans_params = SamplingParams(n=1,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens_ans,
                                stop_token_ids=[tokenizer.eos_token_id])

    # --------------------------- main loop ------------------------------
    answer_trigger = ANSWER_TRIGGER_MAP.get(args.task)
    if answer_trigger is None:
        raise KeyError(f"No answer‑extraction trigger registered for task {args.task}")

    extract_answer = get_answer_extractor(args.task)

    for ex in test_samples:
        q = ex["question"]

        # ---------- stage 1: reasoning ----------
        prompt1 = template.format(question=q, answer=f"{COT_TRIGGER}")
        reasoning = llm.generate([prompt1], cot_params, lora_request=lora_req)[0].outputs[0].text
        ex["reasoning"] = reasoning

        # ---------- stage 2: answer extraction ----------
        prompt2 = prompt1 + reasoning + " " + answer_trigger
        answer_text = llm.generate([prompt2], ans_params, lora_request=lora_req)[0].outputs[0].text
        ex["answer_raw"] = answer_text
        ex["pred"] = extract_answer(answer_text)

    # --------------------------- metrics & save -------------------------
    is_math = args.task in {"GSM8K", "GSM8K-Rev", "MATH", "TabMWP"}
    acc = evaluate(test_samples, is_math=is_math, pred_key="pred")
    print(f"Zero‑shot‑CoT accuracy on {args.task}: {acc}")

    os.makedirs("./results/Zero-Shot-Cot", exist_ok=True)
    out_path = f"./results/Zero-Shot-Cot/{args.model}_{args.task}_zs_cot_{acc}.json"
    with open(out_path, "w") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved detailed predictions to {out_path}")


if __name__ == "__main__":
    main()
