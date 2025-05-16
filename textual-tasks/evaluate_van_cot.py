# coding=utf-8
"""Evaluate a *base* model with **direct zero‑shot answering** (no CoT).

This script mirrors `evaluate_zero_shot_cot.py` but removes the two‑stage
chain‑of‑thought procedure from Kojima et al. (2023). The model is prompted
once per example and must commit to a short final answer without any
intermediate reasoning.

Example
-------
CUDA_VISIBLE_DEVICES=0 python evaluate_direct_answer.py \
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

ANSWER_TRIGGER_MAP: Dict[str, str] = {
    # yes / no
    "SQA": "\nTherefore, the answer (Yes or No) is ",
    "BoolQ": "\nTherefore, the answer (Yes or No) is ",
    # multiple‑choice – 5 options (A‑E)
    "CSQA": "\nTherefore, among (A) through (E), the answer is ",
    "ARC": "\nTherefore, among (A) through (E), the answer is ",
    "ANLI": "\nTherefore, among (A) through (E), the answer is ",
    "OBQA": "\nTherefore, among (A) through (E), the answer is ",
    "ESNLI": "\nTherefore, among (A) through (E), the answer is ",
    # multiple‑choice – 6 options (A‑F)
    "Date": "\nTherefore, among (A) through (F), the answer is ",
    # numeric
    "GSM8K": "\nTherefore, the answer (arabic numerals) is ",
    "GSM8K-Rev": "\nTherefore, the answer (arabic numerals) is ",
    # boxed maths / LaTeX
    "MATH": "\nTherefore, the answer (boxed) is ",
    "TabMWP": "\nTherefore, the answer (boxed) is ",
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
    parser.add_argument("--max_tokens", default=32, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--adapter_path", default="", type=str,
                        help="Optional LoRA adapter to load on top of base model")
    parser.add_argument("--n", default=100, type=int)
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
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.task == "BoolQ":
        adapter_path = f"./checkpoints/{args.model}_SQA_{args.n}" + "_vanilla_cot"
    elif args.task == "OBQA":
        adapter_path = f"./checkpoints/{args.model}_ARC_{args.n}"+ "_vanilla_cot"
    elif args.task == "ESNLI":
        adapter_path = f"./checkpoints/{args.model}_ANLI_{args.n}"+ "_vanilla_cot"
    elif args.task == "GSM8K-Rev":
        adapter_path = f"./checkpoints/{args.model}_GSM8K_{args.n}"+ "_vanilla_cot"
    else:
        adapter_path = f"./checkpoints/{args.model}_{args.task}_{args.n}" + "_vanilla_cot"

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

    llm = LLM(model=base_model,
                enable_lora=True,
                max_lora_rank=32,
                download_dir=args.model_dir,
                tensor_parallel_size=1)

    if adapter_path:
        lora_request = LoRARequest("finetined_adapter", 1, adapter_path)
    else:
        lora_request = None

    # Sampler cfg – deterministic greedy decoding by default.
    gen_params = SamplingParams(n=1,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                stop_token_ids=[tokenizer.eos_token_id])

    # --------------------------- main loop ------------------------------
    answer_trigger = ANSWER_TRIGGER_MAP.get(args.task)
    if answer_trigger is None:
        raise KeyError(f"No answer‑extraction trigger registered for task {args.task}")

    extract_answer = get_answer_extractor(args.task)

    for ex in test_samples:
        q = ex["question"]

        # ---------- single stage: direct answer ----------
        prompt = template.format(question=q, answer=f"{answer_trigger}")
        answer_text = llm.generate([prompt], gen_params, lora_request=lora_request)[0].outputs[0].text
        ex["answer_raw"] = answer_text
        ex["pred"] = extract_answer(answer_text)

    # --------------------------- metrics & save -------------------------
    is_math = args.task in {"GSM8K", "GSM8K-Rev", "MATH", "TabMWP"}
    acc = evaluate(test_samples, is_math=is_math, pred_key="pred")
    print(f"Accuracy on {args.task}: {acc}")

    os.makedirs("./results/van-cot", exist_ok=True)
    out_path = f"./results/van-cot/{args.model}_{args.task}_direct_{acc}.json"
    with open(out_path, "w") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    print(f"Saved detailed predictions to {out_path}")


if __name__ == "__main__":
    main()
