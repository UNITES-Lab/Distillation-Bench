# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate the model on test set (Modified for robust MATH/TabMWP eval)."""

import argparse
import json
import os
import re # Added
import math # Added
import signal # Added
from typing import List, Dict, Any, Optional, Union # Added

import transformers
# Keep original utils import for non-math tasks' answer extraction
from utils import get_alphabet_choice
from utils import get_yes_no
# Original math_utils imports might still be needed if parse_number/parse_math_boxed
# are used by other logic, but the core math comparison is replaced below.
# We avoid importing is_math_correct to prevent its use for MATH/TabMWP.
from math_utils import parse_math_boxed
from math_utils import parse_number

from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# --- Start: Inserted Simplified Evaluation Logic ---

# Timeout handler function for complex comparisons
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Comparison timed out")

def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    """
    Extracts the content from the last \boxed{...} or \fbox{...} in a string.
    """
    if text is None:
        return None
    idx_boxed = text.rfind("\\boxed")
    idx_fbox = text.rfind("\\fbox")
    idx = max(idx_boxed, idx_fbox)
    if idx == -1:
        return None
    brace_open_idx = text.find("{", idx)
    if brace_open_idx == -1:
        return None
    level = 0
    brace_close_idx = -1
    for i in range(brace_open_idx, len(text)):
        if text[i] == '{':
            level += 1
        elif text[i] == '}':
            level -= 1
            if level == 0:
                brace_close_idx = i
                break
    if brace_close_idx == -1:
        return None
    return text[brace_open_idx + 1 : brace_close_idx].strip()

def normalize_answer(ans: Optional[str]) -> Optional[str]:
    """
    Performs basic normalization on an answer string for comparison.
    """
    if ans is None:
        return None
    ans = re.sub(r"\\text\{.*?\}", "", ans)
    ans = re.sub(r"\\(left|right)[.()]", "", ans)
    ans = ans.replace("\\$", "").replace("$", "")
    ans = ans.replace("\\%", "").replace("%", "")
    ans = ans.replace("\\", "")
    ans = ans.replace(",", "")
    ans = ans.strip()
    ans = re.sub(r"frac\{([\d\.]+)\}\{([\d\.]+)\}", r"\1/\2", ans)
    return ans

def compare_answers(pred: Optional[str], gold: Optional[str]) -> bool:
    """
    Compares the predicted answer with the gold answer using simplified logic.
    """
    if pred is None or gold is None:
        return False
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(str(gold)) # Ensure gold is string
    if norm_pred is None or norm_gold is None:
        return False
    if norm_pred == norm_gold:
        return True
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(2) # 2 second timeout
    try:
        try:
            if '/' in norm_pred:
                num, den = norm_pred.split('/', 1)
                pred_float = float(num) / float(den)
            else:
                pred_float = float(norm_pred)
        except (ValueError, TypeError, ZeroDivisionError):
            pred_float = None
        try:
            if '/' in norm_gold:
                 num, den = norm_gold.split('/', 1)
                 gold_float = float(num) / float(den)
            else:
                gold_float = float(norm_gold)
        except (ValueError, TypeError, ZeroDivisionError):
            gold_float = None
        if pred_float is not None and gold_float is not None:
            is_close = math.isclose(pred_float, gold_float, rel_tol=1e-4, abs_tol=1e-6)
            signal.alarm(0)
            return is_close
        else:
            signal.alarm(0)
            return False
    except TimeoutError:
        signal.alarm(0)
        return False
    except Exception as e:
        signal.alarm(0)
        return False
    finally:
        signal.alarm(0) # Ensure alarm is always disabled

# --- End: Inserted Simplified Evaluation Logic ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Keep original arguments
    parser.add_argument("--n", default=0, type=int, help="Legacy argument (not used in this modified eval logic, kept for compatibility)")
    parser.add_argument("--task", required=True, type=str, help="Dataset task to evaluate (e.g., SQA, GSM8K, MATH, TabMWP).")
    parser.add_argument("--model", required=True, type=str, help="Hugging Face model name or path (e.g., 'mistralai/Mistral-7B-Instruct-v0.3').")
    parser.add_argument("--model_dir", default=None, type=str, help="Directory to cache downloaded models.")
    # Keep adapter path logic, but use zero_shot flag to control usage
    parser.add_argument("--adapter_path_base", default="./checkpoints/", type=str, help="Base directory containing adapter checkpoints named like {model}_{task}_{n}")
    parser.add_argument("--specific_adapter_path", default=None, type=str, help="Explicit path to a specific adapter (overrides constructed path).")
    parser.add_argument("--lora_rank", default=32, type=int, help="LoRA rank if using adapter.")
    parser.add_argument("--zero_shot", action='store_true', help="Enable Zero-Shot evaluation (ignore adapter_path).")
    parser.add_argument("--data_dir", default="./data/test_data", type=str, help="Directory containing test data JSON files.")
    parser.add_argument("--output_dir", default="./results", type=str, help="Directory to save detailed results.")
    parser.add_argument("--max_tokens", default=1024, type=int, help="Max generation tokens.")
    parser.add_argument("--tensor_parallel_size", default=1, type=int, help="Tensor parallel size for vLLM.")
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float, help="GPU memory utilization for vLLM.")

    args = parser.parse_args()

    # --- Determine Model Name and Adapter Path ---
    if args.model == "mistral-7b":
        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model == "gemma-2b":
        base_model = "google/gemma-2b-it"
    elif args.model == "gemma-7b":
        base_model = "google/gemma-7b-it"
    elif args.model == "llama-8b": # Assuming this maps to 3.1
         base_model = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        # Allow using full path directly if not a predefined shortcut
        if os.path.isdir(args.model):
             base_model = args.model
        else:
             # Assume it's a huggingface identifier
             base_model = args.model
             print(f"Warning: Model shortcut '{args.model}' not recognized. Using it directly as Hugging Face identifier.")

    adapter_path = None
    run_type_str = "base" # Default if no adapter found/used
    if not args.zero_shot:
        if args.specific_adapter_path:
            adapter_path = args.specific_adapter_path
            if not os.path.exists(adapter_path):
                print(f"Warning: Specified adapter path '{adapter_path}' not found. Running base model.")
                adapter_path = None
            else:
                 run_type_str = f"lora{args.lora_rank}"
        else:
            # Construct path based on convention (handle potential OOD tasks mapping)
            if args.task == "BoolQ":
                adapter_task_name = "SQA"
            elif args.task == "OBQA":
                adapter_task_name = "ARC"
            elif args.task == "ESNLI":
                adapter_task_name = "ANLI"
            elif args.task == "GSM8K-Rev":
                adapter_task_name = "GSM8K"
            else:
                adapter_task_name = args.task

            # Use args.model which is the shortcut name (e.g., 'mistral-7b')
            constructed_path = os.path.join(args.adapter_path_base, f"{args.model}_{adapter_task_name}_{args.n}")
            if os.path.exists(constructed_path):
                adapter_path = constructed_path
                run_type_str = f"lora{args.lora_rank}"
            else:
                 print(f"Warning: Constructed adapter path '{constructed_path}' not found. Running base model.")
    else:
         run_type_str = "zs" # Zero-shot run type

    # --- Print Run Configuration ---
    print(f"--- Starting Evaluation ---")
    print(f"Task: {args.task}")
    print(f"Base Model: {base_model}")
    if args.zero_shot:
        print(f"Mode: Zero-Shot")
        if args.specific_adapter_path or args.adapter_path_base != "./checkpoints/":
             print(f"Adapter Path (Ignored): {args.specific_adapter_path or 'Constructed based on convention'}")
    elif adapter_path:
        print(f"Mode: LoRA (Rank: {args.lora_rank})")
        print(f"Adapter: {adapter_path}")
    else:
        print(f"Mode: Base Model (No adapter found or specified)")
    print(f"---------------------------")


    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit() # Use exit instead of return in main block

    # --- Load Data ---
    data_file = os.path.join(args.data_dir, f"{args.task}_test.json")
    print(f"Loading data from {data_file}...")
    try:
        with open(data_file, "r") as f:
            test_samples = json.load(f)
        print(f"Loaded {len(test_samples)} samples.")
        if not test_samples or not isinstance(test_samples, list) or \
           "question" not in test_samples[0] or "gold_answer" not in test_samples[0]:
             raise ValueError("Test data is missing required keys ('question', 'gold_answer') or has incorrect format.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        exit()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing data file: {e}")
        exit()


    # --- Prepare Prompts ---
    print("Preparing prompts...")
    # Match original prompt selection logic
    if "mistral" in base_model.lower() or "llama" in base_model.lower() or "mixtral" in base_model.lower() :
        # Assuming Mistral/Llama/Mixtral use similar instruct formats
        # Modify specific template based on model type if needed
        if "llama-3.1" in base_model.lower():
             prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer the following question:\n### Question:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n### Answer:\n"
        else: # Default Mistral/Llama 2 style
             prompt_template = "<s>[INST] Answer the following question:\n### Question:\n{question} [/INST] ### Answer:\n"
    elif "gemma" in base_model.lower():
        prompt_template = "<bos><start_of_turn>user\nAnswer the following question:\n### Question: {question}<end_of_turn>\n<start_of_turn>model\n### Answer: " # No answer placeholder needed? Check vLLM handling
    else:
        print(f"Warning: Using generic prompt template for model {base_model}. Check if specific format required.")
        prompt_template = "Answer the following question:\n### Question:\n{question}\n\n### Answer:\n"

    # Format prompts (original code had {answer} placeholder, removed here as model generates answer)
    prompts = [prompt_template.format(question=i["question"]) for i in test_samples]
    print(f"Prepared {len(prompts)} prompts.")


    # --- Load Model ---
    print("Loading model with vLLM...")
    try:
        llm = LLM(
            model=base_model,
            tokenizer=base_model,
            download_dir=args.model_dir,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=adapter_path is not None, # Enable only if adapter path is valid
            max_lora_rank=args.lora_rank,
            # max_model_len=4096 # Set if necessary
            )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model with vLLM: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Setup Generation ---
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
        # Add model-specific stop tokens if needed
        stop=["<|eot_id|>"] if "llama-3.1" in base_model.lower() else []
    )

    lora_request = None
    if adapter_path: # Only create if adapter_path is valid
        lora_request = LoRARequest("eval_adapter", 1, adapter_path)


    # --- Generate Outputs ---
    print(f"Generating outputs for {len(prompts)} prompts...")
    try:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        print("Generation complete.")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Determine Answer Extraction and Evaluation Method ---
    is_math_task = False
    use_simplified_math_eval = False
    answer_extraction_fn = None

    if args.task in ["SQA", "BoolQ"]:
        answer_extraction_fn = get_yes_no
    elif args.task in ["ANLI", "ARC", "Date", "CSQA", "OBQA", "ESNLI"]:
        answer_extraction_fn = get_alphabet_choice
    elif args.task in ["GSM8K", "GSM8K-Rev"]:
        # Still use original extraction for GSM8K, but comparison will be simple equality
        # Or potentially switch to simplified numerical comparison if needed later
        answer_extraction_fn = parse_number
        is_math_task = True
        # Decide if GSM8K should also use simplified eval (currently no, uses simple ==)
        # use_simplified_math_eval = True # Uncomment if needed
    elif args.task in ["TabMWP", "MATH"]:
        # Use NEW simplified extraction and comparison for these tasks
        answer_extraction_fn = extract_last_boxed # Use the new function
        is_math_task = True
        use_simplified_math_eval = True # Flag to use compare_answers
    else:
        # Fallback or raise error for unsupported tasks
        print(f"Error: Unsupported task '{args.task}' for evaluation logic.")
        exit()


    # --- Evaluate ---
    print("Evaluating results...")
    num_correct = 0
    results_to_save = [] # Use a new list to avoid modifying test_samples directly if not needed

    if len(outputs) != len(test_samples):
        print(f"Error: Number of outputs ({len(outputs)}) does not match number of samples ({len(test_samples)}).")
        exit()

    for i, output in enumerate(outputs):
        sample = test_samples[i]
        generated_text = output.outputs[0].text
        gold_answer = sample["gold_answer"]
        pred_answer_extracted = None
        is_correct = False

        # Extract predicted answer
        if answer_extraction_fn:
            try:
                 # The extraction function might return None or "N/A" etc.
                 pred_answer_extracted = answer_extraction_fn(generated_text)
            except Exception as e:
                 print(f"Warning: Error during answer extraction for sample {i}: {e}")
                 pred_answer_extracted = None # Treat extraction error as incorrect prediction

        # Compare answers
        if use_simplified_math_eval:
            # Use the robust comparison for MATH/TabMWP
             is_correct = compare_answers(pred_answer_extracted, str(gold_answer))
        else:
            # Use simple direct comparison for other tasks (after extraction)
            # Handle potential type differences (e.g., float vs string for GSM8K)
            if is_math_task and args.task in ["GSM8K", "GSM8K-Rev"]:
                 # Simple numerical check for GSM8K extracted number
                 try:
                      # parse_number returns float or "N/A"
                      if pred_answer_extracted != "N/A" and pred_answer_extracted is not None:
                           is_correct = math.isclose(float(pred_answer_extracted), float(gold_answer), rel_tol=1e-4, abs_tol=1e-6)
                      else:
                           is_correct = False
                 except (ValueError, TypeError):
                      is_correct = False # Cannot compare if conversion fails
            else:
                 # Direct equality for classification/QA tasks
                 # Ensure case-insensitivity if needed (e.g., for yes/no) by lowercasing
                 pred_str = str(pred_answer_extracted).lower() if pred_answer_extracted is not None else ""
                 gold_str = str(gold_answer).lower() if gold_answer is not None else ""
                 is_correct = pred_str == gold_str


        if is_correct:
            num_correct += 1

        # Store results for saving
        # Note: Storing raw generated text and extracted answer for debugging
        results_to_save.append({
             **sample, # Include original sample data
             "generated_text": generated_text,
             "pred": pred_answer_extracted, # Store the extracted answer
             "is_correct": is_correct
        })


    # --- Report Accuracy ---
    total_samples = len(test_samples)
    accuracy = (num_correct / total_samples) * 100 if total_samples > 0 else 0
    accuracy_str = f"{accuracy:.4f}" # Format for filename

    print("\n--- Evaluation Summary ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {num_correct}")
    print(f"Accuracy: {accuracy:.4f}%")
    print("--------------------------")

    # --- Save Detailed Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    # Use the base model name for consistency in naming
    model_name_short = args.model # Use the shortcut name provided by user
    # Use run_type_str determined earlier
    results_filename = os.path.join(args.output_dir, f"{model_name_short}_{args.task}_{run_type_str}_{accuracy_str}.json")

    print(f"Saving detailed results to {results_filename}...")
    try:
        with open(results_filename, "w") as f:
            json.dump(results_to_save, f, indent=4)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")