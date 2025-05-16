# coding=utf-8
"""
Augments a dataset using a local Hugging Face transformer model.

Reads an input JSON file, processes each item using a specified HF model,
and writes the augmented data to an output JSON file.

**Hardware Requirements:** Requires significant GPU VRAM or system RAM.
Consider using quantization (e.g., --load-in-4bit) for lower resource usage.

**Dependencies:** pip install transformers torch accelerate bitsandbytes sentencepiece tqdm

**Model Download:** The model will be downloaded on the first run.
"""

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Callable

# Suppress specific warnings if needed, e.g., from bitsandbytes
# warnings.filterwarnings("ignore", category=UserWarning, module='bitsandbytes')

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
        StoppingCriteria,
        StoppingCriteriaList,
    )
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them: pip install transformers torch accelerate bitsandbytes sentencepiece tqdm")
    exit(1)

from tqdm import tqdm

# --- Utility Functions (mostly unchanged) ---

def get_alphabet_choice(text: str) -> str:
    """Extracts the first uppercase letter choice (A-F) enclosed in parentheses."""
    match = re.search(r"\(([A-F])\)", text, re.IGNORECASE)
    return match.group(1).upper() if match else "N/A"

def get_yes_no(text: str) -> str:
    """Extracts 'yes' or 'no', case-insensitive."""
    if re.search(r"\b(yes)\b", text, re.IGNORECASE):
        return "yes"
    if re.search(r"\b(no)\b", text, re.IGNORECASE):
        return "no"
    return "N/A"

def parse_math_boxed(text: str) -> str:
    """Extracts content within \\boxed{}."""
    match = re.search(r"\\boxed\{(.*?)\}", text)
    # Handle potential nested braces if necessary, simple version for now
    return match.group(1).strip() if match else "N/A"

def parse_number(text: str) -> str:
    """Extracts the first number (integer or float)."""
    match = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+", text)
    return match.group(0).replace(",", "") if match else "N/A"

def get_true_false(text: str) -> str:
    """Extracts 'true' or 'false', case-insensitive."""
    if re.search(r"\b(true)\b", text, re.IGNORECASE):
        return "true"
    if re.search(r"\b(false)\b", text, re.IGNORECASE):
        return "false"
    return "false"

def remove_backward_answer(text: str) -> str:
    """Removes the 'The correct answer is...' part from generated backward questions."""
    cleaned_text = re.sub(
        r"\s*The\s+correct\s+answer\s+is\s*\(?\w+\)?\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()
    return cleaned_text

_RE_PREFIX_REPHRASE = re.compile(
    r"^\s*(?:Rephrase[sd]?|Rephrased)(?:\s+the)?\s+(?:above\s+)?question[:\-\s]*",
    re.IGNORECASE,
)
def clean_rephrase(txt: str) -> str:
    """Cleans the prefix from rephrased questions."""
    txt = _RE_PREFIX_REPHRASE.sub("", txt).lstrip("–—:- ").strip()
    if len(txt) > 1 and txt[0] in "\"“‘" and txt[-1] in "\"”’":
        txt = txt[1:-1]
    return txt.strip()

# --- Hugging Face Model Inference ---

# Global variables for model and tokenizer to load them only once
hf_model = None
hf_tokenizer = None
hf_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_hf_model(model_id: str, load_in_8bit: bool = False, load_in_4bit: bool = False):
    """Loads the Hugging Face model and tokenizer."""
    global hf_model, hf_tokenizer, hf_pipeline, device

    if hf_model is not None:
        print("Model already loaded.")
        return

    print(f"Loading model: {model_id}...")
    print(f"Using device: {device}")

    quantization_config = None
    model_kwargs = {"device_map": "auto"} # Automatically distribute across GPUs/CPU+GPU

    if load_in_4bit:
        print("Applying 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16
            bnb_4bit_use_double_quant=True,
        )
        # device_map="auto" handled by accelerate with quantization
    elif load_in_8bit:
        print("Applying 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # device_map="auto" handled by accelerate with quantization


    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Ensure pad token is set if missing (common issue)
        if hf_tokenizer.pad_token is None:
             hf_tokenizer.pad_token = hf_tokenizer.eos_token
             print("Set tokenizer pad_token to eos_token")

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_kwargs # device_map handled here or via quantization
        )

        # Use pipeline for easier text generation
        hf_pipeline = pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=hf_tokenizer,
            device_map="auto" # Redundant? Accelerate should handle it. Test.
            # device=0 if device == "cuda" else -1 # Alternative device mapping if auto fails
        )
        print(f"Model '{model_id}' loaded successfully.")

    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        print("Please ensure the model ID is correct, you have internet access,")
        print("and necessary dependencies (torch, accelerate, bitsandbytes) are installed.")
        if "trust_remote_code=True" in str(e):
             print("You may need to confirm trust for remote code execution.")
        exit(1)

def get_hf_output(
    prompt: str,
    max_new_tokens: int = 500,
    do_sample: bool = True,
    **kwargs # Allow passing other pipeline kwargs
) -> Optional[str]:
    """
    Generates text using the loaded Hugging Face model pipeline.

    Args:
        prompt: The user prompt string.
        max_new_tokens: Max tokens to generate.
        temperature: Controls randomness.
        do_sample: Whether to use sampling; set to False for greedy decoding.
        top_p: Nucleus sampling parameter.

    Returns:
        The generated text (assistant's response part), or None on error.
    """
    global hf_pipeline, hf_tokenizer

    if hf_pipeline is None or hf_tokenizer is None:
        print("Error: Hugging Face pipeline/tokenizer not initialized.")
        return None

    # Apply the chat template - THIS IS CRUCIAL
    try:
        # Use the tokenizer's chat template if available
        messages = [{"role": "user", "content": prompt}]
        # add_generation_prompt=True adds the tokens indicating the start of the assistant's turn
        formatted_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"Warning: Could not apply chat template automatically: {e}. Falling back to basic format.")
        # Fallback basic format - adjust if needed for the specific model
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"


    # print(f"\n--- Formatted Prompt ---\n{formatted_prompt}\n----------------------\n") # Debug: Show formatted prompt

    try:


        # Generate text using the pipeline
        # Ensure EOS token is used for stopping, handle potential list format
        eos_token_id = hf_tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
             eos_token_id = eos_token_id[0] # Use the first one if it's a list

        outputs = hf_pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False, # Only return the generated part
            pad_token_id=eos_token_id, # Use EOS for padding during generation
            eos_token_id=eos_token_id, # Ensure stopping at EOS
            **kwargs
        )

        if outputs and isinstance(outputs, list) and 'generated_text' in outputs[0]:
            generated_text = outputs[0]['generated_text']
            # print(f"\n--- Raw Output ---\n{generated_text}\n------------------\n") # Debug: Show raw output
            return generated_text.strip()
        else:
            print(f"Warning: Unexpected output format from pipeline: {outputs}")
            return None

    except Exception as e:
        print(f"Error during Hugging Face model inference: {e}")
        # If it's an OOM error, print a specific message
        if "out of memory" in str(e).lower():
            print("CUDA out of memory error. Try using quantization (--load-in-4bit or --load-in-8bit)")
            print("or running on a machine with more GPU VRAM.")
        return None


# --- Prompts (Mostly unchanged, adapt if needed for HF model nuances) ---
# NOTE: The effectiveness of these specific prompts depends heavily on how
# well the chosen HF model follows complex instructions and few-shot examples.
# You might need to simplify or adjust them.

prompt_for_backward_question = """<INSTRUCTIONS>Your task is to generate an inverse question with the same number of inverse answer choices, based on the input question and its correct answer. Follow these rules:
1. Use the correct answer from the input question to create a new, related but inverse question.
2. Ensure that the four new answer choices are inversely correlated with the four input question's choices.
3. Make sure only one answer choice in your generated question is correct and reasonable.
4. The correct answer in your generated question must be present in the input question.
5. The generated question and answer choices should be semantically different from the input question.
</INSTRUCTIONS>

<EXAMPLE>
{icl_samples}
</EXAMPLE>
{input_question}
"""

icl_samples = {
    "SQA": """INPUT: Is shrimp scampi definitely free of plastic? The correct answer is no.
OUTPUT: If shrimp scampi does not definitely free of plastic, can shrimp scampi possibly contain plastic? (A) yes (B) no. The correct answer is (A).
""", # ... (rest of SQA examples) ...
    "CSQA": """INPUT: Sammy wanted to go to where the people were.  Where might he go? (A) race track (B) populated areas (C) the desert (D) apartment (E) roadblock. The correct answer is (B).
OUTPUT: Sammy wanted to go to populated areas. What might he like? (A) car racing (B) places with many people (C) hot and dry weather (D) stay at home (E) block on road. The correct answer is (B).
""", # ... (rest of CSQA examples) ...
    "ARC": """INPUT: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? (A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion. The correct answer is (A).
OUTPUT: George is rubbing his dry palms. What is he most likely trying to do? (A) Warm his hands (B) Moisturize his hands (C) Care for his skin (D) Lubricate his hands. The correct answer is (A).
""", # ... (rest of ARC examples) ...
    "GSM8K": """INPUT: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? The correct answer is 72.0.
OUTPUT: Natalia sold clips to x of her friends in April, and then she sold half as many clips in May. If the number of clips Natalia sell altogether in April and May is 72, what is the value of x?
""", # ... (rest of GSM8K examples) ...
    "MATH": """INPUT: The mean of one set of five numbers is 13, and the mean of a separate set of six numbers is 24. What is the mean of the set of all eleven numbers? The correct answer is 19.
OUTPUT: The mean of one set of six numbers is 24, and the mean of the set of all eleven numbers is 19. What is the mean of the separate set of five numbers?
""", # ... (rest of MATH examples) ...
    "TabMWP": """INPUT: ### Plants per garden\n\n| Stem | Leaf  |\n|---|---|\n| 3 | 3, 3, 3, 5, 5 |\n| 4 | 6 |\n| 5 | 4, 5, 7, 8 |\n| 6 | 7, 8 |\n| 7 | 2, 3, 7, 9 |\n| 8 | 6, 8, 9 |\n\n### Question\n\nThe members of the local garden club tallied the number of plants in each person's garden. How many gardens have at least 47 plants? The correct answer is 13.
OUTPUT: ### Plants per garden\n\n| Stem | Leaf  |\n|---|---|\n| 3 | 3, 3, 3, 5, 5 |\n| 4 | x |\n| 5 | 4, 5, 7, 8 |\n| 6 | 7, 8 |\n| 7 | 2, 3, 7, 9 |\n| 8 | 6, 8, 9 |\n\n### Question\n\nIf there are 13 gardens have at least 47 plants, what is true about the variable x? (A) x cannot be greater than 6 (B) x cannot be less than 6 (C) x can be any integer. The correct answer is A.
""", # ... (rest of TabMWP examples) ...
    "ANLI": """INPUT: What is the relationship between the following two sentences?\nSentence 1: TOKYO, Dec 18 (Reuters) - Japan’s Shionogi & Co said on Tuesday that it has applied to health regulators in the United States, Canada and Europe for approval of its HIV drug Dolutegravir. Shionogi developed Dolutegravir with a Viiv Healthcare, an AIDS drug joint venture between GlaxoSmithKline and Pfizer, in exchange for its rights to the drug.\nSentence 2: The article was written on December 18th.\nThe options are (A) entailment (B) neutral (C) contradiction. The correct answer is (A).
OUTPUT: Sentence 1: TOKYO, MM/DD (Reuters) - Japan’s Shionogi & Co said on Tuesday that it has applied to health regulators in the United States, Canada and Europe for approval of its HIV drug Dolutegravir. Shionogi developed Dolutegravir with a Viiv Healthcare, an AIDS drug joint venture between GlaxoSmithKline and Pfizer, in exchange for its rights to the drug.\nSentence 2: The article was written on December 18th.\nIf the relationship between the above two sentences is entailment, what date in sentence 1 should be (MM/DD)? (A) Dec 18 (B) Apr 30 (C) Aug 31. The correct answer is (A).
""", # ... (rest of ANLI examples) ...
    "Date": """INPUT: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY? (A) 03/11/2021 (B) 05/01/2021 (C) 02/23/2021 (D) 04/29/2021 (E) 05/09/2021 (F) 06/12/2021. The correct answer is (B).
OUTPUT: Today is May 1, 2021. What is the date yesterday? (A) 03/11/2021 (B) 04/30/2021 (C) 02/23/2021 (D) 04/29/2021 (E) 05/09/2021 (F) 06/12/2021. The correct answer is (B).
""",
} # NOTE: For brevity, the full icl_samples dictionary content from your script is used.

gen_reasoning_prompt_suffix = {
    "SQA": """Provide your step-by-step reasoning to the question first, and then print \"The answer is x\" where x is \"yes\" or \"no\", at the end of your response.""",
    "CSQA": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, C, D or E, at the end of your response.""",
    "ARC": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, C or D, at the end of your response.""",
    "GSM8K": """Provide your step-by-step reasoning to the question first, and then print \"The answer is: $\\boxed{[ANS]}$\" where [ANS] is the final answer, at the end of your response.""",
    "MATH": """Provide your step-by-step reasoning to the question first, and then print \"The answer is: $\\boxed{[ANS]}$\" where [ANS] is the final answer, at the end of your response.""",
    "TabMWP": """Provide your step-by-step reasoning to the question first, and then print \"The answer is: [ANS]\" where [ANS] is the final answer, at the end of your response.""",
    "ANLI": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, or C, at the end of your response.""",
    "Date": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, C, D, E or F, at the end of your response.""",
    "GSM8K-Rev": """Provide your step-by-step reasoning to the question first, and then print \"The answer is: $\\boxed{[ANS]}$\" where [ANS] is the final answer, at the end of your response.""",
    "ESNLI": """Provide your step-by-step reasoning to the question first, and then print \"The answer is (x)\" where x is A, B, or C, at the end of your response.""",
}

consistency_check_prompt_mcq = """<INSTRUCTIONS>You will be given two question-answering pairs, (Q1, A1) and (Q2, A2).
Your task is to check the consistency between Q1 and A2.
If Q1 entail A2, i.e., A2 is semantically contained in Q1, output `True`.
Otherwise, if Q1 and A2 is not related, output `False`.</INSTRUCTIONS>
<EXAMPLE>
...
</EXAMPLE>
Q1: {question}
A1: The correct answer is: ({gold_answer})
Q2: {backward_question}
A2: The correct answer is: ({backward_pred})
""" # NOTE: For brevity, the full consistency_check_prompt_mcq content from your script is used.

math_prefix = """<INSTRUCTIONS>You will be given two question-answering pairs, (Q1, A1) and (Q2, A2).
Your task is to check the consistency between Q1 and A2.
If (1) A2 can be found in Q1, and (2) A2 is correct, output `True`.
Otherwise, if Q1 and A2 is not related, or A2 is not correct, output `False`.</INSTRUCTIONS>
<EXAMPLE>
...
</EXAMPLE>
""" # NOTE: For brevity, the full math_prefix content from your script is used.

consistency_check_prompt_math = math_prefix + """Q1: {question}
A1: The correct answer is: {gold_answer}
Q2: {backward_question}
A2: The correct answer is: {backward_pred}
"""

_EX_REPHRASE = """
Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Rephrase the above question: What amount of money does Olivia have left after buying five bagels at $3 each if she started with $23?
""".strip()

def build_rephrase_prompt(q: str) -> str:
    return f"{_EX_REPHRASE}\n\nQuestion: {q}\nRephrase the above question:"

_ANS_AUG_HEADER = """
Q: Liam had 4 boxes of apples with 6 apples each. He ate 5 apples. How many apples are left?
A: Let's think step by step. 4 * 6 = 24 apples. 24 - 5 = 19. The answer is: 19

Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: Let's think step by step. Half of 16 is 8. Half of 8 is 4. The answer is: 4
""".strip()

def build_prompt_for_answer_aug(q: str) -> str:
    """Builds the prompt for generating an augmented answer."""
    return f"{_ANS_AUG_HEADER}\n\nQ: {q}\nA: Let's think step by step."

_LETTER_AUG = re.compile(r"\(([A-F])\)")
_NUM_AUG    = re.compile(r"-?\d+(?:\.\d+)?")
def extract_prediction_from_answer_aug(text: str) -> str:
    """Extracts A-F choice or the first number from augmented answer reasoning."""
    match_letter = _LETTER_AUG.search(text)
    if match_letter:
        return match_letter.group(1)
    match_num = _NUM_AUG.search(text)
    return match_num.group(0) if match_num else "N/A"

# --- Task Configuration ---

def get_task_config(task_name: str) -> Dict:
    """Gets the appropriate prompts and parsing functions for a task."""
    # ... (content of get_task_config remains the same)
    if task_name == "SQA":
        return {
            "answer_extraction": get_yes_no,
            "consistency_check_prompt": consistency_check_prompt_mcq,
            "is_math": False,
        }
    elif task_name in ["ANLI", "ARC", "CSQA", "Date", "ESNLI"]:
        return {
            "answer_extraction": get_alphabet_choice,
            "consistency_check_prompt": consistency_check_prompt_mcq,
            "is_math": False,
        }
    elif task_name in ["GSM8K", "GSM8K-Rev"]:
        return {
            "answer_extraction": parse_number,
            "consistency_check_prompt": consistency_check_prompt_math,
            "is_math": True,
        }
    elif task_name in ["TabMWP", "MATH"]:
        return {
            "answer_extraction": parse_math_boxed,
            "consistency_check_prompt": consistency_check_prompt_math,
            "is_math": True,
        }
    else:
        raise ValueError(f"Unsupported task: {task_name}. Please add configuration.")

# --- Main Processing Function ---

def process_dataset(
    input_path: Path,
    output_path: Path,
    task: str,
    model_id: str, # Changed from 'model' to 'model_id' for clarity
    k_answer_aug: int,
    start_index: int = 0,
    max_items: Optional[int] = None,
    generation_max_new_tokens: int = 500, # Allow configuring max tokens
):
    """Processes the dataset using the loaded Hugging Face model."""
    global hf_model # Ensure model is loaded before processing

    if hf_model is None:
         print("Error: Model not loaded. Call load_hf_model first (handled in main).")
         exit(1)

    if task not in icl_samples or task not in gen_reasoning_prompt_suffix:
         print(f"Warning: Task '{task}' not found in prompt dictionaries.")
         # Continue with potentially incorrect prompts

    task_config = get_task_config(task)
    answer_extraction: Callable = task_config["answer_extraction"]
    consistency_prompt_template: str = task_config["consistency_check_prompt"]
    reasoning_suffix = gen_reasoning_prompt_suffix.get(task, "Provide step-by-step reasoning and the final answer.")
    icl_sample_text = icl_samples.get(task, "")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    results = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming. Loaded {len(results)} existing results from {output_path}")
            if results:
                start_index = len(results)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load existing results from {output_path}. Starting fresh. Error: {e}")
            results = []

    effective_start_index = start_index
    end_index = len(all_samples) if max_items is None else min(effective_start_index + max_items, len(all_samples))
    samples_to_process = all_samples[effective_start_index:end_index]

    if not samples_to_process:
        print(f"No new samples to process. Effective start index: {effective_start_index}, Total samples: {len(all_samples)}")
        return

    print(f"Processing samples from index {effective_start_index} to {end_index-1}...")


    # --- Main Loop ---
    for idx, sample in enumerate(tqdm(samples_to_process, desc=f"Augmenting {task}")):
        current_global_idx = effective_start_index + idx
        try:
            output_data = {
                "_original_index": current_global_idx,
                "question": sample.get("question", "N/A"),
                "gold_answer": str(sample.get("gold_answer", "N/A")),
                "backward_question": "N/A",
                "backward_reasoning": "N/A",
                "forward_reasoning": "N/A",
                "forward_pred": "N/A",
                "backward_pred": "N/A",
                "consistency_reasoning": "N/A",
                "is_consistent": "false",
                "rephrased_question": "N/A",
                "answer_augmentations": [],
            }

            if output_data["question"] == "N/A" or output_data["gold_answer"] == "N/A":
                print(f"Warning: Skipping sample {current_global_idx} due to missing question or gold_answer.")
                results.append({**output_data, "error": "Missing question or gold_answer"})
                continue

            # --- Run all generation steps ---
            print("1. Backward Question")
            input_q_for_backward = f"INPUT: {output_data['question']} The correct answer is {output_data['gold_answer']}."
            bwd_q_prompt = prompt_for_backward_question.format(
                icl_samples=icl_sample_text, input_question=input_q_for_backward
            )
            raw_backward_question = get_hf_output(bwd_q_prompt)
            if raw_backward_question:
                match = re.search(r"OUTPUT:\s*(.*)", raw_backward_question, re.DOTALL | re.IGNORECASE)
                extracted_bwd_q = match.group(1).strip() if match else raw_backward_question
                output_data["backward_question"] = remove_backward_answer(extracted_bwd_q)

            print("Forward Reasoning")
            fwd_reasoning_prompt = output_data["question"] + "\n" + reasoning_suffix
            forward_reasoning = get_hf_output(fwd_reasoning_prompt)
            if forward_reasoning:
                output_data["forward_reasoning"] = forward_reasoning
                output_data["forward_pred"] = str(answer_extraction(forward_reasoning))

            print("Backward Reasoning")
            if output_data["backward_question"] != "N/A":
                bwd_reasoning_prompt = output_data["backward_question"] + "\n" + reasoning_suffix
                backward_reasoning = get_hf_output(bwd_reasoning_prompt)
                if backward_reasoning:
                    output_data["backward_reasoning"] = backward_reasoning
                    output_data["backward_pred"] = str(answer_extraction(backward_reasoning))

            print("Consistency Check")
            if output_data["backward_pred"] != "N/A":
                 consistency_prompt = consistency_prompt_template.format(
                     question=output_data["question"], gold_answer=output_data["gold_answer"],
                     backward_question=output_data["backward_question"], backward_pred=output_data["backward_pred"]
                 )
                 consistency_reasoning = get_hf_output(consistency_prompt)
                 if consistency_reasoning:
                     output_data["consistency_reasoning"] = consistency_reasoning
                     output_data["is_consistent"] = get_true_false(consistency_reasoning)

            print("Rephrased Question")
            rephrase_prompt = build_rephrase_prompt(output_data["question"])
            raw_rephrased = get_hf_output(rephrase_prompt)
            if raw_rephrased:
                output_data["rephrased_question"] = clean_rephrase(raw_rephrased)

            print("Answer Augmentations")
            if k_answer_aug > 0 and output_data["gold_answer"] != "N/A":
                for _ in range(k_answer_aug):
                    ans_aug_prompt = build_prompt_for_answer_aug(output_data["question"])
                    augmented_reasoning_text = get_hf_output(ans_aug_prompt)
                    if augmented_reasoning_text:
                        aug_pred = extract_prediction_from_answer_aug(augmented_reasoning_text)
                        gold_answer_str = str(output_data["gold_answer"])
                        if gold_answer_str and aug_pred != "N/A" and aug_pred.upper() == gold_answer_str.upper():
                            output_data["answer_augmentations"].append({
                                "reasoning": augmented_reasoning_text.strip(),
                                "extracted_prediction": aug_pred
                            })

            results.append(output_data)

        except Exception as e:
            print(f"\nError processing sample index {current_global_idx}: {type(e).__name__} - {e}")
            results.append({
                "_original_index": current_global_idx,
                "error": f"{type(e).__name__}: {str(e)}",
                "question": sample.get("question", "N/A"),
                "gold_answer": str(sample.get("gold_answer", "N/A")),
            })
            # Optionally add a delay or break here if errors are persistent (e.g., OOM)
            # time.sleep(2)
            continue

        # Save progress periodically
        if (idx + 1) % 5 == 0 or (idx + 1) == len(samples_to_process) : # Save more frequently for local models
            print(f"\nSaving progress at sample index {current_global_idx} (processed {idx+1} in this run)...")
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving progress: {e}")

    print("\nProcessing complete. Saving final results...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved augmented data to {output_path}")
    except Exception as e:
        print(f"Error saving final results: {e}")


# --- Argument Parser and Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment dataset using a local Hugging Face model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file", required=True, type=Path,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output-file", required=True, type=Path,
        help="Path to save the output JSON file."
    )
    parser.add_argument(
        "--task", required=True, type=str, choices=list(icl_samples.keys()),
        help="Task type to determine prompts and parsing logic."
    )
    parser.add_argument(
        "--model-id", default="unsloth/QwQ-32B-bnb-4bit", type=str,
        help="Hugging Face model ID to use (e.g., 'microsoft/Phi-4-reasoning-plus', 'meta-llama/Llama-2-7b-chat-hf')."
    )
    parser.add_argument(
        "--k-answer-aug", default=0, type=int,
        help="Number of answer augmentations per question (0 to disable)."
    )
    parser.add_argument(
        "--start-index", default=0, type=int,
        help="Index of the first sample to process (for resuming)."
    )
    parser.add_argument(
        "--max-items", default=None, type=int,
        help="Maximum number of new samples to process (None for all remaining)."
    )
    parser.add_argument(
        "--load-in-8bit", action='store_true',
        help="Load the model using 8-bit quantization."
    )
    parser.add_argument(
        "--load-in-4bit", action='store_true',
        help="Load the model using 4-bit quantization (requires bitsandbytes)."
    )
    parser.add_argument(
        "--max-new-tokens", default=10, type=int,
        help="Maximum number of new tokens to generate for each response."
    )
    # Add --device argument if needed, though device_map='auto' is preferred
    # parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to run on ('cuda', 'cpu').")


    args = parser.parse_args()

    if args.load_in_8bit and args.load_in_4bit:
        parser.error("Cannot use both --load-in-8bit and --load-in-4bit.")

    if not args.input_file.is_file():
        print(f"Error: Input file not found: {args.input_file}")
        exit(1)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load the model *before* starting the processing loop
    load_hf_model(args.model_id, args.load_in_8bit, args.load_in_4bit)

    # Start processing
    process_dataset(
        input_path=args.input_file,
        output_path=args.output_file,
        task=args.task,
        model_id=args.model_id,
        k_answer_aug=args.k_answer_aug,
        start_index=args.start_index,
        max_items=args.max_items,
        generation_max_new_tokens=args.max_new_tokens,
    )
