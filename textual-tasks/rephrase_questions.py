
import argparse, json, re, time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from utils import batch_call_gemini_api   

_EX_REPHRASE = """
Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Rephrase the above question: What amount of money does Olivia have left after buying five bagels at $3 each if she started with $23?
""".strip()

def build_rephrase_prompt(q: str) -> str:
    return f"{_EX_REPHRASE}\n\nQuestion: {q}\nRephrase the above question:"


_ANS_HEADER = """
Q: Liam had 4 boxes of apples with 6 apples each. He ate 5 apples. How many apples are left?
A: Let's think step by step. 4 * 6 = 24 apples. 24 - 5 = 19. The answer is: 19

Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: Let's think step by step. Half of 16 is 8. Half of 8 is 4. The answer is: 4
""".strip()

def build_answer_prompt(q: str) -> str:
    return f"{_ANS_HEADER}\n\nQ: {q}\nA: Let's think step by step."

_RE_PREFIX = re.compile(
    r"^\s*(?:Rephrase[sd]?|Rephrased)(?:\s+the)?\s+(?:above\s+)?question[:\-\s]*",
    re.I,
)
def clean_rephrase(txt: str) -> str:
    txt = _RE_PREFIX.sub("", txt).lstrip("–—:- ").strip()
    if len(txt) > 1 and txt[0] in "\"“‘" and txt[-1] in "\"”’":
        txt = txt[1:-1]
    return txt.strip()

_LETTER = re.compile(r"\(([A-F])\)")
_NUM    = re.compile(r"-?\d+(?:\.\d+)?")
def extract_pred(text: str) -> str:
    """A‑F if multiple‑choice, else first number."""
    m = _LETTER.search(text)
    if m: return m.group(1)
    m = _NUM.search(text)
    return m.group(0) if m else ""

def add_answer_aug(items: List[Dict], sel: List[int], k: int, model: str):
    prompts, owners = [], []
    for i in sel:
        for _ in range(k):
            prompts.append(build_answer_prompt(items[i]["question"]))
            owners.append(i)

    replies = batch_call_gemini_api(prompts, model_name=model)

    for reply, idx in zip(replies, owners):
        pred = extract_pred(reply)
        gold = str(items[idx].get("gold_answer", "")).strip()
        if gold and pred and pred.upper() == gold.upper():
            items[idx].setdefault("answer_aug", []).append(
                {"reasoning": reply.strip(),
                 "answer_raw": pred,
                 "pred": pred}
            )

# ----------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--model", default="flash", choices=["flash", "pro"])
    ap.add_argument("--batch", type=int, default=16, help="Gemini batch size")
    ap.add_argument("--k", type=int, default=5,  help="answer paths per Q")
    args = ap.parse_args()

    data: List[Dict] = json.loads(args.json_path.read_text())
    out_path = args.json_path.with_name(args.json_path.stem + "_rephrased.json")

    # ---------- pass 1: Re‑phrasing ------------------------------------
    todo = [i for i, d in enumerate(data) if "rephrased_question" not in d]
    for start in tqdm(range(0, len(todo), args.batch), desc="Rephrase", unit="batch"):
        idx = todo[start:start+args.batch]
        prompts = [build_rephrase_prompt(data[i]["question"]) for i in idx]
        replies = batch_call_gemini_api(prompts, model_name=args.model)
        for rep, i in zip(replies, idx):
            data[i]["rephrased_question"] = clean_rephrase(rep)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ---------- pass 2: Answer augmentation ----------------------------
    todo = [i for i, d in enumerate(data) if "gold_answer" in d]   # needs label
    for start in tqdm(range(0, len(todo), args.batch), desc="AnsAug", unit="batch"):
        idx = todo[start:start+args.batch]
        add_answer_aug(data, idx, args.k, args.model)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    print("✅  Done →", out_path)

if __name__ == "__main__":
    main()
