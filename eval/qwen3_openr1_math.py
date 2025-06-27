import os
import re
import time
import json
import random
from tqdm import tqdm
import numpy as np
import argparse
from scipy.special import comb
from datetime import datetime
from evaluate import evaluate, evaluate_on_policy
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from datasets import load_dataset
import wandb

SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}. "


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Evaluation Script for 72B Model")

    parser.add_argument("--model_name_or_path", type=str, default="/oss/public/user/liuts/model/Qwen3-30B-A3B")
    parser.add_argument("--data_dir", type=str, default="/oss/public/user/liuts/datasets/math/OpenR1-Math-220k")
    parser.add_argument("--data_name", type=str, default="openr1-math-220k")
    parser.add_argument("--output_dir", type=str, default="/oss/public/user/liuts/datasets/math/OpenR1-Math-Filtered")
    parser.add_argument("--dataset_size", type=int, default=10000)

    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")

    parser.add_argument("--num_test_sample", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    parser.add_argument("--problem_type", type=str, default="Algebra")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens_per_call", type=int, default=16000)
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=4)
    return parser.parse_args()


def prepare_data(args):
    dataset = load_dataset(args.data_dir, 'default')

    dataset = dataset['train']
    filtered_dataset = dataset.filter(lambda example: example['problem_type'] == args.problem_type).select(range(args.dataset_size))
    out_file = f"{args.output_dir}/{args.problem_type}.jsonl"
    if os.path.isfile(out_file):
        os.remove(out_file)

    return filtered_dataset, out_file


def save_res_jsonl(samples, save_path):
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


# units mainly from MathQA
unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    if not skip_unit:
        # Remove unit: texts
        for _ in range(2):
            for unit_text in unit_texts:
                # use regex, the prefix should be either the start of the string or a non-alphanumeric character
                # the suffix should be either the end of the string or a non-alphanumeric character
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def extract_answer(pred_str, data_name='openr1-math-220k'):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    pred = a
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def main():
    wandb.login(key="2f39b29deef86a6909949ed808c7406a144a0d2b")
    args = parse_args()
    set_seed(args.seed)
    
    wandb.init(
        project=f"qwen3-openr1-math", 
        name=f"eval-0601",
        config=vars(args)
    )


    examples, out_file = prepare_data(args)
    print('Dataset size: ', len(examples))

    chat_prompts = []
    meta_infos = []

    batch_size = 100

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    n = max(args.n, args.k)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=20,
        max_tokens=args.max_tokens_per_call,
        n=n
    )
    for i in tqdm(range(0, len(examples), batch_size)):
        start = i
        end = min(i + batch_size, len(examples))
        problems = examples['problem'][start:end]
        solutions = examples['solution'][start:end]
        answers = examples['answer'][start:end]
        messages = [
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problems}]
            for problems in problems
        ]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        outputs = llm.generate(prompts, sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs_texts = [[o.text for o in output.outputs] for output in outputs]

        results = []
        for output_texts, prob, solu, ans in zip(outputs_texts, problems, solutions, answers):
            predictions = []
            c = 0
            for output_text in output_texts:
                pred_ans = extract_answer(output_text)
                predictions.append(pred_ans)
                if ans == pred_ans:
                    c += 1

            results.append({
                "problem": prob,
                "solution": solu,
                "answer": ans,
                "responses": output_texts,
                "predictions": predictions, 
                "accuracy":  c / n,
                "pass@k": compute_pass_at_k(n, c, args.k),
            })

        save_res_jsonl(results, out_file)
        print(f"Saved {end} results to {out_file}")
        
        wandb_table = wandb.Table(columns=["problem", "solution", "answer", "response", "predictions"])
        for r in results:
            wandb_table.add_data(r["problem"], r["solution"], r["answer"], r["response"], r["predictions"])
        wandb.log({
            f"batch_{i // batch_size}_samples": wandb_table,
            f"batch_{i // batch_size}_acc": sum([r["accuracy"] for r in results]) / len(results),
            "global_step": end
        })

    wandb.finish()
    print("Evaluation finished. Results saved to:", out_file)


if __name__ == "__main__":
    main()

