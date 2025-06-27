import os
import time
import json
import random
import argparse
from datetime import datetime
from evaluate import evaluate, evaluate_on_policy
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
import wandb

SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}. "

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Evaluation Script for 72B Model")

    parser.add_argument("--model_name_or_path", type=str, default="/oss/public/user/liuts/model/Qwen3-30B-A3B")
    parser.add_argument("--data_dir", type=str, default="/cpfs04/user/liutianshuo/math/deepscaler/deepscaler/data/train")
    parser.add_argument("--data_name", type=str, default="deepscaler")
    parser.add_argument("--output_dir", type=str, default="qwen3_30b_a3b_deepscaler--0602/ppo/2")

    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")

    parser.add_argument("--split", type=str, default=None, choices=[None, "train", "dev", "test"])
    parser.add_argument("--num_test_sample", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    parser.add_argument("--prompt_type", type=str, default="qwen25-math-cot")
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens_per_call", type=int, default=16000)
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    return parser.parse_args()


def prepare_data(data_name, args):
    examples = load_data(args.data_name, args.split, args.data_dir)

    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)
    examples = examples[args.start: len(examples) if args.end == -1 else args.end]

    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    return examples, out_file


def save_res_jsonl(samples, save_path):
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)

def main():
    wandb.login(key="2f39b29deef86a6909949ed808c7406a144a0d2b")
    args = parse_args()
    set_seed(args.seed)
    
    wandb.init(
        project=f"qwen3-deepscaler-s{args.start}-e{args.end}", 
        name=f"eval_{args.prompt_type}_seed{args.seed}",
        config=vars(args)
    )


    examples, out_file = prepare_data(args.data_name, args)

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    chat_prompts = []
    meta_infos = []

    for example in examples:
        question = ""
        for key in ["question", "problem", "Question", "input", "Problem"]:
            if key in example:
                question = example[key]
                break
        if question == "":
            continue

        format_input = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        chat_prompt = tokenizer.apply_chat_template(
            format_input, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        chat_prompts.append(chat_prompt)


        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        meta_infos.append((question, gt_cot, gt_ans))

    batch_size = 100 
    for i in range(0, len(chat_prompts), batch_size):
        batch_prompts = chat_prompts[i:i + batch_size]

        outputs = llm.generate(
            batch_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=20,
                max_tokens=args.max_tokens_per_call,
                n=args.n
            )
        )
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        # outputs_text = [o.outputs[0].text for o in outputs]
        outputs_text = [[o.outputs[j].text for j in range(args.n)] for o in outputs]

        # results = []
        # for output_text, (question, gt_cot, gt_ans) in zip(outputs_text, meta_infos[i:i + batch_size]):
        #     ans = extract_answer(output_text, args.data_name)
        #     results.append({
        #         "input": question,
        #         "gt_cot": gt_cot,
        #         "gt_ans": gt_ans,
        #         "response": output_text,
        #         "pred": ans
        #     })
        results = []
        for i, (output_texts, (question, gt_cot, gt_ans)) in enumerate(zip(outputs_text, meta_infos[i:i + batch_size])):
            anss = []
            for output_text in output_texts:
                ans = extract_answer(output_text, args.data_name)
                anss.append(ans)

            results.append({
                "input": question,
                "gt_cot": gt_cot,
                "gt_ans": gt_ans,
                "response": output_texts,
                "pred": anss
            })

        save_res_jsonl(results, out_file)
        print(f"Saved {i + batch_size} results to {out_file}")
        

        wandb_table = wandb.Table(columns=["input", "gt_cot", "gt_ans", "response", "pred"])
        for r in results:
            wandb_table.add_data(r["input"], r["gt_cot"], r["gt_ans"], r["response"], r["pred"])
        wandb.log({
            f"batch_{i // batch_size}_samples": wandb_table,
            f"batch_{i // batch_size}_acc": sum([r["pred"] == r["gt_ans"] for r in results]) / len(results),
            "global_step": i + batch_size
        })


    wandb.finish()
    print("Evaluation finished. Results saved to:", out_file.replace(".jsonl", "_metrics.json"))


if __name__ == "__main__":
    main()

