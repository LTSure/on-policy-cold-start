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


SYSTEM_PROMPT = "Let's think step by step and output the final answer within \\boxed{}. "

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Evaluation Script for 72B Model")

    parser.add_argument("--model_name_or_path", type=str, default="/cpfs04/user/liutianshuo/model/Qwen3-30B-A3B")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="aime24")
    parser.add_argument("--output_dir", type=str, default="qwen3_30b_base/ppo/2")

    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")

    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--num_test_sample", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    parser.add_argument("--prompt_type", type=str, default="qwen25-math-cot")
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens_per_call", type=int, default=4096)
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
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
    examples = load_data(data_name, args.split, args.data_dir)

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
def main():
    args = parse_args()
    set_seed(args.seed)
    examples, out_file = prepare_data(args.data_name, args)

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 构建输入 prompts
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
            format_input, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        chat_prompts.append(chat_prompt)

        # 保存元数据用于后续整理结果
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        meta_infos.append((question, gt_cot, gt_ans))

    # 批量生成
    outputs = llm.generate(
        chat_prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=20, 
            max_tokens=args.max_tokens_per_call,
        )
    )

    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    outputs_text = [o.outputs[0].text for o in outputs]

    results = []
    for output_text, (question, gt_cot, gt_ans) in zip(outputs_text, meta_infos):

        ans = extract_answer(output_text, args.data_name)
        results.append({
            "input": question,
            "gt_cot": gt_cot,
            "gt_ans": gt_ans,
            "response": output_text,
            "pred": ans
        })


    save_jsonl(results, out_file)

    results, result_json = evaluate_on_policy(
        samples=results,
        data_name=args.data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )


    with open(out_file.replace(".jsonl", "_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)

    print("Evaluation finished. Results saved to:", out_file.replace(".jsonl", "_metrics.json"))


if __name__ == "__main__":
    main()