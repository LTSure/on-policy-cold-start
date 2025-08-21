#!/usr/bin/env python3
"""
Modified evaluation script for LiveCodeBench code generation tasks
"""
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
import sys
import os
sys.path.append('/cpfs04/user/liutianshuo/math/simpleRL-reason/Transferability-of-LLM-Reasoning/eval/evalchemy/eval/chat_benchmarks/LiveCodeBenchv5')
from livecodebench_utils import lcb_run, map_to_example, post_process_code, translate_private_test_cases


SYSTEM_PROMPT = "You are a coding assistant. Please write a complete solution to the given programming problem. Make sure your code is correct and efficient."

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Evaluation Script for LiveCodeBench")

    parser.add_argument("--model_name_or_path", type=str, default="/cpfs04/user/liutianshuo/model/Qwen3-30B-A3B")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="livecodebench")
    parser.add_argument("--output_dir", type=str, default="livecodebench_results")

    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_test_sample", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)

    parser.add_argument("--prompt_type", type=str, default="livecodebench")
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens_per_call", type=int, default=2048)
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

def extract_code_from_response(response_text):
    """Extract code from the model response"""
    # Look for code blocks
    if "```python" in response_text:
        start = response_text.find("```python") + 9
        end = response_text.find("```", start)
        if end != -1:
            return response_text[start:end].strip()
    
    # If no code block, try to extract the last class or function definition
    lines = response_text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.strip().startswith('class ') or line.strip().startswith('def '):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return response_text.strip()

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
    
    # Build input prompts
    chat_prompts = []
    meta_infos = []

    for example in examples:
        question = parse_question(example, args.data_name)
        
        # Format for code generation
        format_input = [{"role": "user", "content": question + "\n\n" + SYSTEM_PROMPT}]

        chat_prompt = tokenizer.apply_chat_template(
            format_input, tokenize=False, add_generation_prompt=True) 
        chat_prompts.append(chat_prompt)

        # Save metadata for later processing
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        meta_infos.append((question, gt_cot, gt_ans, example))

    # Batch generation
    outputs = llm.generate(
        chat_prompts,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens_per_call,
            n=args.n
        )
    )

    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    outputs_text = [o.outputs[0].text for o in outputs]

    results = []
    for output_text, (question, gt_cot, gt_ans, example) in zip(outputs_text, meta_infos):
        
        # Extract code from response
        code = extract_code_from_response(output_text)
        
        results.append({
            "input": question,
            "gt_cot": gt_cot,
            "gt_ans": gt_ans,
            "response": output_text,
            "code": code,
            "pred": code,  # For code generation, prediction is the code
            "example": example  # Keep original example for test case evaluation
        })

    save_jsonl(results, out_file)
    
    # Evaluate LiveCodeBench results using test case execution
    print(f"Generated {len(results)} code solutions")
    print("Starting evaluation...")
    
    # Evaluate each result
    evaluation_results = []
    total_passed = 0
    total_tests = 0
    
    for result in results:
        example = result["example"]
        code = result["code"]
        
        # Map to the format expected by lcb_run
        problem = map_to_example(example)
        
        # Post-process the code
        processed_code = post_process_code(code)
        
        # Run tests with timeout
        timeout = 10  # 10 seconds timeout per test case
        is_extracted = False  # We're not extracting from a larger response
        
        test_results = lcb_run(problem, processed_code, timeout, is_extracted)
        
        # Count passed tests
        passed_tests = sum(1 for passed, _, _, _ in test_results if passed)
        total_tests_for_problem = len(test_results)
        
        total_passed += passed_tests
        total_tests += total_tests_for_problem
        
        # Add evaluation results to the result
        result["evaluation"] = {
            "passed_tests": passed_tests,
            "total_tests": total_tests_for_problem,
            "test_results": test_results,
            "pass_rate": passed_tests / total_tests_for_problem if total_tests_for_problem > 0 else 0
        }
        
        evaluation_results.append(result)
    
    # Calculate overall metrics
    overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
    problems_with_all_tests_passed = sum(1 for r in evaluation_results if r["evaluation"]["passed_tests"] == r["evaluation"]["total_tests"])
    
    # Save evaluation results
    save_jsonl(evaluation_results, out_file.replace(".jsonl", "_evaluated.jsonl"))
    
    # Save summary with evaluation metrics
    summary = {
        "total_problems": len(results),
        "total_tests": total_tests,
        "total_passed_tests": total_passed,
        "overall_pass_rate": overall_pass_rate,
        "problems_with_all_tests_passed": problems_with_all_tests_passed,
        "problems_pass_rate": problems_with_all_tests_passed / len(results) if len(results) > 0 else 0,
        "model": args.model_name_or_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens_per_call,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(out_file.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"Evaluation completed!")
    print(f"Overall pass rate: {overall_pass_rate:.2%}")
    print(f"Problems with all tests passed: {problems_with_all_tests_passed}/{len(results)} ({problems_with_all_tests_passed/len(results):.2%})")
    print("Results saved to:", out_file.replace(".jsonl", "_evaluated.jsonl"))
    print("Summary saved to:", out_file.replace(".jsonl", "_summary.json"))

if __name__ == "__main__":
    main() 