#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pca_shift_analysis.py
---------------------
This script computes and saves PCA shift analysis between a base model and a fine-tuned model.
It extracts hidden state features, computes principal components on the base model,
and measures the shift in average projections on these components for the fine-tuned model.
"""

import argparse  # for parsing command-line arguments
import os        # for file and directory operations
import random    # for random sampling
import torch     # for PyTorch model handling and tensor operations
import numpy as np  # for numerical computations
import pandas as pd  # for DataFrame creation and manipulation
import matplotlib as mpl  # for Matplotlib configuration
import matplotlib.pyplot as plt  # for plotting

from transformers import AutoModelForCausalLM, AutoTokenizer  # for loading pre-trained language models
from sklearn.decomposition import PCA  # for principal component analysis
from datasets import load_dataset  # for loading datasets from Hugging Face
import json   # for reading and writing JSON
import gc     # for garbage collection

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Compute PCA shift between two model checkpoints")

parser.add_argument('--base_model', type=str, required=False, default="/oss/public/user/liuts/model/Llama-3.2-3B-Instruct")
parser.add_argument('--fine_tuned_model', type=str, required=False, default="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Llama-3.2-3B-Instruct_sft_alfworld_test——0623/_actor/3", 
                    help="Name or path of the base (original) model checkpoint. Default: SFT model path.")
# parser.add_argument('--fine_tuned_model', type=str, required=False, default="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Llama-3.2-3B-Instruct_ppo_sft_alfworld_muti_turn——0623/_actor/3", 
#                     help="Name or path of the fine-tuned or updated model checkpoint. Default: PPO-SFT model path.")
parser.add_argument('--task_type', type=str, required=False, default="GSM8K", 
                    help="Task identifier for selecting dataset configuration. Default: MATH500.")
parser.add_argument('--output_name', type=str, required=False, default="Llama-3.2-3B-sft_alfworld")
parser.add_argument('--k', type=int, default=30, required=False, 
                    help="Number of samples (queries) to draw from the dataset. Default: 30.")
args = parser.parse_args()

# Assign parsed arguments to variables
checkpoint_before = args.base_model  # path to the original model
checkpoint_after = args.fine_tuned_model  # path to the updated model
data_type = args.task_type  # dataset/task identifier
k = args.k  # number of examples to sample

# === Set random seeds and device configuration ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# === Dataset configuration mapping ===
# Each entry defines the dataset path, split, and relevant columns
DATASET_CFG = {
    "AIME24":  dict(path="math-ai/aime24",            split="test",          cols=["problem", "solution"]),
    "AIME25":  dict(path="math-ai/aime25",            split="test",          cols=["problem", "answer"]),
    "MATH500": dict(path="HuggingFaceH4/MATH-500",    split="test",          cols=["problem", "solution", "answer"]),
    "GSM8K":   dict(path="openai/gsm8k",                     subset="main", split="test",          cols=["question", "answer"]),
    "GPQA-D":  dict(path="Idavidrein/gpqa",           subset="gpqa_diamond", split="train",             cols=[
        "Pre-Revision Question",
        "Pre-Revision Correct Answer",
        "Pre-Revision Incorrect Answer 1",
        "Pre-Revision Incorrect Answer 2",
        "Pre-Revision Incorrect Answer 3",
        "Pre-Revision Explanation",
    ]),               
    "Livecodebench": dict(path="livecodebench/code_generation_lite",
                          split="test",
                          cols=['question_content','public_test_cases'],
                          kwargs=dict(version_tag="release_v5", cache_dir="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/data/livecodebench")),
    "ACPBench": dict(path="ibm-research/acp_bench",   subset="acp_app_bool", split="test",
                     cols=["context", "question", "answer"]),
    "SimpleQA":  dict(path="basicv8vc/SimpleQA",      split="test",          cols=["problem", "answer"]),
    "IFEval":    dict(path="google/IFEval",           split="train",         cols=["prompt"]),
    "TruthfulQA":dict(path="EleutherAI/truthful_qa_mc", subset="multiple_choice",
                      split="validation",             cols=["question", "choices", "label"]),
    "HalluEval": dict(path="pminervini/HaluEval",     subset="qa",          split="data",
                     cols=["knowledge", "dialogue_history", "right_response", "hallucinated_response"]),
    "SuperGPQA_Hard": dict(path="m-a-p/SuperGPQA", split="train",
                     cols=["question","options", "answer"]),
    "ZebraLogicBench":  dict(path="allenai/ZebraLogicBench",    subset="mc_mode",  split="test",   
                     cols=["puzzle", "question","choices"]),
    "COQA": dict(path="stanfordnlp/coqa", split="validation",
                     cols=["story","questions", "answers"]),
    "Head_QA": dict(path="head_qa" ,split="test",
                     cols=['qtext', 'ra', 'answers']),
    "Mc_Taco": dict(path="mc_taco" ,split="test",
                     cols=['question', 'answer']),  
    "Sciq": dict(path="allenai/sciq", split="train",
                     cols=["question","correct_answer", "support"]),
    "Commonsense_qa": dict(path="tau/commonsense_qa", split="train",
                     cols=["questions","choices", "answerKey"]),
    "Mt-bench": dict(path="philschmid/mt-bench", split="train",
                     cols=["turns"]),
}

# === Function to load and sample text data ===
def load_text_samples(task_type: str, k: int = 30):
    """
    Load dataset for the given task and return k randomly sampled text examples.
    Joins specified columns into a single string per example.
    """
    cfg = DATASET_CFG[task_type]
    kwargs = cfg.get("kwargs", {})
    subset = cfg.get("subset", None)
    
    # Load dataset from Hugging Face
    if subset:
        ds = load_dataset(cfg["path"], subset, split=cfg["split"],
                          **kwargs, trust_remote_code=True)
    else:
        ds = load_dataset(cfg["path"], split=cfg["split"],
                          **kwargs, trust_remote_code=True)

    # Select only the columns that exist in the dataset
    cols = [c for c in cfg["cols"] if c in ds.column_names]
    texts = ["  ".join(str(example[c]) for c in cols if example[c] is not None)
             for example in ds]

    random.shuffle(texts)
    return texts[:k]

# Special handling for certain task types
if data_type == "Olympia":
    # Read from a local JSONL file for the "Olympia" task
    file_path = "dataset/olympiadbench.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    sampled = random.sample(data, k)
    texts = [f"{item.get('question', '')} {item.get('solution', '')} {item.get('final_answer', '')}"
             for item in sampled]
elif data_type == "Mt-bench":
    # Use a larger sample size for Mt-bench
    texts = load_text_samples(data_type, k=80)
else:
    texts = load_text_samples(data_type, k=k)

# === Function to load a model checkpoint ===
def load_model(path):
    """
    Load a causal LM model in float16 on a single GPU to avoid device distribution issues.
    """
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None  # Dont use auto device mapping
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.half()
    return model.eval()

# === Function to extract mean hidden-state features ===
def extract_features(model, tokenizer, texts, layer_idx):
    """
    Tokenize input texts and extract mean pooled hidden states from a specific layer.
    Returns a NumPy array of shape (num_examples, hidden_size).
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True,
                       truncation=True, max_length=128)
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx].to(torch.float32)
    return hs.mean(dim=1).detach().cpu().numpy()

# Load tokenizer and ensure pad token is defined
tokenizer = AutoTokenizer.from_pretrained(checkpoint_before, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# === Function to compute centroid L2 distances ===
def compute_centroid_distance(df, states=["Updated"]):
    """
    Compute L2 distance between the centroid of 'Original' state and each specified state.
    """
    orig = df[df["state"] == "Original"][['shift', 'principle']].mean().values
    result = {}
    for state in states:
        if state in df["state"].unique():
            other = df[df["state"] == state][['shift', 'principle']].mean().values
            result[state] = np.linalg.norm(other - orig)
            print(result[state])
    return result

# === Phase 1: Extract PCA components from the base model ===
model_before = load_model(checkpoint_before)
num_layers = model_before.config.num_hidden_layers + 1
orig_stats = []  # to store principal components and means per layer

for layer in range(num_layers):
    # Extract features for the original model
    with torch.no_grad():
        feats_o = extract_features(model_before, tokenizer, texts, layer)

    # Perform PCA on CPU
    pca = PCA(n_components=2).fit(feats_o)
    comp1, comp2 = pca.components_[0], pca.components_[1]
    pc1_o = float(feats_o.dot(comp1).mean())
    pc2_o = float(feats_o.dot(comp2).mean())
    orig_stats.append((comp1, comp2, pc1_o, pc2_o))

    # Clean up to free memory
    del feats_o
    gc.collect()

# Clean up the base model
del model_before
torch.cuda.empty_cache()
gc.collect()

# === Phase 2: Compute shifts on the updated model ===
model_un = load_model(checkpoint_after)
records = []  # list to hold shift and principle values for each layer/state

for layer, (comp1, comp2, pc1_o, pc2_o) in enumerate(orig_stats):
    with torch.no_grad():
        feats_u = extract_features(model_un, tokenizer, texts, layer)

    # Compute new projections
    pc1_u = float(feats_u.dot(comp1).mean())
    pc2_u = float(feats_u.dot(comp2).mean())
    shift = pc1_u - pc1_o

    # Record original (no shift) and updated states
    records.append({"layer": layer, "state": "Original", "shift": 0.0, "principle": pc2_o})
    records.append({"layer": layer, "state": "Updated",  "shift": shift, "principle": pc2_u})

    # Free up memory
    del feats_u
    gc.collect()

# Clean up the updated model
del model_un
torch.cuda.empty_cache()
gc.collect()

# Create a DataFrame from the records
df = pd.DataFrame(records)

# Prepare JSON entry for output
new_entry = {
    "benchmark": f"{data_type}_PCA_Shift",
    "data": records
}

# Determine output file path based on checkpoint name
basename = args.output_name
task = args.task_type
output_dir = f"/cpfs04/user/liutianshuo/math/simpleRL-reason/Transferability-of-LLM-Reasoning/analyse_PCA_Shift/{task}"
output_path = f"{output_dir}/{basename}_pca_shift.json"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Append to existing data if file exists
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        existing_data = json.load(f)
    if isinstance(existing_data, list):
        existing_data.append(new_entry)
    else:
        existing_data = [existing_data, new_entry]
else:
    existing_data = [new_entry]

# Write final JSON data
with open(output_path, "w") as f:
    json.dump(existing_data, f, indent=2)

# Compute centroid distances
result_distance = compute_centroid_distance(df)
