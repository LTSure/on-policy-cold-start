# Clipping Low-Probability Tokens in SFT

This repository contains the training code for **Off-Policy token Clipped SFT (OPC-SFT)**, introduced in:

> Clipping Low-Probability Tokens in SFT Yields a Generalizable Initialization for RL. ICML 2026.

OPC-SFT clips token-wise policy-ratio updates during supervised fine-tuning. The method is designed to reduce overly large updates on tokens that have low probability under the base model, preserving prior knowledge while adapting to expert demonstrations.

The implementation is based on OpenRLHF and adds the OPC-SFT training path used for ALFWorld and ScienceWorld experiments.

## Repository Layout

```text
openrlhf/                         Core training code
openrlhf/cli/train_sft_ppo.py      OPC-SFT training entry point
openrlhf/models/loss.py            Token-wise clipped objective
examples/script/                   Reproduction scripts
data/                              Small packaged ALFWorld/ScienceWorld SFT data
recipes/                           DeepSpeed configs
```

## Installation

Use Python 3.10 and install this repository in editable mode so the local OPC-SFT changes are used.

```bash
conda create -n opc-sft python=3.10 -y
conda activate opc-sft

pip install -r requirements.txt
pip install -e .
```

For full GPU training with FlashAttention, install the optional CUDA-dependent package after PyTorch is correctly installed for your CUDA version:

```bash
pip install -r requirements-flash-attn.txt
```

**Note:** this repo requires `vllm==0.6.3`. Install it separately after PyTorch/CUDA are set up:

```bash
pip install vllm==0.6.3
```

Do not install `openrlhf` from PyPI on top of this checkout, because that may replace the local OPC-SFT implementation.

## Data

The repository includes packaged JSON examples under `data/`:

- `data/alfworld_sft.json`
- `data/sciworld_sft.json`

For full-scale reproduction, prepare the corresponding ALFWorld and ScienceWorld SFT datasets and pass their paths through `PROMPT_DATA`.

### ALFWorld version pinning (important for reproduction)

`data/alfworld_sft.json` and all ALFWorld evaluation prompts/ICL examples in this
project were written against the **ALFWorld 2.1.1 grammar**, i.e. actions such as:

```text
put {obj} in/on {recep}
toggle {obj} {recep}
```

Newer ALFWorld releases (`alfworld` package `0.4.0`/`0.4.2`, using the
`json_2.1.3_tw-pddl.zip` game-file overlay) patch this grammar to:

```text
move {obj} to {recep}
use {obj}
```

**If you download/generate the ALFWorld game data (`.tw-pddl` files) with the
newer grammar while training or evaluating with the SFT data/prompts in this
repo, the environment will reject almost every `put`/`toggle` action as
invalid ("Nothing happened"), and success rate will collapse (we observed
~7.86% instead of the expected ~72.86% success rate on the Qwen2.5-1.5B-Instruct
ALFWorld-Seen setting when this mismatch happened).**

To reproduce the paper's numbers, make sure the ALFWorld data is downloaded
with the grammar overlay matching `json_2.1.1_*`, e.g. via `alfworld-download`
with the `alfworld` package pinned to `0.2.2`, or by explicitly using:

```text
https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip
https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip
https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_tw-pddl.zip
```

You can sanity-check a downloaded `.tw-pddl` file with:

```bash
python -c "import json; g=json.load(open('<path>/game.tw-pddl'))['grammar']; \
print('OK (2.1.1 grammar)' if 'in/on' in g else 'WRONG VERSION: got move-based grammar')"
```

## Training

The main scripts are parameterized through environment variables and contain no machine-specific paths.

ALFWorld:

```bash
MODEL=meta-llama/Llama-3.2-3B-Instruct \
GPU_IDS=0,1,2,3 \
bash examples/script/train_pposft_alfworld.sh
```

ScienceWorld:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
GPU_IDS=0,1,2,3 \
bash examples/script/train_pposft_sciworld.sh
```

Common overrides:

```bash
PROMPT_DATA=/path/to/train.json \
OUTPUT_DIR=/path/to/checkpoints \
RUN_NAME=my_run \
EPS_CLIP=0.5 \
MAX_SAMPLES=200000 \
WANDB_API_KEY=... \
bash examples/script/train_pposft_alfworld.sh
```

If `WANDB_API_KEY` is omitted, Weights & Biases logging is disabled.

## Important Arguments

- `--eps_clip`: token-wise clipping range used by OPC-SFT.
- `--use_muti_turn`: enables the multi-turn formatting used by ALFWorld and ScienceWorld.
- `--no_ppo`: disables the OPC-SFT/PPO-style path and runs the SFT-only path.
- `--dft_baseline`: enables the confidence-based DFT baseline mode.
- `--neftune_alpha`: enables NEFTune embedding noise during training.

## Sanity Checks

Before launching multi-GPU training, run:

```bash
python -X pycache_prefix=.pycache_check -m compileall -q openrlhf
find examples/script -name "*.sh" -print0 | xargs -0 -n1 bash -n
```

If you are training or evaluating on ALFWorld, also verify the game-data
grammar version as described in [ALFWorld version pinning](#alfworld-version-pinning-important-for-reproduction)
above — this is the single most common cause of unreproducible/near-zero
ALFWorld success rates.

The full experiments require CUDA GPUs, model access on Hugging Face or local model paths, and enough disk space for checkpoints.

## Acknowledgements

This code builds on OpenRLHF. We also thank the maintainers of the ALFWorld and ScienceWorld benchmarks and the open datasets used in the paper experiments.

## Citation

```bibtex
@inproceedings{liu2026clipping,
  title={Clipping Low-Probability Tokens in SFT Yields a Generalizable Initialization for RL},
  author={Liu, Tian-Shuo and Jia, Chengxing and Liu, Haoyu and Wang, Pengyuan and Zhang, Shiyuan and Fu, Jie and Yu, Yang},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
