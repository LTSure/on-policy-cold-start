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

Do not install `openrlhf` from PyPI on top of this checkout, because that may replace the local OPC-SFT implementation.

## Data

The repository includes packaged JSON examples under `data/`:

- `data/alfworld_sft.json`
- `data/sciworld_sft.json`

For full-scale reproduction, prepare the corresponding ALFWorld and ScienceWorld SFT datasets and pass their paths through `PROMPT_DATA`.

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
