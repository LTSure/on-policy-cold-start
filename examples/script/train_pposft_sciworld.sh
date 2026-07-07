#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MODEL="${MODEL:-${MODEL_NAME}}"
PROMPT_DATA="${PROMPT_DATA:-${PROJECT_ROOT}/data/sciworld_sft.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints}"
RUN_NAME="${RUN_NAME:-$(basename "${MODEL_NAME}")_opc_sft_sciworld}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-512}"
EPS_CLIP="${EPS_CLIP:-0.6}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_SAMPLES="${MAX_SAMPLES:-200000}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
NUM_EPISODES="${NUM_EPISODES:-5}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-4000}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-4000}"
ZERO_STAGE="${ZERO_STAGE:-3}"
ACTOR_LR="${ACTOR_LR:-5e-5}"
CRITIC_LR="${CRITIC_LR:-9e-6}"
SAVE_STEPS="${SAVE_STEPS:-200}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS+=(--use_wandb "${WANDB_API_KEY}")
fi

mkdir -p "${OUTPUT_DIR}"

deepspeed --include "localhost:${GPU_IDS}" --module openrlhf.cli.train_sft_ppo \
  --pretrain "${MODEL}" \
  --save_path "${OUTPUT_DIR}/${RUN_NAME}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}" \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --eps_clip "${EPS_CLIP}" \
  --temperature "${TEMPERATURE}" \
  --buffer_limit 0 \
  --n_samples_per_prompt 1 \
  --max_samples "${MAX_SAMPLES}" \
  --max_epochs "${MAX_EPOCHS}" \
  --num_episodes "${NUM_EPISODES}" \
  --prompt_max_len "${PROMPT_MAX_LEN}" \
  --generate_max_len "${GENERATE_MAX_LEN}" \
  --zero_stage "${ZERO_STAGE}" \
  --bf16 \
  --actor_learning_rate "${ACTOR_LR}" \
  --critic_learning_rate "${CRITIC_LR}" \
  --init_kl_coef 0.00 \
  --prompt_data "${PROMPT_DATA}" \
  --input_key conversations \
  --normalize_reward \
  --flash_attn \
  --adam_offload \
  --gradient_checkpointing \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps 1 \
  --wandb_run_name "${RUN_NAME}" \
  --ckpt_path "${OUTPUT_DIR}/${RUN_NAME}" \
  --max_ckpt_num 20000 \
  --use_muti_turn \
  "${WANDB_ARGS[@]}"

