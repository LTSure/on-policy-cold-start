#!/bin/bash

HOME="/cpfs04/user/liutianshuo/math/simpleRL-reason/train"

# Use the existing checkpoint as the pretrain model
MODEL="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Qwen2.5_Math_7B_sft_from_base_deepscaler——0529"

export PYTHONPATH=$PYTHONPATH:/cpfs04/user/liutianshuo/math/simpleRL-reason/train

# You can change this to your desired training data
PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/math_level3to5_data_processed_with_qwen_prompt.json"

# Login to wandb (you may want to update this key)
wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 

RUN_NAME="Qwen2.5_Math_7B_ppo_from_checkpoint_$(date +%m%d)"

# PPO training - simplified for Ray job execution
# Ray will handle the distributed setup automatically
python3 openrlhf/cli/train_ppo_ray_box.py \
    --pretrain $MODEL \
    --save_path $HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --temperature 0.6 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 20 \
    --prompt_max_len 1024 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data $PROMPT_DATA \
    --input_key input \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 4 \
    --load_checkpoint \
    --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HOME/checkpoints/$RUN_NAME \
    --max_ckpt_num 20000 