

HOME="/cpfs04/user/liutianshuo/math/simpleRL-reason/train"


# MODEL_NAME="Qwen2.5_Math_1.5B"
# MODEL_NAME="Qwen2.5_Math_7B"
# MODEL_NAME="Llama-3.2-3B-Instruct"

# MODEL_NAME="Qwen2.5-1.5B-instruct"
MODEL_NAME="Qwen2.5-7B-instruct"

MODEL="/oss/public/user/liuts/model/${MODEL_NAME}"


export PYTHONPATH=$PYTHONPATH:/cpfs04/user/liutianshuo/math/simpleRL-reason/train


PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/sciworld_train_2.json"

wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 

RUN_NAME="${MODEL_NAME}_sft_sciworld_0728_1"


# 7B instruct 1
deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
    --pretrain $MODEL \
    --save_path $HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 256 \
    --temperature 0.6 \
    --buffer_limit 0 \
    --n_samples_per_prompt 1 \
    --max_samples 200000 \
    --max_epochs 3 \
    --num_episodes 5 \
    --prompt_max_len 4000 \
    --generate_max_len 4000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 2e-5 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.00 \
    --entropy_coef 0.05 \
    --prompt_data  $PROMPT_DATA \
    --input_key conversations \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 200 \
    --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000 \
    --no_ppo \
    --use_muti_turn


# # llama3.2-3B
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 1 \
#     --train_batch_size 32 \
#     --micro_rollout_batch_size 1 \
#     --rollout_batch_size 256 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 4000 \
#     --generate_max_len 4000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.05 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key conversations \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --no_ppo \
#     --use_muti_turn






# # 1.5B instruct 1
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 1 \
#     --train_batch_size 32 \
#     --micro_rollout_batch_size 1 \
#     --rollout_batch_size 256 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 4000 \
#     --generate_max_len 4000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.05 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key conversations \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --no_ppo \
#     --use_muti_turn