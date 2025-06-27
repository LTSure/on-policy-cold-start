# RUN_NAME=Qwen2.5_Math_1.5B_sft_from_base_deepscaler_shortest——0517
# RUN_NAME=Qwen2.5_Math_1.5B_sft_from_base_deepscaler_max——0517
# RUN_NAME=Qwen2.5_Math_7B_sft_from_base_deepscaler_0_10000_shortest——0519

# RUN_NAME=Qwen2.5-7B_ppo_from_base_openr1
HOME="/cpfs04/user/liutianshuo/math/simpleRL-reason/train"
# MODEL="/oss/public/user/liuts/model/Qwen2.5_1.5B"
# MODEL="/oss/public/user/liuts/model/Qwen2.5_Math_7B"
# MODEL="/oss/public/user/liuts/model/Qwen2.5_Math_1.5B"
# MODEL="/oss/public/user/liuts/model/Qwen2.5_Math_7B"

# MODEL_NAME="Qwen2.5_Math_1.5B"
# MODEL_NAME="Qwen2.5_Math_7B"

MODEL_NAME="Llama-3.2-3B-Instruct"


MODEL="/oss/public/user/liuts/model/${MODEL_NAME}"



export PYTHONPATH=$PYTHONPATH:/cpfs04/user/liutianshuo/math/simpleRL-reason/train

# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge_new.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_max.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge-0-10000_filtered_no_templete_shortest.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_closed_3500_less_7000.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_2.json"
PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt_2.json"


RUN_NAME="${MODEL_NAME}_sft_alfworld_test——0625_2"


wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 

# 0625
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
    --max_epochs 1 \
    --num_episodes 5 \
    --prompt_max_len 4000 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-5 \
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


# # 0623
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 1 \
#     --train_batch_size 32 \
#     --micro_rollout_batch_size 1 \
#     --rollout_batch_size 256 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 4000 \
#     --generate_max_len 3000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
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



# # 0615 alfworld-gpt4-45k_2.json
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 16 \
#     --train_batch_size 512 \
#     --micro_rollout_batch_size 16 \
#     --rollout_batch_size 4096 \
#     --temperature 0.6 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 2700 \
#     --generate_max_len 1600 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key conversations \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps -1 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --no_ppo \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \




# # 0611
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --temperature 0.6 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps -1 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --no_ppo \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --temperature 0.6 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 1 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 5e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps -1 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --no_ppo \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
